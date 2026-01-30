"""
Intent Classifier - Unified Classification + Symbol Normalization

Merges classification and symbol resolution into a SINGLE LLM call.
Eliminates the need for separate Router by determining complexity directly.

Key Benefits:
- Single LLM call (vs 3 in previous architecture)
- Symbol normalization happens IN the classification prompt
- Direct complexity determination (DIRECT vs AGENT_LOOP)
- Simpler, faster, production-ready

Architecture:
```
User Query → IntentClassifier (1 LLM call)
                    ↓
         IntentResult:
           - validated_symbols: ["GOOGL", "AMZN"]  ← Already normalized!
           - complexity: "direct" | "agent_loop"
           - market_type: "stock" | "crypto" | "both"
           - requires_tools: bool
           - reasoning: str

If AGENT_LOOP:
    → UnifiedAgent (sees ALL tools, THINK→ACT→OBSERVE loop)
If DIRECT:
    → Simple response (no tools needed)
```

Usage:
    classifier = IntentClassifier()
    result = await classifier.classify(query, ui_context)

    if result.complexity == "agent_loop":
        agent = UnifiedAgent()
        response = await agent.run_with_all_tools(...)
    else:
        response = await direct_response(...)
"""

import json
import re
import hashlib
import asyncio
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.helpers.redis_cache import redis_manager
from src.utils.circuit_breaker import CircuitBreakerOpenError, get_circuit_breaker


# ============================================================================
# ENUMS AND DATA MODELS
# ============================================================================

class IntentComplexity(str, Enum):
    """Execution complexity levels."""
    DIRECT = "direct"           # No tools needed, simple conversational response
    AGENT_LOOP = "agent_loop"   # Tools needed, use iterative agent loop


class IntentMarketType(str, Enum):
    """Market type for symbol interpretation."""
    STOCK = "stock"
    CRYPTO = "crypto"
    BOTH = "both"
    NONE = "none"


class AnalysisType(str, Enum):
    """Analysis type for skill selection.

    Determines which specialized skill/workflow to use for processing.
    """
    BASIC = "basic"              # Simple price/info queries
    TECHNICAL = "technical"      # Technical analysis (RSI, MACD, charts)
    FUNDAMENTAL = "fundamental"  # Fundamental analysis (financials, ratios)
    VALUATION = "valuation"      # DCF, Graham, DDM valuation queries
    PORTFOLIO = "portfolio"      # Portfolio analysis queries
    BACKTEST = "backtest"        # Strategy backtesting queries
    COMPARISON = "comparison"    # Comparative analysis
    GENERAL = "general"          # General/conversational


@dataclass
class IntentResult:
    """
    Result from IntentClassifier.

    Contains all information needed to route to agent or direct response.
    Symbols are ALREADY normalized (GOOGLE → GOOGL).
    """
    # Core intent
    intent_summary: str
    reasoning: str

    # Validated/normalized symbols
    validated_symbols: List[str] = field(default_factory=list)

    # Market type and complexity
    market_type: IntentMarketType = IntentMarketType.NONE
    complexity: IntentComplexity = IntentComplexity.DIRECT
    requires_tools: bool = False

    # Analysis type for skill selection (Phase 6)
    analysis_type: AnalysisType = AnalysisType.GENERAL

    # Response language
    response_language: str = "en"

    # Confidence
    confidence: float = 0.9

    # Query type for compatibility
    query_type: str = "general"

    # Metadata
    classification_method: str = "intent_classifier"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_summary": self.intent_summary,
            "reasoning": self.reasoning,
            "validated_symbols": self.validated_symbols,
            "market_type": self.market_type.value,
            "complexity": self.complexity.value,
            "requires_tools": self.requires_tools,
            "analysis_type": self.analysis_type.value,
            "response_language": self.response_language,
            "confidence": self.confidence,
            "query_type": self.query_type,
            "classification_method": self.classification_method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntentResult":
        """Create from dict (for cache deserialization)."""
        try:
            market_type = IntentMarketType(data.get("market_type", "none"))
        except ValueError:
            market_type = IntentMarketType.NONE

        try:
            complexity = IntentComplexity(data.get("complexity", "direct"))
        except ValueError:
            complexity = IntentComplexity.DIRECT

        try:
            analysis_type = AnalysisType(data.get("analysis_type", "general"))
        except ValueError:
            analysis_type = AnalysisType.GENERAL

        # Get values from data
        validated_symbols = data.get("validated_symbols", [])
        requires_tools = data.get("requires_tools", False)
        query_type = data.get("query_type", "general")

        # =====================================================================
        # VALIDATION: Force requires_tools=True when symbols + data query
        # Same validation as _parse_result to fix cached wrong classifications
        # =====================================================================
        tool_required_query_types = {
            "stock_analysis", "crypto_analysis", "comparison",
            "screener", "real_time_info", "fundamental_analysis",
            "technical_analysis", "news", "financial_data"
        }

        # Force for query_type that always needs tools (even without symbols)
        if not requires_tools and query_type in tool_required_query_types:
            requires_tools = True
            complexity = IntentComplexity.AGENT_LOOP

        if validated_symbols and not requires_tools:
            if query_type in tool_required_query_types:
                requires_tools = True
                complexity = IntentComplexity.AGENT_LOOP

            # Check intent summary for data-related keywords
            intent_lower = data.get("intent_summary", "").lower()
            data_keywords = [
                "price", "giá", "analysis", "phân tích", "report", "báo cáo",
                "financial", "tài chính", "earnings", "revenue", "doanh thu",
                "news", "tin tức", "chart", "biểu đồ", "technical", "kỹ thuật",
                "fundamental", "cơ bản", "compare", "so sánh", "data", "dữ liệu",
                "quote", "current", "hiện tại", "real-time", "retrieve", "lấy"
            ]
            if any(kw in intent_lower for kw in data_keywords):
                requires_tools = True
                complexity = IntentComplexity.AGENT_LOOP

        # Realtime data validation (weather, commodity prices, news, etc.)
        if not requires_tools:
            intent_lower = data.get("intent_summary", "").lower()
            realtime_keywords = [
                "today", "hôm nay", "now", "bây giờ", "hiện tại", "current",
                "latest", "mới nhất", "gần đây", "recent",
                "weather", "thời tiết", "nhiệt độ", "temperature",
                "gold price", "giá vàng", "oil price", "giá dầu",
                "exchange rate", "tỷ giá", "breaking news", "tin nóng",
            ]
            if any(kw in intent_lower for kw in realtime_keywords):
                requires_tools = True
                complexity = IntentComplexity.AGENT_LOOP
                if query_type in ("general_knowledge", "conversational", "general"):
                    query_type = "real_time_info"

        return cls(
            intent_summary=data.get("intent_summary", ""),
            reasoning=data.get("reasoning", ""),
            validated_symbols=validated_symbols,
            market_type=market_type,
            complexity=complexity,
            requires_tools=requires_tools,
            analysis_type=analysis_type,
            response_language=data.get("response_language", "en"),
            confidence=data.get("confidence", 0.9),
            query_type=query_type,
            classification_method=data.get("classification_method", "intent_classifier"),
        )

    @classmethod
    def fallback(cls, reason: str = "Classification failed") -> "IntentResult":
        """Create fallback result when classification fails."""
        return cls(
            intent_summary="Unknown intent",
            reasoning=f"Fallback: {reason}",
            complexity=IntentComplexity.DIRECT,
            requires_tools=False,
            confidence=0.5,
            classification_method="fallback",
        )

    # Compatibility properties for existing code
    @property
    def symbols(self) -> List[str]:
        """Alias for validated_symbols (compatibility)."""
        return self.validated_symbols

    @property
    def tool_categories(self) -> List[str]:
        """Return empty list - Agent sees ALL tools now."""
        return []


# ============================================================================
# CACHE
# ============================================================================

CACHE_KEY_PREFIX = "intent:"
CACHE_TTL_SECONDS = 300  # 5 minutes


class IntentCache(LoggerMixin):
    """Redis-backed cache for intent classification results."""

    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS):
        super().__init__()
        self.ttl_seconds = ttl_seconds
        self._local_cache: Dict[str, tuple] = {}
        self._local_max_size = 100

    def _make_key(
        self,
        query: str,
        ui_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create cache key from query and UI context."""
        normalized = query.lower().strip()
        ui_str = ""
        if ui_context:
            active_tab = ui_context.get("active_tab", "none")
            ui_str = f"|tab:{active_tab}"
        key_str = f"{normalized}{ui_str}"
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"{CACHE_KEY_PREFIX}{key_hash}"

    async def get(
        self,
        query: str,
        ui_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[IntentResult]:
        """Get cached result."""
        cache_key = self._make_key(query, ui_context)

        # Try Redis first
        try:
            redis_client = await redis_manager.get_client()
            if redis_client:
                cached_json = await asyncio.wait_for(
                    redis_client.get(cache_key),
                    timeout=3.0
                )
                if cached_json:
                    if isinstance(cached_json, bytes):
                        cached_json = cached_json.decode('utf-8')
                    data = json.loads(cached_json)
                    result = IntentResult.from_dict(data)
                    self.logger.debug(f"[INTENT_CACHE] Redis HIT: {cache_key[:20]}...")
                    return result
        except asyncio.TimeoutError:
            self.logger.warning(f"[INTENT_CACHE] Redis timeout")
        except Exception as e:
            self.logger.warning(f"[INTENT_CACHE] Redis error: {e}")

        # Fallback to local cache
        if cache_key in self._local_cache:
            result, timestamp = self._local_cache[cache_key]
            age = (datetime.now() - timestamp).total_seconds()
            if age <= self.ttl_seconds:
                self.logger.debug(f"[INTENT_CACHE] Local HIT")
                return result
            else:
                del self._local_cache[cache_key]

        return None

    async def set(
        self,
        query: str,
        result: IntentResult,
        ui_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Cache result."""
        cache_key = self._make_key(query, ui_context)
        json_str = json.dumps(result.to_dict(), ensure_ascii=False)

        # Try Redis
        try:
            redis_client = await redis_manager.get_client()
            if redis_client:
                await asyncio.wait_for(
                    redis_client.set(cache_key, json_str, ex=self.ttl_seconds),
                    timeout=3.0
                )
                self.logger.debug(f"[INTENT_CACHE] Redis SET: {cache_key[:20]}...")
                return
        except Exception as e:
            self.logger.warning(f"[INTENT_CACHE] Redis SET error: {e}")

        # Fallback to local
        if len(self._local_cache) >= self._local_max_size:
            oldest_key = min(self._local_cache, key=lambda k: self._local_cache[k][1])
            del self._local_cache[oldest_key]

        self._local_cache[cache_key] = (result, datetime.now())


# Global cache instance
_intent_cache = IntentCache()


# ============================================================================
# INTENT CLASSIFIER
# ============================================================================

class IntentClassifier(LoggerMixin):
    """
    Unified Intent Classifier with built-in symbol normalization.

    Single LLM call that:
    1. Understands user intent
    2. Extracts AND normalizes symbols (GOOGLE → GOOGL)
    3. Determines complexity (DIRECT vs AGENT_LOOP)
    4. Determines market type (stock/crypto/both)

    Key difference from old Classifier:
    - Symbol normalization is IN the prompt, not a separate call
    - No Router needed - complexity determined here
    - Agent sees ALL tools, not pre-filtered

    Usage:
        classifier = IntentClassifier()
        result = await classifier.classify(
            query="Phân tích Google và Amazon",
            ui_context={"active_tab": "stock"}
        )
        # result.validated_symbols = ["GOOGL", "AMZN"]  ← Already normalized!
    """

    # Use a capable model for accurate classification
    DEFAULT_MODEL = "gpt-4.1-mini"
    DEFAULT_PROVIDER = ProviderType.OPENAI

    # Fallback providers when primary circuit breaker is open
    # Order matters: first available will be used
    FALLBACK_PROVIDERS = [
        {"provider": ProviderType.GEMINI, "model": "gemini-2.0-flash"},
        # Add more fallback providers here as needed
    ]

    def __init__(
        self,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None,
        max_retries: int = 2,
    ):
        super().__init__()

        self.model_name = model_name or settings.CLASSIFIER_MODEL or self.DEFAULT_MODEL
        self.provider_type = provider_type or settings.CLASSIFIER_PROVIDER or self.DEFAULT_PROVIDER
        self.max_retries = max_retries

        self.llm_provider = LLMGeneratorProvider()
        self.api_key = ModelProviderFactory._get_api_key(self.provider_type)

        self.logger.info(
            f"[INTENT_CLASSIFIER] Initialized: model={self.model_name}, provider={self.provider_type}"
        )

    async def classify(
        self,
        query: str,
        ui_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_cache: bool = True,
        working_memory_symbols: Optional[List[str]] = None,
        core_memory_context: Optional[str] = None,
        conversation_summary: Optional[str] = None,
    ) -> IntentResult:
        """
        Classify query and extract normalized symbols in a single LLM call.

        Args:
            query: User query
            ui_context: UI context (active_tab, recent_symbols)
            conversation_history: Recent K messages for immediate context
            use_cache: Whether to use caching
            working_memory_symbols: Symbols from previous turns (Working Memory)
            core_memory_context: User profile from Core Memory (portfolio, preferences)
            conversation_summary: Summary of older messages (compressed context)

        Returns:
            IntentResult with validated_symbols already normalized
        """
        start_time = datetime.now()

        # Check cache first - BUT skip cache if working_memory_symbols exist
        # This ensures context-dependent queries are re-classified
        if use_cache and not working_memory_symbols:
            cached = await _intent_cache.get(query, ui_context)
            if cached:
                elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                self._log_result(cached, elapsed_ms, is_cached=True)
                return cached

        try:
            # Build prompt with symbol normalization instructions
            prompt = self._build_prompt(
                query, ui_context, conversation_history,
                working_memory_symbols, core_memory_context,
                conversation_summary
            )

            # Call LLM with primary provider
            result_data = await self._call_llm_with_fallback(prompt)

            # Parse result
            result = self._parse_result(result_data)

            # =====================================================================
            # CRITICAL FIX: Inherit symbols from working_memory when query uses
            # reference words but LLM didn't extract symbols
            # =====================================================================
            if not result.validated_symbols and working_memory_symbols:
                query_lower = query.lower()
                # Reference words that indicate user is talking about previous context
                reference_patterns = [
                    "công ty", "cổ phiếu", "mã", "symbol", "ticker",
                    "nó", "này", "đó", "the company", "the stock", "it",
                    "this", "that", "these", "those", "them",
                    "xem tiếp", "phân tích tiếp", "cho tôi xem", "show me",
                    "báo cáo", "report", "financial", "tài chính",
                    "price", "giá", "chart", "biểu đồ"
                ]

                # Check if query uses reference words without explicit symbol
                uses_reference = any(ref in query_lower for ref in reference_patterns)

                # Also check if query has NO uppercase words (potential symbols)
                has_explicit_symbol = bool(re.search(r'\b[A-Z]{2,5}\b', query))

                if uses_reference and not has_explicit_symbol:
                    self.logger.warning(
                        f"[INTENT_CLASSIFIER] Inheriting symbols from working memory: "
                        f"{working_memory_symbols} (query uses reference words without explicit symbol)"
                    )
                    result.validated_symbols = working_memory_symbols.copy()
                    # Also ensure requires_tools is True since we have symbols now
                    if result.validated_symbols:
                        result.requires_tools = True
                        result.complexity = IntentComplexity.AGENT_LOOP

            # Log
            elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self._log_result(result, elapsed_ms, is_cached=False)

            # Cache asynchronously - only if no working_memory context
            if use_cache and not working_memory_symbols:
                asyncio.create_task(_intent_cache.set(query, result, ui_context))

            return result

        except Exception as e:
            self.logger.error(f"[INTENT_CLASSIFIER] Error: {e}", exc_info=True)
            # Use heuristic fallback instead of simple fallback
            return self._heuristic_fallback(query, ui_context, working_memory_symbols, str(e))

    def _build_prompt(
        self,
        query: str,
        ui_context: Optional[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]],
        working_memory_symbols: Optional[List[str]] = None,
        core_memory_context: Optional[str] = None,
        conversation_summary: Optional[str] = None,
    ) -> str:
        """Build classification prompt with symbol normalization and memory context."""

        current_date = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        # UI context hint
        ui_hint = ""
        if ui_context:
            active_tab = ui_context.get("active_tab", "none")
            recent_ui_symbols = ui_context.get("recent_symbols", [])
            if active_tab != "none" or recent_ui_symbols:
                ui_hint = f"""
<ui_context>
Active Tab: {active_tab}
Recent UI Symbols: {recent_ui_symbols if recent_ui_symbols else 'None'}
IMPORTANT: When Active Tab is "{active_tab}", interpret ambiguous symbols accordingly:
- If "stock": BTC = Grayscale Bitcoin Trust (stock), SOL = stock symbol
- If "crypto": BTC = Bitcoin cryptocurrency, SOL = Solana cryptocurrency
</ui_context>
"""

        # Working Memory Symbols (CRITICAL for cross-turn context)
        wm_hint = ""
        if working_memory_symbols and len(working_memory_symbols) > 0:
            # Symbols are in chronological order: oldest first, newest LAST
            most_recent = working_memory_symbols[-1]  # LAST symbol is MOST RECENT
            older_symbols = working_memory_symbols[:-1] if len(working_memory_symbols) > 1 else []

            # Format with clear chronological indication
            symbols_timeline = ""
            if older_symbols:
                symbols_timeline = f"Older symbols (in order): {' → '.join(older_symbols)}\n"
            symbols_timeline += f"**MOST RECENT SYMBOL: {most_recent}** ← This is the symbol from the LAST user query"

            wm_hint = f"""
<working_memory>
{symbols_timeline}

CRITICAL: The symbols are listed in CHRONOLOGICAL ORDER.
- When user says "symbol gần nhất" (most recent symbol) → Use: {most_recent}
- When user says "mã đó", "nó", "this stock", "công ty này" without explicit name → Use: {most_recent}
- When user asks about "các mã đã hỏi" (previously asked symbols) → Use all: {', '.join(working_memory_symbols)}
</working_memory>
"""

        # Core Memory context (user profile)
        cm_hint = ""
        if core_memory_context:
            cm_hint = f"""
<user_profile>
{core_memory_context[:500]}
</user_profile>
"""

        # Conversation Summary (compressed context from older messages)
        summary_hint = ""
        if conversation_summary:
            summary_hint = f"""
<conversation_summary>
(Summary of earlier messages in this session - use for understanding broader context)
{conversation_summary[:1000]}
</conversation_summary>
"""

        # History context - Format with clear chronological ordering
        history_hint = ""
        if conversation_history and len(conversation_history) > 0:
            recent = conversation_history[-5:]  # Last 5 messages
            total_msgs = len(recent)

            # Format with turn numbers to show chronological order
            history_lines = []
            for i, msg in enumerate(recent):
                turn_label = f"[Turn {i + 1}/{total_msgs}]"
                if i == total_msgs - 1:
                    turn_label = f"[Turn {i + 1}/{total_msgs} - MOST RECENT]"
                role = msg.get('role', 'user').upper()
                content = msg.get('content', '')[:500]
                history_lines.append(f"{turn_label} {role}: {content}")

            history_text = "\n".join(history_lines)
            history_hint = f"""
<conversation_history>
(Listed in chronological order, oldest first, newest last)
{history_text}
</conversation_history>
"""

        return f"""<current_context>
Date: {current_date}
Data Status: Real-time market data available
</current_context>

You are a financial intent classifier. Analyze the user query and provide a structured classification.

<query>{query}</query>

{ui_hint}
{wm_hint}
{cm_hint}
{summary_hint}
{history_hint}

<instructions>
## STEP 1: Understand Intent
What does the user want? Be specific about their goal.

## STEP 2: Extract and NORMALIZE Symbols
Extract any financial symbols mentioned and convert to official ticker format:

CRITICAL SYMBOL NORMALIZATION RULES:
- Company names → Tickers: "Google" → "GOOGL", "Amazon" → "AMZN", "Microsoft" → "MSFT"
- "Apple" → "AAPL", "Tesla" → "TSLA", "Netflix" → "NFLX", "Meta" → "META"
- "Nvidia" → "NVDA", "AMD" → "AMD", "Intel" → "INTC"
- Vietnamese: "Vinamilk" → "VNM", "Hòa Phát" → "HPG", "FPT" → "FPT"
- Crypto (if crypto context): "Bitcoin" → "BTC", "Ethereum" → "ETH", "Solana" → "SOL"
- Keep already-correct tickers: "AAPL" → "AAPL", "NVDA" → "NVDA"
- Handle variations: "NVIDIA", "nvidia", "NVidia" → "NVDA"
- Special formats: "Berkshire" → "BRK-B", "BRK.A" → "BRK-A"

If unsure about a symbol, make your best guess based on context.

## STEP 3: Determine Market Type (CRITICAL - USE UI CONTEXT!)

**RULE 1: UI Tab Context is the STRONGEST signal for ambiguous symbols:**
- If active_tab = "stock" → BTC = Grayscale Bitcoin Trust (stock), SOL = stock symbol
- If active_tab = "crypto" → BTC = Bitcoin cryptocurrency, SOL = Solana cryptocurrency
- If active_tab = "none" or missing → Use context clues from query

**RULE 2: Market Type Classification:**
- "stock": Stocks, ETFs, indices (AAPL, NVDA, SPY, VNM, HPG)
- "crypto": Cryptocurrencies (BTC, ETH, SOL when crypto context)
- "both": Query explicitly involves BOTH stock and crypto (e.g., "compare NVDA with BTC")
- "none": ONLY for non-market queries (greetings, general knowledge)

**RULE 3: If symbols are present, market_type MUST NOT be "none":**
- If validated_symbols is not empty, market_type must be "stock", "crypto", or "both"
- Default to "stock" if unsure but symbols are present

**RULE 4: Ambiguous Symbols Reference:**
| Symbol | Stock Meaning | Crypto Meaning |
|--------|--------------|----------------|
| BTC | Grayscale Bitcoin Mini Trust | Bitcoin |
| ETH | Ethan Allen Interiors | Ethereum |
| SOL | Emeren Group Ltd | Solana |
| COIN | Coinbase Global | (Not a crypto) |

**ALWAYS check active_tab first for these ambiguous symbols!**

## STEP 4: Determine Complexity
"direct" - NO tools needed:
- Greetings, thanks, goodbye
- Static financial concepts (what is P/E ratio?)
- General knowledge that NEVER changes (definitions, formulas, historical facts with specific dates)

"agent_loop" - Tools NEEDED:
- Real-time prices, quotes, market data
- Technical analysis (RSI, MACD, etc.)
- Fundamental data (earnings, financials)
- News and current events (ANY news query)
- Stock screening
- Any query about current state of markets
- **Weather, temperature, climate conditions** (requires real-time data)
- **Commodity prices** (gold/vàng, oil/dầu, silver/bạc - requires real-time data)
- **Exchange rates, forex** (requires real-time data)
- **Any question implying "today", "now", "current", "latest", "recent", "hôm nay", "hiện tại", "mới nhất", "gần đây"**
- **Any question where the answer changes over time** (e.g., "who is the CEO of X?", "what is the GDP of Y?")

## STEP 5: Determine Analysis Type (CRITICAL for skill routing)
Classify what TYPE of analysis the user wants:

- "basic": Simple price/quote queries, basic stock info
  Examples: "Giá AAPL?", "What's NVDA trading at?", "Show me MSFT"

- "technical": Technical analysis with indicators (RSI, MACD, Bollinger, etc.)
  Examples: "RSI của AAPL", "Phân tích kỹ thuật NVDA", "MACD signals for TSLA"

- "fundamental": Fundamental analysis (P/E, financials, earnings, revenue)
  Examples: "P/E ratio của AAPL", "Báo cáo tài chính NVDA", "MSFT earnings report"

- "valuation": Stock valuation calculations (DCF, Graham, DDM)
  Examples: "Định giá AAPL theo DCF", "Graham number cho NVDA", "Fair value of MSFT"

- "portfolio": Portfolio analysis (optimization, correlation, diversification)
  Examples: "Phân tích danh mục", "Optimize my portfolio", "Portfolio correlation"

- "backtest": Strategy backtesting queries
  Examples: "Backtest SMA crossover", "Test RSI strategy on AAPL", "Compare strategies"

- "comparison": Comparing multiple stocks or strategies
  Examples: "So sánh AAPL và NVDA", "Compare tech stocks", "MSFT vs GOOGL"

- "general": General knowledge, concepts, greetings, non-financial queries
  Examples: "What is P/E?", "Hello", "Thanks"

## STEP 6: Detect Language
"vi" for Vietnamese, "en" for English, "zh" for Chinese

</instructions>

<output_format>
First, provide your reasoning inside <reasoning> tags.
Then, provide the classification inside <classification> tags as valid JSON:

<reasoning>
[Your step-by-step analysis]
</reasoning>

<classification>
{{
  "intent_summary": "Brief description of user's goal",
  "validated_symbols": ["SYMBOL1", "SYMBOL2"],
  "market_type": "stock|crypto|both|none",
  "complexity": "direct|agent_loop",
  "requires_tools": true|false,
  "analysis_type": "basic|technical|fundamental|valuation|portfolio|backtest|comparison|general",
  "response_language": "vi|en|zh",
  "query_type": "stock_analysis|crypto_analysis|comparison|screener|conversational|general_knowledge|real_time_info",
  "confidence": 0.0-1.0
}}
</classification>
</output_format>

<few_shot_examples>
Example 1 - Technical Analysis (Symbol Normalization):
Query: "Phân tích kỹ thuật Google và Amazon"
<reasoning>
User wants technical analysis of Google and Amazon stocks.
- "Google" → normalized to "GOOGL" (official ticker)
- "Amazon" → normalized to "AMZN" (official ticker)
This requires technical indicators (RSI, MACD, etc.), so analysis_type is "technical".
</reasoning>
<classification>
{{"intent_summary": "Technical analysis of Google and Amazon stocks", "validated_symbols": ["GOOGL", "AMZN"], "market_type": "stock", "complexity": "agent_loop", "requires_tools": true, "analysis_type": "technical", "response_language": "vi", "query_type": "stock_analysis", "confidence": 0.95}}
</classification>

Example 2 - Greeting (No Tools):
Query: "Xin chào"
<reasoning>
Simple greeting. No financial query. No tools needed. Analysis type is "general".
</reasoning>
<classification>
{{"intent_summary": "Greeting", "validated_symbols": [], "market_type": "none", "complexity": "direct", "requires_tools": false, "analysis_type": "general", "response_language": "vi", "query_type": "conversational", "confidence": 0.99}}
</classification>

Example 3 - Comparison Analysis:
Query: "So sánh NVDA, Apple và Microsoft"
<reasoning>
User wants comparison of 3 stocks.
- "NVDA" → already correct ticker
- "Apple" → normalized to "AAPL"
- "Microsoft" → normalized to "MSFT"
Comparison of multiple stocks → analysis_type is "comparison".
</reasoning>
<classification>
{{"intent_summary": "Compare NVIDIA, Apple, and Microsoft stocks", "validated_symbols": ["NVDA", "AAPL", "MSFT"], "market_type": "stock", "complexity": "agent_loop", "requires_tools": true, "analysis_type": "comparison", "response_language": "vi", "query_type": "comparison", "confidence": 0.94}}
</classification>

Example 4 - General Knowledge (No Tools):
Query: "P/E ratio là gì?"
<reasoning>
User asks about P/E ratio definition - a static concept that doesn't need live data.
This is general knowledge, not requiring data retrieval.
</reasoning>
<classification>
{{"intent_summary": "Explain P/E ratio concept", "validated_symbols": [], "market_type": "none", "complexity": "direct", "requires_tools": false, "analysis_type": "general", "response_language": "vi", "query_type": "general_knowledge", "confidence": 0.95}}
</classification>

Example 5 - Basic Price Query:
Query: "Giá BTC" (UI Context: active_tab = crypto)
<reasoning>
User asks about BTC price. UI context shows crypto tab, so BTC = Bitcoin cryptocurrency.
Simple price query → analysis_type is "basic".
</reasoning>
<classification>
{{"intent_summary": "Bitcoin cryptocurrency price", "validated_symbols": ["BTC"], "market_type": "crypto", "complexity": "agent_loop", "requires_tools": true, "analysis_type": "basic", "response_language": "vi", "query_type": "crypto_analysis", "confidence": 0.93}}
</classification>

Example 6 - Fundamental Analysis:
Query: "Báo cáo tài chính của Apple"
<reasoning>
User wants Apple's financial report - this is fundamental analysis.
- "Apple" → normalized to "AAPL"
Financial statements require fundamental data tools.
</reasoning>
<classification>
{{"intent_summary": "Apple financial report and fundamental analysis", "validated_symbols": ["AAPL"], "market_type": "stock", "complexity": "agent_loop", "requires_tools": true, "analysis_type": "fundamental", "response_language": "vi", "query_type": "fundamental_analysis", "confidence": 0.94}}
</classification>

Example 7 - Valuation Query:
Query: "Định giá NVDA theo phương pháp DCF"
<reasoning>
User wants DCF valuation for NVIDIA.
- "NVDA" → already correct ticker
DCF is a valuation method → analysis_type is "valuation".
</reasoning>
<classification>
{{"intent_summary": "DCF valuation of NVIDIA stock", "validated_symbols": ["NVDA"], "market_type": "stock", "complexity": "agent_loop", "requires_tools": true, "analysis_type": "valuation", "response_language": "vi", "query_type": "stock_analysis", "confidence": 0.95}}
</classification>

Example 8 - Backtest Query:
Query: "Backtest chiến lược SMA crossover trên Tesla"
<reasoning>
User wants to backtest SMA crossover strategy on Tesla.
- "Tesla" → normalized to "TSLA"
Backtesting a trading strategy → analysis_type is "backtest".
</reasoning>
<classification>
{{"intent_summary": "Backtest SMA crossover strategy on Tesla stock", "validated_symbols": ["TSLA"], "market_type": "stock", "complexity": "agent_loop", "requires_tools": true, "analysis_type": "backtest", "response_language": "vi", "query_type": "stock_analysis", "confidence": 0.94}}
</classification>

Example 9 - Portfolio Analysis:
Query: "Phân tích danh mục đầu tư của tôi gồm AAPL, NVDA, MSFT"
<reasoning>
User wants portfolio analysis of their holdings.
All symbols are already correct tickers.
Portfolio analysis → analysis_type is "portfolio".
</reasoning>
<classification>
{{"intent_summary": "Portfolio analysis of AAPL, NVDA, MSFT holdings", "validated_symbols": ["AAPL", "NVDA", "MSFT"], "market_type": "stock", "complexity": "agent_loop", "requires_tools": true, "analysis_type": "portfolio", "response_language": "vi", "query_type": "stock_analysis", "confidence": 0.93}}
</classification>

Example 10 - Context Reference:
Query: "Phân tích công ty này" (Working Memory: NVDA mentioned previously)
<reasoning>
User refers to "công ty này" (this company). Working memory shows NVDA was discussed.
NVDA is a stock. General analysis request → analysis_type is "technical" (default for stock analysis).
</reasoning>
<classification>
{{"intent_summary": "Analysis of NVIDIA stock (referenced from context)", "validated_symbols": ["NVDA"], "market_type": "stock", "complexity": "agent_loop", "requires_tools": true, "analysis_type": "technical", "response_language": "vi", "query_type": "stock_analysis", "confidence": 0.88}}
</classification>

Example 11 - Real-time Information (Weather):
Query: "Thời tiết hôm nay ở thành phố Hồ Chí Minh"
<reasoning>
User asks about today's weather in Ho Chi Minh City. Weather is real-time data that changes constantly. This REQUIRES tools (web search) to get current, accurate information. The model's training data does NOT contain today's weather. This is NOT general knowledge - it is real-time information.
</reasoning>
<classification>
{{"intent_summary": "Current weather in Ho Chi Minh City", "validated_symbols": [], "market_type": "none", "complexity": "agent_loop", "requires_tools": true, "analysis_type": "general", "response_language": "vi", "query_type": "real_time_info", "confidence": 0.95}}
</classification>

Example 12 - Real-time Information (Commodity Price):
Query: "Giá vàng hôm nay"
<reasoning>
User asks about today's gold price. Gold price changes in real-time and requires web search or data tools to get current, accurate price. The model cannot know today's gold price from training data alone.
</reasoning>
<classification>
{{"intent_summary": "Current gold price today", "validated_symbols": [], "market_type": "none", "complexity": "agent_loop", "requires_tools": true, "analysis_type": "general", "response_language": "vi", "query_type": "real_time_info", "confidence": 0.95}}
</classification>

Example 13 - Real-time Information (News):
Query: "Tin tức mới nhất về kinh tế Việt Nam"
<reasoning>
User asks for latest economic news about Vietnam. News is time-sensitive real-time information. MUST use tools (web search) to get current, up-to-date news. Model's training data is outdated for this.
</reasoning>
<classification>
{{"intent_summary": "Latest economic news about Vietnam", "validated_symbols": [], "market_type": "none", "complexity": "agent_loop", "requires_tools": true, "analysis_type": "general", "response_language": "vi", "query_type": "real_time_info", "confidence": 0.94}}
</classification>
</few_shot_examples>

Now analyze the query and provide your classification:
"""

    async def _call_llm(
        self,
        prompt: str,
        provider_type: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call LLM and parse response.

        Args:
            prompt: The classification prompt
            provider_type: Override provider (for fallback)
            model_name: Override model name (for fallback)

        Returns:
            Parsed classification result dict

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            RuntimeError: If all retries failed
        """
        use_provider = provider_type or self.provider_type
        use_model = model_name or self.model_name
        use_api_key = ModelProviderFactory._get_api_key(use_provider)

        for attempt in range(self.max_retries + 1):
            try:
                params = {
                    "model_name": use_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a financial intent classifier. "
                                "Provide structured output with <reasoning> and <classification> tags."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "provider_type": use_provider,
                    "api_key": use_api_key,
                    "max_tokens": 1000,
                    "temperature": 0.1,
                }

                response = await self.llm_provider.generate_response(**params)
                content = (
                    response.get("content", "")
                    if isinstance(response, dict)
                    else str(response)
                )
                content = content.strip()

                # Extract reasoning
                reasoning = self._extract_tag_content(content, "reasoning") or ""

                # Extract classification JSON
                json_str = self._extract_tag_content(content, "classification")
                if not json_str:
                    json_str = self._extract_json(content)

                if not json_str:
                    raise ValueError("No classification found in response")

                result = json.loads(json_str)
                result["reasoning"] = reasoning
                return result

            except CircuitBreakerOpenError:
                # Don't retry on circuit breaker - propagate immediately
                raise
            except json.JSONDecodeError as e:
                self.logger.warning(
                    f"[INTENT_CLASSIFIER] JSON parse error (attempt {attempt + 1}): {e}"
                )
                if attempt == self.max_retries:
                    raise
            except Exception as e:
                self.logger.warning(
                    f"[INTENT_CLASSIFIER] Error (attempt {attempt + 1}): {e}"
                )
                if attempt == self.max_retries:
                    raise

        raise RuntimeError("Failed to classify after retries")

    async def _call_llm_with_fallback(self, prompt: str) -> Dict[str, Any]:
        """
        Call LLM with automatic fallback to secondary providers.

        Tries primary provider first. If circuit breaker is open,
        automatically tries fallback providers in order.

        Args:
            prompt: The classification prompt

        Returns:
            Parsed classification result dict

        Raises:
            CircuitBreakerOpenError: If ALL providers' circuits are open
            Exception: If all providers fail with other errors
        """
        # Track which providers we've tried
        providers_tried = []
        last_error = None

        # 1. Try primary provider first
        try:
            self.logger.debug(
                f"[INTENT_CLASSIFIER] Trying primary provider: {self.provider_type}"
            )
            return await self._call_llm(prompt)
        except CircuitBreakerOpenError as e:
            providers_tried.append(self.provider_type)
            last_error = e
            self.logger.warning(
                f"[INTENT_CLASSIFIER] Primary provider circuit open: {e.service_name}, "
                f"trying fallback providers..."
            )
        except Exception as e:
            # For non-circuit-breaker errors, don't fallback (could be prompt issue)
            raise

        # 2. Try fallback providers
        for fallback in self.FALLBACK_PROVIDERS:
            fallback_provider = fallback["provider"]
            fallback_model = fallback["model"]

            # Skip if same as primary
            if fallback_provider == self.provider_type:
                continue

            try:
                self.logger.info(
                    f"[INTENT_CLASSIFIER] Trying fallback provider: {fallback_provider} "
                    f"with model: {fallback_model}"
                )
                result = await self._call_llm(
                    prompt,
                    provider_type=fallback_provider,
                    model_name=fallback_model,
                )
                self.logger.info(
                    f"[INTENT_CLASSIFIER] Fallback provider succeeded: {fallback_provider}"
                )
                return result

            except CircuitBreakerOpenError as e:
                providers_tried.append(fallback_provider)
                last_error = e
                self.logger.warning(
                    f"[INTENT_CLASSIFIER] Fallback provider circuit also open: {e.service_name}"
                )
                continue
            except Exception as e:
                # Log but continue to next fallback
                self.logger.warning(
                    f"[INTENT_CLASSIFIER] Fallback provider {fallback_provider} failed: {e}"
                )
                last_error = e
                continue

        # 3. All providers failed
        providers_str = ", ".join(str(p) for p in providers_tried)
        self.logger.error(
            f"[INTENT_CLASSIFIER] All providers failed. Tried: {providers_str}"
        )

        # Re-raise the last error (preferably CircuitBreakerOpenError for clarity)
        if last_error:
            raise last_error
        raise RuntimeError(f"All classification providers failed: {providers_str}")

    def _parse_result(self, data: Dict[str, Any]) -> IntentResult:
        """Parse LLM response into IntentResult."""

        # Parse complexity
        complexity_str = data.get("complexity", "direct").lower()
        try:
            complexity = IntentComplexity(complexity_str)
        except ValueError:
            complexity = IntentComplexity.DIRECT

        # Parse market type
        market_str = data.get("market_type", "none").lower()
        try:
            market_type = IntentMarketType(market_str)
        except ValueError:
            market_type = IntentMarketType.NONE

        # Parse analysis type (Phase 6)
        analysis_str = data.get("analysis_type", "general").lower()
        try:
            analysis_type = AnalysisType(analysis_str)
        except ValueError:
            analysis_type = AnalysisType.GENERAL

        # Extract validated symbols (already normalized by LLM)
        validated_symbols = data.get("validated_symbols", [])
        if not isinstance(validated_symbols, list):
            validated_symbols = []

        # Clean symbols (uppercase, remove invalid chars)
        validated_symbols = [
            s.upper().strip()
            for s in validated_symbols
            if isinstance(s, str) and s.strip()
        ]

        # Get requires_tools from LLM
        requires_tools = data.get("requires_tools", False)
        query_type = data.get("query_type", "general")

        # =====================================================================
        # VALIDATION: Force requires_tools=True when symbols + data query
        # This catches LLM classification errors where it incorrectly says
        # tools are not needed for queries that clearly need real-time data.
        # =====================================================================
        tool_required_query_types = {
            "stock_analysis", "crypto_analysis", "comparison",
            "screener", "real_time_info", "fundamental_analysis",
            "technical_analysis", "news", "financial_data"
        }

        # VALIDATION 1: query_type indicates tools needed
        if not requires_tools and query_type in tool_required_query_types:
            self.logger.warning(
                f"[INTENT_CLASSIFIER] Forcing requires_tools=True "
                f"(query_type={query_type})"
            )
            requires_tools = True
            complexity = IntentComplexity.AGENT_LOOP

        if validated_symbols and not requires_tools:
            # If symbols are present AND query type suggests data retrieval
            if query_type in tool_required_query_types:
                self.logger.warning(
                    f"[INTENT_CLASSIFIER] Forcing requires_tools=True "
                    f"(symbols={validated_symbols}, query_type={query_type})"
                )
                requires_tools = True
                complexity = IntentComplexity.AGENT_LOOP

            # Also check intent summary for data-related keywords
            intent_lower = data.get("intent_summary", "").lower()
            data_keywords = [
                "price", "giá", "analysis", "phân tích", "report", "báo cáo",
                "financial", "tài chính", "earnings", "revenue", "doanh thu",
                "news", "tin tức", "chart", "biểu đồ", "technical", "kỹ thuật",
                "fundamental", "cơ bản", "compare", "so sánh", "data", "dữ liệu",
                "quote", "current", "hiện tại", "real-time", "retrieve", "lấy"
            ]
            if any(kw in intent_lower for kw in data_keywords):
                if not requires_tools:
                    self.logger.warning(
                        f"[INTENT_CLASSIFIER] Forcing requires_tools=True based on intent keywords "
                        f"(intent={intent_lower[:50]}...)"
                    )
                    requires_tools = True
                    complexity = IntentComplexity.AGENT_LOOP

        # =====================================================================
        # VALIDATION 2: Force requires_tools=True for realtime-data queries
        # Queries about weather, commodity prices, news, current events, etc.
        # ALWAYS need tools (web search) even without financial symbols.
        # The model's training data cannot answer these accurately.
        # =====================================================================
        if not requires_tools:
            intent_lower = data.get("intent_summary", "").lower()
            query_lower = data.get("_original_query", "").lower() if "_original_query" in data else intent_lower

            realtime_keywords = [
                # Time-sensitive indicators
                "today", "hôm nay", "now", "bây giờ", "hiện tại", "current",
                "latest", "mới nhất", "gần đây", "recent", "this week", "tuần này",
                "this month", "tháng này", "tonight", "tối nay",
                # Weather
                "weather", "thời tiết", "nhiệt độ", "temperature", "forecast",
                "dự báo", "mưa", "rain", "nắng", "sunny",
                # Commodity prices (without specific symbols)
                "gold price", "giá vàng", "oil price", "giá dầu", "giá xăng",
                "silver price", "giá bạc", "gas price",
                # Exchange rates
                "exchange rate", "tỷ giá", "USD/VND", "EUR/USD",
                # News/events
                "breaking news", "tin nóng", "tin mới", "happening",
                "đang xảy ra", "diễn biến",
                # Sports/events
                "score", "tỷ số", "kết quả", "result",
            ]

            if any(kw in intent_lower for kw in realtime_keywords):
                self.logger.warning(
                    f"[INTENT_CLASSIFIER] Forcing requires_tools=True for realtime query "
                    f"(intent={intent_lower[:80]}...)"
                )
                requires_tools = True
                complexity = IntentComplexity.AGENT_LOOP
                if query_type in ("general_knowledge", "conversational", "general"):
                    query_type = "real_time_info"

        return IntentResult(
            intent_summary=data.get("intent_summary", ""),
            reasoning=data.get("reasoning", ""),
            validated_symbols=validated_symbols,
            market_type=market_type,
            complexity=complexity,
            requires_tools=requires_tools,
            analysis_type=analysis_type,
            response_language=data.get("response_language", "en"),
            confidence=data.get("confidence", 0.9),
            query_type=query_type,
        )

    def _extract_tag_content(self, content: str, tag: str) -> Optional[str]:
        """Extract content from XML-style tags."""
        pattern = rf"<{tag}[^>]*>(.*?)</{tag}>"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_json(self, content: str) -> Optional[str]:
        """Extract JSON from content (fallback)."""
        start = content.find("{")
        if start == -1:
            return None

        depth = 0
        for i, char in enumerate(content[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return content[start:i + 1]

        return None

    def _log_result(
        self,
        result: IntentResult,
        elapsed_ms: int,
        is_cached: bool,
    ) -> None:
        """Log classification result with visual structure."""
        cache_info = "💾 CACHED" if is_cached else "LLM"
        self.logger.info("─" * 50)
        self.logger.info(f"🎯 INTENT CLASSIFICATION ({cache_info})")
        self.logger.info("─" * 50)
        self.logger.info(f"  ├─ Intent: {result.intent_summary}")
        self.logger.info(f"  ├─ Symbols: {result.validated_symbols}")
        self.logger.info(f"  ├─ Market: {result.market_type.value}")
        self.logger.info(f"  ├─ Complexity: {result.complexity.value}")
        self.logger.info(f"  ├─ Analysis Type: {result.analysis_type.value}")
        self.logger.info(f"  ├─ Requires Tools: {result.requires_tools}")
        self.logger.info(f"  └─ ⏱️ Time: {elapsed_ms}ms")

    def _heuristic_fallback(
        self,
        query: str,
        ui_context: Optional[Dict[str, Any]],
        working_memory_symbols: Optional[List[str]],
        error_reason: str,
    ) -> IntentResult:
        """
        Rule-based classification fallback when all LLM providers fail.

        Uses regex patterns and keyword matching to provide a best-effort
        classification. This ensures the system can still function even
        when all LLM services are unavailable.

        Args:
            query: User query
            ui_context: UI context (active_tab, etc.)
            working_memory_symbols: Symbols from previous turns
            error_reason: Why LLM classification failed

        Returns:
            IntentResult with heuristic classification
        """
        self.logger.warning(
            f"[INTENT_CLASSIFIER] Using heuristic fallback due to: {error_reason}"
        )

        query_lower = query.lower()

        # =====================================================================
        # 1. Extract symbols using regex patterns
        # =====================================================================
        # Match 1-5 uppercase letters that look like tickers
        potential_symbols = re.findall(r'\b[A-Z]{1,5}\b', query)

        # Filter out common non-symbol words
        non_symbols = {
            "I", "A", "THE", "AND", "OR", "FOR", "TO", "OF", "IN", "ON",
            "IS", "IT", "AT", "BY", "BE", "AS", "AN", "SO", "IF", "NO",
            "YES", "OK", "HI", "RSI", "MACD", "SMA", "EMA", "ATR", "DCF",
            "PE", "PB", "ROE", "ROA", "EPS", "CEO", "CFO", "IPO", "ETF",
        }
        validated_symbols = [s for s in potential_symbols if s not in non_symbols]

        # If no symbols found, try working memory
        if not validated_symbols and working_memory_symbols:
            # Check if query references previous context
            reference_words = [
                "này", "đó", "nó", "this", "that", "it", "the stock",
                "company", "công ty", "cổ phiếu", "symbol", "mã"
            ]
            if any(ref in query_lower for ref in reference_words):
                validated_symbols = working_memory_symbols.copy()
                self.logger.info(
                    f"[HEURISTIC] Inherited symbols from working memory: {validated_symbols}"
                )

        # =====================================================================
        # 2. Detect market type
        # =====================================================================
        crypto_keywords = [
            "bitcoin", "btc", "ethereum", "eth", "crypto", "coin",
            "solana", "sol", "binance", "defi", "blockchain"
        ]
        is_crypto = any(kw in query_lower for kw in crypto_keywords)

        # Check UI context for market type hint
        active_tab = ui_context.get("active_tab", "none") if ui_context else "none"
        if active_tab == "crypto":
            market_type = IntentMarketType.CRYPTO
        elif active_tab == "stock" or not is_crypto:
            market_type = IntentMarketType.STOCK if validated_symbols else IntentMarketType.NONE
        else:
            market_type = IntentMarketType.CRYPTO

        # =====================================================================
        # 3. Detect if tools are needed
        # =====================================================================
        tool_keywords = [
            "price", "giá", "quote", "analysis", "phân tích",
            "news", "tin", "chart", "biểu đồ", "technical", "kỹ thuật",
            "fundamental", "cơ bản", "financial", "tài chính",
            "report", "báo cáo", "earnings", "revenue", "doanh thu",
            "compare", "so sánh", "screen", "lọc", "search", "tìm"
        ]
        requires_tools = (
            len(validated_symbols) > 0 or
            any(kw in query_lower for kw in tool_keywords)
        )

        # =====================================================================
        # 4. Detect greeting/conversational queries (no tools needed)
        # =====================================================================
        greeting_patterns = [
            "hello", "hi", "xin chào", "chào", "thanks", "cảm ơn",
            "thank you", "bye", "goodbye", "tạm biệt", "help", "giúp"
        ]
        is_greeting = any(pattern in query_lower for pattern in greeting_patterns)
        if is_greeting and not validated_symbols:
            requires_tools = False

        # =====================================================================
        # 5. Detect analysis type
        # =====================================================================
        analysis_type = AnalysisType.GENERAL

        if validated_symbols:
            # Check for specific analysis types
            if any(kw in query_lower for kw in ["rsi", "macd", "sma", "ema", "kỹ thuật", "technical", "chart"]):
                analysis_type = AnalysisType.TECHNICAL
            elif any(kw in query_lower for kw in ["pe", "pb", "roe", "earnings", "financial", "tài chính", "cơ bản"]):
                analysis_type = AnalysisType.FUNDAMENTAL
            elif any(kw in query_lower for kw in ["dcf", "định giá", "valuation", "fair value", "graham"]):
                analysis_type = AnalysisType.VALUATION
            elif any(kw in query_lower for kw in ["compare", "so sánh", "vs", "versus"]):
                analysis_type = AnalysisType.COMPARISON
            elif any(kw in query_lower for kw in ["portfolio", "danh mục", "allocation"]):
                analysis_type = AnalysisType.PORTFOLIO
            elif any(kw in query_lower for kw in ["backtest", "strategy", "chiến lược"]):
                analysis_type = AnalysisType.BACKTEST
            else:
                analysis_type = AnalysisType.BASIC

        # =====================================================================
        # 6. Detect language
        # =====================================================================
        vietnamese_chars = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
        has_vietnamese = any(c in vietnamese_chars for c in query_lower)
        response_language = "vi" if has_vietnamese else "en"

        # =====================================================================
        # 7. Build result
        # =====================================================================
        complexity = IntentComplexity.AGENT_LOOP if requires_tools else IntentComplexity.DIRECT

        # Build intent summary
        if validated_symbols:
            intent_summary = f"Query about {', '.join(validated_symbols)}"
        elif is_greeting:
            intent_summary = "Greeting or conversational"
        else:
            intent_summary = "General query"

        result = IntentResult(
            intent_summary=intent_summary,
            reasoning=f"Heuristic fallback (LLM unavailable: {error_reason[:100]})",
            validated_symbols=validated_symbols,
            market_type=market_type,
            complexity=complexity,
            requires_tools=requires_tools,
            analysis_type=analysis_type,
            response_language=response_language,
            confidence=0.6,  # Lower confidence for heuristic
            query_type="stock_analysis" if validated_symbols else "general",
            classification_method="heuristic_fallback",
        )

        self.logger.info(
            f"[HEURISTIC] Result: symbols={validated_symbols}, "
            f"requires_tools={requires_tools}, market={market_type.value}"
        )

        return result


# ============================================================================
# SINGLETON
# ============================================================================

_intent_classifier_instance: Optional[IntentClassifier] = None


def get_intent_classifier(
    model_name: Optional[str] = None,
    provider_type: Optional[str] = None,
) -> IntentClassifier:
    """Get singleton IntentClassifier instance."""
    global _intent_classifier_instance

    if _intent_classifier_instance is None:
        _intent_classifier_instance = IntentClassifier(
            model_name=model_name,
            provider_type=provider_type,
        )

    return _intent_classifier_instance


def reset_intent_classifier():
    """Reset singleton (for testing)."""
    global _intent_classifier_instance
    _intent_classifier_instance = None


async def clear_intent_cache():
    """Clear the intent cache."""
    await _intent_cache._local_cache.clear()
