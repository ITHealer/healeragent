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

        return cls(
            intent_summary=data.get("intent_summary", ""),
            reasoning=data.get("reasoning", ""),
            validated_symbols=data.get("validated_symbols", []),
            market_type=market_type,
            complexity=complexity,
            requires_tools=data.get("requires_tools", False),
            response_language=data.get("response_language", "en"),
            confidence=data.get("confidence", 0.9),
            query_type=data.get("query_type", "general"),
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
    ) -> IntentResult:
        """
        Classify query and extract normalized symbols in a single LLM call.

        Args:
            query: User query
            ui_context: UI context (active_tab, recent_symbols)
            conversation_history: Previous messages for context
            use_cache: Whether to use caching
            working_memory_symbols: Symbols from previous turns (Working Memory)
            core_memory_context: User profile from Core Memory (portfolio, preferences)

        Returns:
            IntentResult with validated_symbols already normalized
        """
        start_time = datetime.now()

        # Check cache first
        if use_cache:
            cached = await _intent_cache.get(query, ui_context)
            if cached:
                elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                self._log_result(cached, elapsed_ms, is_cached=True)
                return cached

        try:
            # Build prompt with symbol normalization instructions
            prompt = self._build_prompt(
                query, ui_context, conversation_history,
                working_memory_symbols, core_memory_context
            )

            # Call LLM
            result_data = await self._call_llm(prompt)

            # Parse result
            result = self._parse_result(result_data)

            # Log
            elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self._log_result(result, elapsed_ms, is_cached=False)

            # Cache asynchronously
            if use_cache:
                asyncio.create_task(_intent_cache.set(query, result, ui_context))

            return result

        except Exception as e:
            self.logger.error(f"[INTENT_CLASSIFIER] Error: {e}", exc_info=True)
            return IntentResult.fallback(str(e))

    def _build_prompt(
        self,
        query: str,
        ui_context: Optional[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]],
        working_memory_symbols: Optional[List[str]] = None,
        core_memory_context: Optional[str] = None,
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
            wm_hint = f"""
<working_memory>
SYMBOLS FROM RECENT TURNS: {', '.join(working_memory_symbols)}
CRITICAL: When user refers to "nó", "this stock", "công ty này", "symbol đó", "these stocks", etc.
without explicit names, they likely refer to these symbols from recent conversation.
Use these symbols to resolve ambiguous references!
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

        # History context - INCREASED from 100 to 500 chars per message
        history_hint = ""
        if conversation_history and len(conversation_history) > 0:
            recent = conversation_history[-5:]  # Last 5 messages (increased from 3)
            history_text = "\n".join([
                f"- {msg.get('role', 'user')}: {msg.get('content', '')[:500]}"
                for msg in recent
            ])
            history_hint = f"""
<conversation_history>
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

## STEP 3: Determine Market Type
- "stock": Stocks, ETFs, indices
- "crypto": Cryptocurrencies
- "both": Query involves both stock and crypto
- "none": No specific market

## STEP 4: Determine Complexity
"direct" - NO tools needed:
- Greetings, thanks, goodbye
- Static financial concepts (what is P/E ratio?)
- General knowledge that doesn't change

"agent_loop" - Tools NEEDED:
- Real-time prices, quotes
- Technical analysis (RSI, MACD, etc.)
- Fundamental data (earnings, financials)
- News and current events
- Stock screening
- Any query about current state of markets

## STEP 5: Detect Language
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
  "response_language": "vi|en|zh",
  "query_type": "stock_analysis|crypto_analysis|comparison|screener|conversational|general_knowledge|real_time_info",
  "confidence": 0.0-1.0
}}
</classification>
</output_format>

<few_shot_examples>
Example 1 - Stock Analysis (Symbol Normalization):
Query: "Phân tích Google và Amazon"
<reasoning>
User wants analysis of Google and Amazon stocks.
- "Google" → normalized to "GOOGL" (official ticker)
- "Amazon" → normalized to "AMZN" (official ticker)
This requires real-time data (price, technicals), so complexity is agent_loop.
</reasoning>
<classification>
{{"intent_summary": "Technical analysis of Google and Amazon stocks", "validated_symbols": ["GOOGL", "AMZN"], "market_type": "stock", "complexity": "agent_loop", "requires_tools": true, "response_language": "vi", "query_type": "stock_analysis", "confidence": 0.95}}
</classification>

Example 2 - Greeting (No Tools):
Query: "Xin chào"
<reasoning>
Simple greeting. No financial query. No tools needed.
</reasoning>
<classification>
{{"intent_summary": "Greeting", "validated_symbols": [], "market_type": "none", "complexity": "direct", "requires_tools": false, "response_language": "vi", "query_type": "conversational", "confidence": 0.99}}
</classification>

Example 3 - Multiple Symbols:
Query: "So sánh NVDA, Apple và Microsoft"
<reasoning>
User wants comparison of 3 stocks.
- "NVDA" → already correct ticker
- "Apple" → normalized to "AAPL"
- "Microsoft" → normalized to "MSFT"
Comparison requires multiple data points, so agent_loop.
</reasoning>
<classification>
{{"intent_summary": "Compare NVIDIA, Apple, and Microsoft stocks", "validated_symbols": ["NVDA", "AAPL", "MSFT"], "market_type": "stock", "complexity": "agent_loop", "requires_tools": true, "response_language": "vi", "query_type": "comparison", "confidence": 0.94}}
</classification>

Example 4 - General Knowledge (No Tools):
Query: "P/E ratio là gì?"
<reasoning>
User asks about P/E ratio definition - a static concept that doesn't need live data.
</reasoning>
<classification>
{{"intent_summary": "Explain P/E ratio concept", "validated_symbols": [], "market_type": "none", "complexity": "direct", "requires_tools": false, "response_language": "vi", "query_type": "general_knowledge", "confidence": 0.95}}
</classification>

Example 5 - Crypto with UI Context:
Query: "Giá BTC" (UI Context: active_tab = crypto)
<reasoning>
User asks about BTC price. UI context shows crypto tab, so BTC = Bitcoin cryptocurrency.
Price data requires tools.
</reasoning>
<classification>
{{"intent_summary": "Bitcoin cryptocurrency price", "validated_symbols": ["BTC"], "market_type": "crypto", "complexity": "agent_loop", "requires_tools": true, "response_language": "vi", "query_type": "crypto_analysis", "confidence": 0.93}}
</classification>

Example 6 - Real-Time Info:
Query: "Who is Apple's CEO?"
<reasoning>
User asks about current Apple CEO. This is real-time info that could change.
Need web search to verify current state.
</reasoning>
<classification>
{{"intent_summary": "Find current Apple CEO", "validated_symbols": ["AAPL"], "market_type": "stock", "complexity": "agent_loop", "requires_tools": true, "response_language": "en", "query_type": "real_time_info", "confidence": 0.92}}
</classification>
</few_shot_examples>

Now analyze the query and provide your classification:
"""

    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM and parse response."""

        for attempt in range(self.max_retries + 1):
            try:
                params = {
                    "model_name": self.model_name,
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
                    "provider_type": self.provider_type,
                    "api_key": self.api_key,
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

        return IntentResult(
            intent_summary=data.get("intent_summary", ""),
            reasoning=data.get("reasoning", ""),
            validated_symbols=validated_symbols,
            market_type=market_type,
            complexity=complexity,
            requires_tools=data.get("requires_tools", False),
            response_language=data.get("response_language", "en"),
            confidence=data.get("confidence", 0.9),
            query_type=data.get("query_type", "general"),
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
        self.logger.info(f"  ├─ Requires Tools: {result.requires_tools}")
        self.logger.info(f"  └─ ⏱️ Time: {elapsed_ms}ms")


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
