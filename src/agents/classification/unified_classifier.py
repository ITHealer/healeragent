"""
Unified Classifier

Merges query classification and tool necessity validation into a single LLM call.
Follows Anthropic's structured output pattern with <reasoning> + <classification> tags.

Before (2 LLM calls):
- Stage 1: Classification → query_type, categories, symbols
- Thinking: Validation → need_tools, final_intent

After (1 LLM call):
- Unified Classification → query_type, categories, symbols, requires_tools, intent_summary
"""

import json
import re
import hashlib
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from functools import lru_cache

from src.utils.logger.custom_logging import LoggerMixin
from src.utils.logger.log_formatter import LogFormatter
from src.utils.config import settings
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.helpers.redis_cache import redis_manager

from .models import (
    UnifiedClassificationResult,
    ClassifierContext,
    QueryType,
    MarketType,
    VALID_CATEGORIES,
)

# Symbol resolution imports (for Soft Context Inheritance)
from src.services.asset.symbol_resolution_models import (
    UIContext,
    ActiveTab,
)
from src.services.asset.symbol_resolver import (
    SymbolResolver,
    get_symbol_resolver,
)


# ============================================================================
# CLASSIFICATION CACHE (Redis-backed with in-memory fallback)
# ============================================================================

# Cache key prefix for Redis
CACHE_KEY_PREFIX = "classification:"
CACHE_TTL_SECONDS = 120  # 2 minutes


class ClassificationCache(LoggerMixin):
    """
    Redis-backed TTL cache for classification results.

    Features:
    - Primary: Redis for distributed caching across instances
    - Fallback: In-memory cache if Redis unavailable
    - Shared across all users (query-based, not user-based)
    - TTL-based expiration (120s default)

    Benefits over in-memory only:
    - Survives server restarts
    - Shared across multiple server instances
    - No memory pressure on application server
    """

    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS):
        super().__init__()
        self.ttl_seconds = ttl_seconds
        # Fallback in-memory cache (used when Redis unavailable)
        self._local_cache: Dict[str, tuple] = {}
        self._local_max_size = 100

    def _make_key(self, query: str, symbols_context: List[str], ui_context: Optional[Dict[str, Any]] = None) -> str:
        """Create cache key from query, context, and UI context."""
        normalized = query.lower().strip()
        context_str = ",".join(sorted(symbols_context)) if symbols_context else ""
        # Include UI context in cache key (active_tab affects classification)
        ui_context_str = ""
        if ui_context:
            active_tab = ui_context.get("active_tab", "none")
            ui_context_str = f"|tab:{active_tab}"
        key_str = f"{normalized}|{context_str}{ui_context_str}"
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"{CACHE_KEY_PREFIX}{key_hash}"

    async def get(self, query: str, symbols_context: List[str], ui_context: Optional[Dict[str, Any]] = None) -> Optional[UnifiedClassificationResult]:
        """Get cached result from Redis (with in-memory fallback)."""
        cache_key = self._make_key(query, symbols_context, ui_context)

        # Try Redis first
        try:
            redis_client = await redis_manager.get_client()
            if redis_client:
                cached_json = await asyncio.wait_for(
                    redis_client.get(cache_key),
                    timeout=5.0
                )
                if cached_json:
                    # Parse JSON string to dict
                    if isinstance(cached_json, bytes):
                        cached_json = cached_json.decode('utf-8')
                    data = json.loads(cached_json)
                    result = UnifiedClassificationResult.from_dict(data)
                    self.logger.debug(f"[CACHE] Redis HIT: {cache_key[:20]}...")
                    return result
        except asyncio.TimeoutError:
            self.logger.warning(f"[CACHE] Redis GET timeout for: {cache_key[:20]}...")
        except Exception as e:
            self.logger.warning(f"[CACHE] Redis GET error: {e}")

        # Fallback to local cache
        if cache_key in self._local_cache:
            result, timestamp = self._local_cache[cache_key]
            age = (datetime.now() - timestamp).total_seconds()
            if age <= self.ttl_seconds:
                self.logger.debug(f"[CACHE] Local HIT: {cache_key[:20]}...")
                return result
            else:
                del self._local_cache[cache_key]

        return None

    async def set(
        self,
        query: str,
        symbols_context: List[str],
        result: UnifiedClassificationResult,
        ui_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Cache result to Redis (with in-memory fallback)."""
        cache_key = self._make_key(query, symbols_context, ui_context)

        # Serialize result to JSON
        result_dict = {
            "query_type": result.query_type.value,
            "symbols": result.symbols,
            "tool_categories": result.tool_categories,
            "market_type": result.market_type.value if result.market_type else None,
            "requires_tools": result.requires_tools,
            "confidence": result.confidence,
            "intent_summary": result.intent_summary,
            "response_language": result.response_language,
            "reasoning": result.reasoning,
            "classification_method": result.classification_method,
        }
        json_str = json.dumps(result_dict, ensure_ascii=False)

        # Try Redis first
        try:
            redis_client = await redis_manager.get_client()
            if redis_client:
                await asyncio.wait_for(
                    redis_client.set(cache_key, json_str, ex=self.ttl_seconds),
                    timeout=5.0
                )
                self.logger.debug(f"[CACHE] Redis SET: {cache_key[:20]}... TTL={self.ttl_seconds}s")
                return
        except asyncio.TimeoutError:
            self.logger.warning(f"[CACHE] Redis SET timeout for: {cache_key[:20]}...")
        except Exception as e:
            self.logger.warning(f"[CACHE] Redis SET error: {e}")

        # Fallback to local cache
        if len(self._local_cache) >= self._local_max_size:
            # Evict oldest
            oldest_key = min(self._local_cache, key=lambda k: self._local_cache[k][1])
            del self._local_cache[oldest_key]

        self._local_cache[cache_key] = (result, datetime.now())
        self.logger.debug(f"[CACHE] Local SET: {cache_key[:20]}...")

    async def clear(self) -> None:
        """Clear all cached results (both Redis and local)."""
        # Clear local cache
        self._local_cache.clear()

        # Clear Redis keys with pattern
        try:
            redis_client = await redis_manager.get_client()
            if redis_client:
                # Use SCAN to find and delete keys
                cursor = 0
                deleted_count = 0
                while True:
                    cursor, keys = await redis_client.scan(
                        cursor=cursor,
                        match=f"{CACHE_KEY_PREFIX}*",
                        count=100
                    )
                    if keys:
                        await redis_client.delete(*keys)
                        deleted_count += len(keys)
                    if cursor == 0:
                        break
                self.logger.info(f"[CACHE] Cleared {deleted_count} Redis keys")
        except Exception as e:
            self.logger.warning(f"[CACHE] Redis clear error: {e}")

    def clear_sync(self) -> None:
        """Synchronous clear (local cache only, for backwards compatibility)."""
        self._local_cache.clear()


# Global cache instance
_classification_cache = ClassificationCache(ttl_seconds=CACHE_TTL_SECONDS)


class UnifiedClassifier(LoggerMixin):
    """
    Unified query classifier with integrated tool necessity validation.

    Key features:
    - Single LLM call (vs 2 in previous implementation)
    - Anthropic-style structured output
    - Few-shot examples for accuracy
    - Graceful fallback handling
    - Soft Context Inheritance for symbol resolution

    Usage:
        classifier = UnifiedClassifier(model_name="gpt-4.1-nano")
        result = await classifier.classify(context, resolve_symbols=True)
    """

    def __init__(
        self,
        model_name: str = None,
        provider_type: str = None,
        max_retries: int = 2,
    ):
        super().__init__()

        self.model_name = model_name or settings.CLASSIFIER_MODEL or "gpt-4.1-nano"
        self.provider_type = provider_type or settings.CLASSIFIER_PROVIDER or ProviderType.OPENAI
        self.max_retries = max_retries

        # Initialize LLM provider
        self.llm_provider = LLMGeneratorProvider()
        self.api_key = ModelProviderFactory._get_api_key(self.provider_type)

        # Symbol resolver for Soft Context Inheritance
        self._symbol_resolver: Optional[SymbolResolver] = None

        self.logger.info(
            f"[CLASSIFIER] Initialized: model={self.model_name}, provider={self.provider_type}"
        )

    def _get_symbol_resolver(self) -> SymbolResolver:
        """Lazy initialization of symbol resolver"""
        if self._symbol_resolver is None:
            self._symbol_resolver = get_symbol_resolver()
        return self._symbol_resolver

    async def classify(
        self,
        context: ClassifierContext,
        use_cache: bool = True,
        resolve_symbols: bool = True,
    ) -> UnifiedClassificationResult:
        """
        Classify a query and determine if tools are needed.

        Now includes Soft Context Inheritance for symbol resolution.

        Args:
            context: ClassifierContext with query, history, memory, and ui_context
            use_cache: Whether to use caching (default True)
            resolve_symbols: Whether to resolve symbols with context (default True)

        Returns:
            UnifiedClassificationResult with classification, validation, and resolved symbols
        """
        start_time = datetime.now()

        # Extract symbols from working memory for cache key
        symbols_context = self._extract_symbols_from_context(context)

        # Check cache first (async Redis) - include ui_context in cache key
        if use_cache:
            cached = await _classification_cache.get(
                context.query, symbols_context, context.ui_context
            )
            if cached:
                # Still run symbol resolution on cache hit if symbols exist
                # This ensures UI context is applied to resolve ambiguous symbols
                if resolve_symbols and cached.symbols:
                    cached = await self._resolve_classification_symbols(cached, context)
                    # Apply UI context override to query type
                    cached = self._apply_ui_context_override(cached, context)

                elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                # Enhanced logging with visual structure
                self.logger.info("─" * 50)
                self.logger.info(f"🎯 CLASSIFICATION (cached)")
                self.logger.info("─" * 50)
                self.logger.info(f"  ├─ 💾 [CACHE HIT] key={context.query[:20]}...")
                self.logger.info(f"  ├─ Type: {cached.query_type.value}")
                self.logger.info(f"  ├─ Symbols: {cached.symbols}")
                self.logger.info(f"  ├─ Categories: {cached.tool_categories}")
                self.logger.info(f"  └─ ⏱️ Time: {elapsed_ms}ms")
                return cached

        try:
            # Build prompt
            prompt = self._build_prompt(context)

            # Call LLM
            result = await self._call_llm(prompt)

            # Parse result
            classification = UnifiedClassificationResult.from_dict(result)
            classification.classification_method = "llm"

            # Post-processing
            classification = self._post_process(classification, context)

            # Apply UI context override to query type (before symbol resolution)
            classification = self._apply_ui_context_override(classification, context)

            # Symbol Resolution (Soft Context Inheritance)
            if resolve_symbols and classification.symbols:
                classification = await self._resolve_classification_symbols(
                    classification, context
                )

            elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            # Enhanced logging with visual structure
            self.logger.info("─" * 50)
            self.logger.info(f"🎯 CLASSIFICATION (LLM)")
            self.logger.info("─" * 50)
            self.logger.info(f"  ├─ Type: {classification.query_type.value}")
            self.logger.info(f"  ├─ Symbols: {classification.symbols}")
            self.logger.info(f"  ├─ Categories: {classification.tool_categories}")
            self.logger.info(f"  ├─ Requires Tools: {classification.requires_tools}")
            if classification.clarification_needed:
                self.logger.info(f"  ├─ ⚠️ Clarification Needed: {classification.clarification_messages}")
            self.logger.info(f"  └─ ⏱️ Time: {elapsed_ms}ms")

            # Cache the result (async Redis, fire and forget style) - include ui_context
            if use_cache:
                asyncio.create_task(_classification_cache.set(
                    context.query, symbols_context, classification, context.ui_context
                ))

            return classification

        except Exception as e:
            self.logger.error(f"[CLASSIFIER] Failed: {e}", exc_info=True)
            return UnifiedClassificationResult.fallback(str(e))

    def _apply_ui_context_override(
        self,
        classification: UnifiedClassificationResult,
        context: ClassifierContext,
    ) -> UnifiedClassificationResult:
        """
        Apply UI context override to query type.

        If user is on Stock tab and query is classified as crypto_specific,
        override to stock_specific if the symbol could be a stock.

        Principle: "Assume smartly" based on UI context.
        """
        if not context.ui_context:
            return classification

        active_tab = context.ui_context.get("active_tab", "none")

        # Stock tab + crypto_specific → check if could be stock
        if active_tab == "stock" and classification.query_type == QueryType.CRYPTO_SPECIFIC:
            self.logger.info(f"  ├─ 🔄 UI Context Override:")
            self.logger.info(f"  │    crypto_specific → stock_specific")
            self.logger.info(f"  │    (active_tab=stock, symbols={classification.symbols})")
            classification.query_type = QueryType.STOCK_SPECIFIC
            classification.market_type = MarketType.STOCK
            # Update tool categories: remove crypto, add stock-related
            if "crypto" in classification.tool_categories:
                classification.tool_categories.remove("crypto")
            if "price" not in classification.tool_categories:
                classification.tool_categories.append("price")

        # Crypto tab + stock_specific → check if could be crypto
        elif active_tab == "crypto" and classification.query_type == QueryType.STOCK_SPECIFIC:
            self.logger.info(f"  ├─ 🔄 UI Context Override:")
            self.logger.info(f"  │    stock_specific → crypto_specific")
            self.logger.info(f"  │    (active_tab=crypto, symbols={classification.symbols})")
            classification.query_type = QueryType.CRYPTO_SPECIFIC
            classification.market_type = MarketType.CRYPTO
            # Update tool categories: add crypto
            if "crypto" not in classification.tool_categories:
                classification.tool_categories.append("crypto")

        return classification

    async def _resolve_classification_symbols(
        self,
        classification: UnifiedClassificationResult,
        context: ClassifierContext,
    ) -> UnifiedClassificationResult:
        """
        Resolve extracted symbols using Soft Context Inheritance.

        Args:
            classification: Classification result with raw symbols
            context: Classifier context with UI context

        Returns:
            Classification with resolved_symbols and clarification info
        """
        if not classification.symbols:
            return classification

        try:
            resolver = self._get_symbol_resolver()

            # Build UI context from classifier context
            ui_context = self._build_ui_context(context)

            # Resolve symbols
            result = await resolver.resolve(
                symbols=classification.symbols,
                query=context.query,
                ui_context=ui_context,
                conversation_history=context.conversation_history,
                use_cache=True,
            )

            # Update classification with resolution results
            classification.resolved_symbols = [
                rs.to_dict() for rs in result.resolved_symbols
            ]

            if result.has_ambiguity():
                classification.clarification_needed = True
                classification.clarification_messages = result.clarification_messages

                # Add ambiguous symbols to resolved_symbols for UI display
                for ambiguous in result.needs_clarification:
                    classification.resolved_symbols.append(ambiguous.to_dict())

            self.logger.debug(
                f"[CLASSIFIER] Symbol resolution: "
                f"resolved={len(result.resolved_symbols)}, "
                f"ambiguous={len(result.needs_clarification)}, "
                f"unresolved={len(result.unresolved)}"
            )

        except Exception as e:
            self.logger.warning(f"[CLASSIFIER] Symbol resolution failed: {e}")
            # Continue without symbol resolution

        return classification

    def _build_ui_context(self, context: ClassifierContext) -> UIContext:
        """Build UIContext from ClassifierContext"""
        if not context.ui_context:
            return UIContext()

        try:
            active_tab = ActiveTab.NONE
            tab_value = context.ui_context.get("active_tab", "none")
            try:
                active_tab = ActiveTab(tab_value)
            except ValueError:
                pass

            return UIContext(
                active_tab=active_tab,
                recent_symbols=context.ui_context.get("recent_symbols", []),
                preferred_quote_currency=context.ui_context.get(
                    "preferred_quote_currency", "USD"
                ),
            )
        except Exception:
            return UIContext()

    def _extract_symbols_from_context(self, context: ClassifierContext) -> List[str]:
        """Extract current symbols from working memory context for cache key."""
        symbols = []

        # Parse working memory summary for symbols
        if context.working_memory_summary:
            # Look for patterns like "current_symbols: AAPL, NVDA"
            import re
            pattern = r"current_symbols?[:\s]+\[?([A-Z,\s]+)\]?"
            match = re.search(pattern, context.working_memory_summary, re.IGNORECASE)
            if match:
                symbols_str = match.group(1)
                symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]

        return symbols

    def _build_prompt(self, context: ClassifierContext) -> str:
        """Build the unified classification prompt"""

        # Format history
        history_text = context.format_history(max_turns=5)

        # Format memory context
        memory_context = context.format_memory_context()

        # Format UI context (Soft Context Inheritance)
        ui_context = context.format_ui_context()

        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        return f"""<current_context>
Date: {current_date}
Data Status: Real-time market data available through tools
</current_context>

You are a financial query classification system. Analyze the user's query and provide a unified classification that includes:
1. What type of query this is
2. What tools/categories are needed
3. Whether external tools are actually required

<query>{context.query}</query>

<conversation_history>
{history_text}
</conversation_history>

{memory_context}

{ui_context}

<analysis_instructions>
Analyze the query carefully:

STEP 1: Determine Query Type
- stock_specific: About specific stock(s) with symbols
- crypto_specific: About specific cryptocurrency
- screener: Finding stocks by criteria ("find stocks with P/E < 20")
- market_level: Market overview, indices, sector performance
- conversational: ONLY greetings, thanks, goodbye
- memory_recall: Questions about past conversations
- general_knowledge: Static financial concepts that never change (what is P/E ratio?)
- real_time_info: Current events, latest news, current leaders/positions, recent changes - anything that requires up-to-date information from the web

IMPORTANT - UI Context (Soft Context Inheritance):
If <ui_context> is provided, the Active Tab strongly indicates user intent:
- If Active Tab is "stock": Interpret ambiguous symbols (like BTC, SOL, COMP) as STOCKS, use query_type=stock_specific
- If Active Tab is "crypto": Interpret ambiguous symbols as CRYPTOCURRENCY, use query_type=crypto_specific
- The Active Tab reflects which section of the app the user is currently viewing
- Follow the principle: "Assume smartly" based on UI context

STEP 2: Extract Symbols
- Look for explicit tickers: AAPL, NVDA, BTC
- Resolve references from Working Memory: "it", "that stock", "cổ phiếu đó"
- Consider UI context when interpreting ambiguous symbols

STEP 3: Determine Tool Categories Needed
Categories: price, technical, fundamentals, news, market, risk, crypto, discovery, memory, web
- If Active Tab is "stock": Use price, technical, fundamentals, news, etc. (NOT crypto)
- If Active Tab is "crypto": Include crypto category
- Use "web" category ONLY when other categories cannot provide the needed information

STEP 4: Determine if Tools Are Required
Tools ARE needed for:
- Real-time prices, quotes
- Current technical indicators
- Recent news, earnings
- Live market data
- Past conversation recall (memory tools)
- Current events, current leaders, recent changes (use web search)
- Questions about "who is", "what is current", "latest" anything

Tools are NOT needed for:
- Greetings, thanks, goodbye
- Static financial concepts (what is P/E ratio? how does compound interest work?)
- Definitions and explanations of unchanging concepts
</analysis_instructions>

<output_format>
First, provide your reasoning inside <reasoning> tags.
Then, provide the classification inside <classification> tags as valid JSON:

<reasoning>
[Your step-by-step analysis]
</reasoning>

<classification>
{{
  "query_type": "stock_specific|crypto_specific|screener|market_level|conversational|memory_recall|general_knowledge|real_time_info",
  "symbols": ["SYMBOL1", "SYMBOL2"],
  "tool_categories": ["category1", "category2"],
  "market_type": "stock|crypto|both|null",
  "requires_tools": true|false,
  "confidence": 0.0-1.0,
  "intent_summary": "Brief description of what user wants",
  "response_language": "vi|en|zh"
}}
</classification>
</output_format>

<few_shot_examples>
Example 1 - Stock Analysis:
Query: "Phân tích kỹ thuật NVDA"
<reasoning>
User asks for technical analysis of NVDA. This is stock-specific with explicit symbol. Requires real-time technical indicators.
</reasoning>
<classification>
{{"query_type": "stock_specific", "symbols": ["NVDA"], "tool_categories": ["technical", "price"], "market_type": "stock", "requires_tools": true, "confidence": 0.95, "intent_summary": "Technical analysis for NVDA", "response_language": "vi"}}
</classification>

Example 2 - Greeting (No Tools):
Query: "Xin chào"
<reasoning>
Simple Vietnamese greeting. No financial request. No tools needed.
</reasoning>
<classification>
{{"query_type": "conversational", "symbols": [], "tool_categories": [], "market_type": null, "requires_tools": false, "confidence": 0.99, "intent_summary": "Greeting", "response_language": "vi"}}
</classification>

Example 3 - General Knowledge (No Tools):
Query: "What is a good P/E ratio for tech stocks?"
<reasoning>
General question about P/E ratio concepts. Can answer from knowledge without real-time data.
</reasoning>
<classification>
{{"query_type": "general_knowledge", "symbols": [], "tool_categories": [], "market_type": null, "requires_tools": false, "confidence": 0.90, "intent_summary": "Explain P/E ratio benchmarks", "response_language": "en"}}
</classification>

Example 4 - Memory Recall:
Query: "Chúng ta vừa nói về cổ phiếu gì?"
<reasoning>
User asks about previous conversation. This is memory recall, requires memory tools.
</reasoning>
<classification>
{{"query_type": "memory_recall", "symbols": [], "tool_categories": ["memory"], "market_type": null, "requires_tools": true, "confidence": 0.92, "intent_summary": "Recall previous stock discussion", "response_language": "vi"}}
</classification>

Example 5 - Screener:
Query: "Find me tech stocks with RSI below 30"
<reasoning>
User wants to find stocks meeting criteria. This is a screening request.
</reasoning>
<classification>
{{"query_type": "screener", "symbols": [], "tool_categories": ["discovery", "technical"], "market_type": "stock", "requires_tools": true, "confidence": 0.93, "intent_summary": "Screen for oversold tech stocks", "response_language": "en"}}
</classification>

Example 6 - Reference Resolution:
Query: "Còn về tin tức thì sao?" (Working Memory has: current_symbols=["AAPL"])
<reasoning>
User asks about news for the stock in context. Working Memory shows AAPL. This is stock-specific.
</reasoning>
<classification>
{{"query_type": "stock_specific", "symbols": ["AAPL"], "tool_categories": ["news"], "market_type": "stock", "requires_tools": true, "confidence": 0.88, "intent_summary": "News for AAPL from context", "response_language": "vi"}}
</classification>

Example 7 - Fundamental Analysis:
Query: "Triển vọng cơ bản của NVDA như thế nào? Xem báo cáo tài chính"
<reasoning>
User asks about NVDA's fundamental outlook and financial reports. This requires fundamentals tools (income statement, balance sheet, financial ratios).
</reasoning>
<classification>
{{"query_type": "stock_specific", "symbols": ["NVDA"], "tool_categories": ["fundamentals", "price"], "market_type": "stock", "requires_tools": true, "confidence": 0.94, "intent_summary": "Fundamental analysis for NVDA", "response_language": "vi"}}
</classification>

Example 8 - Comprehensive Analysis (Technical + Fundamental):
Query: "Phân tích toàn diện cổ phiếu AAPL, cả kỹ thuật lẫn cơ bản"
<reasoning>
User wants comprehensive analysis including both technical and fundamental aspects. Needs multiple categories.
</reasoning>
<classification>
{{"query_type": "stock_specific", "symbols": ["AAPL"], "tool_categories": ["technical", "fundamentals", "price", "news"], "market_type": "stock", "requires_tools": true, "confidence": 0.96, "intent_summary": "Comprehensive analysis (technical + fundamental) for AAPL", "response_language": "vi"}}
</classification>

Example 9 - Current Events (Real-Time Info - Web Search Required):
Query: "Thủ tướng nhật bản là ai?"
<reasoning>
User asks about the current Prime Minister of Japan. This is a real-time information query - political leaders change over time. Must use web search to get current, accurate information.
</reasoning>
<classification>
{{"query_type": "real_time_info", "symbols": [], "tool_categories": ["web"], "market_type": null, "requires_tools": true, "confidence": 0.95, "intent_summary": "Find current Prime Minister of Japan", "response_language": "vi"}}
</classification>

Example 10 - Latest News/Events (Real-Time Info):
Query: "What happened with Tesla stock today?"
<reasoning>
User asks about what happened TODAY with Tesla. This requires current real-time information. Needs both price tools and web search for news context.
</reasoning>
<classification>
{{"query_type": "real_time_info", "symbols": ["TSLA"], "tool_categories": ["price", "news", "web"], "market_type": "stock", "requires_tools": true, "confidence": 0.94, "intent_summary": "Latest news and events for Tesla today", "response_language": "en"}}
</classification>

Example 11 - Current Position/State (Real-Time Info):
Query: "Ai là CEO của Apple hiện tại?"
<reasoning>
User asks about the CURRENT CEO of Apple. CEO positions can change. This requires web search for accurate current information.
</reasoning>
<classification>
{{"query_type": "real_time_info", "symbols": ["AAPL"], "tool_categories": ["web"], "market_type": "stock", "requires_tools": true, "confidence": 0.93, "intent_summary": "Find current Apple CEO", "response_language": "vi"}}
</classification>

Example 12 - General Knowledge (Static Concept - No Tools):
Query: "Lãi suất kép là gì?"
<reasoning>
User asks about compound interest - a static financial concept that doesn't change. Can answer from knowledge without tools.
</reasoning>
<classification>
{{"query_type": "general_knowledge", "symbols": [], "tool_categories": [], "market_type": null, "requires_tools": false, "confidence": 0.95, "intent_summary": "Explain compound interest concept", "response_language": "vi"}}
</classification>
</few_shot_examples>

Now analyze the query and provide your classification:
"""

    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM and parse response"""

        for attempt in range(self.max_retries + 1):
            try:
                params = {
                    "model_name": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a financial query classification system. "
                                "Provide structured output with <reasoning> tags "
                                "followed by <classification> tags containing valid JSON."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "provider_type": self.provider_type,
                    "api_key": self.api_key,
                    "enable_thinking": False,
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

                # Extract reasoning for logging
                reasoning = self._extract_tag_content(content, "reasoning")
                if reasoning:
                    self.logger.debug(f"[CLASSIFIER] Reasoning: {reasoning[:200]}...")

                # Extract classification JSON
                json_str = self._extract_tag_content(content, "classification")
                if not json_str:
                    # Fallback: try to find JSON directly
                    json_str = self._extract_json(content)

                if not json_str:
                    raise ValueError("No classification found in response")

                # Parse JSON
                result = json.loads(json_str)
                result["reasoning"] = reasoning or ""
                return result

            except json.JSONDecodeError as e:
                self.logger.warning(
                    f"[CLASSIFIER] JSON parse error (attempt {attempt + 1}): {e}"
                )
                if attempt == self.max_retries:
                    raise
            except Exception as e:
                self.logger.warning(
                    f"[CLASSIFIER] Error (attempt {attempt + 1}): {e}"
                )
                if attempt == self.max_retries:
                    raise

        raise RuntimeError("Failed to classify after retries")

    def _extract_tag_content(self, content: str, tag: str) -> Optional[str]:
        """Extract content from XML-style tags"""
        pattern = rf"<{tag}[^>]*>(.*?)</{tag}>"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_json(self, content: str) -> Optional[str]:
        """Extract JSON from content (fallback)"""
        # Try to find JSON object
        start = content.find("{")
        if start == -1:
            return None

        # Find matching closing brace
        depth = 0
        for i, char in enumerate(content[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return content[start : i + 1]

        return None

    def _post_process(
        self,
        classification: UnifiedClassificationResult,
        context: ClassifierContext,
    ) -> UnifiedClassificationResult:
        """
        Post-process classification result.

        Handles edge cases and ensures consistency.
        """

        # Auto-add price category if symbols present
        if classification.symbols and "price" not in classification.tool_categories:
            if classification.query_type in [QueryType.STOCK_SPECIFIC, QueryType.CRYPTO_SPECIFIC]:
                classification.tool_categories.append("price")

        # Ensure crypto category for crypto queries
        if classification.query_type == QueryType.CRYPTO_SPECIFIC:
            if "crypto" not in classification.tool_categories:
                classification.tool_categories.append("crypto")

        # Screener always needs discovery
        if classification.query_type == QueryType.SCREENER:
            if "discovery" not in classification.tool_categories:
                classification.tool_categories.append("discovery")

        # Strong financial signals override conversational
        strong_categories = {"discovery", "risk", "technical", "fundamentals"}
        if (
            classification.query_type == QueryType.CONVERSATIONAL
            and classification.symbols
        ):
            # Has symbols but marked conversational - likely wrong
            classification.query_type = QueryType.STOCK_SPECIFIC
            classification.requires_tools = True
            self.logger.info(f"  ├─ 🔄 Override: conversational → stock_specific (has symbols)")

        return classification


# Singleton instance
_classifier_instance: Optional[UnifiedClassifier] = None


def get_unified_classifier(
    model_name: str = None,
    provider_type: str = None,
) -> UnifiedClassifier:
    """Get singleton UnifiedClassifier instance"""
    global _classifier_instance

    if _classifier_instance is None:
        _classifier_instance = UnifiedClassifier(
            model_name=model_name,
            provider_type=provider_type,
        )

    return _classifier_instance


def reset_classifier():
    """Reset singleton instance (for testing)"""
    global _classifier_instance
    _classifier_instance = None


async def clear_classification_cache():
    """Clear the classification cache (async - clears both Redis and local)"""
    await _classification_cache.clear()


def clear_classification_cache_sync():
    """Clear the classification cache (sync - local cache only)"""
    _classification_cache.clear_sync()