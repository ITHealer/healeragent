"""
Mode Router - LLM-based Complexity Classification

Implements semantic complexity classification for AUTO mode routing.
Uses LLM understanding instead of hardcoded keywords for multilingual support.

Based on industry best practices:
- Claude AI: Model self-decides thinking budget
- GPT-5: Auto-routing with 94% accuracy
- Hybrid approach: LLM semantic + context signals

Key Features:
- NO hardcoded keywords (works in Vietnamese, English, Chinese, etc.)
- Context continuity (inherits previous mode if appropriate)
- Fast classification (~150-300ms with gpt-4o-mini)
- Caching for similar queries
"""

import json
import hashlib
import asyncio
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.utils.logger.custom_logging import LoggerMixin
from src.config.mode_config import (
    ResponseMode,
    ModeConfig,
    ModeClassificationResult,
    get_mode_config,
    get_effective_config,
    FAST_MODE_CONFIG,
    EXPERT_MODE_CONFIG,
    is_feature_enabled,
)


class QueryComplexity(str, Enum):
    """
    Query complexity levels for routing decisions

    SIMPLE: Direct lookups, single data points, definitions
    COMPLEX: Multi-step analysis, comparisons, research
    """
    SIMPLE = "simple"
    COMPLEX = "complex"


@dataclass
class RouterContext:
    """
    Context for routing decisions

    Contains signals that help determine query complexity.
    """
    query: str
    recent_symbols: List[str] = None
    previous_mode: Optional[str] = None
    conversation_turn: int = 0
    user_preference: Optional[str] = None
    session_complexity_history: List[str] = None

    def __post_init__(self):
        self.recent_symbols = self.recent_symbols or []
        self.session_complexity_history = self.session_complexity_history or []


class ModeRouter(LoggerMixin):
    """
    LLM-based Mode Router for AUTO mode complexity classification

    Determines whether a query should be routed to FAST or EXPERT mode
    using semantic understanding instead of hardcoded keywords.

    Usage:
        router = ModeRouter()
        result = await router.classify(
            query="So sánh NVDA và AMD",
            context=RouterContext(
                recent_symbols=["NVDA", "AMD"],
                previous_mode="expert"
            )
        )
        # result.effective_mode == ResponseMode.EXPERT
    """

    # Classification prompt template
    CLASSIFICATION_PROMPT = """Classify query complexity for a financial AI assistant.

Query: "{query}"
Recent symbols: {recent_symbols}
Previous complexity: {previous_mode}
Conversation turn: {turn}

Classification Rules:
- SIMPLE = Single lookups, definitions, greetings, price checks, single indicator
- COMPLEX = Multi-symbol comparison, deep analysis, research, strategy, screening

Examples of SIMPLE:
- "Giá AAPL?" (Price lookup)
- "PE ratio của MSFT?" (Single metric)
- "RSI là gì?" (Definition)
- "NVDA đang bullish hay bearish?" (Quick sentiment)

Examples of COMPLEX:
- "So sánh NVDA và AMD" (Comparison)
- "Phân tích kỹ thuật + định giá GOOGL" (Multi-aspect analysis)
- "Chiến lược đầu tư AI stocks" (Strategy/Research)
- "Tìm cổ phiếu P/E < 30 và revenue growth > 20%" (Screening)

Context Signals:
- Multiple symbols (≥2) usually means COMPLEX
- If previous was COMPLEX and this continues discussion, stay COMPLEX
- Short queries (<20 chars) with single symbol usually SIMPLE

Return ONLY valid JSON:
{{"complexity": "simple" | "complex", "reason": "brief explanation", "confidence": 0.0-1.0}}"""

    # Cache TTL
    CACHE_TTL_SECONDS = 300  # 5 minutes

    def __init__(
        self,
        classifier_model: str = "gpt-4o-mini",
        classifier_timeout_ms: int = 500,
        enable_cache: bool = True,
    ):
        super().__init__()
        self.classifier_model = classifier_model
        self.classifier_timeout_ms = classifier_timeout_ms
        self.enable_cache = enable_cache

        # Simple in-memory cache
        self._cache: Dict[str, tuple] = {}  # hash -> (result, timestamp)

    def _get_cache_key(self, query: str, context: RouterContext) -> str:
        """Generate cache key for query + context"""
        key_data = {
            "query": query.lower().strip(),
            "symbols": sorted(context.recent_symbols) if context.recent_symbols else [],
            "prev_mode": context.previous_mode,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[ModeClassificationResult]:
        """Check if result is cached and still valid"""
        if not self.enable_cache or cache_key not in self._cache:
            return None

        result, timestamp = self._cache[cache_key]
        if datetime.utcnow() - timestamp < timedelta(seconds=self.CACHE_TTL_SECONDS):
            self.logger.debug(f"[ModeRouter] Cache hit for key {cache_key[:8]}")
            return result

        # Expired, remove from cache
        del self._cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, result: ModeClassificationResult):
        """Cache the classification result"""
        if self.enable_cache:
            self._cache[cache_key] = (result, datetime.utcnow())
            # Limit cache size
            if len(self._cache) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(
                    self._cache.keys(),
                    key=lambda k: self._cache[k][1]
                )[:100]
                for k in oldest_keys:
                    del self._cache[k]

    async def classify(
        self,
        query: str,
        context: Optional[RouterContext] = None,
        provider=None,
    ) -> ModeClassificationResult:
        """
        Classify query complexity using LLM semantic understanding

        Args:
            query: User's query text
            context: Optional context with signals
            provider: Optional LLM provider (will create if not provided)

        Returns:
            ModeClassificationResult with effective mode and reasoning
        """
        context = context or RouterContext(query=query)

        # Check feature flag
        if not is_feature_enabled("llm_router_enabled"):
            return self._fallback_classification(query, context, "feature_disabled")

        # Check cache first
        cache_key = self._get_cache_key(query, context)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result

        # Quick heuristics before LLM call
        quick_result = self._quick_heuristics(query, context)
        if quick_result:
            self._set_cache(cache_key, quick_result)
            return quick_result

        # LLM classification
        try:
            result = await self._llm_classify(query, context, provider)
            self._set_cache(cache_key, result)
            return result

        except asyncio.TimeoutError:
            self.logger.warning(f"[ModeRouter] Classification timeout, using fallback")
            return self._fallback_classification(query, context, "timeout")

        except Exception as e:
            self.logger.error(f"[ModeRouter] Classification error: {e}")
            return self._fallback_classification(query, context, f"error: {str(e)}")

    def _quick_heuristics(
        self,
        query: str,
        context: RouterContext
    ) -> Optional[ModeClassificationResult]:
        """
        Quick heuristics that can skip LLM call

        Returns result if confident, None if should proceed to LLM.
        """
        query_lower = query.lower().strip()
        query_length = len(query_lower)

        # Very short queries (< 15 chars) are usually simple
        if query_length < 15 and len(context.recent_symbols) <= 1:
            return ModeClassificationResult(
                effective_mode=ResponseMode.FAST,
                reason="very_short_query",
                confidence=0.85,
                detection_method="heuristic_short",
                query_features={"length": query_length}
            )

        # Multiple symbols (>= 2) strongly suggests comparison/complex
        if len(context.recent_symbols) >= 2:
            return ModeClassificationResult(
                effective_mode=ResponseMode.EXPERT,
                reason="multi_symbol_detected",
                confidence=0.90,
                detection_method="heuristic_multi_symbol",
                query_features={"symbol_count": len(context.recent_symbols)}
            )

        # Context continuity: if previous was EXPERT and query continues discussion
        if (context.previous_mode == "expert" and
            context.conversation_turn > 0 and
            query_length > 30):
            return ModeClassificationResult(
                effective_mode=ResponseMode.EXPERT,
                reason="context_continuity",
                confidence=0.80,
                detection_method="heuristic_continuity",
                query_features={"previous_mode": context.previous_mode}
            )

        # Can't decide quickly, need LLM
        return None

    async def _llm_classify(
        self,
        query: str,
        context: RouterContext,
        provider=None,
    ) -> ModeClassificationResult:
        """
        Use LLM for semantic complexity classification

        This is the core multilingual classification that works
        in Vietnamese, English, Chinese, Japanese, etc.
        """
        # Build prompt
        prompt = self.CLASSIFICATION_PROMPT.format(
            query=query,
            recent_symbols=context.recent_symbols or [],
            previous_mode=context.previous_mode or "unknown",
            turn=context.conversation_turn,
        )

        # Use provided provider or create one
        if provider is None:
            from src.providers.provider_factory import ProviderFactory
            provider = ProviderFactory.get_provider("openai")

        try:
            # Make LLM call with timeout
            response = await asyncio.wait_for(
                provider.generate(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.classifier_model,
                    response_format={"type": "json_object"},
                    max_tokens=100,
                    temperature=0.0,  # Deterministic for consistency
                ),
                timeout=self.classifier_timeout_ms / 1000.0
            )

            # Parse response
            content = response.get("content", "{}")
            if isinstance(content, str):
                result_data = json.loads(content)
            else:
                result_data = content

            complexity = result_data.get("complexity", "simple").lower()
            reason = result_data.get("reason", "llm_classification")
            confidence = float(result_data.get("confidence", 0.7))

            # Map to response mode
            effective_mode = (
                ResponseMode.EXPERT if complexity == "complex"
                else ResponseMode.FAST
            )

            return ModeClassificationResult(
                effective_mode=effective_mode,
                reason=reason,
                confidence=confidence,
                detection_method="llm_semantic",
                query_features={
                    "raw_complexity": complexity,
                    "symbol_count": len(context.recent_symbols),
                    "query_length": len(query),
                }
            )

        except json.JSONDecodeError as e:
            self.logger.warning(f"[ModeRouter] JSON parse error: {e}")
            return self._fallback_classification(query, context, "json_parse_error")

    def _fallback_classification(
        self,
        query: str,
        context: RouterContext,
        reason: str
    ) -> ModeClassificationResult:
        """
        Fallback classification when LLM is unavailable

        Default to FAST for safety (prefer speed when unsure).
        """
        # Exception: multiple symbols still route to EXPERT
        if len(context.recent_symbols) >= 2:
            return ModeClassificationResult(
                effective_mode=ResponseMode.EXPERT,
                reason=f"fallback_multi_symbol ({reason})",
                confidence=0.70,
                detection_method="fallback",
                query_features={"fallback_reason": reason}
            )

        return ModeClassificationResult(
            effective_mode=ResponseMode.FAST,
            reason=f"fallback_default ({reason})",
            confidence=0.60,
            detection_method="fallback",
            query_features={"fallback_reason": reason}
        )

    async def route(
        self,
        query: str,
        user_mode: str,
        context: Optional[RouterContext] = None,
        provider=None,
    ) -> tuple[ModeConfig, ModeClassificationResult]:
        """
        Main routing method - returns mode config and classification result

        Args:
            query: User's query
            user_mode: User-selected mode ("auto", "fast", "expert")
            context: Optional routing context
            provider: Optional LLM provider

        Returns:
            Tuple of (ModeConfig to use, ClassificationResult)
        """
        context = context or RouterContext(query=query)

        # Explicit mode selection - skip classification
        if user_mode.lower() == ResponseMode.FAST.value:
            result = ModeClassificationResult(
                effective_mode=ResponseMode.FAST,
                reason="explicit_user_selection",
                confidence=1.0,
                detection_method="explicit_user",
            )
            return FAST_MODE_CONFIG, result

        if user_mode.lower() == ResponseMode.EXPERT.value:
            result = ModeClassificationResult(
                effective_mode=ResponseMode.EXPERT,
                reason="explicit_user_selection",
                confidence=1.0,
                detection_method="explicit_user",
            )
            return EXPERT_MODE_CONFIG, result

        # AUTO mode - classify
        classification = await self.classify(query, context, provider)

        # Get effective config based on classification
        config = get_effective_config(
            base_mode=user_mode,
            classified_complexity=classification.effective_mode.value
        )

        return config, classification

    def clear_cache(self):
        """Clear the classification cache"""
        self._cache.clear()
        self.logger.info("[ModeRouter] Cache cleared")


# ============================================================================
# MODULE-LEVEL SINGLETON
# ============================================================================

_mode_router_instance: Optional[ModeRouter] = None


def get_mode_router() -> ModeRouter:
    """Get singleton ModeRouter instance"""
    global _mode_router_instance
    if _mode_router_instance is None:
        _mode_router_instance = ModeRouter()
    return _mode_router_instance
