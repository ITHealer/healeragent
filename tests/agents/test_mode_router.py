"""
Unit tests for ModeRouter

Tests the LLM-based complexity classification for AUTO mode routing.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.agents.routing.mode_router import (
    ModeRouter,
    RouterContext,
    QueryComplexity,
    get_mode_router,
)
from src.config.mode_config import (
    ResponseMode,
    ModeConfig,
    ModeClassificationResult,
    get_mode_config,
    get_effective_config,
    FAST_MODE_CONFIG,
    EXPERT_MODE_CONFIG,
    AUTO_MODE_CONFIG,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def router():
    """Create a ModeRouter instance for testing"""
    return ModeRouter(enable_cache=False)


@pytest.fixture
def router_with_cache():
    """Create a ModeRouter with cache enabled"""
    return ModeRouter(enable_cache=True)


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider"""
    provider = AsyncMock()
    return provider


# ============================================================================
# QUICK HEURISTICS TESTS
# ============================================================================

class TestQuickHeuristics:
    """Test the quick heuristic rules that bypass LLM classification"""

    def test_very_short_query_routes_to_fast(self, router):
        """Very short queries (<15 chars) should route to FAST"""
        context = RouterContext(query="AAPL?", recent_symbols=["AAPL"])
        result = router._quick_heuristics("AAPL?", context)

        assert result is not None
        assert result.effective_mode == ResponseMode.FAST
        assert result.detection_method == "heuristic_short"

    def test_multi_symbol_routes_to_expert(self, router):
        """Multiple symbols should route to EXPERT"""
        context = RouterContext(
            query="Compare NVDA and AMD",
            recent_symbols=["NVDA", "AMD"]
        )
        result = router._quick_heuristics("Compare NVDA and AMD", context)

        assert result is not None
        assert result.effective_mode == ResponseMode.EXPERT
        assert result.detection_method == "heuristic_multi_symbol"

    def test_context_continuity_stays_expert(self, router):
        """If previous mode was EXPERT and long query, stay EXPERT"""
        context = RouterContext(
            query="Can you also analyze their growth prospects?",
            recent_symbols=["NVDA"],
            previous_mode="expert",
            conversation_turn=2
        )
        result = router._quick_heuristics(
            "Can you also analyze their growth prospects?",
            context
        )

        assert result is not None
        assert result.effective_mode == ResponseMode.EXPERT
        assert result.detection_method == "heuristic_continuity"

    def test_medium_query_needs_llm(self, router):
        """Medium-length queries with single symbol need LLM classification"""
        context = RouterContext(
            query="What is the PE ratio of AAPL?",
            recent_symbols=["AAPL"]
        )
        result = router._quick_heuristics(
            "What is the PE ratio of AAPL?",
            context
        )

        # Should return None to proceed to LLM classification
        assert result is None


# ============================================================================
# FALLBACK CLASSIFICATION TESTS
# ============================================================================

class TestFallbackClassification:
    """Test fallback behavior when LLM is unavailable"""

    def test_fallback_defaults_to_fast(self, router):
        """Default fallback should be FAST mode"""
        context = RouterContext(query="Test query", recent_symbols=[])
        result = router._fallback_classification("Test query", context, "test_reason")

        assert result.effective_mode == ResponseMode.FAST
        assert result.detection_method == "fallback"
        assert "test_reason" in result.reason

    def test_fallback_multi_symbol_routes_to_expert(self, router):
        """Fallback with multiple symbols should still route to EXPERT"""
        context = RouterContext(
            query="Compare stocks",
            recent_symbols=["AAPL", "MSFT"]
        )
        result = router._fallback_classification("Compare stocks", context, "timeout")

        assert result.effective_mode == ResponseMode.EXPERT
        assert "multi_symbol" in result.reason


# ============================================================================
# LLM CLASSIFICATION TESTS
# ============================================================================

class TestLLMClassification:
    """Test LLM-based semantic classification"""

    @pytest.mark.asyncio
    async def test_llm_classifies_simple_query(self, router, mock_provider):
        """LLM should classify simple queries as SIMPLE"""
        mock_provider.generate = AsyncMock(return_value={
            "content": json.dumps({
                "complexity": "simple",
                "reason": "single price lookup",
                "confidence": 0.95
            })
        })

        context = RouterContext(query="Giá AAPL?", recent_symbols=["AAPL"])
        result = await router._llm_classify("Giá AAPL?", context, mock_provider)

        assert result.effective_mode == ResponseMode.FAST
        assert result.detection_method == "llm_semantic"
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_llm_classifies_complex_query(self, router, mock_provider):
        """LLM should classify complex queries as COMPLEX"""
        mock_provider.generate = AsyncMock(return_value={
            "content": json.dumps({
                "complexity": "complex",
                "reason": "multi-aspect analysis required",
                "confidence": 0.88
            })
        })

        context = RouterContext(
            query="So sánh NVDA và AMD về P/E, revenue growth và market cap",
            recent_symbols=["NVDA", "AMD"]
        )
        result = await router._llm_classify(
            "So sánh NVDA và AMD về P/E, revenue growth và market cap",
            context,
            mock_provider
        )

        assert result.effective_mode == ResponseMode.EXPERT
        assert result.detection_method == "llm_semantic"

    @pytest.mark.asyncio
    async def test_llm_timeout_uses_fallback(self, router, mock_provider):
        """LLM timeout should trigger fallback classification"""
        mock_provider.generate = AsyncMock(side_effect=asyncio.TimeoutError())

        context = RouterContext(query="Test query", recent_symbols=[])
        result = await router.classify("Test query", context, mock_provider)

        assert result.detection_method == "fallback"
        assert "timeout" in result.reason


# ============================================================================
# CACHING TESTS
# ============================================================================

class TestCaching:
    """Test classification result caching"""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(self, router_with_cache, mock_provider):
        """Second call with same query should use cache"""
        mock_provider.generate = AsyncMock(return_value={
            "content": json.dumps({
                "complexity": "simple",
                "reason": "cached test",
                "confidence": 0.9
            })
        })

        context = RouterContext(query="AAPL price", recent_symbols=["AAPL"])

        # First call - should call LLM
        result1 = await router_with_cache.classify("AAPL price", context, mock_provider)

        # Second call - should use cache
        result2 = await router_with_cache.classify("AAPL price", context, mock_provider)

        # LLM should only be called once
        assert mock_provider.generate.call_count == 1
        assert result1.effective_mode == result2.effective_mode

    def test_clear_cache(self, router_with_cache):
        """Cache should be clearable"""
        # Add something to cache manually
        router_with_cache._cache["test_key"] = (
            ModeClassificationResult(
                effective_mode=ResponseMode.FAST,
                reason="test",
                confidence=1.0,
                detection_method="test"
            ),
            None
        )

        router_with_cache.clear_cache()
        assert len(router_with_cache._cache) == 0


# ============================================================================
# FULL ROUTING TESTS
# ============================================================================

class TestFullRouting:
    """Test the complete route() method"""

    @pytest.mark.asyncio
    async def test_explicit_fast_mode_skips_classification(self, router, mock_provider):
        """Explicit FAST mode should skip classification entirely"""
        config, result = await router.route(
            query="Any query",
            user_mode="fast",
            provider=mock_provider
        )

        assert config.mode == ResponseMode.FAST
        assert result.detection_method == "explicit_user"
        mock_provider.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_expert_mode_skips_classification(self, router, mock_provider):
        """Explicit EXPERT mode should skip classification entirely"""
        config, result = await router.route(
            query="Any query",
            user_mode="expert",
            provider=mock_provider
        )

        assert config.mode == ResponseMode.EXPERT
        assert result.detection_method == "explicit_user"
        mock_provider.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_mode_classifies_query(self, router, mock_provider):
        """AUTO mode should classify the query"""
        mock_provider.generate = AsyncMock(return_value={
            "content": json.dumps({
                "complexity": "complex",
                "reason": "comparison needed",
                "confidence": 0.85
            })
        })

        context = RouterContext(
            query="Compare AAPL and MSFT",
            recent_symbols=["AAPL", "MSFT"]
        )
        config, result = await router.route(
            query="Compare AAPL and MSFT",
            user_mode="auto",
            context=context,
            provider=mock_provider
        )

        # Should route to EXPERT because multi-symbol heuristic
        assert config.mode == ResponseMode.EXPERT


# ============================================================================
# MODE CONFIG TESTS
# ============================================================================

class TestModeConfig:
    """Test mode configuration helpers"""

    def test_get_mode_config_fast(self):
        """get_mode_config should return FAST config"""
        config = get_mode_config("fast")
        assert config.mode == ResponseMode.FAST
        assert config.max_turns == 2
        assert config.enable_web_search is False

    def test_get_mode_config_expert(self):
        """get_mode_config should return EXPERT config"""
        config = get_mode_config("expert")
        assert config.mode == ResponseMode.EXPERT
        assert config.max_turns == 6
        assert config.enable_web_search is True

    def test_get_mode_config_invalid_defaults_to_auto(self):
        """Invalid mode should default to AUTO"""
        config = get_mode_config("invalid_mode")
        assert config.mode == ResponseMode.AUTO

    def test_get_effective_config_auto_fast(self):
        """AUTO mode with fast classification should return FAST config"""
        config = get_effective_config("auto", "fast")
        assert config.mode == ResponseMode.FAST

    def test_get_effective_config_auto_expert(self):
        """AUTO mode with expert classification should return EXPERT config"""
        config = get_effective_config("auto", "expert")
        assert config.mode == ResponseMode.EXPERT

    def test_fast_mode_config_values(self):
        """Verify FAST mode configuration values"""
        config = FAST_MODE_CONFIG
        assert config.use_classifier is True
        assert config.tool_selection == "filtered"
        assert config.max_tools == 8
        assert config.system_prompt_version == "condensed"

    def test_expert_mode_config_values(self):
        """Verify EXPERT mode configuration values"""
        config = EXPERT_MODE_CONFIG
        assert config.use_classifier is False
        assert config.tool_selection == "all"
        assert config.max_tools == 31
        assert config.system_prompt_version == "full"


# ============================================================================
# SINGLETON TESTS
# ============================================================================

class TestSingleton:
    """Test singleton pattern for ModeRouter"""

    def test_get_mode_router_returns_same_instance(self):
        """get_mode_router should return the same instance"""
        router1 = get_mode_router()
        router2 = get_mode_router()
        assert router1 is router2


# ============================================================================
# MULTILINGUAL TESTS
# ============================================================================

class TestMultilingual:
    """Test multilingual query handling"""

    @pytest.mark.asyncio
    async def test_vietnamese_simple_query(self, router, mock_provider):
        """Vietnamese simple query should classify correctly"""
        mock_provider.generate = AsyncMock(return_value={
            "content": json.dumps({
                "complexity": "simple",
                "reason": "price lookup in Vietnamese",
                "confidence": 0.92
            })
        })

        context = RouterContext(query="Giá AAPL?", recent_symbols=["AAPL"])
        result = await router._llm_classify("Giá AAPL?", context, mock_provider)

        assert result.effective_mode == ResponseMode.FAST

    @pytest.mark.asyncio
    async def test_chinese_complex_query(self, router, mock_provider):
        """Chinese complex query should classify correctly"""
        mock_provider.generate = AsyncMock(return_value={
            "content": json.dumps({
                "complexity": "complex",
                "reason": "comparison in Chinese",
                "confidence": 0.88
            })
        })

        context = RouterContext(
            query="详细比较苹果和微软",
            recent_symbols=["AAPL", "MSFT"]
        )
        result = await router._llm_classify("详细比较苹果和微软", context, mock_provider)

        assert result.effective_mode == ResponseMode.EXPERT


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
