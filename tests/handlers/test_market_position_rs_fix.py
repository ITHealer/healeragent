"""
Test to verify that multi-timeframe RS data (21d, 63d, 126d, 252d)
is correctly passed from GetRelativeStrengthTool's formatted_context
to the _build_market_position_summary() prompt builder.

This validates the fix for the bug where result.formatted_context
(containing the detailed RS analysis) was being discarded, and
rs_data.get("llm_summary") returned "" because the key doesn't exist
in result.data.
"""

import sys
import os
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import Dict, Any, Optional
from types import ModuleType

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Set required env vars before importing any src module
_required_envs = {
    "AES_ENCRYPTION_KEY": "test_key_1234567890",
    "ALCHEMY_API_KEY": "test_alchemy",
    "ETHERSCAN_API_KEY": "test_etherscan",
    "TWITTER_API_IO_KEY": "test_twitter",
    "OLLAMA_ENDPOINT": "http://localhost:11434",
    "QDRANT_ENDPOINT": "http://localhost:6333",
    "QDRANT_COLLECTION_NAME": "test_collection",
}
for k, v in _required_envs.items():
    os.environ.setdefault(k, v)

# Auto-mock any missing heavy dependency so we can import MarketScannerHandler
# without needing the full environment (pandas_ta, scipy, yfinance, etc.)
import importlib
import importlib.abc
import importlib.machinery

class _MockFinder(importlib.abc.MetaPathFinder):
    """Auto-mock missing top-level packages and all their submodules."""
    _AUTO_MOCK = {
        "pandas_ta", "scipy", "aioredis", "qdrant_client", "yfinance",
        "sentence_transformers", "bs4", "playwright", "openai",
        "google", "langchain", "langchain_core", "langchain_openai",
        "langchain_community", "mysql", "feedparser", "newsapi",
        "trafilatura", "sqlalchemy", "redis", "sklearn",
        "aiohttp", "websockets", "uvicorn", "fastapi", "starlette",
        "jose", "passlib", "bcrypt", "celery", "kombu",
        "telegram", "tweepy", "anthropic", "groq", "ollama",
        "tiktoken", "tokenizers", "transformers",
    }

    def find_module(self, fullname, path=None):
        top = fullname.split(".")[0]
        if top in self._AUTO_MOCK:
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        # Create a MagicMock that also acts as a package
        mock = MagicMock()
        mock.__path__ = []  # Make it a package so submodule imports work
        mock.__name__ = fullname
        mock.__loader__ = self
        mock.__package__ = fullname
        mock.__spec__ = importlib.machinery.ModuleSpec(fullname, self)
        sys.modules[fullname] = mock
        return mock

sys.meta_path.insert(0, _MockFinder())

from src.agents.tools.base import ToolOutput


class TestMarketPositionRSFix:
    """Test that RS formatted_context is correctly propagated."""

    def _create_mock_tool_output(
        self,
        formatted_context: str = "",
        data: Optional[Dict[str, Any]] = None
    ) -> ToolOutput:
        """Create a mock ToolOutput similar to GetRelativeStrengthTool's output."""
        return ToolOutput(
            tool_name="getRelativeStrength",
            status="success",
            data=data or {
                "symbol": "NVDA",
                "benchmark": "SPY",
                "timeframe": "multi",
                "relative_performance": 5.2,
                "is_outperforming": True,
                "trend": "IMPROVING",
                "percentile_rank": 75,
                "strength_score": 72,
                "summary": "NVDA outperforms SPY",
                "timestamp": "2025-01-30T10:00:00"
            },
            formatted_context=formatted_context,
            metadata={"source": "RelativeStrengthHandler"}
        )

    def test_build_market_position_summary_with_formatted_context(self):
        """
        Test that _build_market_position_summary uses rs_formatted_context
        when provided (the fix).
        """
        from src.handlers.market_scanner_handler import MarketScannerHandler

        handler = MarketScannerHandler()

        rs_data = {
            "symbol": "NVDA",
            "benchmark": "SPY",
            "relative_performance": 5.2,
            "is_outperforming": True,
        }

        rs_formatted_context = """## METHODOLOGY
RS Score = percentile rank of excess return (stock return - benchmark return)
Scale: 1 (weakest) to 99 (strongest), 50 = market-perform

## MULTI-TIMEFRAME RETURNS
- 21d: NVDA 8.50% vs SPY 3.20% = 5.30% (OUTPERFORMING)
  └─ RS Score: 78/100
- 63d: NVDA 15.20% vs SPY 7.10% = 8.10% (OUTPERFORMING)
  └─ RS Score: 82/100
- 126d: NVDA 25.40% vs SPY 12.30% = 13.10% (OUTPERFORMING)
  └─ RS Score: 85/100
- 252d: NVDA 45.60% vs SPY 20.10% = 25.50% (OUTPERFORMING)
  └─ RS Score: 90/100

### OVERALL CLASSIFICATION: LEADER
### RS TREND: IMPROVING"""

        result = handler._build_market_position_summary(
            symbol="NVDA",
            benchmark="SPY",
            rs_data=rs_data,
            sector_context=None,
            rs_formatted_context=rs_formatted_context
        )

        # Verify that multi-timeframe data is in the output
        assert "21d" in result, "21d timeframe data missing from summary"
        assert "63d" in result, "63d timeframe data missing from summary"
        assert "126d" in result, "126d timeframe data missing from summary"
        assert "252d" in result, "252d timeframe data missing from summary"
        assert "LEADER" in result, "Classification missing from summary"
        assert "IMPROVING" in result, "RS trend missing from summary"
        assert "MULTI-TIMEFRAME" in result or "Multi-timeframe" in result.lower(), \
            "Multi-timeframe header missing"
        assert "RS Score: 78/100" in result, "21d RS score missing"
        assert "RS Score: 90/100" in result, "252d RS score missing"

    def test_build_market_position_summary_without_formatted_context(self):
        """
        Test that without rs_formatted_context, the summary falls back
        to rs_data.get("llm_summary") - which is empty in the old behavior.
        """
        from src.handlers.market_scanner_handler import MarketScannerHandler

        handler = MarketScannerHandler()

        rs_data = {
            "symbol": "NVDA",
            "benchmark": "SPY",
            "relative_performance": 5.2,
        }

        result = handler._build_market_position_summary(
            symbol="NVDA",
            benchmark="SPY",
            rs_data=rs_data,
            sector_context=None,
            rs_formatted_context=""  # Empty - simulates old behavior
        )

        # Without formatted_context and no llm_summary in rs_data,
        # the RS section should NOT appear
        assert "RELATIVE STRENGTH DATA" not in result

    def test_build_market_position_summary_fallback_to_rs_data_llm_summary(self):
        """
        Test backward compatibility: if rs_formatted_context is empty but
        rs_data has "llm_summary" key, it should still work.
        """
        from src.handlers.market_scanner_handler import MarketScannerHandler

        handler = MarketScannerHandler()

        rs_data = {
            "symbol": "NVDA",
            "benchmark": "SPY",
            "llm_summary": "Fallback RS summary with 21d 63d 126d 252d data",
        }

        result = handler._build_market_position_summary(
            symbol="NVDA",
            benchmark="SPY",
            rs_data=rs_data,
            sector_context=None,
            rs_formatted_context=""  # Empty - trigger fallback
        )

        assert "Fallback RS summary" in result
        assert "RELATIVE STRENGTH DATA" in result

    def test_get_market_position_extracts_formatted_context(self):
        """
        Integration test: verify get_market_position() correctly extracts
        formatted_context from ToolOutput and passes it through.
        """
        from src.handlers.market_scanner_handler import MarketScannerHandler

        handler = MarketScannerHandler()

        mock_formatted_context = "## RS Analysis\n- 21d: Score 78\n- 63d: Score 82\n- 126d: Score 85\n- 252d: Score 90"

        mock_result = self._create_mock_tool_output(
            formatted_context=mock_formatted_context
        )

        # Mock the rs_tool.execute to return our mock (use _rs_tool as it's a lazy property)
        mock_tool = AsyncMock()
        mock_tool.execute = AsyncMock(return_value=mock_result)
        handler._rs_tool = mock_tool

        # Mock _get_sector_context
        handler._get_sector_context = AsyncMock(return_value=None)

        # Run the async method
        result = asyncio.get_event_loop().run_until_complete(
            handler.get_market_position(symbol="NVDA", benchmark="SPY")
        )

        assert result["success"] is True
        assert "llm_summary" in result

        # The llm_summary should contain the RS data from formatted_context
        llm_summary = result["llm_summary"]
        assert "21d" in llm_summary, "21d missing from get_market_position output"
        assert "63d" in llm_summary, "63d missing from get_market_position output"
        assert "126d" in llm_summary, "126d missing from get_market_position output"
        assert "252d" in llm_summary, "252d missing from get_market_position output"

    def test_tool_output_data_does_not_contain_llm_summary(self):
        """
        Verify that the ToolOutput.data dict from GetRelativeStrengthTool
        does NOT contain 'llm_summary' key - confirming the original bug.
        """
        mock_result = self._create_mock_tool_output(
            formatted_context="Detailed RS analysis here"
        )

        # This is the root cause: data dict doesn't have llm_summary
        assert "llm_summary" not in mock_result.data
        # But formatted_context does have the analysis
        assert mock_result.formatted_context == "Detailed RS analysis here"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
