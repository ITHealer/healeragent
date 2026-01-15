# File: src/agents/tools/fundamentals/get_analyst_ratings.py

"""
GetAnalystRatingsTool - Analyst Ratings, Price Targets & Recommendations

Fetches Wall Street analyst data including:
- Price Target Consensus (high, low, median, average)
- Analyst Grades Summary (Strong Buy, Buy, Hold, Sell, Strong Sell counts)
- Price Target Summary with historical trends

FMP Stable APIs:
- /stable/price-target-consensus - Consensus price targets
- /stable/grades-summary - Analyst grades/ratings
- /stable/price-target-summary - Price target history
"""

import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

import httpx

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    create_success_output,
    create_error_output
)


class GetAnalystRatingsTool(BaseTool):
    """
    Atomic tool to fetch Analyst Ratings & Price Targets

    Data Source: FMP Stable API

    Usage:
        tool = GetAnalystRatingsTool()
        result = await tool.safe_execute(symbol="AAPL")

    Returns:
        - price_targets: High, Low, Median, Average price targets
        - analyst_ratings: Count of Strong Buy, Buy, Hold, Sell, Strong Sell
        - consensus: Overall recommendation (e.g., "Moderate Buy")
        - total_analysts: Number of analysts covering the stock
    """

    FMP_STABLE_URL = "https://financialmodelingprep.com/stable"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize tool"""
        super().__init__()

        # Get API key
        if api_key is None:
            api_key = os.environ.get("FMP_API_KEY")

        if not api_key:
            raise ValueError("FMP_API_KEY required")

        self.api_key = api_key
        self.logger = logging.getLogger(__name__)

        # Define schema
        self.schema = ToolSchema(
            name="getAnalystRatings",
            category="fundamentals",
            description=(
                "Fetch Wall Street analyst ratings, price targets, and consensus recommendations. "
                "Returns price target range (high/low/median), analyst grades (Strong Buy to Strong Sell), "
                "and overall market sentiment. Use when user asks about analyst opinions, price targets, "
                "or Wall Street recommendations."
            ),
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol (e.g., AAPL, NVDA, TSLA)",
                    required=True
                )
            ],
            returns={
                "price_targets": "object - High, Low, Median, Average price targets from analysts",
                "analyst_ratings": "object - Count of Strong Buy, Buy, Hold, Sell, Strong Sell ratings",
                "consensus": "string - Overall recommendation (e.g., 'Strong Buy', 'Moderate Buy', 'Hold')",
                "total_analysts": "number - Total number of analysts covering the stock",
                "upside_potential": "number - % upside from current price to consensus target",
                "formatted_summary": "string - LLM-friendly summary of analyst sentiment"
            },
            typical_execution_time_ms=1200,
            requires_symbol=True
        )

    async def execute(
        self,
        symbol: str,
        **kwargs
    ) -> ToolOutput:
        """
        Execute analyst ratings retrieval.

        Args:
            symbol: Stock symbol (e.g., AAPL)

        Returns:
            ToolOutput with analyst data
        """
        start_time = datetime.now()
        symbol_upper = symbol.upper()

        self.logger.info(f"[getAnalystRatings] Executing for symbol={symbol_upper}")

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Fetch data from multiple endpoints in parallel
                price_target_task = self._fetch_price_target_consensus(client, symbol_upper)
                grades_task = self._fetch_grades_summary(client, symbol_upper)

                # Wait for both
                price_target_data, grades_data = await price_target_task, await grades_task

            # Combine results
            result = self._combine_analyst_data(
                symbol_upper,
                price_target_data,
                grades_data
            )

            # Generate LLM-friendly summary
            result["formatted_summary"] = self._format_analyst_summary(result)

            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(f"[getAnalystRatings] SUCCESS ({elapsed_ms:.0f}ms)")

            return create_success_output(
                tool_name=self.schema.name,
                data=result,
                formatted_context=result["formatted_summary"]
            )

        except Exception as e:
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.error(f"[getAnalystRatings] ERROR ({elapsed_ms:.0f}ms): {e}", exc_info=True)

            return create_error_output(
                tool_name=self.schema.name,
                error=str(e),
                metadata={"symbol": symbol_upper}
            )

    async def _fetch_price_target_consensus(
        self,
        client: httpx.AsyncClient,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch price target consensus from FMP stable API"""
        try:
            url = f"{self.FMP_STABLE_URL}/price-target-consensus"
            params = {"symbol": symbol, "apikey": self.api_key}

            response = await client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    return data[0]
                elif data and isinstance(data, dict):
                    return data
            else:
                self.logger.warning(
                    f"[getAnalystRatings] Price target API failed: {response.status_code}"
                )

            return None

        except Exception as e:
            self.logger.warning(f"[getAnalystRatings] Price target fetch error: {e}")
            return None

    async def _fetch_grades_summary(
        self,
        client: httpx.AsyncClient,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch analyst grades summary from FMP stable API"""
        try:
            url = f"{self.FMP_STABLE_URL}/grades-summary"
            params = {"symbol": symbol, "apikey": self.api_key}

            response = await client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    return data[0]
                elif data and isinstance(data, dict):
                    return data
            else:
                self.logger.warning(
                    f"[getAnalystRatings] Grades API failed: {response.status_code}"
                )

            return None

        except Exception as e:
            self.logger.warning(f"[getAnalystRatings] Grades fetch error: {e}")
            return None

    def _combine_analyst_data(
        self,
        symbol: str,
        price_target_data: Optional[Dict],
        grades_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """Combine data from multiple sources into unified result"""
        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "price_targets": None,
            "analyst_ratings": None,
            "consensus": None,
            "total_analysts": 0,
            "upside_potential": None
        }

        # Process price targets
        if price_target_data:
            target_high = price_target_data.get("targetHigh")
            target_low = price_target_data.get("targetLow")
            target_median = price_target_data.get("targetMedian")
            target_consensus = price_target_data.get("targetConsensus")

            result["price_targets"] = {
                "high": target_high,
                "low": target_low,
                "median": target_median,
                "consensus": target_consensus,
                "range": f"${target_low:.2f} - ${target_high:.2f}" if target_low and target_high else None
            }

        # Process analyst grades
        if grades_data:
            strong_buy = grades_data.get("strongBuy", 0) or 0
            buy = grades_data.get("buy", 0) or 0
            hold = grades_data.get("hold", 0) or 0
            sell = grades_data.get("sell", 0) or 0
            strong_sell = grades_data.get("strongSell", 0) or 0

            total = strong_buy + buy + hold + sell + strong_sell

            result["analyst_ratings"] = {
                "strong_buy": strong_buy,
                "buy": buy,
                "hold": hold,
                "sell": sell,
                "strong_sell": strong_sell
            }
            result["total_analysts"] = total

            # Calculate consensus recommendation
            if total > 0:
                bullish = strong_buy + buy
                bearish = sell + strong_sell
                bullish_pct = bullish / total * 100
                bearish_pct = bearish / total * 100

                if bullish_pct >= 70:
                    result["consensus"] = "Strong Buy"
                elif bullish_pct >= 50:
                    result["consensus"] = "Moderate Buy"
                elif bearish_pct >= 50:
                    result["consensus"] = "Sell"
                elif bearish_pct >= 70:
                    result["consensus"] = "Strong Sell"
                else:
                    result["consensus"] = "Hold"

                result["bullish_pct"] = round(bullish_pct, 1)
                result["bearish_pct"] = round(bearish_pct, 1)

        return result

    def _format_analyst_summary(self, data: Dict[str, Any]) -> str:
        """Format analyst data as LLM-friendly summary"""
        symbol = data.get("symbol", "N/A")
        lines = [
            f"## Analyst Ratings & Price Targets: {symbol}",
            ""
        ]

        # Price targets
        pt = data.get("price_targets")
        if pt:
            lines.extend([
                "### Price Targets",
                f"- **Consensus Target**: ${pt.get('consensus', 'N/A')}",
                f"- **Range**: {pt.get('range', 'N/A')}",
                f"- **Median**: ${pt.get('median', 'N/A')}",
                ""
            ])

        # Analyst ratings
        ar = data.get("analyst_ratings")
        total = data.get("total_analysts", 0)
        if ar and total > 0:
            lines.extend([
                f"### Analyst Recommendations ({total} analysts)",
                f"- Strong Buy: {ar.get('strong_buy', 0)}",
                f"- Buy: {ar.get('buy', 0)}",
                f"- Hold: {ar.get('hold', 0)}",
                f"- Sell: {ar.get('sell', 0)}",
                f"- Strong Sell: {ar.get('strong_sell', 0)}",
                ""
            ])

        # Consensus
        consensus = data.get("consensus")
        if consensus:
            bullish_pct = data.get("bullish_pct", 0)
            lines.extend([
                f"### Overall Consensus: **{consensus}**",
                f"- Bullish: {bullish_pct:.0f}% of analysts",
                f"- Bearish: {data.get('bearish_pct', 0):.0f}% of analysts",
                ""
            ])

        return "\n".join(lines)


# =============================================================================
# Standalone Test
# =============================================================================
if __name__ == "__main__":
    import asyncio

    async def test_tool():
        tool = GetAnalystRatingsTool()
        print(f"Testing {tool.schema.name}...")

        result = await tool.safe_execute(symbol="AAPL")

        if result.status == "success":
            print("\nSUCCESS")
            print(result.formatted_context)
            print("\nRaw data:")
            import json
            print(json.dumps(result.data, indent=2, default=str))
        else:
            print(f"\nERROR: {result.error}")

    asyncio.run(test_tool())
