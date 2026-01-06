"""
Get Crypto Top Losers Tool

Fetches the top losing cryptocurrencies from internal API.
Returns coins with highest negative price change percentage.

Internal API: http://10.10.0.2:20073/api/v1/market/crypto/top-losers
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.agents.tools.base import ToolOutput, ToolSchema, ToolParameter
from src.agents.tools.crypto.base_crypto_tool import BaseCryptoTool


class GetCryptoTopLosersTool(BaseCryptoTool):
    """
    Tool: getCryptoTopLosers

    Fetch top losing cryptocurrencies by price change percentage.
    Uses internal API for real-time market data.

    Use when user asks about:
    - Worst performing cryptos
    - Top losers in crypto market
    - Which coins are dumping
    - Crypto to avoid
    """

    CACHE_TTL = 60  # 1 minute - market moves fast

    def __init__(self):
        """Initialize GetCryptoTopLosersTool"""
        super().__init__()

        self.schema = ToolSchema(
            name="getCryptoTopLosers",
            category="crypto",
            description=(
                "Fetch top losing cryptocurrencies ranked by price change percentage. "
                "Returns the worst performing coins in the market right now. "
                "Use when user asks about worst performers, top losers, dumping coins, or what to avoid."
            ),
            capabilities=[
                "Real-time top losers from internal API",
                "Ranked by 24h price change percentage (most negative first)",
                "Includes price, volume, market cap data",
                "Configurable result limit (1-100)",
            ],
            limitations=[
                "Data from internal crypto API only",
                "1-minute cache for performance",
                "Maximum 100 results",
            ],
            usage_hints=[
                "User asks: 'top crypto losers' → USE THIS",
                "User asks: 'which coins dumping today' → USE THIS",
                "User asks: 'worst performing crypto' → USE THIS",
                "Vietnamese: 'coin giảm mạnh nhất' → USE THIS",
            ],
            parameters=[
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Number of top losers to return (default: 10, max: 100)",
                    required=False,
                    default=10,
                    min_value=1,
                    max_value=100,
                ),
            ],
            returns={
                "losers": "array - List of top losers with symbol, name, price, change_pct, volume",
                "count": "number - Number of results returned",
                "timestamp": "string - Data timestamp",
            },
            typical_execution_time_ms=500,
            requires_symbol=False,
        )

    async def execute(self, limit: int = 10, **kwargs) -> ToolOutput:
        """
        Execute getCryptoTopLosers

        Args:
            limit: Number of results to return (1-100)

        Returns:
            ToolOutput with top losers data
        """
        start_time = time.time()

        # Validate limit
        limit = min(max(1, limit), 100)

        self.logger.info(f"  ┌─ 🔧 TOOL: {self.schema.name}")
        self.logger.info(f"  │  Input: {{limit={limit}}}")

        try:
            # Check cache first
            cache_key = f"crypto:top_losers:{limit}"
            cached_result = await self._get_cached_result(cache_key)

            if cached_result:
                execution_time = int((time.time() - start_time) * 1000)
                self.logger.info(f"  │  🎯 [CACHE HIT]")
                self.logger.info(f"  └─ ✅ SUCCESS ({execution_time}ms)")

                return ToolOutput(
                    tool_name=self.schema.name,
                    status="success",
                    data=cached_result,
                    formatted_context=self._build_formatted_context(cached_result),
                    metadata={
                        "limit": limit,
                        "from_cache": True,
                        "execution_time_ms": execution_time,
                    },
                )

            # Fetch from internal API
            response = await self._fetch_api(
                endpoint="/top-losers",
                params={"limit": limit},
            )

            # Process response
            losers = response.get("data", []) if isinstance(response, dict) else response

            if not losers:
                self.logger.warning(f"[{self.schema.name}] No losers data returned")
                losers = []

            # Format result data
            result_data = {
                "losers": self._format_losers(losers[:limit]),
                "count": len(losers[:limit]),
                "timestamp": datetime.now().isoformat(),
            }

            # Cache result
            await self._set_cached_result(cache_key, result_data, ttl=self.CACHE_TTL)

            execution_time = int((time.time() - start_time) * 1000)

            self.logger.info(f"  │  Result: {result_data['count']} losers")
            self.logger.info(f"  └─ ✅ SUCCESS ({execution_time}ms)")

            return ToolOutput(
                tool_name=self.schema.name,
                status="success",
                data=result_data,
                formatted_context=self._build_formatted_context(result_data),
                execution_time_ms=execution_time,
                metadata={
                    "limit": limit,
                    "from_cache": False,
                },
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"  │  Error: {str(e)[:100]}")
            self.logger.info(f"  └─ ❌ FAILED ({execution_time}ms)")

            return ToolOutput(
                tool_name=self.schema.name,
                status="error",
                error=f"Failed to fetch top losers: {str(e)}",
                execution_time_ms=execution_time,
                metadata={"limit": limit},
            )

    def _format_losers(self, losers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format raw loser data for output"""
        formatted = []

        for item in losers:
            formatted.append({
                "symbol": item.get("symbol", ""),
                "name": item.get("name", item.get("symbol", "")),
                "price": float(item.get("price", 0)),
                "change_24h": float(item.get("change_24h", item.get("change", 0))),
                "change_pct_24h": float(item.get("change_pct_24h", item.get("changesPercentage", 0))),
                "volume_24h": float(item.get("volume_24h", item.get("volume", 0))),
                "market_cap": float(item.get("market_cap", item.get("marketCap", 0))),
            })

        return formatted

    def _build_formatted_context(self, data: Dict[str, Any]) -> str:
        """Build human-readable formatted context for LLM"""
        losers = data.get("losers", [])

        if not losers:
            return "🚫 No top losers data available at this time."

        lines = [
            f"📉 CRYPTO TOP LOSERS ({data['count']} coins):",
            "",
        ]

        for i, loser in enumerate(losers[:10], 1):
            price_str = self._format_price(loser["price"])
            change_pct = loser["change_pct_24h"]

            lines.append(
                f"{i}. {loser['symbol']} - {price_str} "
                f"(🔴 {change_pct:.2f}%)"
            )

        if data["count"] > 10:
            lines.append(f"\n... and {data['count'] - 10} more losers")

        return "\n".join(lines)


# Standalone test
if __name__ == "__main__":
    import asyncio

    async def test():
        tool = GetCryptoTopLosersTool()

        print("\n" + "=" * 60)
        print("Testing getCryptoTopLosers Tool")
        print("=" * 60)

        result = await tool.execute(limit=5)

        if result.status == "success":
            print(f"\n✅ Success: {result.data['count']} losers")
            for l in result.data["losers"][:5]:
                print(f"   {l['symbol']}: ${l['price']:.4f} ({l['change_pct_24h']:.2f}%)")
        else:
            print(f"\n❌ Error: {result.error}")

    asyncio.run(test())
