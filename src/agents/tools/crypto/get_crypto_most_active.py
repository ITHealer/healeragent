"""
Get Crypto Most Active Tool

Fetches the most actively traded cryptocurrencies from internal API.
Returns coins with highest trading volume.

Internal API: http://10.10.0.2:20073/api/v1/market/crypto/most-active
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.agents.tools.base import ToolOutput, ToolSchema, ToolParameter
from src.agents.tools.crypto.base_crypto_tool import BaseCryptoTool


class GetCryptoMostActiveTool(BaseCryptoTool):
    """
    Tool: getCryptoMostActive

    Fetch most actively traded cryptocurrencies by volume.
    Uses internal API for real-time market data.

    Use when user asks about:
    - Most traded cryptos
    - Highest volume coins
    - Active crypto markets
    - Where the volume is
    """

    CACHE_TTL = 60  # 1 minute - volume changes frequently

    def __init__(self):
        """Initialize GetCryptoMostActiveTool"""
        super().__init__()

        self.schema = ToolSchema(
            name="getCryptoMostActive",
            category="crypto",
            description=(
                "Fetch most actively traded cryptocurrencies ranked by trading volume. "
                "Returns coins with highest 24h trading volume. "
                "Use when user asks about most traded coins, volume leaders, or active markets."
            ),
            capabilities=[
                "Real-time most active from internal API",
                "Ranked by 24h trading volume",
                "Includes price, change, market cap data",
                "Configurable result limit (1-100)",
            ],
            limitations=[
                "Data from internal crypto API only",
                "1-minute cache for performance",
                "Maximum 100 results",
            ],
            usage_hints=[
                "User asks: 'most traded crypto' → USE THIS",
                "User asks: 'highest volume coins' → USE THIS",
                "User asks: 'where is the volume' → USE THIS",
                "Vietnamese: 'coin giao dịch nhiều nhất' → USE THIS",
            ],
            parameters=[
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Number of most active to return (default: 10, max: 100)",
                    required=False,
                    default=10,
                    min_value=1,
                    max_value=100,
                ),
            ],
            returns={
                "active": "array - List of most active with symbol, name, price, volume, change_pct",
                "count": "number - Number of results returned",
                "timestamp": "string - Data timestamp",
            },
            typical_execution_time_ms=500,
            requires_symbol=False,
        )

    async def execute(self, limit: int = 10, **kwargs) -> ToolOutput:
        """
        Execute getCryptoMostActive

        Args:
            limit: Number of results to return (1-100)

        Returns:
            ToolOutput with most active data
        """
        start_time = time.time()

        # Validate limit
        limit = min(max(1, limit), 100)

        self.logger.info(f"  ┌─ 🔧 TOOL: {self.schema.name}")
        self.logger.info(f"  │  Input: {{limit={limit}}}")

        try:
            # Check cache first
            cache_key = f"crypto:most_active:{limit}"
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
                endpoint="/most-active",
                params={"limit": limit},
            )

            # Process response
            active = response.get("data", []) if isinstance(response, dict) else response

            if not active:
                self.logger.warning(f"[{self.schema.name}] No active data returned")
                active = []

            # Format result data
            result_data = {
                "active": self._format_active(active[:limit]),
                "count": len(active[:limit]),
                "timestamp": datetime.now().isoformat(),
            }

            # Cache result
            await self._set_cached_result(cache_key, result_data, ttl=self.CACHE_TTL)

            execution_time = int((time.time() - start_time) * 1000)

            self.logger.info(f"  │  Result: {result_data['count']} active coins")
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
                error=f"Failed to fetch most active: {str(e)}",
                execution_time_ms=execution_time,
                metadata={"limit": limit},
            )

    def _format_active(self, active: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format raw active data for output"""
        formatted = []

        for item in active:
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
        active = data.get("active", [])

        if not active:
            return "🚫 No most active data available at this time."

        lines = [
            f"📊 CRYPTO MOST ACTIVE ({data['count']} coins by volume):",
            "",
        ]

        for i, coin in enumerate(active[:10], 1):
            price_str = self._format_price(coin["price"])
            volume_str = self._format_large_number(coin["volume_24h"])
            change_pct = coin["change_pct_24h"]
            emoji = "🟢" if change_pct >= 0 else "🔴"
            sign = "+" if change_pct >= 0 else ""

            lines.append(
                f"{i}. {coin['symbol']} - {price_str} | "
                f"Vol: {volume_str} | {emoji} {sign}{change_pct:.2f}%"
            )

        if data["count"] > 10:
            lines.append(f"\n... and {data['count'] - 10} more active coins")

        return "\n".join(lines)


# Standalone test
if __name__ == "__main__":
    import asyncio

    async def test():
        tool = GetCryptoMostActiveTool()

        print("\n" + "=" * 60)
        print("Testing getCryptoMostActive Tool")
        print("=" * 60)

        result = await tool.execute(limit=5)

        if result.status == "success":
            print(f"\n✅ Success: {result.data['count']} active coins")
            for a in result.data["active"][:5]:
                print(f"   {a['symbol']}: ${a['price']:.4f} | Vol: {a['volume_24h']:,.0f}")
        else:
            print(f"\n❌ Error: {result.error}")

    asyncio.run(test())
