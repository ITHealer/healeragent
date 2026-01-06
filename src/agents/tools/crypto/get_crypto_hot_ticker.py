"""
Get Crypto Hot Ticker Tool

Fetches trending/hot cryptocurrencies from internal API.
Returns coins that are currently trending based on momentum and interest.

Internal API: http://10.10.0.2:20073/api/v1/market/crypto/hot-ticker
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.agents.tools.base import ToolOutput, ToolSchema, ToolParameter
from src.agents.tools.crypto.base_crypto_tool import BaseCryptoTool


class GetCryptoHotTickerTool(BaseCryptoTool):
    """
    Tool: getCryptoHotTicker

    Fetch trending/hot cryptocurrencies based on momentum and interest.
    Uses internal API for real-time market data.

    Use when user asks about:
    - Trending cryptos
    - Hot coins right now
    - What's hot in crypto
    - Momentum plays
    """

    CACHE_TTL = 120  # 2 minutes - trends change moderately

    def __init__(self):
        """Initialize GetCryptoHotTickerTool"""
        super().__init__()

        self.schema = ToolSchema(
            name="getCryptoHotTicker",
            category="crypto",
            description=(
                "Fetch trending/hot cryptocurrencies based on momentum and market interest. "
                "Returns coins that are currently trending in the market. "
                "Use when user asks about trending coins, hot cryptos, or momentum plays."
            ),
            capabilities=[
                "Real-time hot/trending from internal API",
                "Based on momentum, volume surge, and interest",
                "Includes price, change, volume data",
                "Configurable result limit (1-50)",
            ],
            limitations=[
                "Data from internal crypto API only",
                "2-minute cache for performance",
                "Maximum 50 results",
                "Trending algorithm may vary",
            ],
            usage_hints=[
                "User asks: 'trending crypto' → USE THIS",
                "User asks: 'hot coins today' → USE THIS",
                "User asks: 'what's hot in crypto' → USE THIS",
                "Vietnamese: 'coin đang hot' → USE THIS",
                "Vietnamese: 'coin trending' → USE THIS",
            ],
            parameters=[
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Number of hot tickers to return (default: 10, max: 50)",
                    required=False,
                    default=10,
                    min_value=1,
                    max_value=50,
                ),
            ],
            returns={
                "hot_tickers": "array - List of hot tickers with symbol, name, price, momentum metrics",
                "count": "number - Number of results returned",
                "timestamp": "string - Data timestamp",
            },
            typical_execution_time_ms=500,
            requires_symbol=False,
        )

    async def execute(self, limit: int = 10, **kwargs) -> ToolOutput:
        """
        Execute getCryptoHotTicker

        Args:
            limit: Number of results to return (1-50)

        Returns:
            ToolOutput with hot ticker data
        """
        start_time = time.time()

        # Validate limit
        limit = min(max(1, limit), 50)

        self.logger.info(f"  ┌─ 🔧 TOOL: {self.schema.name}")
        self.logger.info(f"  │  Input: {{limit={limit}}}")

        try:
            # Check cache first
            cache_key = f"crypto:hot_ticker:{limit}"
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
                endpoint="/hot-ticker",
                params={"limit": limit},
            )

            # Process response
            hot_tickers = response.get("data", []) if isinstance(response, dict) else response

            if not hot_tickers:
                self.logger.warning(f"[{self.schema.name}] No hot ticker data returned")
                hot_tickers = []

            # Format result data
            result_data = {
                "hot_tickers": self._format_hot_tickers(hot_tickers[:limit]),
                "count": len(hot_tickers[:limit]),
                "timestamp": datetime.now().isoformat(),
            }

            # Cache result
            await self._set_cached_result(cache_key, result_data, ttl=self.CACHE_TTL)

            execution_time = int((time.time() - start_time) * 1000)

            self.logger.info(f"  │  Result: {result_data['count']} hot tickers")
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
                error=f"Failed to fetch hot tickers: {str(e)}",
                execution_time_ms=execution_time,
                metadata={"limit": limit},
            )

    def _format_hot_tickers(self, tickers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format raw hot ticker data for output"""
        formatted = []

        for item in tickers:
            formatted.append({
                "symbol": item.get("symbol", ""),
                "name": item.get("name", item.get("symbol", "")),
                "price": float(item.get("price", 0)),
                "change_24h": float(item.get("change_24h", item.get("change", 0))),
                "change_pct_24h": float(item.get("change_pct_24h", item.get("changesPercentage", 0))),
                "volume_24h": float(item.get("volume_24h", item.get("volume", 0))),
                "volume_change_pct": float(item.get("volume_change_pct", 0)),
                "market_cap": float(item.get("market_cap", item.get("marketCap", 0))),
                "hot_score": float(item.get("hot_score", item.get("score", 0))),
            })

        return formatted

    def _build_formatted_context(self, data: Dict[str, Any]) -> str:
        """Build human-readable formatted context for LLM"""
        hot_tickers = data.get("hot_tickers", [])

        if not hot_tickers:
            return "🚫 No hot ticker data available at this time."

        lines = [
            f"🔥 CRYPTO HOT TICKERS ({data['count']} trending coins):",
            "",
        ]

        for i, ticker in enumerate(hot_tickers[:10], 1):
            price_str = self._format_price(ticker["price"])
            change_pct = ticker["change_pct_24h"]
            emoji = "🟢" if change_pct >= 0 else "🔴"
            sign = "+" if change_pct >= 0 else ""

            hot_indicator = "🔥" * min(3, max(1, int(ticker.get("hot_score", 1) / 30) + 1))

            lines.append(
                f"{i}. {ticker['symbol']} {hot_indicator} - {price_str} | "
                f"{emoji} {sign}{change_pct:.2f}%"
            )

        if data["count"] > 10:
            lines.append(f"\n... and {data['count'] - 10} more trending coins")

        return "\n".join(lines)


# Standalone test
if __name__ == "__main__":
    import asyncio

    async def test():
        tool = GetCryptoHotTickerTool()

        print("\n" + "=" * 60)
        print("Testing getCryptoHotTicker Tool")
        print("=" * 60)

        result = await tool.execute(limit=5)

        if result.status == "success":
            print(f"\n✅ Success: {result.data['count']} hot tickers")
            for t in result.data["hot_tickers"][:5]:
                print(f"   {t['symbol']}: ${t['price']:.4f} ({t['change_pct_24h']:+.2f}%)")
        else:
            print(f"\n❌ Error: {result.error}")

    asyncio.run(test())
