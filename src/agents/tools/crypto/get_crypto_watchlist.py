"""
Get Crypto Watchlist Tool

Fetches user's cryptocurrency watchlist from internal API.
Returns coins that the user is tracking/watching.

Internal API: http://10.10.0.2:20073/api/v1/market/crypto/watch-list
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.agents.tools.base import ToolOutput, ToolSchema, ToolParameter
from src.agents.tools.crypto.base_crypto_tool import BaseCryptoTool


class GetCryptoWatchlistTool(BaseCryptoTool):
    """
    Tool: getCryptoWatchlist

    Fetch user's cryptocurrency watchlist.
    Uses internal API for real-time market data.

    Use when user asks about:
    - Their watchlist
    - Coins they're watching
    - Portfolio overview
    - Tracked cryptocurrencies
    """

    CACHE_TTL = 30  # 30 seconds - watchlist updates frequently

    def __init__(self):
        """Initialize GetCryptoWatchlistTool"""
        super().__init__()

        self.schema = ToolSchema(
            name="getCryptoWatchlist",
            category="crypto",
            description=(
                "Fetch user's cryptocurrency watchlist with current prices and changes. "
                "Returns all coins the user is tracking. "
                "Use when user asks about their watchlist, tracked coins, or portfolio."
            ),
            capabilities=[
                "Real-time watchlist data from internal API",
                "Current prices for all watched coins",
                "24h change and volume for each coin",
                "User-specific watchlist",
            ],
            limitations=[
                "Data from internal crypto API only",
                "30-second cache for performance",
                "Requires user_id for personalized list",
                "Returns default watchlist if no user_id",
            ],
            usage_hints=[
                "User asks: 'my watchlist' → USE THIS",
                "User asks: 'coins I'm watching' → USE THIS",
                "User asks: 'show my crypto' → USE THIS",
                "Vietnamese: 'danh sách theo dõi' → USE THIS",
                "Vietnamese: 'coin của tôi' → USE THIS",
            ],
            parameters=[
                ToolParameter(
                    name="user_id",
                    type="integer",
                    description="User ID for personalized watchlist (optional, uses default if not provided)",
                    required=False,
                ),
            ],
            returns={
                "watchlist": "array - List of watched coins with symbol, name, price, change_pct, volume",
                "count": "number - Number of coins in watchlist",
                "timestamp": "string - Data timestamp",
            },
            typical_execution_time_ms=500,
            requires_symbol=False,
        )

    async def execute(self, user_id: Optional[int] = None, **kwargs) -> ToolOutput:
        """
        Execute getCryptoWatchlist

        Args:
            user_id: Optional user ID for personalized watchlist

        Returns:
            ToolOutput with watchlist data
        """
        start_time = time.time()

        self.logger.info(f"  ┌─ 🔧 TOOL: {self.schema.name}")
        self.logger.info(f"  │  Input: {{user_id={user_id}}}")

        try:
            # Check cache first
            cache_key = f"crypto:watchlist:{user_id or 'default'}"
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
                        "user_id": user_id,
                        "from_cache": True,
                        "execution_time_ms": execution_time,
                    },
                )

            # Build params
            params = {}
            if user_id:
                params["user_id"] = user_id

            # Fetch from internal API
            response = await self._fetch_api(
                endpoint="/watch-list",
                params=params if params else None,
            )

            # Process response
            watchlist = response.get("data", []) if isinstance(response, dict) else response

            if not watchlist:
                self.logger.info(f"[{self.schema.name}] Empty watchlist returned")
                watchlist = []

            # Format result data
            result_data = {
                "watchlist": self._format_watchlist(watchlist),
                "count": len(watchlist),
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
            }

            # Cache result
            await self._set_cached_result(cache_key, result_data, ttl=self.CACHE_TTL)

            execution_time = int((time.time() - start_time) * 1000)

            self.logger.info(f"  │  Result: {result_data['count']} coins in watchlist")
            self.logger.info(f"  └─ ✅ SUCCESS ({execution_time}ms)")

            return ToolOutput(
                tool_name=self.schema.name,
                status="success",
                data=result_data,
                formatted_context=self._build_formatted_context(result_data),
                execution_time_ms=execution_time,
                metadata={
                    "user_id": user_id,
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
                error=f"Failed to fetch watchlist: {str(e)}",
                execution_time_ms=execution_time,
                metadata={"user_id": user_id},
            )

    def _format_watchlist(self, watchlist: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format raw watchlist data for output"""
        formatted = []

        for item in watchlist:
            formatted.append({
                "symbol": item.get("symbol", ""),
                "name": item.get("name", item.get("symbol", "")),
                "price": float(item.get("price", 0)),
                "change_24h": float(item.get("change_24h", item.get("change", 0))),
                "change_pct_24h": float(item.get("change_pct_24h", item.get("changesPercentage", 0))),
                "volume_24h": float(item.get("volume_24h", item.get("volume", 0))),
                "market_cap": float(item.get("market_cap", item.get("marketCap", 0))),
                "added_at": item.get("added_at", item.get("addedAt")),
            })

        return formatted

    def _build_formatted_context(self, data: Dict[str, Any]) -> str:
        """Build human-readable formatted context for LLM"""
        watchlist = data.get("watchlist", [])

        if not watchlist:
            return "📋 Your watchlist is empty. Add some coins to track!"

        lines = [
            f"📋 CRYPTO WATCHLIST ({data['count']} coins):",
            "",
        ]

        # Calculate totals for summary
        total_positive = sum(1 for w in watchlist if w["change_pct_24h"] >= 0)
        total_negative = len(watchlist) - total_positive

        lines.append(f"Summary: 🟢 {total_positive} up | 🔴 {total_negative} down")
        lines.append("")

        for i, coin in enumerate(watchlist, 1):
            price_str = self._format_price(coin["price"])
            change_pct = coin["change_pct_24h"]
            emoji = "🟢" if change_pct >= 0 else "🔴"
            sign = "+" if change_pct >= 0 else ""

            lines.append(
                f"{i}. {coin['symbol']} - {price_str} | {emoji} {sign}{change_pct:.2f}%"
            )

        return "\n".join(lines)


# Standalone test
if __name__ == "__main__":
    import asyncio

    async def test():
        tool = GetCryptoWatchlistTool()

        print("\n" + "=" * 60)
        print("Testing getCryptoWatchlist Tool")
        print("=" * 60)

        result = await tool.execute()

        if result.status == "success":
            print(f"\n✅ Success: {result.data['count']} coins in watchlist")
            for w in result.data["watchlist"]:
                print(f"   {w['symbol']}: ${w['price']:.4f} ({w['change_pct_24h']:+.2f}%)")
        else:
            print(f"\n❌ Error: {result.error}")

    asyncio.run(test())
