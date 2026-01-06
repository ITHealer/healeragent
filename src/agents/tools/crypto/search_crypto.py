"""
Search Crypto Tool

Search for cryptocurrencies by name or symbol.
Returns matching coins from internal API.

Internal API: http://10.10.0.2:20073/api/v1/market/crypto/search
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.agents.tools.base import ToolOutput, ToolSchema, ToolParameter
from src.agents.tools.crypto.base_crypto_tool import BaseCryptoTool


class SearchCryptoTool(BaseCryptoTool):
    """
    Tool: searchCrypto

    Search for cryptocurrencies by name or symbol.
    Uses internal API for comprehensive crypto database.

    Use when user asks about:
    - Finding a specific crypto
    - What is [coin name]
    - Searching for coins
    - Looking up cryptocurrencies
    """

    CACHE_TTL = 300  # 5 minutes - search results don't change often

    def __init__(self):
        """Initialize SearchCryptoTool"""
        super().__init__()

        self.schema = ToolSchema(
            name="searchCrypto",
            category="crypto",
            description=(
                "Search for cryptocurrencies by name or symbol. "
                "Returns matching coins with basic info and current prices. "
                "Use when user wants to find a specific crypto or explore coins."
            ),
            capabilities=[
                "Search by coin name (partial match)",
                "Search by symbol (exact or partial)",
                "Returns current price and change",
                "Includes market cap and rank info",
            ],
            limitations=[
                "Data from internal crypto API only",
                "5-minute cache for performance",
                "Maximum 50 results per search",
                "English names work best",
            ],
            usage_hints=[
                "User asks: 'find solana' → USE THIS with query='solana'",
                "User asks: 'what is SOL' → USE THIS with query='SOL'",
                "User asks: 'search for meme coins' → USE THIS with query='meme'",
                "Vietnamese: 'tìm coin bitcoin' → USE THIS with query='bitcoin'",
            ],
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query - coin name or symbol (e.g., 'bitcoin', 'BTC', 'sol', 'ethereum')",
                    required=True,
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of results (default: 10, max: 50)",
                    required=False,
                    default=10,
                    min_value=1,
                    max_value=50,
                ),
            ],
            returns={
                "results": "array - List of matching coins with symbol, name, price, market_cap",
                "count": "number - Number of results found",
                "query": "string - Original search query",
                "timestamp": "string - Data timestamp",
            },
            typical_execution_time_ms=500,
            requires_symbol=False,
        )

    async def execute(self, query: str, limit: int = 10, **kwargs) -> ToolOutput:
        """
        Execute searchCrypto

        Args:
            query: Search query (name or symbol)
            limit: Maximum number of results (1-50)

        Returns:
            ToolOutput with search results
        """
        start_time = time.time()

        # Validate and clean query
        query = query.strip()
        if not query:
            return ToolOutput(
                tool_name=self.schema.name,
                status="error",
                error="Search query cannot be empty",
                metadata={},
            )

        # Validate limit
        limit = min(max(1, limit), 50)

        self.logger.info(f"  ┌─ 🔧 TOOL: {self.schema.name}")
        self.logger.info(f"  │  Input: {{query='{query}', limit={limit}}}")

        try:
            # Check cache first
            cache_key = f"crypto:search:{query.lower()}:{limit}"
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
                        "query": query,
                        "limit": limit,
                        "from_cache": True,
                        "execution_time_ms": execution_time,
                    },
                )

            # Fetch from internal API
            response = await self._fetch_api(
                endpoint="/search",
                params={
                    "query": query,
                    "limit": limit,
                },
            )

            # Process response
            results = response.get("data", []) if isinstance(response, dict) else response

            if not results:
                self.logger.info(f"[{self.schema.name}] No results for query: {query}")
                results = []

            # Format result data
            result_data = {
                "results": self._format_results(results[:limit]),
                "count": len(results[:limit]),
                "query": query,
                "timestamp": datetime.now().isoformat(),
            }

            # Cache result
            await self._set_cached_result(cache_key, result_data, ttl=self.CACHE_TTL)

            execution_time = int((time.time() - start_time) * 1000)

            self.logger.info(f"  │  Result: {result_data['count']} coins found")
            self.logger.info(f"  └─ ✅ SUCCESS ({execution_time}ms)")

            return ToolOutput(
                tool_name=self.schema.name,
                status="success",
                data=result_data,
                formatted_context=self._build_formatted_context(result_data),
                execution_time_ms=execution_time,
                metadata={
                    "query": query,
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
                error=f"Failed to search crypto: {str(e)}",
                execution_time_ms=execution_time,
                metadata={"query": query, "limit": limit},
            )

    def _format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format raw search results for output"""
        formatted = []

        for item in results:
            formatted.append({
                "symbol": item.get("symbol", ""),
                "name": item.get("name", item.get("symbol", "")),
                "price": float(item.get("price", 0)),
                "change_24h": float(item.get("change_24h", item.get("change", 0))),
                "change_pct_24h": float(item.get("change_pct_24h", item.get("changesPercentage", 0))),
                "volume_24h": float(item.get("volume_24h", item.get("volume", 0))),
                "market_cap": float(item.get("market_cap", item.get("marketCap", 0))),
                "rank": item.get("rank", item.get("market_cap_rank")),
            })

        return formatted

    def _build_formatted_context(self, data: Dict[str, Any]) -> str:
        """Build human-readable formatted context for LLM"""
        results = data.get("results", [])
        query = data.get("query", "")

        if not results:
            return f"🔍 No cryptocurrencies found matching '{query}'."

        lines = [
            f"🔍 SEARCH RESULTS for '{query}' ({data['count']} matches):",
            "",
        ]

        for i, coin in enumerate(results[:10], 1):
            price_str = self._format_price(coin["price"])
            change_pct = coin["change_pct_24h"]
            emoji = "🟢" if change_pct >= 0 else "🔴"
            sign = "+" if change_pct >= 0 else ""

            rank_str = f"#{coin['rank']}" if coin.get("rank") else ""

            lines.append(
                f"{i}. {coin['symbol']} ({coin['name']}) {rank_str}"
            )
            lines.append(
                f"   {price_str} | {emoji} {sign}{change_pct:.2f}% | "
                f"MCap: {self._format_large_number(coin['market_cap'])}"
            )

        if data["count"] > 10:
            lines.append(f"\n... and {data['count'] - 10} more matches")

        return "\n".join(lines)


# Standalone test
if __name__ == "__main__":
    import asyncio

    async def test():
        tool = SearchCryptoTool()

        print("\n" + "=" * 60)
        print("Testing searchCrypto Tool")
        print("=" * 60)

        # Test various searches
        test_queries = ["bitcoin", "sol", "meme"]

        for query in test_queries:
            print(f"\n--- Searching: {query} ---")
            result = await tool.execute(query=query, limit=3)

            if result.status == "success":
                print(f"✅ Found {result.data['count']} results")
                for r in result.data["results"]:
                    print(f"   {r['symbol']}: {r['name']} @ ${r['price']:.4f}")
            else:
                print(f"❌ Error: {result.error}")

    asyncio.run(test())
