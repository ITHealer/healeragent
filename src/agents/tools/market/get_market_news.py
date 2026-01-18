"""
GetMarketNewsTool - General market news from FMP

Uses: src.helpers.redis_cache helpers
Endpoint: https://financialmodelingprep.com/stable/news/general-latest
"""

import httpx
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    create_success_output,
    create_error_output
)

from src.helpers.redis_cache import get_redis_client_llm


class GetMarketNewsTool(BaseTool):
    """
    Atomic tool for fetching general market news (no symbol required)
    
    Category: news
    Data Source: FMP /stable/news/general-latest
    Cache: Uses aioredis via get_redis_client_llm()
    """

    FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"
    CACHE_TTL = 900  # 15 minutes

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()

        if api_key is None:
            import os
            api_key = os.environ.get("FMP_API_KEY")

        if not api_key:
            raise ValueError("FMP_API_KEY required for GetMarketNewsTool")

        self.api_key = api_key
        self.logger = logging.getLogger(__name__)

        # Tool schema: NO symbol required
        self.schema = ToolSchema(
            name="getMarketNews",
            category="news",
            description=(
                "Fetch latest general market news (macro, markets, economy, major headlines) "
                "from multiple financial news sources via FMP General News API. "
                "Does NOT require a stock symbol. "
                "Use when the user asks for overall market news, global headlines, or general updates."
            ),
            capabilities=[
                "✅ Fetch latest general market news (no symbol required)",
                "✅ Provide headlines, snippets, dates, sources, and URLs",
                "✅ Pagination support via page & limit",
                "✅ Can be used as background context for sentiment / market regime"
            ],
            limitations=[
                "❌ News may be delayed 5–15 minutes compared to real-time",
                "❌ Not filtered by specific stock symbol (use getStockNews for that)",
                "❌ Limited to publicly available sources (no paywalled content)"
            ],
            usage_hints=[
                # When to use
                "User asks: 'Tin tức thị trường hôm nay', 'market news today', 'what's happening in the market?'",
                "User asks: 'Global financial news', 'tin tức tài chính chung', 'tóm tắt tin tức hôm nay'.",
                "User wants a high-level view of macro / market sentiment, not a specific stock.",
                # When NOT to use
                "If user asks about a specific symbol (AAPL, TSLA, NVDA), prefer getStockNews.",
            ],
            parameters=[
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Number of news articles to return (default: 20, max: 50).",
                    required=False,
                    default=20,
                    min_value=1,
                    max_value=50,
                ),
                ToolParameter(
                    name="page",
                    type="integer",
                    description="Page number for pagination (default: 0).",
                    required=False,
                    default=0,
                    min_value=0,
                ),
            ],
            returns={
                "article_count": "number",
                "page": "number",
                "limit": "number",
                "articles": "array - General market news articles",
                "timestamp": "string (ISO-8601)",
            },
            typical_execution_time_ms=1200,
            requires_symbol=False,  # CRITICAL: no symbol needed
        )

    async def execute(
        self,
        limit: int = 20,
        page: int = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute general market news fetch with Redis cache.

        Args:
            limit: Number of articles (default: 20, max: 50)
            page: Page number (default: 0)

        Returns:
            ToolOutput with market news articles
        """
        start_time = datetime.now()

        # Validate limits
        limit = min(max(1, int(limit)), 50)
        page = max(0, int(page))

        self.logger.info(
            f"[getMarketNews] Executing: limit={limit}, page={page}"
        )

        try:
            # Build cache key
            cache_key = f"getMarketNews_{limit}_{page}"

            # Get Redis client
            redis_client = await get_redis_client_llm()

            # Try cache first
            cached_data = None
            if redis_client:
                try:
                    cached_data = await redis_client.get(cache_key)
                    if cached_data:
                        # Handle both bytes and str (depending on decode_responses setting)
                        if isinstance(cached_data, bytes):
                            cached_str = cached_data.decode('utf-8')
                        else:
                            cached_str = cached_data
                        return json.loads(cached_str)
                except Exception as e:
                    self.logger.warning(f"[CACHE] Error reading: {e}")

            if cached_data:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                self.logger.info(
                    f"[getMarketNews] ✅ CACHED ({int(execution_time)}ms)"
                )

                # Generate formatted context for LLM
                formatted_context = self._generate_formatted_context(cached_data)

                return create_success_output(
                    tool_name="getMarketNews",
                    data=cached_data,
                    metadata={
                        "execution_time_ms": int(execution_time),
                        "limit": limit,
                        "page": page,
                        "from_cache": True,
                    },
                    formatted_context=formatted_context
                )

            # Fetch from API
            news_data = await self._fetch_news(limit, page)

            if not news_data:
                return create_error_output(
                    tool_name="getMarketNews",
                    error="No general market news data available",
                    metadata={
                        "limit": limit,
                        "page": page,
                    },
                )

            # Format response
            result_data = self._format_news_data(news_data, limit, page)

            # Cache the result
            if redis_client:
                try:
                    json_string = json.dumps(result_data)
                    await redis_client.set(cache_key, json_string, ex=self.CACHE_TTL)
                    self.logger.info(
                        f"[CACHE SET] {cache_key} (TTL={self.CACHE_TTL}s)"
                    )
                except Exception as e:
                    self.logger.warning(f"[CACHE] Error writing: {e}")

            # Close Redis connection
            if redis_client:
                try:
                    await redis_client.close()
                except Exception as e:
                    self.logger.debug(f"[CACHE] Error closing Redis: {e}")

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            self.logger.info(
                f"[getMarketNews] ✅ SUCCESS ({int(execution_time)}ms) - "
                f"{result_data['article_count']} articles"
            )

            # Generate formatted context for LLM
            formatted_context = self._generate_formatted_context(result_data)

            return create_success_output(
                tool_name="getMarketNews",
                data=result_data,
                metadata={
                    "execution_time_ms": int(execution_time),
                    "limit": limit,
                    "page": page,
                    "from_cache": False,
                    "article_count": result_data["article_count"],
                },
                formatted_context=formatted_context
            )

        except Exception as e:
            self.logger.error(f"[getMarketNews] Error: {e}", exc_info=True)
            return create_error_output(
                tool_name="getMarketNews",
                error=str(e),
                metadata={
                    "limit": limit,
                    "page": page,
                },
            )

    async def _fetch_news(
        self,
        limit: int,
        page: int,
    ) -> Optional[Any]:
        """
        Fetch general market news from FMP Stable API
        Endpoint: /stable/news/general-latest?page={page}&limit={limit}
        """

        url = f"{self.FMP_STABLE_BASE}/news/general-latest"
        params = {
            "page": page,
            "limit": limit,
            "apikey": self.api_key,
        }

        self.logger.info(f"[FMP] GET {url} with params: {params}")

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params=params)

                if response.status_code != 200:
                    self.logger.error(
                        f"[FMP] HTTP {response.status_code}: {response.text[:200]}"
                    )
                    return None

                data = response.json()

                # FMP error pattern
                if isinstance(data, dict) and "Error Message" in data:
                    self.logger.error(f"[FMP] API Error: {data['Error Message']}")
                    return None

                self.logger.info(
                    f"[FMP] ✅ Success: {len(data) if isinstance(data, list) else 1} items"
                )

                return data

        except httpx.TimeoutException:
            self.logger.error("[FMP] Timeout fetching general market news")
            return None
        except Exception as e:
            self.logger.error(f"[FMP] Error: {e}", exc_info=True)
            return None

    def _format_news_data(
        self,
        raw_data: Any,
        limit: int,
        page: int,
    ) -> Dict[str, Any]:
        """
        Format general market news data to structured output.
        We normalize fields similar to GetStockNewsTool but without symbol.
        """

        if not isinstance(raw_data, list):
            raw_data = []

        articles = []
        for item in raw_data[:limit]:
            article = {
                "title": item.get("title") or item.get("newsTitle", "Untitled"),
                "published_date": item.get("publishedDate", ""),
                "text": item.get("text", "")[:1000],  # Limit to 500 chars
                "site": item.get("site", "Unknown"),
                "url": item.get("url") or item.get("newsURL", ""),
                "image": item.get("image", ""),
                # Many FMP news items have a list of symbols/tickers
                "symbols": item.get("symbols") or item.get("symbol") or [],
                "category": item.get("category", ""),
            }
            articles.append(article)

        return {
            "article_count": len(articles),
            "page": page,
            "limit": limit,
            "articles": articles,
            "timestamp": datetime.now().isoformat(),
        }

    def _generate_formatted_context(self, data: Dict[str, Any]) -> str:
        """
        Generate human-readable formatted context for LLM consumption.

        Args:
            data: Formatted news data from _format_news_data()

        Returns:
            Human-readable string summary of market news
        """
        articles = data.get("articles", [])
        article_count = data.get("article_count", 0)

        lines = [
            "=== GENERAL MARKET NEWS ===",
            f"Total Articles: {article_count}",
            ""
        ]

        if not articles:
            lines.append("No recent market news available.")
            return "\n".join(lines)

        # List top articles
        for i, article in enumerate(articles[:10], 1):  # Limit to top 10
            title = article.get("title", "Untitled")
            source = article.get("site", "Unknown")
            pub_date = article.get("published_date", "")
            symbols = article.get("symbols", [])
            text_preview = article.get("text", "")[:150]

            lines.append(f"[{i}] {title}")
            lines.append(f"    Source: {source} | Date: {pub_date}")
            if symbols:
                symbols_str = ", ".join(symbols[:5]) if isinstance(symbols, list) else str(symbols)
                lines.append(f"    Related: {symbols_str}")
            if text_preview:
                lines.append(f"    Preview: {text_preview}...")
            lines.append("")

        if article_count > 10:
            lines.append(f"(Showing top 10 of {article_count} articles)")

        return "\n".join(lines)


# Standalone test
if __name__ == "__main__":
    import asyncio
    import os

    async def test():
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            print("❌ FMP_API_KEY not set")
            return

        tool = GetMarketNewsTool(api_key=api_key)

        print("\n" + "=" * 60)
        print("Testing getMarketNews Tool")
        print("=" * 60)

        result = await tool.execute(limit=5, page=0)

        if result["status"] == "success":
            data = result["data"]
            print(f"✅ Success: {data['article_count']} articles")
            for i, article in enumerate(data["articles"][:3], 1):
                print(f"\n  Article {i}:")
                print(f"  - Title: {article['title'][:80]}")
                print(f"  - Source: {article['site']}")
                print(f"  - Date: {article['published_date']}")
        else:
            print(f"❌ Error: {result.get('error')}")

    asyncio.run(test())
