"""
GetStockNewsTool - FIXED with proper Redis cache pattern

Uses: src.helpers.redis_cache helpers
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

# Use existing Redis cache helpers
from src.helpers.redis_cache import get_redis_client_llm


class GetStockNewsTool(BaseTool):
    """
    Atomic tool for fetching stock news
    
    Category: news
    Data Source: FMP /stable/news/stock
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
            raise ValueError("FMP_API_KEY required for GetStockNewsTool")
        
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Define tool schema
        self.schema = ToolSchema(
            name="getStockNews",
            category="news",
            description=(
                "Fetch latest news articles for a stock symbol. "
                "Returns recent news from major financial sources."
            ),
            capabilities=[
                "Fetch latest news articles",
                "Filter by stock symbol",
                "Pagination support"
            ],
            limitations=[
                "News may have 5-15 minute delay",
                "Limited to public news sources"
            ],
            usage_hints=[
                "Use for sentiment analysis",
                "Check recent company announcements"
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock symbol (e.g., AAPL, TSLA, NVDA)",
                    required=True,
                    pattern="^[A-Z]{1,10}$"
                ),
                ToolParameter(
                    name="limit",
                    type="number",
                    description="Number of news articles (default: 20, max: 50)",
                    required=False,
                    default=20,
                    min_value=1,
                    max_value=50
                ),
                ToolParameter(
                    name="page",
                    type="number",
                    description="Page number for pagination (default: 0)",
                    required=False,
                    default=0,
                    min_value=0
                )
            ],
            returns={
                "symbol": "string",
                "article_count": "number",
                "articles": "array"
            },
            requires_symbol=True
        )
                    
    async def execute(
        self,
        symbol: str,
        limit: int = 20,
        page: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute stock news fetch with Redis cache
        
        Args:
            symbol: Stock symbol
            limit: Number of articles (default: 20, max: 50)
            page: Page number (default: 0)
            
        Returns:
            ToolOutput with news articles
        """
        start_time = datetime.now()
        symbol = symbol.upper()
        
        # Validate limits
        limit = min(max(1, limit), 50)
        page = max(0, page)
        
        self.logger.info(
            f"[getStockNews] Executing: symbol={symbol}, limit={limit}, page={page}"
        )
        
        try:
            # Build cache key
            cache_key = f"getStockNews_{symbol}_{limit}_{page}"
            
            # Get Redis client
            redis_client = await get_redis_client_llm()
            
            # Try cache first
            cached_data = None
            if redis_client:
                try:
                    cached_bytes = await redis_client.get(cache_key)
                    if cached_bytes:
                        self.logger.info(f"[CACHE HIT] {cache_key}")
                        cached_data = json.loads(cached_bytes.decode('utf-8'))
                except Exception as e:
                    self.logger.warning(f"[CACHE] Error reading: {e}")
            
            if cached_data:
                # Return cached result
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                self.logger.info(
                    f"[getStockNews] ✅ CACHED ({int(execution_time)}ms)"
                )
                
                return create_success_output(
                    tool_name="getStockNews",
                    data=cached_data,
                    metadata={
                        "symbol": symbol,
                        "execution_time_ms": int(execution_time),
                        "limit": limit,
                        "page": page,
                        "from_cache": True
                    }
                )
            
            # Fetch from API
            news_data = await self._fetch_news(symbol, limit, page)
            
            if not news_data:
                return create_error_output(
                    tool_name="getStockNews",
                    error=f"No news data available for {symbol}",
                    metadata={
                        "symbol": symbol,
                        "limit": limit,
                        "page": page
                    }
                )
            
            # Format response
            result_data = self._format_news_data(news_data, symbol, limit, page)
            
            # Cache the result
            if redis_client:
                try:
                    json_string = json.dumps(result_data)
                    await redis_client.set(cache_key, json_string, ex=self.CACHE_TTL)
                    self.logger.info(f"[CACHE SET] {cache_key} (TTL={self.CACHE_TTL}s)")
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
                f"[getStockNews] ✅ SUCCESS ({int(execution_time)}ms) - "
                f"{result_data['article_count']} articles"
            )
            
            return create_success_output(
                tool_name="getStockNews",
                data=result_data,
                metadata={
                    "symbol": symbol,
                    "execution_time_ms": int(execution_time),
                    "limit": limit,
                    "page": page,
                    "from_cache": False,
                    "article_count": result_data['article_count']
                }
            )
            
        except Exception as e:
            self.logger.error(f"[getStockNews] Error for {symbol}: {e}", exc_info=True)
            
            return create_error_output(
                tool_name="getStockNews",
                error=str(e),
                metadata={
                    "symbol": symbol,
                    "limit": limit,
                    "page": page
                }
            )
    
    async def _fetch_news(
        self,
        symbol: str,
        limit: int,
        page: int
    ) -> Optional[Any]:
        """Fetch news from FMP Stable API"""
        
        url = f"{self.FMP_STABLE_BASE}/news/stock"
        params = {
            "symbols": symbol,
            "page": page,
            "limit": limit,
            "apikey": self.api_key
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
                
                if isinstance(data, dict) and "Error Message" in data:
                    self.logger.error(f"[FMP] API Error: {data['Error Message']}")
                    return None
                
                self.logger.info(
                    f"[FMP] ✅ Success: {len(data) if isinstance(data, list) else 1} items"
                )
                
                return data
                
        except httpx.TimeoutException:
            self.logger.error(f"[FMP] Timeout fetching news")
            return None
        except Exception as e:
            self.logger.error(f"[FMP] Error: {e}", exc_info=True)
            return None
    
    def _format_news_data(
        self,
        raw_data: Any,
        symbol: str,
        limit: int,
        page: int
    ) -> Dict[str, Any]:
        """Format news data to structured output"""
        
        if not isinstance(raw_data, list):
            raw_data = []
        
        # Parse articles
        articles = []
        for item in raw_data[:limit]:
            article = {
                "title": item.get("title") or item.get("newsTitle", "Untitled"),
                "published_date": item.get("publishedDate", ""),
                "text": item.get("text", "")[:500],  # Limit to 500 chars
                "site": item.get("site", "Unknown"),
                "url": item.get("url") or item.get("newsURL", ""),
                "image": item.get("image", ""),
                "symbol": item.get("symbol", symbol)
            }
            articles.append(article)
        
        return {
            "symbol": symbol,
            "article_count": len(articles),
            "page": page,
            "limit": limit,
            "articles": articles,
            "timestamp": datetime.now().isoformat()
        }


# Standalone test
if __name__ == "__main__":
    import asyncio
    import os
    
    async def test():
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            print("❌ FMP_API_KEY not set")
            return
        
        tool = GetStockNewsTool(api_key=api_key)
        
        print("\n" + "="*60)
        print("Testing getStockNews Tool")
        print("="*60)
        
        # Test: AAPL news
        print("\nTest: AAPL latest news")
        result = await tool.execute(symbol="AAPL", limit=5)
        
        if result['status'] == 'success':
            data = result['data']
            print(f"✅ Success: {data['article_count']} articles")
            for i, article in enumerate(data['articles'][:3], 1):
                print(f"\n  Article {i}:")
                print(f"  - Title: {article['title'][:80]}")
                print(f"  - Source: {article['site']}")
                print(f"  - Date: {article['published_date']}")
        else:
            print(f"❌ Error: {result.get('error')}")
    
    asyncio.run(test())