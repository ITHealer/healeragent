# src/news_aggregator/providers/tavily_provider.py
"""
Tavily News Provider
Uses Tavily Search API for supplementary news search

Only uses tavily_search - no crawl4AI or selenium
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from tavily import TavilyClient

from src.news_aggregator.providers.base_provider import BaseNewsProvider
from src.news_aggregator.schemas.unified_news import UnifiedNewsItem, NewsProvider
from src.news_aggregator.schemas.fmp_news import NewsCategory


class TavilyNewsProvider(BaseNewsProvider):
    """
    Tavily News Provider implementation.
    
    Uses Tavily Search API for web search.
    Good for supplementary news that FMP might miss.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tavily provider.
        
        Args:
            api_key: Tavily API key. If not provided, reads from TAVILY_API_KEY env var.
        """
        super().__init__()
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is required")
        self.client = TavilyClient(api_key=self.api_key)
    
    @property
    def provider_name(self) -> NewsProvider:
        return NewsProvider.TAVILY
    
    @property
    def priority(self) -> int:
        # Tavily is supplementary, lower priority than FMP
        return 2
    
    def _build_search_query(
        self,
        keywords: List[str],
        categories: List[NewsCategory]
    ) -> str:
        """
        Build search query from keywords and categories.
        
        Args:
            keywords: User-provided keywords
            categories: News categories
            
        Returns:
            Search query string
        """
        parts = []
        
        # Add keywords
        if keywords:
            parts.extend(keywords[:5])  # Limit to 5 keywords
        
        # Add category context
        category_terms = {
            NewsCategory.STOCK: "stock market",
            NewsCategory.CRYPTO: "cryptocurrency",
            NewsCategory.FOREX: "forex currency",
            NewsCategory.GENERAL: "financial news",
            NewsCategory.PRESS_RELEASE: "press release",
        }
        
        for cat in categories[:2]:  # Limit to 2 categories
            if cat in category_terms:
                parts.append(category_terms[cat])
        
        # Always add "news" to focus on news content
        parts.append("latest news")
        
        return " ".join(parts)
    
    def _convert_tavily_result(
        self,
        item: Dict[str, Any],
        category: NewsCategory = NewsCategory.GENERAL
    ) -> Optional[UnifiedNewsItem]:
        """
        Convert Tavily search result to unified format.
        
        Tavily result format:
        {
            "title": "Article Title",
            "url": "https://...",
            "content": "Article content/snippet...",
            "score": 0.95,
            "published_date": "2024-01-15"  # Optional
        }
        """
        try:
            url = item.get("url", "")
            if not url:
                return None
            
            title = item.get("title", "")
            if not title:
                return None
            
            # Parse published date if available
            published_date = datetime.utcnow()
            if "published_date" in item and item["published_date"]:
                try:
                    pub_str = item["published_date"]
                    for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                        try:
                            published_date = datetime.strptime(pub_str, fmt)
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass
            
            # Extract source from URL
            source_site = None
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                source_site = parsed.netloc.replace("www.", "")
            except Exception:
                pass
            
            # Get relevance score
            score = float(item.get("score", 0.5))
            
            return UnifiedNewsItem(
                provider=NewsProvider.TAVILY,
                category=category,
                title=title,
                content=item.get("content", "")[:1000],  # Limit content length
                url=url,
                image_url=None,  # Tavily search doesn't return images
                source_site=source_site,
                published_at=published_date,
                symbols=[],  # Will be extracted during matching phase
                relevance_score=score,
            )
        except Exception as e:
            self.logger.warning(f"Failed to convert Tavily result: {e}")
            return None
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        search_depth: str = "advanced",
        time_range: str = "week",
        topic: str = "news",
    ) -> List[UnifiedNewsItem]:
        """
        Search Tavily for news.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            search_depth: "basic" or "advanced"
            time_range: "day", "week", "month", "year"
            topic: "news", "finance", "general"
            
        Returns:
            List of UnifiedNewsItem
        """
        self.logger.info(f"[Tavily] Searching: '{query[:50]}...' max_results={max_results}")
        start_time = time.time()
        
        try:
            # Build search parameters
            search_params = {
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "time_range": time_range,
                "topic": topic,
                "exclude_domains": ["reddit.com", "twitter.com", "x.com"],  # Exclude social media
            }
            
            # Execute search
            response = self.client.search(**search_params)
            
            # Parse results
            results = response.get("results", []) if isinstance(response, dict) else []
            
            # Convert to unified format
            unified_results = []
            for item in results:
                unified = self._convert_tavily_result(item)
                if unified:
                    unified_results.append(unified)
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.info(f"[Tavily] Found {len(unified_results)} results in {elapsed_ms}ms")
            
            return unified_results
            
        except Exception as e:
            self.logger.error(f"[Tavily] Search error: {str(e)}")
            return []
    
    async def fetch_news(
        self,
        categories: List[NewsCategory],
        page: int = 0,
        limit: int = 10,
        keywords: List[str] = None,
        symbols: List[str] = None,
        **kwargs
    ) -> List[UnifiedNewsItem]:
        """
        Fetch news using Tavily search.
        
        Args:
            categories: News categories to search
            page: Ignored (Tavily doesn't support pagination in same way)
            limit: Max results
            keywords: Keywords to include in search
            symbols: Symbols to search for
            
        Returns:
            List of UnifiedNewsItem
        """
        all_keywords = list(keywords or [])
        
        # Add symbols as keywords
        if symbols:
            all_keywords.extend(symbols)
        
        # Build query
        query = self._build_search_query(all_keywords, categories)
        
        if not query.strip():
            query = "financial market news today"
        
        # Determine topic based on categories
        topic = "news"
        if NewsCategory.CRYPTO in categories:
            topic = "news"  # Tavily topic for finance/crypto
        elif NewsCategory.STOCK in categories or NewsCategory.FOREX in categories:
            topic = "finance"
        
        return await self.search(
            query=query,
            max_results=limit,
            search_depth="advanced",
            time_range="day" if page == 0 else "week",
            topic=topic,
        )
    
    async def search_for_symbols(
        self,
        symbols: List[str],
        max_results_per_symbol: int = 5,
    ) -> List[UnifiedNewsItem]:
        """
        Search for news about specific symbols.
        
        Args:
            symbols: List of stock/crypto symbols
            max_results_per_symbol: Max results per symbol
            
        Returns:
            Combined list of UnifiedNewsItem
        """
        all_results = []
        
        for symbol in symbols[:5]:  # Limit to 5 symbols to avoid rate limits
            query = f"{symbol} stock news latest"
            results = await self.search(
                query=query,
                max_results=max_results_per_symbol,
                time_range="day",
                topic="finance",
            )
            
            # Tag results with the symbol
            for item in results:
                if symbol not in item.symbols:
                    item.symbols.append(symbol)
                # Determine category based on symbol format
                if "-USD" in symbol.upper() or symbol.upper() in ["BTC", "ETH", "DOGE"]:
                    item.category = NewsCategory.CRYPTO
                else:
                    item.category = NewsCategory.STOCK
            
            all_results.extend(results)
        
        return all_results