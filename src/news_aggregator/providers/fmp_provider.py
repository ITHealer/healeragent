# src/news_aggregator/providers/fmp_provider.py
"""
FMP News Provider
Fetches news from Financial Modeling Prep Stable API endpoints

Endpoints:
- /stable/news/stock-latest
- /stable/news/general-latest
- /stable/news/crypto-latest
- /stable/news/forex-latest
- /stable/news/press-releases-latest
- /stable/fmp-articles
"""

import os
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode

import httpx

from src.news_aggregator.providers.base_provider import BaseNewsProvider
from src.news_aggregator.schemas.unified_news import UnifiedNewsItem, NewsProvider
from src.news_aggregator.schemas.fmp_news import (
    NewsCategory,
    FMPStockNewsItem,
    FMPGeneralNewsItem,
    FMPCryptoNewsItem,
    FMPForexNewsItem,
    FMPPressReleaseItem,
    FMPArticleItem,
)


class FMPNewsProvider(BaseNewsProvider):
    """
    FMP News Provider implementation.
    
    Uses Financial Modeling Prep's Stable API for news data.
    High reliability, structured data, good for production.
    """
    
    BASE_URL = "https://financialmodelingprep.com/stable"
    
    # Endpoint mapping for each category
    ENDPOINTS = {
        NewsCategory.STOCK: "news/stock-latest",
        NewsCategory.GENERAL: "news/general-latest",
        NewsCategory.CRYPTO: "news/crypto-latest",
        NewsCategory.FOREX: "news/forex-latest",
        NewsCategory.PRESS_RELEASE: "news/press-releases-latest",
        NewsCategory.FMP_ARTICLE: "fmp-articles",
    }
    
    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0):
        """
        Initialize FMP provider.
        
        Args:
            api_key: FMP API key. If not provided, reads from FMP_API_KEY env var.
            timeout: Request timeout in seconds
        """
        super().__init__()
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP_API_KEY is required")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def provider_name(self) -> NewsProvider:
        return NewsProvider.FMP
    
    @property
    def priority(self) -> int:
        # FMP is primary source, highest priority
        return 1
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self):
        """Close HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def _fetch_endpoint(
        self,
        endpoint: str,
        page: int = 0,
        limit: int = 20,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch data from a specific FMP endpoint.
        
        Args:
            endpoint: API endpoint path
            page: Page number
            limit: Items per page
            extra_params: Additional query parameters
            
        Returns:
            List of raw response items
        """
        params = {
            "page": page,
            "limit": limit,
            "apikey": self.api_key,
        }
        if extra_params:
            params.update(extra_params)
        
        url = f"{self.BASE_URL}/{endpoint}?{urlencode(params)}"
        
        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # FMP returns array directly or wrapped in object
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "data" in data:
                return data["data"]
            elif isinstance(data, dict) and "error" in data:
                self.logger.error(f"FMP API error: {data['error']}")
                return []
            else:
                return []
                
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error fetching {endpoint}: {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            self.logger.error(f"Request error fetching {endpoint}: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error fetching {endpoint}: {str(e)}")
            return []
    
    def _convert_stock_news(self, item: Dict[str, Any]) -> Optional[UnifiedNewsItem]:
        """Convert FMP stock news to unified format"""
        try:
            fmp_item = FMPStockNewsItem(**item)
            
            # Extract symbols
            symbols = []
            if fmp_item.symbol:
                symbols = [s.strip() for s in fmp_item.symbol.split(",") if s.strip()]
            
            return UnifiedNewsItem(
                provider=NewsProvider.FMP,
                category=NewsCategory.STOCK,
                title=fmp_item.title,
                content=fmp_item.text,
                url=fmp_item.url,
                image_url=fmp_item.image if fmp_item.image and fmp_item.image.strip() else None,
                source_site=fmp_item.site,
                published_at=fmp_item.published_date,
                symbols=symbols,
            )
        except Exception as e:
            self.logger.warning(f"Failed to convert stock news item: {e}")
            return None
    
    def _convert_general_news(self, item: Dict[str, Any]) -> Optional[UnifiedNewsItem]:
        """Convert FMP general news to unified format"""
        try:
            fmp_item = FMPGeneralNewsItem(**item)
            
            return UnifiedNewsItem(
                provider=NewsProvider.FMP,
                category=NewsCategory.GENERAL,
                title=fmp_item.title,
                content=fmp_item.text,
                url=fmp_item.url,
                image_url=fmp_item.image if fmp_item.image and fmp_item.image.strip() else None,
                source_site=fmp_item.site,
                published_at=fmp_item.published_date,
                symbols=[],
            )
        except Exception as e:
            self.logger.warning(f"Failed to convert general news item: {e}")
            return None
    
    def _convert_crypto_news(self, item: Dict[str, Any]) -> Optional[UnifiedNewsItem]:
        """Convert FMP crypto news to unified format"""
        try:
            fmp_item = FMPCryptoNewsItem(**item)
            
            symbols = []
            if fmp_item.symbol:
                # Convert BTCUSD -> BTC-USD format for consistency
                symbol = fmp_item.symbol.strip()
                if "USD" in symbol and "-" not in symbol:
                    symbol = symbol.replace("USD", "-USD")
                symbols = [symbol]
            
            return UnifiedNewsItem(
                provider=NewsProvider.FMP,
                category=NewsCategory.CRYPTO,
                title=fmp_item.title,
                content=fmp_item.text,
                url=fmp_item.url,
                image_url=fmp_item.image if fmp_item.image and fmp_item.image.strip() else None,
                source_site=fmp_item.site,
                published_at=fmp_item.published_date,
                symbols=symbols,
            )
        except Exception as e:
            self.logger.warning(f"Failed to convert crypto news item: {e}")
            return None
    
    def _convert_forex_news(self, item: Dict[str, Any]) -> Optional[UnifiedNewsItem]:
        """Convert FMP forex news to unified format"""
        try:
            fmp_item = FMPForexNewsItem(**item)
            
            symbols = []
            if fmp_item.symbol:
                symbols = [fmp_item.symbol.strip()]
            
            return UnifiedNewsItem(
                provider=NewsProvider.FMP,
                category=NewsCategory.FOREX,
                title=fmp_item.title,
                content=fmp_item.text,
                url=fmp_item.url,
                image_url=fmp_item.image if fmp_item.image and fmp_item.image.strip() else None,
                source_site=fmp_item.site,
                published_at=fmp_item.published_date,
                symbols=symbols,
            )
        except Exception as e:
            self.logger.warning(f"Failed to convert forex news item: {e}")
            return None
    
    def _convert_press_release(self, item: Dict[str, Any]) -> Optional[UnifiedNewsItem]:
        """Convert FMP press release to unified format"""
        try:
            fmp_item = FMPPressReleaseItem(**item)
            
            symbols = []
            if fmp_item.symbol:
                symbols = [fmp_item.symbol.strip()]
            
            return UnifiedNewsItem(
                provider=NewsProvider.FMP,
                category=NewsCategory.PRESS_RELEASE,
                title=fmp_item.title,
                content=fmp_item.text,
                url=fmp_item.url,
                image_url=fmp_item.image if fmp_item.image and fmp_item.image.strip() else None,
                source_site=fmp_item.site or "Company Press Release",
                published_at=fmp_item.published_date,
                symbols=symbols,
            )
        except Exception as e:
            self.logger.warning(f"Failed to convert press release item: {e}")
            return None
    
    def _convert_fmp_article(self, item: Dict[str, Any]) -> Optional[UnifiedNewsItem]:
        """Convert FMP article to unified format"""
        try:
            fmp_item = FMPArticleItem(**item)
            
            return UnifiedNewsItem(
                provider=NewsProvider.FMP,
                category=NewsCategory.FMP_ARTICLE,
                title=fmp_item.title,
                content=fmp_item.text,
                url=fmp_item.url,
                image_url=fmp_item.image if fmp_item.image and fmp_item.image.strip() else None,
                source_site="Financial Modeling Prep",
                published_at=fmp_item.published_date,
                symbols=fmp_item.tickers or [],
            )
        except Exception as e:
            self.logger.warning(f"Failed to convert FMP article item: {e}")
            return None
    
    async def fetch_category(
        self,
        category: NewsCategory,
        page: int = 0,
        limit: int = 20,
        tickers: Optional[List[str]] = None,
    ) -> List[UnifiedNewsItem]:
        """
        Fetch news for a specific category.

        Args:
            category: News category
            page: Page number
            limit: Items per page
            tickers: Optional list of stock symbols to filter (for STOCK category)

        Returns:
            List of UnifiedNewsItem
        """
        endpoint = self.ENDPOINTS.get(category)
        if not endpoint:
            self.logger.warning(f"Unknown category: {category}")
            return []

        # Log with tickers info if provided
        tickers_info = f" | tickers={tickers}" if tickers else ""
        self.logger.info(f"[fmp] Fetching {category.value} news - page={page}, limit={limit}{tickers_info}")
        start_time = time.time()

        # Add tickers filter for stock news
        extra_params = None
        if tickers and category == NewsCategory.STOCK:
            extra_params = {"tickers": ",".join(tickers)}

        raw_items = await self._fetch_endpoint(endpoint, page, limit, extra_params)

        # Convert based on category
        converter_map = {
            NewsCategory.STOCK: self._convert_stock_news,
            NewsCategory.GENERAL: self._convert_general_news,
            NewsCategory.CRYPTO: self._convert_crypto_news,
            NewsCategory.FOREX: self._convert_forex_news,
            NewsCategory.PRESS_RELEASE: self._convert_press_release,
            NewsCategory.FMP_ARTICLE: self._convert_fmp_article,
        }

        converter = converter_map.get(category)
        if not converter:
            return []

        results = []
        for item in raw_items:
            unified = converter(item)
            if unified:
                results.append(unified)

        elapsed_ms = int((time.time() - start_time) * 1000)
        self._log_fetch_complete(category.value, len(results), elapsed_ms)

        return results
    
    async def fetch_news(
        self,
        categories: List[NewsCategory],
        page: int = 0,
        limit: int = 20,
        tickers: Optional[List[str]] = None,
        **kwargs
    ) -> List[UnifiedNewsItem]:
        """
        Fetch news from all specified categories.

        Args:
            categories: List of categories to fetch
            page: Page number for each category
            limit: Items per page per category
            tickers: Optional stock symbols to filter (applied to STOCK category)

        Returns:
            Combined list of UnifiedNewsItem from all categories
        """
        tickers_str = f" | tickers={tickers}" if tickers else ""
        self.logger.info(f"[FMP] Fetching news for categories: {[c.value for c in categories]}{tickers_str}")
        start_time = time.time()

        # Fetch all categories concurrently
        tasks = [
            self.fetch_category(category, page, limit, tickers=tickers)
            for category in categories
        ]

        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_results = []
        for i, result in enumerate(results_lists):
            if isinstance(result, Exception):
                self.logger.error(f"Error fetching {categories[i].value}: {result}")
            elif isinstance(result, list):
                all_results.extend(result)

        total_time_ms = int((time.time() - start_time) * 1000)
        self.logger.info(f"[FMP] Total: {len(all_results)} articles in {total_time_ms}ms")

        return all_results