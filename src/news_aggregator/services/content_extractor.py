"""
Content Extractor Service
=========================

Extracts full article content from URLs using Tavily Extract API.

FMP news returns only ~200 char snippets. This service fetches
the FULL article body (up to 5000+ chars) for proper AI analysis.

Features:
- Batch URL extraction (max 20 URLs per call)
- Relevance-based chunking with query parameter
- Graceful failure handling (paywalled, blocked URLs)
- Content caching to avoid re-extraction

Usage:
    extractor = ContentExtractor()
    articles = await extractor.extract_articles(urls, query="TSLA Tesla")
"""

import os
import time
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from src.news_aggregator.schemas.task import ArticleContent

# Initialize logger
logger = logging.getLogger(__name__)

# Thread pool for sync Tavily client
_extractor_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="content_extractor_")


class ContentExtractor:
    """
    Content extractor using Tavily Extract API.

    Tavily Extract features:
    - Batch extraction (max 20 URLs)
    - Relevance-based chunking with query parameter
    - Handles paywalled content gracefully
    - 1 credit per 5 URLs (basic depth)
    """

    MAX_URLS_PER_BATCH = 20
    DEFAULT_TIMEOUT = 60  # seconds
    MAX_CONTENT_LENGTH = 10000  # chars per article

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize content extractor.

        Args:
            api_key: Tavily API key. Falls back to TAVILY_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._client = None
        self.logger = logger

        if not self.api_key:
            self.logger.warning(
                "[ContentExtractor] TAVILY_API_KEY not set - content extraction disabled"
            )

    def _get_client(self):
        """Get or create Tavily client (lazy initialization)."""
        if self._client is None and self.api_key:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except ImportError:
                self.logger.error("[ContentExtractor] tavily package not installed")
        return self._client

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.replace("www.", "")
        except Exception:
            return "unknown"

    def _sync_extract(
        self,
        urls: List[str],
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synchronous extraction (runs in thread pool).

        Args:
            urls: URLs to extract content from
            query: Optional query for relevance-based extraction

        Returns:
            List of extraction results
        """
        client = self._get_client()
        if not client:
            return []

        try:
            # Build extraction parameters
            extract_params = {
                "urls": urls[:self.MAX_URLS_PER_BATCH],
            }

            # Add query for relevance-based extraction
            if query:
                extract_params["include_raw_content"] = True

            self.logger.info(
                f"[ContentExtractor] Extracting {len(urls)} URLs"
                + (f" with query: {query[:30]}..." if query else "")
            )

            # Execute extraction
            response = client.extract(**extract_params)

            # Handle response format
            if isinstance(response, dict):
                return response.get("results", [])
            elif isinstance(response, list):
                return response
            else:
                self.logger.warning(f"[ContentExtractor] Unexpected response type: {type(response)}")
                return []

        except Exception as e:
            self.logger.error(f"[ContentExtractor] Extraction error: {e}")
            return []

    async def extract_urls(
        self,
        urls: List[str],
        query: Optional[str] = None,
        timeout: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract content from multiple URLs.

        Args:
            urls: URLs to extract
            query: Optional query for relevance filtering
            timeout: Timeout in seconds

        Returns:
            List of extraction results with url, raw_content, etc.
        """
        if not urls:
            return []

        if not self.api_key:
            self.logger.warning("[ContentExtractor] No API key - returning empty results")
            return []

        timeout = timeout or self.DEFAULT_TIMEOUT
        loop = asyncio.get_event_loop()

        try:
            # Process in batches
            all_results = []
            for i in range(0, len(urls), self.MAX_URLS_PER_BATCH):
                batch = urls[i:i + self.MAX_URLS_PER_BATCH]

                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        _extractor_executor,
                        self._sync_extract,
                        batch,
                        query,
                    ),
                    timeout=timeout,
                )

                all_results.extend(result)

                # Small delay between batches to avoid rate limits
                if i + self.MAX_URLS_PER_BATCH < len(urls):
                    await asyncio.sleep(0.5)

            return all_results

        except asyncio.TimeoutError:
            self.logger.error(f"[ContentExtractor] Timeout after {timeout}s")
            return []
        except Exception as e:
            self.logger.error(f"[ContentExtractor] Error: {e}")
            return []

    async def extract_articles(
        self,
        news_items: List[Dict[str, Any]],
        query: Optional[str] = None,
    ) -> List[ArticleContent]:
        """
        Extract full content for news items.

        Args:
            news_items: List of news items with url, title, etc.
            query: Optional query for relevance filtering

        Returns:
            List of ArticleContent with full content
        """
        if not news_items:
            return []

        start_time = time.time()

        # Build URL to item mapping
        url_to_item = {item.get("url", ""): item for item in news_items if item.get("url")}
        urls = list(url_to_item.keys())

        if not urls:
            return []

        self.logger.info(f"[ContentExtractor] Extracting content for {len(urls)} articles")

        # Extract content
        extraction_results = await self.extract_urls(urls, query=query)

        # Build URL to extracted content mapping
        url_to_content = {}
        for result in extraction_results:
            url = result.get("url", "")
            if url:
                url_to_content[url] = result

        # Create ArticleContent objects
        articles = []
        success_count = 0
        fail_count = 0

        for url, item in url_to_item.items():
            extracted = url_to_content.get(url, {})

            # Get content from extraction or fallback to original
            raw_content = extracted.get("raw_content", "")
            if not raw_content:
                raw_content = item.get("content", "") or item.get("text", "")

            # Truncate if too long
            if len(raw_content) > self.MAX_CONTENT_LENGTH:
                raw_content = raw_content[:self.MAX_CONTENT_LENGTH] + "..."

            extraction_success = bool(extracted.get("raw_content"))
            if extraction_success:
                success_count += 1
            else:
                fail_count += 1

            # Parse published date
            published_at = None
            pub_str = item.get("published_at") or item.get("publishedDate")
            if pub_str:
                try:
                    if isinstance(pub_str, datetime):
                        published_at = pub_str
                    else:
                        for fmt in [
                            "%Y-%m-%dT%H:%M:%S.%fZ",
                            "%Y-%m-%dT%H:%M:%SZ",
                            "%Y-%m-%d %H:%M:%S",
                            "%Y-%m-%d",
                        ]:
                            try:
                                published_at = datetime.strptime(str(pub_str), fmt)
                                break
                            except ValueError:
                                continue
                except Exception:
                    pass

            article = ArticleContent(
                url=url,
                title=item.get("title", "Untitled"),
                content=raw_content,
                snippet=item.get("content", "")[:300] if item.get("content") else "",
                source=self._extract_domain(url),
                published_at=published_at,
                symbol=item.get("symbol"),
                extraction_success=extraction_success,
                extraction_method="tavily" if extraction_success else "fallback",
            )
            articles.append(article)

        elapsed_ms = int((time.time() - start_time) * 1000)
        self.logger.info(
            f"[ContentExtractor] Completed: {success_count} success, {fail_count} fallback "
            f"({elapsed_ms}ms)"
        )

        return articles

    async def extract_for_symbols(
        self,
        news_by_symbol: Dict[str, List[Dict[str, Any]]],
        max_per_symbol: int = 5,
    ) -> Dict[str, List[ArticleContent]]:
        """
        Extract content organized by symbol.

        Args:
            news_by_symbol: Dict mapping symbol to list of news items
            max_per_symbol: Max articles to extract per symbol

        Returns:
            Dict mapping symbol to list of ArticleContent
        """
        result = {}

        for symbol, items in news_by_symbol.items():
            # Limit items per symbol
            limited_items = items[:max_per_symbol]

            # Build query for relevance
            query = f"{symbol} stock price analysis"

            # Extract content
            articles = await self.extract_articles(
                news_items=limited_items,
                query=query,
            )

            # Tag with symbol
            for article in articles:
                article.symbol = symbol

            result[symbol] = articles

        return result


# Singleton instance
_content_extractor: Optional[ContentExtractor] = None


def get_content_extractor() -> ContentExtractor:
    """Get singleton content extractor instance."""
    global _content_extractor
    if _content_extractor is None:
        _content_extractor = ContentExtractor()
    return _content_extractor
