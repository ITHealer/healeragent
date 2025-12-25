# src/news_aggregator/services/aggregator_service.py
"""
News Aggregator Service
Main orchestration service that combines all providers and services
"""

import time
import asyncio
import logging
from typing import List, Optional, Dict
from datetime import datetime, timedelta

from src.news_aggregator.schemas.request import NewsAggregatorRequest
from src.news_aggregator.schemas.response import (
    NewsAggregatorResponse,
    NewsArticleResponse,
    AggregatorMetadata,
)
from src.news_aggregator.schemas.unified_news import UnifiedNewsItem
from src.news_aggregator.schemas.fmp_news import NewsCategory

from src.news_aggregator.providers.fmp_provider import FMPNewsProvider
from src.news_aggregator.providers.tavily_provider import TavilyNewsProvider
from src.news_aggregator.services.deduplication import DeduplicationService
from src.news_aggregator.services.keyword_matcher import KeywordMatcher
from src.news_aggregator.services.digest_generator import DigestGenerator

logger = logging.getLogger(__name__)


class NewsAggregatorService:
    """
    Main service for news aggregation.
    
    Pipeline:
    1. Fetch from FMP (primary)
    2. Fetch from Tavily (supplementary, if enabled)
    3. Normalize to UnifiedNewsItem
    4. Deduplicate
    5. Match keywords/symbols
    6. Sort by relevance
    7. Generate digest (if enabled)
    """
    
    def __init__(
        self,
        fmp_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
    ):
        """
        Initialize aggregator service.
        
        Args:
            fmp_api_key: FMP API key (defaults to env var)
            tavily_api_key: Tavily API key (defaults to env var)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize providers
        try:
            self.fmp_provider = FMPNewsProvider(api_key=fmp_api_key)
            self.logger.info("[Aggregator] FMP provider initialized")
        except ValueError as e:
            self.logger.warning(f"[Aggregator] FMP provider not available: {e}")
            self.fmp_provider = None
        
        try:
            self.tavily_provider = TavilyNewsProvider(api_key=tavily_api_key)
            self.logger.info("[Aggregator] Tavily provider initialized")
        except ValueError as e:
            self.logger.warning(f"[Aggregator] Tavily provider not available: {e}")
            self.tavily_provider = None
        
        # Initialize services
        self.dedup_service = DeduplicationService()
        self.keyword_matcher = KeywordMatcher()
        self.digest_generator = DigestGenerator()
    
    async def _fetch_from_fmp(
        self,
        categories: List[NewsCategory],
        page: int,
        limit: int,
    ) -> tuple[List[UnifiedNewsItem], int]:
        """
        Fetch news from FMP.
        
        Returns:
            Tuple of (items, time_ms)
        """
        if not self.fmp_provider:
            return [], 0
        
        start_time = time.time()
        
        try:
            items = await self.fmp_provider.fetch_news(
                categories=categories,
                page=page,
                limit=limit,
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            return items, elapsed_ms
        except Exception as e:
            self.logger.error(f"[Aggregator] FMP fetch error: {e}")
            return [], int((time.time() - start_time) * 1000)
    
    async def _fetch_from_tavily(
        self,
        categories: List[NewsCategory],
        keywords: List[str],
        symbols: List[str],
        limit: int,
    ) -> tuple[List[UnifiedNewsItem], int]:
        """
        Fetch news from Tavily.
        
        Returns:
            Tuple of (items, time_ms)
        """
        if not self.tavily_provider:
            return [], 0
        
        start_time = time.time()
        
        try:
            items = await self.tavily_provider.fetch_news(
                categories=categories,
                limit=limit,
                keywords=keywords,
                symbols=symbols,
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            return items, elapsed_ms
        except Exception as e:
            self.logger.error(f"[Aggregator] Tavily fetch error: {e}")
            return [], int((time.time() - start_time) * 1000)
    
    def _filter_by_time(
        self,
        items: List[UnifiedNewsItem],
        hours: int
    ) -> List[UnifiedNewsItem]:
        """Filter items to only those within time range"""
        if hours <= 0:
            return items
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        filtered = [
            item for item in items
            if item.published_at and item.published_at >= cutoff
        ]
        
        self.logger.info(
            f"[Aggregator] Time filter ({hours}h): {len(items)} -> {len(filtered)}"
        )
        
        return filtered
    
    async def aggregate(
        self,
        request: NewsAggregatorRequest
    ) -> NewsAggregatorResponse:
        """
        Main aggregation method.
        
        Args:
            request: Aggregation request with filters
            
        Returns:
            NewsAggregatorResponse with articles and optional digest
        """
        total_start = time.time()
        metadata = AggregatorMetadata()
        warnings: List[str] = []
        
        self.logger.info(
            f"[Aggregator] Starting aggregation: "
            f"keywords={request.keywords}, symbols={request.symbols}, "
            f"categories={request.categories}"
        )
        
        # Get categories
        categories = request.get_categories()
        
        # ========================================
        # PHASE 1: FETCH FROM PROVIDERS
        # ========================================
        
        all_items: List[UnifiedNewsItem] = []
        
        # Fetch from FMP (primary)
        fmp_items, fmp_time = await self._fetch_from_fmp(
            categories=categories,
            page=request.page,
            limit=request.limit_per_category,
        )
        all_items.extend(fmp_items)
        metadata.fmp_fetched = len(fmp_items)
        metadata.fmp_time_ms = fmp_time
        
        # Fetch from Tavily (supplementary)
        if request.use_tavily and self.tavily_provider:
            tavily_items, tavily_time = await self._fetch_from_tavily(
                categories=categories,
                keywords=request.keywords,
                symbols=request.symbols,
                limit=request.tavily_max_results,
            )
            all_items.extend(tavily_items)
            metadata.tavily_fetched = len(tavily_items)
            metadata.tavily_time_ms = tavily_time
        
        metadata.total_fetched = len(all_items)
        
        if not all_items:
            return NewsAggregatorResponse(
                status="success",
                articles=[],
                metadata=metadata,
            )
        
        # ========================================
        # PHASE 2: FILTER BY TIME
        # ========================================
        
        filtered_items = self._filter_by_time(all_items, request.time_range_hours)
        
        # ========================================
        # PHASE 3: DEDUPLICATE
        # ========================================
        
        dedup_start = time.time()
        unique_items = self.dedup_service.deduplicate(filtered_items)
        metadata.after_dedup = len(unique_items)
        metadata.dedup_time_ms = int((time.time() - dedup_start) * 1000)
        
        # ========================================
        # PHASE 4: KEYWORD MATCHING
        # ========================================
        
        # Extract symbols from Tavily results
        unique_items = self.keyword_matcher.extract_and_tag_symbols(unique_items)
        
        # Match and score
        if request.keywords or request.symbols:
            matched_results = self.keyword_matcher.filter_and_score(
                unique_items,
                request.keywords,
                request.symbols,
            )
            matched_items = [item for item, _ in matched_results]
            matched_keywords_map = {item.id: kws for item, kws in matched_results}
        else:
            matched_items = unique_items
            matched_keywords_map = {}
        
        metadata.matched = len(matched_items)
        
        # ========================================
        # PHASE 5: SORT AND LIMIT
        # ========================================
        
        # Sort by relevance score then by published date
        sorted_items = sorted(
            matched_items,
            key=lambda x: (x.relevance_score, x.published_at.timestamp() if x.published_at else 0),
            reverse=True,
        )
        
        # Apply limit
        final_items = sorted_items[:request.max_articles]
        metadata.returned = len(final_items)
        
        # Count by category
        for item in final_items:
            cat = item.category.value
            metadata.category_counts[cat] = metadata.category_counts.get(cat, 0) + 1
        
        # ========================================
        # PHASE 6: GENERATE DIGEST (optional)
        # ========================================
        
        digest = None
        if request.generate_digest and final_items:
            digest_start = time.time()
            try:
                digest = await self.digest_generator.generate_digest(
                    items=final_items,
                    target_language=request.target_language,
                    model_name=request.model_name,
                    provider_type=request.provider_type,
                )
            except Exception as e:
                self.logger.error(f"[Aggregator] Digest generation failed: {e}")
                warnings.append(f"Digest generation failed: {str(e)}")
            metadata.digest_time_ms = int((time.time() - digest_start) * 1000)
        
        # ========================================
        # PHASE 7: BUILD RESPONSE
        # ========================================
        
        # Convert to response format
        articles = [
            NewsArticleResponse.from_unified(
                item,
                matched_keywords=matched_keywords_map.get(item.id, [])
            )
            for item in final_items
        ]
        
        # Finalize metadata
        metadata.processing_time_ms = int((time.time() - total_start) * 1000)
        metadata.warnings = warnings
        
        self.logger.info(
            f"[Aggregator] Complete: {metadata.returned} articles in {metadata.processing_time_ms}ms"
        )
        
        return NewsAggregatorResponse(
            status="success",
            articles=articles,
            digest=digest,
            metadata=metadata,
        )
    
    async def close(self):
        """Cleanup resources"""
        if self.fmp_provider:
            await self.fmp_provider.close()