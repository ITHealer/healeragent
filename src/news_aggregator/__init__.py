"""
Request -> FMP Provider ──┐
        Tavily Provider ──┼→ Normalize → Dedupe → Match → Sort → Digest → Response
                        ──┘
"""
from src.news_aggregator.schemas.request import NewsAggregatorRequest, DigestRequest
from src.news_aggregator.schemas.response import (
    NewsAggregatorResponse,
    DigestResponse,
    NewsDigest,
    NewsArticleResponse,
)
from src.news_aggregator.schemas.unified_news import UnifiedNewsItem, NewsProvider
from src.news_aggregator.schemas.fmp_news import NewsCategory

from src.news_aggregator.services.aggregator_service import NewsAggregatorService
from src.news_aggregator.services.deduplication import DeduplicationService
from src.news_aggregator.services.keyword_matcher import KeywordMatcher
from src.news_aggregator.services.digest_generator import DigestGenerator

from src.news_aggregator.providers.fmp_provider import FMPNewsProvider
from src.news_aggregator.providers.tavily_provider import TavilyNewsProvider

__version__ = "1.0.0"
__all__ = [
    # Schemas
    "NewsAggregatorRequest",
    "DigestRequest",
    "NewsAggregatorResponse",
    "DigestResponse",
    "NewsDigest",
    "NewsArticleResponse",
    "UnifiedNewsItem",
    "NewsProvider",
    "NewsCategory",
    # Services
    "NewsAggregatorService",
    "DeduplicationService",
    "KeywordMatcher",
    "DigestGenerator",
    # Providers
    "FMPNewsProvider",
    "TavilyNewsProvider",
]