# src/news_aggregator/schemas/response.py
"""
API Response Schemas for News Aggregator
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from src.news_aggregator.schemas.unified_news import UnifiedNewsItem


class NewsArticleResponse(BaseModel):
    """
    Simplified article format for API response.
    Derived from UnifiedNewsItem but with cleaner structure.
    """
    id: str
    title: str
    content: Optional[str] = None
    url: str
    image_url: Optional[str] = None
    source: Optional[str] = None
    published_at: str  # ISO format string
    category: str
    provider: str
    
    # Matching info
    symbols: List[str] = Field(default_factory=list)
    matched_keywords: List[str] = Field(default_factory=list)
    relevance_score: float = 0.0
    importance_score: Optional[float] = None
    
    @classmethod
    def from_unified(cls, item: UnifiedNewsItem, matched_keywords: List[str] = None) -> "NewsArticleResponse":
        """Convert UnifiedNewsItem to response format"""
        return cls(
            id=item.id or "",
            title=item.title,
            content=item.content[:500] if item.content else None,
            url=item.url,
            image_url=item.image_url,
            source=item.source_site,
            published_at=item.published_at.isoformat(),
            category=item.category.value,
            provider=item.provider.value,
            symbols=item.symbols,
            matched_keywords=matched_keywords or [],
            relevance_score=item.relevance_score,
            importance_score=item.importance_score,
        )


class TopStory(BaseModel):
    """A highlighted top story in the digest"""
    rank: int = Field(..., ge=1, le=10)
    article_id: str
    title: str
    summary: str = Field(..., description="1-2 sentence summary")
    url: str
    source: Optional[str] = None
    importance_reason: str = Field(..., description="Why this story is important")


class NewsDigest(BaseModel):
    """
    LLM-generated digest/summary of news articles.
    """
    summary_text: str = Field(..., description="Overall summary of all news")
    top_stories: List[TopStory] = Field(default_factory=list)
    market_sentiment: str = Field(default="neutral", description="bullish/bearish/neutral/mixed")
    key_themes: List[str] = Field(default_factory=list, description="Main themes identified")
    
    # Stats
    articles_analyzed: int = 0
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    target_language: str = "en"


class AggregatorMetadata(BaseModel):
    """
    Metadata about the aggregation process.
    Useful for debugging and monitoring.
    """
    # Counts
    total_fetched: int = Field(default=0, description="Total articles fetched from all sources")
    fmp_fetched: int = Field(default=0, description="Articles from FMP")
    tavily_fetched: int = Field(default=0, description="Articles from Tavily")
    after_dedup: int = Field(default=0, description="Articles after deduplication")
    matched: int = Field(default=0, description="Articles matching filters")
    returned: int = Field(default=0, description="Articles returned to client")
    
    # Performance
    processing_time_ms: int = 0
    fmp_time_ms: int = 0
    tavily_time_ms: int = 0
    dedup_time_ms: int = 0
    digest_time_ms: int = 0
    
    # Categories breakdown
    category_counts: Dict[str, int] = Field(default_factory=dict)
    
    # Errors (non-fatal)
    warnings: List[str] = Field(default_factory=list)


class NewsAggregatorResponse(BaseModel):
    """
    Main response from news aggregation endpoint.
    """
    status: str = Field(default="success", description="success or error")
    
    # News articles (sorted by relevance/importance)
    articles: List[NewsArticleResponse] = Field(default_factory=list)
    
    # Optional digest (if generate_digest=true)
    digest: Optional[NewsDigest] = None
    
    # Metadata for debugging/monitoring
    metadata: AggregatorMetadata = Field(default_factory=AggregatorMetadata)
    
    # Error info (if status=error)
    error_message: Optional[str] = None
    error_code: Optional[str] = None


class DigestResponse(BaseModel):
    """
    Response for digest-only generation endpoint.
    """
    status: str = "success"
    digest: Optional[NewsDigest] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None