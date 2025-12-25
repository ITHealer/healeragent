# src/news_aggregator/schemas/unified_news.py
"""
Unified News Item Schema
Normalized format that all providers convert to
"""

import hashlib
from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, computed_field

from src.news_aggregator.schemas.fmp_news import NewsCategory


class NewsProvider(str, Enum):
    """News data providers"""
    FMP = "fmp"
    TAVILY = "tavily"


class UnifiedNewsItem(BaseModel):
    """
    Unified news item format.
    All providers convert their responses to this format.
    
    This is the internal representation used throughout the aggregator.
    """
    # Identification
    id: Optional[str] = Field(None, description="Unique ID (generated from URL hash)")
    external_id: Optional[str] = Field(None, description="Original ID from source if available")
    provider: NewsProvider = Field(..., description="Source provider: fmp, tavily")
    category: NewsCategory = Field(..., description="News category")
    
    # Content
    title: str = Field(..., description="News headline")
    content: Optional[str] = Field(None, description="Summary/snippet (first 500 chars)")
    full_content: Optional[str] = Field(None, description="Full article text if available")
    url: str = Field(..., description="Original article URL")
    image_url: Optional[str] = Field(None, description="Thumbnail/image URL")
    
    # Metadata
    source_site: Optional[str] = Field(None, description="Source website name (e.g., Reuters)")
    published_at: datetime = Field(..., description="Publication datetime")
    fetched_at: datetime = Field(default_factory=datetime.utcnow, description="When we fetched this")
    
    # Matching fields (extracted for keyword matching)
    symbols: List[str] = Field(default_factory=list, description="Related symbols [AAPL, BTC-USD]")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    
    # Scoring (computed during aggregation)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Match score 0-1")
    importance_score: Optional[float] = Field(None, description="LLM-computed importance 1-10")
    sentiment: Optional[str] = Field(None, description="positive/negative/neutral")
    
    # For deduplication
    url_hash: Optional[str] = Field(None, description="SHA256 hash of URL")
    title_normalized: Optional[str] = Field(None, description="Lowercase, stripped title")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def model_post_init(self, __context) -> None:
        """Generate computed fields after initialization"""
        # Generate URL hash for deduplication
        if not self.url_hash and self.url:
            self.url_hash = hashlib.sha256(self.url.encode()).hexdigest()[:16]
        
        # Generate normalized title for similarity matching
        if not self.title_normalized and self.title:
            self.title_normalized = self.title.lower().strip()
        
        # Generate ID if not provided
        if not self.id and self.url_hash:
            self.id = f"{self.provider.value}_{self.url_hash}"
    
    @computed_field
    @property
    def has_symbol(self) -> bool:
        """Check if this news item has associated symbols"""
        return len(self.symbols) > 0
    
    def matches_keywords(self, keywords: List[str]) -> bool:
        """
        Check if this news item matches any of the given keywords.
        Searches in title, content, and symbols.
        """
        if not keywords:
            return True
        
        search_text = f"{self.title} {self.content or ''} {' '.join(self.symbols)}".lower()
        
        for keyword in keywords:
            if keyword.lower() in search_text:
                return True
        return False
    
    def matches_symbols(self, symbols: List[str]) -> bool:
        """Check if this news item matches any of the given symbols"""
        if not symbols:
            return True
        
        symbols_lower = [s.lower() for s in symbols]
        item_symbols_lower = [s.lower() for s in self.symbols]
        
        # Check direct symbol match
        for symbol in symbols_lower:
            if symbol in item_symbols_lower:
                return True
            # Also check in title (e.g., "AAPL" mentioned in title)
            if symbol in self.title.lower():
                return True
        
        return False
    
    def to_summary_dict(self) -> dict:
        """Convert to a simplified dict for digest generation"""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content[:300] if self.content else "",
            "url": self.url,
            "source": self.source_site,
            "published_at": self.published_at.isoformat(),
            "symbols": self.symbols,
            "category": self.category.value,
            "importance_score": self.importance_score,
        }