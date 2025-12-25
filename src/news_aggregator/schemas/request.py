# src/news_aggregator/schemas/request.py
"""
API Request Schemas for News Aggregator
Stateless design - Web BE provides all filter criteria
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from src.news_aggregator.schemas.fmp_news import NewsCategory


class NewsAggregatorRequest(BaseModel):
    """
    Main request model for news aggregation.
    
    Web BE sends this with user's subscription criteria.
    AI service fetches, filters, and returns matching news.
    
    Example:
    {
        "keywords": ["AAPL", "Apple", "iPhone"],
        "symbols": ["AAPL", "NVDA"],
        "categories": ["stock", "general"],
        "time_range_hours": 24,
        "max_articles": 50,
        "generate_digest": true,
        "target_language": "vi"
    }
    """
    # Filter criteria
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords to match in title/content (e.g., ['AAPL', 'Apple', 'earnings'])"
    )
    symbols: List[str] = Field(
        default_factory=list,
        description="Stock/crypto symbols to filter (e.g., ['AAPL', 'BTC-USD'])"
    )
    categories: List[str] = Field(
        default=["stock", "crypto", "general"],
        description="News categories: stock, crypto, forex, general, press_release, fmp_article"
    )
    
    # Time range
    time_range_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # Max 1 week
        description="Fetch news from last N hours"
    )
    
    # Pagination
    max_articles: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Maximum articles to return after filtering"
    )
    page: int = Field(
        default=0,
        ge=0,
        description="Page number for FMP pagination"
    )
    limit_per_category: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Max articles to fetch per category from FMP"
    )
    
    # Options
    generate_digest: bool = Field(
        default=False,
        description="Generate LLM summary digest"
    )
    target_language: str = Field(
        default="en",
        description="Target language for digest (en, vi, zh, ja, ko)"
    )
    include_full_content: bool = Field(
        default=False,
        description="Include full article content (uses Tavily extract)"
    )
    
    # Provider options
    use_tavily: bool = Field(
        default=True,
        description="Also search Tavily for supplementary news"
    )
    tavily_max_results: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Max results from Tavily search"
    )
    
    # LLM options (for digest)
    model_name: str = Field(
        default="gpt-4.1-nano",
        description="LLM model for digest generation"
    )
    provider_type: str = Field(
        default="openai",
        description="LLM provider: openai, ollama, gemini"
    )
    
    def get_categories(self) -> List[NewsCategory]:
        """Convert string categories to NewsCategory enum"""
        result = []
        for cat in self.categories:
            try:
                result.append(NewsCategory(cat.lower()))
            except ValueError:
                continue
        return result if result else [NewsCategory.STOCK, NewsCategory.GENERAL]


class DigestRequest(BaseModel):
    """
    Request to generate digest from already-fetched articles.
    
    Use case: Web BE already has articles from previous call,
    just wants to regenerate summary.
    """
    article_ids: List[str] = Field(
        ...,
        min_length=1,
        description="List of article IDs to summarize"
    )
    target_language: str = Field(
        default="en",
        description="Target language for summary"
    )
    model_name: str = Field(
        default="gpt-4.1-nano",
        description="LLM model for digest"
    )
    provider_type: str = Field(
        default="openai",
        description="LLM provider"
    )
    max_top_stories: int = Field(
        default=5,
        ge=3,
        le=10,
        description="Number of top stories to highlight"
    )