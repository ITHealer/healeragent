# src/news_aggregator/schemas/fmp_news.py
"""
FMP News Response Models
Based on Financial Modeling Prep Stable API endpoints

Endpoints covered:
- /stable/news/stock-latest
- /stable/news/general-latest
- /stable/news/crypto-latest
- /stable/news/forex-latest
- /stable/news/press-releases-latest
- /stable/fmp-articles
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class NewsCategory(str, Enum):
    """News categories supported by FMP"""
    STOCK = "stock"
    GENERAL = "general"
    CRYPTO = "crypto"
    FOREX = "forex"
    PRESS_RELEASE = "press_release"
    FMP_ARTICLE = "fmp_article"


class FMPStockNewsItem(BaseModel):
    """
    Stock News from FMP /stable/news/stock-latest

    Example response:
    {
        "symbol": "AAPL",
        "publishedDate": "2024-02-28 14:30:00",
        "title": "Apple Releases New iPhone Model",
        "image": "https://...",
        "site": "Reuters",
        "text": "Apple has announced...",
        "url": "https://example.com/article"
    }
    """
    symbol: Optional[str] = Field(None, description="Stock symbol (e.g., AAPL)")
    published_date: Optional[datetime] = Field(None, alias="publishedDate", description="Publication datetime")
    title: str = Field(..., description="News headline")
    image: Optional[str] = Field(None, description="Image URL")
    site: Optional[str] = Field(None, description="Source website name")
    text: Optional[str] = Field(None, description="News snippet/summary")
    url: str = Field(..., description="Full article URL")

    class Config:
        populate_by_name = True

    @field_validator("published_date", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        """Parse datetime from FMP format. Returns None if unparseable."""
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            return None  # Don't auto-assign current date
        return None


class FMPGeneralNewsItem(BaseModel):
    """
    General News from FMP /stable/news/general-latest
    Similar structure to stock news but without symbol
    """
    published_date: Optional[datetime] = Field(None, alias="publishedDate")
    title: str = Field(...)
    image: Optional[str] = None
    site: Optional[str] = None
    text: Optional[str] = None
    url: str = Field(...)

    class Config:
        populate_by_name = True

    @field_validator("published_date", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        """Parse datetime. Returns None if unparseable."""
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            return None
        return None


class FMPCryptoNewsItem(BaseModel):
    """
    Crypto News from FMP /stable/news/crypto-latest
    Note: symbol format is like "BTCUSD"
    """
    symbol: Optional[str] = Field(None, description="Crypto pair (e.g., BTCUSD)")
    published_date: Optional[datetime] = Field(None, alias="publishedDate")
    title: str = Field(...)
    image: Optional[str] = None
    site: Optional[str] = None
    text: Optional[str] = None
    url: str = Field(...)

    class Config:
        populate_by_name = True

    @field_validator("published_date", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            return None
        return None


class FMPForexNewsItem(BaseModel):
    """Forex News from FMP /stable/news/forex-latest"""
    symbol: Optional[str] = Field(None, description="Currency pair (e.g., EURUSD)")
    published_date: Optional[datetime] = Field(None, alias="publishedDate")
    title: str = Field(...)
    image: Optional[str] = None
    site: Optional[str] = None
    text: Optional[str] = None
    url: str = Field(...)

    class Config:
        populate_by_name = True

    @field_validator("published_date", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            return None
        return None


class FMPPressReleaseItem(BaseModel):
    """Press Releases from FMP /stable/news/press-releases-latest"""
    symbol: Optional[str] = Field(None, description="Company symbol")
    published_date: Optional[datetime] = Field(None, alias="publishedDate")
    title: str = Field(...)
    image: Optional[str] = None
    site: Optional[str] = None
    text: Optional[str] = None
    url: str = Field(...)

    class Config:
        populate_by_name = True

    @field_validator("published_date", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            return None
        return None


class FMPArticleItem(BaseModel):
    """FMP Articles from /stable/fmp-articles - written by FMP analysts"""
    published_date: Optional[datetime] = Field(None, alias="publishedDate")
    title: str = Field(...)
    image: Optional[str] = None
    site: str = Field(default="Financial Modeling Prep")
    text: Optional[str] = Field(None, alias="content")
    url: str = Field(..., alias="link")
    author: Optional[str] = None
    tickers: Optional[List[str]] = Field(default_factory=list)

    class Config:
        populate_by_name = True

    @field_validator("published_date", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            return None
        return None