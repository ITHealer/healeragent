from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime
from datetime import datetime
from src.database.models.base import Base
from typing import Optional

from pydantic import BaseModel, HttpUrl, Field

class StockPrice(Base):
    __tablename__ = "stock_prices"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    ticker = Column(String, index=True)
    date = Column(DateTime)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    dividends = Column(Float)
    stock_splits = Column(Float)

class NewsSchema(Base):
    __tablename__ = "news"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    title = Column(String(512))
    content = Column(String)
    author = Column(String(256))
    url = Column(String(1024))
    created_at = Column(DateTime)

class NewsSchemaBase(BaseModel):
    """Pydantic model for news data validation and extraction"""
    title: str = Field(..., description="Article title")
    content: str = Field(..., description="Article content")
    author: str = Field(..., description="Article author")
    url: HttpUrl = Field(..., description="Article URL")
    created_at: Optional[datetime] = Field(None, description="Publication date")

    class Config:
        from_attributes = True # For ORM compatibility