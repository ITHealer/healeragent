# src/routers/news_agent.py - CLEANED VERSION

import os
import logging
from typing import Any, Dict, List, Optional

import aioredis
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.helpers.redis_cache import get_redis_client
from src.services.news_agent_service import (
    execute_news_intelligence_standard,
    execute_news_intelligence_personalized,
    execute_symbol_news,
    execute_sector_analysis,
)

app_logger = logging.getLogger("main_app")
app_logger.setLevel(logging.INFO)

router = APIRouter()
api_key_auth = APIKeyAuth()


class NewsIntelligenceRequest(BaseModel):
    """Request for daily market digest."""
    topics: Optional[str] = Field(default=None, description="Custom topics; defaults to standard financial ones.")
    max_searches: int = Field(default=3, ge=1, le=5, description="Number of searches (1-5).")
    target_language: str = Field(default="en", description="Target language (en, vi, zh, ja, ko, fr, de, es).")
    include_videos: bool = Field(default=True, description="Include video content.")
    include_embedded_links: bool = Field(default=True, description="Process embedded links.")
    max_results_per_search: int = Field(default=10, ge=1, le=20, description="Max results per search (1-20).")
    processing_method: str = Field(default="tavily", description="'content_processor' or 'tavily'.")
    model_name: str = Field(default="gpt-4.1-nano", description="LLM model name.")
    provider_type: str = Field(default="openai", description="LLM provider: openai, ollama.")


class PersonalizedNewsRequest(BaseModel):
    """Request for personalized news."""
    topics: str = Field(description="Topics string (e.g., 'AI stocks, crypto, tech news').")
    max_searches: int = Field(default=3, ge=1, le=5, description="Number of searches (1-5).")
    target_language: str = Field(default="en", description="Target language (en, vi, zh, ja, ko, fr, de, es).")
    include_videos: bool = Field(default=True, description="Include video content.")
    include_embedded_links: bool = Field(default=True, description="Process embedded links.")
    max_results_per_search: int = Field(default=10, ge=1, le=20, description="Max results per search (1-20).")
    processing_method: str = Field(default="tavily", description="'content_processor' or 'tavily'.")
    model_name: str = Field(default="gpt-4.1-nano", description="LLM model name.")
    provider_type: str = Field(default="openai", description="LLM provider: openai, ollama.")


class SymbolNewsRequest(BaseModel):
    """Request for symbol-specific news"""
    symbols: List[str] = Field(description="Symbols to track (e.g., ['AAPL','TSLA','BTC-USD']).")
    topics: Optional[str] = Field(default=None, description="Symbol-focused topics.")
    max_searches: int = Field(default=3, ge=1, le=5, description="Number of searches (1-5).")
    target_language: str = Field(default="en", description="Target language.")
    include_videos: bool = Field(default=True, description="Include video content.")
    include_embedded_links: bool = Field(default=True, description="Process embedded links.")
    max_results_per_search: int = Field(default=10, ge=1, le=20, description="Max results per search (1-20).")
    processing_method: str = Field(default="tavily", description="'content_processor' or 'tavily'.")
    model_name: str = Field(default="gpt-4.1-nano", description="LLM model name.")
    provider_type: str = Field(default="openai", description="LLM provider: openai, ollama.")


class SectorAnalysisRequest(BaseModel):
    """Request for sector analysis."""
    sectors: List[str] = Field(description="Sectors to analyze (e.g., ['Technology','Healthcare']).")
    analysis_depth: str = Field(default="standard", description="'quick' (1 search), 'standard' (2), 'deep' (3).")
    topics: Optional[str] = Field(default=None, description="Optional sector topics.")
    target_language: str = Field(default="en", description="Target language.")
    include_videos: bool = Field(default=True, description="Include video content.")
    max_results_per_search: int = Field(default=15, ge=1, le=20, description="Max results per search (1-20).")
    processing_method: str = Field(default="tavily", description="'content_processor' or 'tavily'.")
    model_name: str = Field(default="gpt-4.1-nano", description="LLM model name.")
    provider_type: str = Field(default="openai", description="LLM provider: openai, ollama.")


class NewsIntelligenceData(BaseModel):
    """Data payload for news response."""
    thread_id: str
    summaries: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_stats: Dict[str, Any]


class NewsIntelligenceResponse(BaseModel):
    """Standard response model for news endpoints."""
    status: str
    message: str
    data: Optional[NewsIntelligenceData]


@router.post("/news/daily_news_summarize", response_model=NewsIntelligenceResponse)
async def get_daily_news_summaries(
    request: NewsIntelligenceRequest,
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Get daily market digest with general market news.
    Cache TTL: 15 minutes
    """
    
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    if not os.getenv("TAVILY_API_KEY"):
        raise HTTPException(status_code=500, detail="Tavily API key not configured")
    
    result = await execute_news_intelligence_standard(
        topics=request.topics,
        max_searches=request.max_searches,
        target_language=request.target_language,
        include_videos=request.include_videos,
        include_embedded_links=request.include_embedded_links,
        max_results_per_search=request.max_results_per_search,
        processing_method=request.processing_method,
        model=request.model_name,
        provider_type=request.provider_type,
        redis_client=redis_client
    )
    
    if result["status"] == "error":
        return NewsIntelligenceResponse(
            status="error",
            message=f"Failed: {result['metadata'].get('error', 'Unknown error')}",
            data=None
        )
    
    return NewsIntelligenceResponse(
        status="success",
        message=f"Processed {result['processing_stats']['total_articles']} articles in {result['metadata']['target_language']}",
        data=NewsIntelligenceData(
            thread_id=result["metadata"]["thread_id"],
            summaries=result["summaries"],
            metadata=result["metadata"],
            processing_stats=result["processing_stats"]
        )
    )


@router.post("/news/personalized_news_summarize", response_model=NewsIntelligenceResponse)
async def get_personalized_news_summaries(
    request: PersonalizedNewsRequest,
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Personalized news - FE provides full topics string.
    Cache TTL: 30 minutes
    
    FE should manage user preferences (localStorage/IndexedDB) and send full topics.
    """
    
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    if not os.getenv("TAVILY_API_KEY"):
        raise HTTPException(status_code=500, detail="Tavily API key not configured")
    
    result = await execute_news_intelligence_personalized(
        topics=request.topics,
        max_searches=request.max_searches,
        target_language=request.target_language,
        include_videos=request.include_videos,
        include_embedded_links=request.include_embedded_links,
        max_results_per_search=request.max_results_per_search,
        processing_method=request.processing_method,
        model=request.model_name,
        provider_type=request.provider_type,
        redis_client=redis_client
    )
    
    if result["status"] == "error":
        return NewsIntelligenceResponse(
            status="error",
            message=f"Failed: {result['metadata'].get('error', 'Unknown error')}",
            data=None
        )
    
    return NewsIntelligenceResponse(
        status="success",
        message=f"Processed {result['processing_stats']['total_articles']} personalized articles",
        data=NewsIntelligenceData(
            thread_id=result["metadata"]["thread_id"],
            summaries=result["summaries"],
            metadata=result["metadata"],
            processing_stats=result["processing_stats"]
        )
    )


@router.post("/news/symbol_news", response_model=NewsIntelligenceResponse)
async def get_symbol_specific_news(
    request: SymbolNewsRequest,
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Symbol-specific news - FE provides symbols list.
    Cache TTL: 15 minutes
    
    FE should manage user watchlist and send symbols array.
    """
    
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    if not os.getenv("TAVILY_API_KEY"):
        raise HTTPException(status_code=500, detail="Tavily API key not configured")
    
    result = await execute_symbol_news(
        symbols=request.symbols,
        topics=request.topics,
        max_searches=request.max_searches,
        target_language=request.target_language,
        include_videos=request.include_videos,
        include_embedded_links=request.include_embedded_links,
        max_results_per_search=request.max_results_per_search,
        processing_method=request.processing_method,
        model=request.model_name,
        provider_type=request.provider_type,
        redis_client=redis_client
    )
    
    if result["status"] == "error":
        return NewsIntelligenceResponse(
            status="error",
            message=f"Failed: {result['metadata'].get('error', 'Unknown error')}",
            data=None
        )
    
    symbols_tracked = result['metadata'].get('symbols', [])
    
    return NewsIntelligenceResponse(
        status="success",
        message=f"Tracked {len(symbols_tracked)} symbols: {', '.join(symbols_tracked[:3])}{'...' if len(symbols_tracked) > 3 else ''}",
        data=NewsIntelligenceData(
            thread_id=result["metadata"]["thread_id"],
            summaries=result["summaries"],
            metadata=result["metadata"],
            processing_stats=result["processing_stats"]
        )
    )


@router.post("/news/sector_analysis", response_model=NewsIntelligenceResponse)
async def get_sector_analysis(
    request: SectorAnalysisRequest,
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Sector analysis with industry trends.
    Cache TTL: 60 minutes
    """
    
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    if not os.getenv("TAVILY_API_KEY"):
        raise HTTPException(status_code=500, detail="Tavily API key not configured")
    
    result = await execute_sector_analysis(
        sectors=request.sectors,
        analysis_depth=request.analysis_depth,
        topics=request.topics,
        target_language=request.target_language,
        include_videos=request.include_videos,
        max_results_per_search=request.max_results_per_search,
        processing_method=request.processing_method,
        model=request.model_name,
        provider_type=request.provider_type,
        redis_client=redis_client
    )
    
    if result["status"] == "error":
        return NewsIntelligenceResponse(
            status="error",
            message=f"Failed: {result['metadata'].get('error', 'Unknown error')}",
            data=None
        )
    
    return NewsIntelligenceResponse(
        status="success",
        message=f"Analyzed {len(request.sectors)} sectors: {', '.join(request.sectors)}",
        data=NewsIntelligenceData(
            thread_id=result["metadata"]["thread_id"],
            summaries=result["summaries"],
            metadata=result["metadata"],
            processing_stats=result["processing_stats"]
        )
    )