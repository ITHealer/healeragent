import os
import logging
from typing import Any, Dict, List, Optional
import asyncio
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse
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
from src.helpers.llm_chat_helper import (
    SSE_HEADERS,
    DEFAULT_HEARTBEAT_SEC,
    sse_error,
    sse_done
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


@router.post("/news/daily_news_summarize/stream")
async def get_daily_news_summaries_stream(
    request: NewsIntelligenceRequest,
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Stream daily market digest with heartbeat support.
    Prevents timeout during long-running news processing.
    """
    
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    if not os.getenv("TAVILY_API_KEY"):
        raise HTTPException(status_code=500, detail="Tavily API key not configured")
    
    async def generate() -> AsyncGenerator[str, None]:
        import json
        
        try:
            # Send initial progress
            yield f"{json.dumps({'type': 'progress', 'message': 'Starting daily news analysis...'}, ensure_ascii=False)}\n\n"
            
            # Create task for long-running operation
            news_task = asyncio.create_task(
                execute_news_intelligence_standard(
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
            )
            
            # Wait with heartbeat
            result = None
            iteration = 0
            while True:
                done, _ = await asyncio.wait({news_task}, timeout=DEFAULT_HEARTBEAT_SEC)
                
                if done:
                    try:
                        result = news_task.result()
                    except Exception as e:
                        app_logger.error(f"Daily news task error: {e}")
                        yield sse_error(str(e))
                        yield sse_done()
                        return
                    break
                else:
                    # Send heartbeat
                    yield ": heartbeat\n\n"
                    iteration += 1
                    
                    # Send progress update every 2 heartbeats
                    if iteration % 2 == 0:
                        yield f"{json.dumps({'type': 'progress', 'message': f'Processing news... ({iteration * DEFAULT_HEARTBEAT_SEC}s elapsed)'}, ensure_ascii=False)}\n\n"
            
            # Process result
            if result["status"] == "error":
                yield sse_error(result['metadata'].get('error', 'Unknown error'))
                yield sse_done()
                return
            
            # Build success response
            response_data = {
                "type": "result",
                "status": "success",
                "message": f"Processed {result['processing_stats']['total_articles']} articles in {result['metadata']['target_language']}",
                "data": {
                    "thread_id": result["metadata"]["thread_id"],
                    "summaries": result["summaries"],
                    "metadata": result["metadata"],
                    "processing_stats": result["processing_stats"]
                }
            }
            
            yield f"{json.dumps(response_data, ensure_ascii=False)}\n\n"
            yield sse_done()
            
        except asyncio.CancelledError:
            app_logger.info("Daily news stream cancelled")
            raise
        except Exception as e:
            app_logger.error(f"Daily news stream error: {e}", exc_info=True)
            yield sse_error(str(e))
            yield sse_done()
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )


@router.post("/news/personalized_news_summarize/stream")
async def get_personalized_news_summaries_stream(
    request: PersonalizedNewsRequest,
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Stream personalized news with heartbeat support.
    Prevents timeout during long-running news processing.
    """
    
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    if not os.getenv("TAVILY_API_KEY"):
        raise HTTPException(status_code=500, detail="Tavily API key not configured")
    
    async def generate() -> AsyncGenerator[str, None]:
        import json
        
        try:
            # Send initial progress
            yield f"{json.dumps({'type': 'progress', 'message': 'Starting personalized news analysis...'}, ensure_ascii=False)}\n\n"
            
            # Create task for long-running operation
            news_task = asyncio.create_task(
                execute_news_intelligence_personalized(
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
            )
            
            # Wait with heartbeat
            result = None
            iteration = 0
            while True:
                done, _ = await asyncio.wait({news_task}, timeout=DEFAULT_HEARTBEAT_SEC)
                
                if done:
                    try:
                        result = news_task.result()
                    except Exception as e:
                        app_logger.error(f"Personalized news task error: {e}")
                        yield sse_error(str(e))
                        yield sse_done()
                        return
                    break
                else:
                    # Send heartbeat
                    yield ": heartbeat\n\n"
                    iteration += 1
                    
                    if iteration % 2 == 0:
                        yield f"{json.dumps({'type': 'progress', 'message': f'Gathering personalized news... ({iteration * DEFAULT_HEARTBEAT_SEC}s elapsed)'}, ensure_ascii=False)}\n\n"
            
            # Process result
            if result["status"] == "error":
                yield sse_error(result['metadata'].get('error', 'Unknown error'))
                yield sse_done()
                return
            
            # Build success response
            response_data = {
                "type": "result",
                "status": "success",
                "message": f"Processed {result['processing_stats']['total_articles']} personalized articles",
                "data": {
                    "thread_id": result["metadata"]["thread_id"],
                    "summaries": result["summaries"],
                    "metadata": result["metadata"],
                    "processing_stats": result["processing_stats"]
                }
            }
            
            yield f"{json.dumps(response_data, ensure_ascii=False)}\n\n"
            yield sse_done()
            
        except asyncio.CancelledError:
            app_logger.info("Personalized news stream cancelled")
            raise
        except Exception as e:
            app_logger.error(f"Personalized news stream error: {e}", exc_info=True)
            yield sse_error(str(e))
            yield sse_done()
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )


@router.post("/news/symbol_news/stream")
async def get_symbol_specific_news_stream(
    request: SymbolNewsRequest,
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Stream symbol-specific news with heartbeat support.
    Prevents timeout during long-running news processing.
    """
    
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    if not os.getenv("TAVILY_API_KEY"):
        raise HTTPException(status_code=500, detail="Tavily API key not configured")
    
    async def generate() -> AsyncGenerator[str, None]:
        import json
        
        try:
            symbols_str = ', '.join(request.symbols[:3])
            suffix = '...' if len(request.symbols) > 3 else ''
            
            # Send initial progress
            yield f"{json.dumps({'type': 'progress', 'message': f'Starting symbol news analysis for {symbols_str}{suffix}...'}, ensure_ascii=False)}\n\n"
            
            # Create task for long-running operation
            news_task = asyncio.create_task(
                execute_symbol_news(
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
            )
            
            # Wait with heartbeat
            result = None
            iteration = 0
            while True:
                done, _ = await asyncio.wait({news_task}, timeout=DEFAULT_HEARTBEAT_SEC)
                
                if done:
                    try:
                        result = news_task.result()
                    except Exception as e:
                        app_logger.error(f"Symbol news task error: {e}")
                        yield sse_error(str(e))
                        yield sse_done()
                        return
                    break
                else:
                    # Send heartbeat
                    yield ": heartbeat\n\n"
                    iteration += 1
                    
                    if iteration % 2 == 0:
                        yield f"{json.dumps({'type': 'progress', 'message': f'Tracking {len(request.symbols)} symbols... ({iteration * DEFAULT_HEARTBEAT_SEC}s elapsed)'}, ensure_ascii=False)}\n\n"
            
            # Process result
            if result["status"] == "error":
                yield sse_error(result['metadata'].get('error', 'Unknown error'))
                yield sse_done()
                return
            
            # Build success response
            symbols_tracked = result['metadata'].get('symbols', [])
            
            response_data = {
                "type": "result",
                "status": "success",
                "message": f"Tracked {len(symbols_tracked)} symbols: {', '.join(symbols_tracked[:3])}{'...' if len(symbols_tracked) > 3 else ''}",
                "data": {
                    "thread_id": result["metadata"]["thread_id"],
                    "summaries": result["summaries"],
                    "metadata": result["metadata"],
                    "processing_stats": result["processing_stats"]
                }
            }
            
            yield f"{json.dumps(response_data, ensure_ascii=False)}\n\n"
            yield sse_done()
            
        except asyncio.CancelledError:
            app_logger.info("Symbol news stream cancelled")
            raise
        except Exception as e:
            app_logger.error(f"Symbol news stream error: {e}", exc_info=True)
            yield sse_error(str(e))
            yield sse_done()
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )


@router.post("/news/sector_analysis/stream")
async def get_sector_analysis_stream(
    request: SectorAnalysisRequest,
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Stream sector analysis with heartbeat support.
    Prevents timeout during long-running analysis.
    """
    
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    if not os.getenv("TAVILY_API_KEY"):
        raise HTTPException(status_code=500, detail="Tavily API key not configured")
    
    async def generate() -> AsyncGenerator[str, None]:
        import json
        
        try:
            sectors_str = ', '.join(request.sectors)
            
            # Send initial progress
            yield f"{json.dumps({'type': 'progress', 'message': f'Starting {request.analysis_depth} analysis for sectors: {sectors_str}...'}, ensure_ascii=False)}\n\n"
            
            # Create task for long-running operation
            news_task = asyncio.create_task(
                execute_sector_analysis(
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
            )
            
            # Wait with heartbeat
            result = None
            iteration = 0
            while True:
                done, _ = await asyncio.wait({news_task}, timeout=DEFAULT_HEARTBEAT_SEC)
                
                if done:
                    try:
                        result = news_task.result()
                    except Exception as e:
                        app_logger.error(f"Sector analysis task error: {e}")
                        yield sse_error(str(e))
                        yield sse_done()
                        return
                    break
                else:
                    # Send heartbeat
                    yield ": heartbeat\n\n"
                    iteration += 1
                    
                    if iteration % 2 == 0:
                        yield f"{json.dumps({'type': 'progress', 'message': f'Analyzing {len(request.sectors)} sectors ({request.analysis_depth})... ({iteration * DEFAULT_HEARTBEAT_SEC}s elapsed)'}, ensure_ascii=False)}\n\n"
            
            # Process result
            if result["status"] == "error":
                yield sse_error(result['metadata'].get('error', 'Unknown error'))
                yield sse_done()
                return
            
            # Build success response
            response_data = {
                "type": "result",
                "status": "success",
                "message": f"Analyzed {len(request.sectors)} sectors: {sectors_str}",
                "data": {
                    "thread_id": result["metadata"]["thread_id"],
                    "summaries": result["summaries"],
                    "metadata": result["metadata"],
                    "processing_stats": result["processing_stats"]
                }
            }
            
            yield f"{json.dumps(response_data, ensure_ascii=False)}\n\n"
            yield sse_done()
            
        except asyncio.CancelledError:
            app_logger.info("Sector analysis stream cancelled")
            raise
        except Exception as e:
            app_logger.error(f"Sector analysis stream error: {e}", exc_info=True)
            yield sse_error(str(e))
            yield sse_done()
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )