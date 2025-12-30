import json
import aioredis
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends, Query, HTTPException, Request
from datetime import datetime, timedelta
from src.services.news_service import NewsService
from src.models.equity import NewsItemOutput, APIResponse, APIResponseData
from src.handlers.news_analysis_handler import NewsAnalysisHandler
from src.helpers.redis_cache import get_redis_client, get_cache, set_cache
from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.helpers.llm_helper import LLMGeneratorProvider
from src.helpers.chat_management_helper import ChatService
from src.handlers.llm_chat_handler import ChatMessageHistory
from src.helpers.llm_chat_helper import (
    stream_with_heartbeat,
    analyze_conversation_importance,
    SSE_HEADERS,
    DEFAULT_HEARTBEAT_SEC,
    sse_error,
    sse_done
)
from src.agents.memory.memory_manager import get_memory_manager
from src.services.background_tasks import trigger_summary_update_nowait

# Initialize router and services
router = APIRouter()

api_key_auth = APIKeyAuth()
news_analysis_handler = NewsAnalysisHandler()
news_service = NewsService()
logger_mixin = LoggerMixin()
logger = logger_mixin.logger
llm_generator = LLMGeneratorProvider()
chat_service = ChatService()
memory_manager = get_memory_manager()



class NewsAnalysisRequest(BaseModel):
    session_id: str = Field(None, description="Chat session ID for context")
    question_input: Optional[str] = None
    target_language: Optional[str] = None
    symbol: str = Field(..., description="Stock symbol to analyze")
    # limit: int = Field(10, ge=1, le=50, description="Number of news items to analyze"),
    model_name: str = Field("gpt-4.1-nano-2025-04-14", description="LLM model to use")
    provider_type: str = Field("openai", description="Provider type: openai, ollama, gemini")
    # include_trading_signals: bool = Field(True, description="Include trading recommendations in analysis")

class NewsAnalysisResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


@router.post("/news/analysis",
            response_model=NewsAnalysisResponse,
            summary="Get company news with AI-powered market impact analysis")

async def get_news_with_analysis(
    request: Request,
    chat_request: NewsAnalysisRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Get company news with AI-powered analysis of market impact.
    
    This endpoint fetches recent news and provides:
    - Sentiment analysis
    - Market impact assessment
    - Trading implications
    - Key themes and insights
    """

    user_id = getattr(request.state, "user_id", None)

    # Config
    limit = 10
    include_trading_signals = True

    # Extract values from request
    symbol = chat_request.symbol
    limit = limit
    model_name = chat_request.model_name
    provider_type = chat_request.provider_type
    include_trading_signals = include_trading_signals

    # Cache key for raw news data
    news_cache_key = f"company_news_{symbol.upper()}_limit_{limit}"
    
    # Get news data (from cache or API)
    news_data = None
    
    if redis_client:
        try:
            # Try to get cached news
            cached_response = await get_cache(redis_client, news_cache_key, APIResponse[NewsItemOutput])
            
            if cached_response and cached_response.data and cached_response.data.data:
                logger.info(f"Cache HIT for news data: {news_cache_key}")
                news_data = [item.model_dump() if hasattr(item, 'model_dump') else item 
                            for item in cached_response.data.data]
        except Exception as e:
            logger.error(f"Cache error: {e}")
    
    # If not cached, fetch from API
    if news_data is None:
        try:
            news_items = await news_service.get_company_news(symbol.upper(), limit)
            
            if news_items:
                # Convert to dict format
                news_data = [item.model_dump() if hasattr(item, 'model_dump') else item.__dict__ 
                            for item in news_items]
                
                # Cache the news data
                if redis_client:
                    response_data_payload = APIResponseData[NewsItemOutput](data=news_items)
                    api_response = APIResponse[NewsItemOutput](
                        message="OK",
                        status="200", 
                        provider_used="fmp",
                        data=response_data_payload
                    )
                    await set_cache(redis_client, news_cache_key, api_response, expiry=settings.CACHE_TTL_CHART)
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return NewsAnalysisResponse(
                status="500",
                message=f"Error fetching news for {symbol}: {str(e)}",
                data=None
            )
    
    # Check if we have news to analyze
    if not news_data:
        return NewsAnalysisResponse(
            status="404",
            message=f"No news found for {symbol}",
            data={
                "symbol": symbol,
                # "news_count": 0,
                "interpretation": "No recent news available for analysis.",
                # "market_impact": "neutral",
                # "sentiment_score": 0
            }
        )
    
    # Now perform AI analysis
    try:
        # Get API Key
        api_key = ModelProviderFactory._get_api_key(provider_type)
        
        # Check for analysis cache
        analysis_cache_key = f"news_analysis_{symbol}_{limit}_{model_name}_{provider_type}_{include_trading_signals}"
        
        if redis_client:
            try:
                cached_analysis = await redis_client.get(analysis_cache_key)
                if cached_analysis:
                    logger.info(f"Cache HIT for news analysis: {analysis_cache_key}")
                    cached_data = json.loads(cached_analysis)
                    return NewsAnalysisResponse(
                            status=cached_data.get("status", "200"),
                            message=cached_data.get("message", "News analysis completed successfully"),
                            data=cached_data
                        )
            except Exception as e:
                logger.error(f"Analysis cache error: {e}")
        
        # Generate new analysis
        analysis_result = await news_analysis_handler.analyze_company_news(
            symbol=symbol,
            news_data=news_data,
            model_name=model_name,
            provider_type=provider_type,
            api_key=api_key,
            include_trading_signals=include_trading_signals,
            question_input=chat_request.question_input,
            target_language=chat_request.target_language
        )
        
        # Prepare response
        response_data = {
            "analysis": analysis_result
        }
        
        # Cache the complete response
        if redis_client:
            try:
                cache_data = {
                    "status": "200",
                    "message": "News analysis completed successfully",
                    **response_data
                }
                # Analysis cache for shorter time (1 hour)
                await redis_client.set(
                    analysis_cache_key,
                    json.dumps(cache_data),
                    ex=300
                )
            except Exception as e:
                logger.error(f"Analysis cache set error: {e}")
        
        if chat_request.session_id and user_id:
            try:
                from src.helpers.chat_management_helper import ChatService
                chat_service = ChatService()
                
                question_content = chat_request.question_input or f"Analyze technical indicators for {chat_request.symbol}"
                question_id = chat_service.save_user_question(
                    session_id=chat_request.session_id,
                    created_at=datetime.now(),
                    created_by=user_id,
                    content=question_content
                )
                
                chat_service.save_assistant_response(
                    session_id=chat_request.session_id,
                    created_at=datetime.now(),
                    question_id=question_id,
                    content=analysis_result["analysis"],
                    response_time=0.1 
                )
            except Exception as e:
                logger.error(f"Error saving to chat history: {str(e)}")


        return NewsAnalysisResponse(
            status="200",
            message="News analysis completed successfully",
            data=analysis_result["analysis"]
        )
        
    except Exception as e:
        logger.error(f"News analysis error for {symbol}: {str(e)}")
        return NewsAnalysisResponse(
            status="500",
            message=f"Error analyzing news for {symbol}: {str(e)}",
            data={
                "symbol": symbol,
                # "news_count": len(news_data),
                "interpretation": None,
                # "market_impact": "unknown",
                # "sentiment_score": 0
            }
        )


@router.post("/news/analysis/stream",
            summary="Get company news with AI-powered market impact analysis (STREAMING)")
async def get_news_with_analysis_stream(
    request: Request,
    chat_request: NewsAnalysisRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Get company news with streaming AI-powered analysis and memory integration
    """
    user_id = getattr(request.state, "user_id", None)
  
    async def generate_stream():
        full_response = []
        news_data = None
        
        try:
            # Get chat history
            chat_history = ""
            if chat_request.session_id:
                try:
                    chat_history = ChatMessageHistory.string_message_chat_history(
                        session_id=chat_request.session_id
                    )
                except Exception as e:
                    logger.error(f"Error fetching history: {e}")

            # Get memory context
            context = ""
            memory_stats = {}
            if chat_request.session_id and user_id:
                try:
                    query_text = chat_request.question_input or f"Analyze news for {chat_request.symbol}"
                    context, memory_stats, _ = await memory_manager.get_relevant_context(
                        session_id=chat_request.session_id,
                        user_id=user_id,
                        current_query=query_text,
                        llm_provider=llm_generator,
                        max_short_term=5,
                        max_long_term=3
                    )
                except Exception as e:
                    logger.error(f"Error getting memory context: {e}")
            
            # Build enhanced history
            enhanced_history = ""
            if context:
                enhanced_history = f"{context}\n\n"
            if chat_history:
                enhanced_history += f"[Conversation History]\n{chat_history}"
            
            # Config
            limit = 10
            include_trading_signals = True
            symbol = chat_request.symbol
            model_name = chat_request.model_name
            provider_type = chat_request.provider_type
            
            # Save user question first
            question_id = None
            if chat_request.session_id and user_id:
                try:
                    question_content = chat_request.question_input or f"Analyze news for {symbol}"
                    question_id = chat_service.save_user_question(
                        session_id=chat_request.session_id,
                        created_at=datetime.now(),
                        created_by=user_id,
                        content=question_content
                    )
                except Exception as e:
                    logger.error(f"Error saving question: {e}")
            
            # Cache key for raw news data 
            news_cache_key = f"company_news_{symbol.upper()}_limit_{limit}"
            
            # Get news data 
            if redis_client:
                try:
                    cached_response = await get_cache(redis_client, news_cache_key, APIResponse[NewsItemOutput])
                    if cached_response and cached_response.data and cached_response.data.data:
                        logger.info(f"Cache HIT for news data: {news_cache_key}")
                        news_data = [item.model_dump() if hasattr(item, 'model_dump') else item 
                                    for item in cached_response.data.data]
                except Exception as e:
                    logger.error(f"Cache error: {e}")
            
            # If not cached, fetch from API 
            if news_data is None:
                try:
                    logger.info(f"Cache MISS for news data: {news_cache_key}, fetching from API")
                    news_items = await news_service.get_company_news(symbol.upper(), limit)
                    
                    if news_items:
                        news_data = [item.model_dump() if hasattr(item, 'model_dump') else item.__dict__ 
                                    for item in news_items]
                        
                        # Cache the news data
                        if redis_client:
                            response_data_payload = APIResponseData[NewsItemOutput](data=news_items)
                            api_response = APIResponse[NewsItemOutput](
                                message="OK",
                                status="200", 
                                provider_used="fmp",
                                data=response_data_payload
                            )
                            await set_cache(redis_client, news_cache_key, api_response, expiry=settings.CACHE_TTL_CHART)
                except Exception as e:
                    logger.error(f"Error fetching news: {e}")
                    yield sse_error(f"Error fetching news for {symbol}: {str(e)}")
                    yield sse_done()
                    return
            
            # Check if we have news
            if not news_data:
                no_news_msg = f"No recent news available for analysis for {symbol}."
                yield f"{json.dumps({'content': no_news_msg}, ensure_ascii=False)}\n\n"
                yield sse_done()
                return
            
            # Prepare news summary
            news_count = len(news_data)
            sorted_news = sorted(
                news_data, 
                key=lambda x: news_analysis_handler._parse_date(x.get("publishedDate", "")), 
                reverse=True
            )
            
            # Get API Key
            api_key = ModelProviderFactory._get_api_key(provider_type)
            
            # Get LLM generator
            llm_gen = news_analysis_handler.stream_company_news_analysis(
                symbol=symbol,
                news_data=news_data,
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key,
                include_trading_signals=include_trading_signals,
                memory_context=enhanced_history,
                question_input=chat_request.question_input,
                target_language=chat_request.target_language
            )
            
            # Stream with heartbeat
            async for event in stream_with_heartbeat(llm_gen, DEFAULT_HEARTBEAT_SEC):
                if event["type"] == "content":
                    full_response.append(event["chunk"])
                    yield f"{json.dumps({'content': event['chunk']}, ensure_ascii=False)}\n\n"
                elif event["type"] == "heartbeat":
                    yield ": heartbeat\n\n"
                elif event["type"] == "error":
                    yield sse_error(event["error"])
                    break
                elif event["type"] == "done":
                    break
            
            # Join full response 
            analysis_text = ''.join(full_response)
            
            # Analyze conversation importance 
            importance_score = 0.5
            if chat_request.session_id and user_id and analysis_text:
                try:
                    analysis_model = "gpt-4.1-nano" if provider_type == ProviderType.OPENAI else model_name
                    
                    importance_score = await analyze_conversation_importance(
                        query=chat_request.question_input or f"Analyze news for {symbol}",
                        response=analysis_text,
                        llm_provider=llm_generator,
                        model_name=analysis_model,
                        provider_type=provider_type
                    )
                    
                    if "breaking" in analysis_text.lower() or "major" in analysis_text.lower():
                        importance_score = min(1.0, importance_score + 0.1)
                except Exception as e:
                    logger.error(f"Error analyzing importance: {e}")
            
            # Store in memory system 
            if chat_request.session_id and user_id and analysis_text:
                try:
                    metadata = {
                        "type": "news_analysis",
                        "symbol": symbol,
                        "news_count": news_count,
                        "latest_news_date": sorted_news[0].get("publishedDate") if sorted_news else None,
                        "include_trading_signals": include_trading_signals,
                        "news_titles": [n.get("title", "") for n in sorted_news[:5]]
                    }
                    
                    await memory_manager.store_conversation_turn(
                        session_id=chat_request.session_id,
                        user_id=user_id,
                        query=chat_request.question_input or f"Analyze news for {symbol}",
                        response=analysis_text,
                        metadata=metadata,
                        importance_score=importance_score
                    )
                    
                    if question_id:
                        chat_service.save_assistant_response(
                            session_id=chat_request.session_id,
                            created_at=datetime.now(),
                            question_id=question_id,
                            content=analysis_text,
                            response_time=0.1
                        )

                    trigger_summary_update_nowait(
                        session_id=chat_request.session_id,
                        user_id=user_id
                    )
                except Exception as e:
                    logger.error(f"Error saving to memory: {e}")
            
            yield sse_done()
            
        except Exception as e:
            logger.error(f"News analysis streaming error for {chat_request.symbol}: {e}")
            yield sse_error(str(e))
            yield sse_done()
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )