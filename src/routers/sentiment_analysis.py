import json
import aioredis
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from src.utils.logger.custom_logging import LoggerMixin
from src.handlers.sentiment_analysis_handler import sentiment_analysis_handler
from src.utils.config import settings
from src.routers.equity import get_redis_client
from src.services.news_service import NewsService
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.helpers.chat_management_helper import ChatService
from src.handlers.llm_chat_handler import ChatMessageHistory
from src.routers.llm_chat import analyze_conversation_importance
from src.agents.memory.memory_manager import MemoryManager
from src.helpers.llm_helper import LLMGeneratorProvider
from datetime import datetime
from src.services.background_tasks import trigger_summary_update_nowait

router = APIRouter()

logger_mixin = LoggerMixin()
logger = logger_mixin.logger
news_service = NewsService()
FMP_API_KEY = settings.FMP_API_KEY
chat_service = ChatService()
memory_manager = MemoryManager()
llm_generator = LLMGeneratorProvider()


class SentimentAnalysisRequest(BaseModel):
    session_id: str = Field(None, description="Chat session ID for context")
    question_input: Optional[str] = None
    target_language: Optional[str] = None
    symbol: str = Field(..., description="Stock symbol to analyze")
    model_name: str = Field("gpt-4.1-nano-2025-04-14", description="LLM model to use")
    provider_type: str = Field("openai", description="Provider type: openai, ollama, gemini")
    # include_raw_data: bool = Field(False, description="Include raw sentiment data in response")


@router.post("/social-sentiment/analyze", summary="Capturing and Analyzing Social Emotional Data with LLM")
async def get_analyzed_social_sentiment(
    request_body: SentimentAnalysisRequest,
    page: int = Query(1, ge=0, description="Page number to get (starts from 0)"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Get sentiment data from FMP and analyze market impact through LLM
    """
    if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
        logger.error(f"FMP API key is not configured for sentiment analysis {request_body.symbol}.")
        return {
            "message": "FMP API key is not configured on the server.",
            "status": "500",
            "data": None
        }
    
    # Try cache first for raw sentiment data
    cache_key = f"sentiment_analysis_{request_body.symbol.upper()}_page_{page}_{request_body.model_name}"
    
    if redis_client:
        try:
            cached_analysis = await redis_client.get(cache_key)
            if cached_analysis:
                logger.info(f"Cache HIT cho sentiment analysis ({request_body.symbol}, page {page})")
                return json.loads(cached_analysis)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
    
    # Get sentiment data from FMP
    sentiment_data_list = await news_service.get_historical_social_sentiment(
        symbol=request_body.symbol, page=page
    )
    
    if sentiment_data_list is None:
        return {
            "message": f"Unable to get sentiment data for {request_body.symbol}",
            "status": "502",
            "data": None
        }
    
    # Convert Pydantic models to dicts for processing
    sentiment_dicts = []
    for item in sentiment_data_list:
        if hasattr(item, 'model_dump'):
            sentiment_dicts.append(item.model_dump())
        elif hasattr(item, 'dict'):
            sentiment_dicts.append(item.dict())
        else:
            sentiment_dicts.append(dict(item))
    
    # Get API Key
    api_key = ModelProviderFactory._get_api_key(request_body.provider_type)
            
    try:
        # Analyze with LLM
        analysis_result = await sentiment_analysis_handler.analyze_sentiment_impact(
            symbol=request_body.symbol,
            query=request_body.question_input,
            target_language=request_body.target_language,
            sentiment_data=sentiment_dicts,
            model_name=request_body.model_name,
            provider_type=request_body.provider_type,
            api_key=api_key 
        )
        
        # Prepare response
        response_data = {
            "symbol": request_body.symbol,
            # "page": page,
            # "sentiment_summary": analysis_result["sentiment_summary"],
            "interpretation": analysis_result["market_impact_analysis"],
            # "analysis_timestamp": analysis_result["analysis_timestamp"]
        }
        
        # Include raw data if requested
        # if request_body.include_raw_data:
        #     response_data["raw_sentiment_data"] = sentiment_dicts
        
        response = {
            "message": "Successful sentiment analysis",
            "status": "200",
            "data": response_data
        }
        
        # Cache the analysis
        if redis_client and len(sentiment_dicts) > 0:
            try:
                await redis_client.set(
                    cache_key,
                    json.dumps(response),
                    ex=300 
                )
                logger.info(f"Cached sentiment analysis for {request_body.symbol}")
            except Exception as e:
                logger.error(f"Redis SET error: {e}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return {
            "message": f"Sentiment analysis error: {str(e)}",
            "status": "500", 
            "data": None
        }


# @router.post("/social-sentiment/analyze/stream", summary="Streaming Social Sentiment Analysis with LLM")
# async def get_analyzed_social_sentiment_stream(
#     request_body: SentimentAnalysisRequest,
#     page: int = Query(1, ge=0, description="Page number to get (starts from 0)"),
#     redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
# ):
#     """
#     Get sentiment data from FMP and stream market impact analysis through LLM
#     """
#     async def format_sse():
#         try:
#             if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
#                 logger.error(f"FMP API key is not configured for sentiment analysis {request_body.symbol}.")
#                 yield f"{json.dumps({'error': 'FMP API key is not configured on the server.'})}\n\n"
#                 yield "[DONE]\n\n"
#                 return
            
#             # # Try cache first for raw sentiment data
#             # cache_key = f"sentiment_analysis_{request_body.symbol.upper()}_page_{page}_{request_body.model_name}"
            
#             # cached_analysis = None
#             # if redis_client:
#             #     try:
#             #         cached_analysis = await redis_client.get(cache_key)
#             #         if cached_analysis:
#             #             logger.info(f"Cache HIT for sentiment analysis ({request_body.symbol}, page {page})")
#             #             # Send cached data
#             #             cached_data = json.loads(cached_analysis)
#             #             yield f"{json.dumps({'cached': True, 'content': cached_data['data']['interpretation']})}\n\n"
#             #             yield "[DONE]\n\n"
#             #             return
#             #     except Exception as e:
#             #         logger.error(f"Redis GET error: {e}")
            
                    
#             # Get sentiment data from FMP
#             sentiment_data_list = await news_service.get_historical_social_sentiment(
#                 symbol=request_body.symbol, page=page
#             )
            
#             if sentiment_data_list is None:
#                 yield f"{json.dumps({'error': f'Unable to get sentiment data for {request_body.symbol}'})}\n\n"
#                 yield "[DONE]\n\n"
#                 return
            
#             # Convert Pydantic models to dicts for processing
#             sentiment_dicts = []
#             for item in sentiment_data_list:
#                 if hasattr(item, 'model_dump'):
#                     sentiment_dicts.append(item.model_dump())
#                 elif hasattr(item, 'dict'):
#                     sentiment_dicts.append(item.dict())
#                 else:
#                     sentiment_dicts.append(dict(item))
            
#             # Get API Key
#             api_key = ModelProviderFactory._get_api_key(request_body.provider_type)
            
#             # Track full response for caching
#             full_response = []
            
#             # Stream LLM analysis
#             async for chunk in sentiment_analysis_handler.stream_sentiment_impact(
#                 symbol=request_body.symbol,
#                 sentiment_data=sentiment_dicts,
#                 model_name=request_body.model_name,
#                 provider_type=request_body.provider_type,
#                 api_key=api_key
#             ):
#                 if chunk:
#                     full_response.append(chunk)
#                     yield f"{json.dumps({'content': chunk})}\n\n"
            
#             # Join full response for caching
#             market_impact_analysis = ''.join(full_response)
            
#             # Prepare cache data
#             response_data = {
#                 "symbol": request_body.symbol,
#                 "interpretation": market_impact_analysis,
#             }
            
#             cache_response = {
#                 "message": "Successful sentiment analysis",
#                 "status": "200",
#                 "data": response_data
#             }
            
#             # # Cache the analysis
#             # if redis_client and len(sentiment_dicts) > 0 and market_impact_analysis:
#             #     try:
#             #         await redis_client.set(
#             #             cache_key,
#             #             json.dumps(cache_response),
#             #             ex=300
#             #         )
#             #         logger.info(f"Cached sentiment analysis for {request_body.symbol}")
#             #     except Exception as e:
#             #         logger.error(f"Redis SET error: {e}")
            
#             yield "[DONE]\n\n"
            
#         except Exception as e:
#             logger.error(f"Error in sentiment analysis streaming: {str(e)}")
#             yield f"{json.dumps({'error': str(e)})}\n\n"
#             yield "[DONE]\n\n"
    
#     return StreamingResponse(
#         format_sse(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#             "X-Accel-Buffering": "no"
#         }
#     )
@router.post("/social-sentiment/analyze/stream", summary="Streaming Social Sentiment Analysis with LLM")
async def get_analyzed_social_sentiment_stream(
    request: Request,  # Add Request parameter
    request_body: SentimentAnalysisRequest,
    page: int = Query(1, ge=0, description="Page number to get (starts from 0)"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Get sentiment data from FMP and stream market impact analysis through LLM with memory integration
    """
    user_id = getattr(request.state, "user_id", None)
    
    # Save user question first
    question_id = None
    if request_body.session_id and user_id:
        try:
            question_content = request_body.question_input or f"Analyze social sentiment for {request_body.symbol}"
            question_id = chat_service.save_user_question(
                session_id=request_body.session_id,
                created_at=datetime.now(),
                created_by=user_id,
                content=question_content
            )
        except Exception as save_error:
            logger.error(f"Error saving user question: {str(save_error)}")
    
    # Get memory context
    # Get chat history
    chat_history = ""
    if request_body.session_id:
        try:
            chat_history = ChatMessageHistory.string_message_chat_history(
                session_id=request_body.session_id
            )
        except Exception as e:
            logger.error(f"Error fetching history: {e}")

    context = ""
    memory_stats = {}
    if request_body.session_id and user_id:
        try:
            query_text = request_body.question_input or f"Analyze social sentiment for {request_body.symbol}"
            context, memory_stats = await memory_manager.get_relevant_context(
                session_id=request_body.session_id,
                user_id=user_id,
                current_query=query_text,
                llm_provider=llm_generator,
                max_short_term=5,
                max_long_term=3
            )
            logger.info(f"Retrieved memory context for sentiment analysis: {memory_stats}")
        except Exception as e:
            logger.error(f"Error getting memory context: {e}")
    
    enhanced_history = ""
    if context:
        enhanced_history = f"{context}\n\n"
    if chat_history:
        enhanced_history += f"[Conversation History]\n{chat_history}"
    
    async def format_sse():
        full_response = []
        sentiment_data = None
        
        try:
            if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
                logger.error(f"FMP API key is not configured for sentiment analysis {request_body.symbol}.")
                yield f"{json.dumps({'error': 'FMP API key'})}\n\n"
                yield "[DONE]\n\n"
                # error_data = {
                #     "session_id": request_body.session_id,
                #     "type": "error",
                #     "data": "FMP API key is not configured on the server."
                # }
                # yield f"{json.dumps(error_data)}\n\n"
                return
            
            # Get sentiment data from FMP
            sentiment_data_list = await news_service.get_historical_social_sentiment(
                symbol=request_body.symbol, 
                page=page
            )
            
            if sentiment_data_list is None:
                yield f"{json.dumps({'error': 'Unable to get sentiment data'})}\n\n"
                yield "[DONE]\n\n"
                # error_data = {
                #     "session_id": request_body.session_id,
                #     "type": "error",
                #     "data": f"Unable to get sentiment data for {request_body.symbol}"
                # }
                # yield f"{json.dumps(error_data)}\n\n"
                return
            
            # Convert Pydantic models to dicts
            sentiment_dicts = []
            for item in sentiment_data_list:
                if hasattr(item, 'model_dump'):
                    sentiment_dicts.append(item.model_dump())
                elif hasattr(item, 'dict'):
                    sentiment_dicts.append(item.dict())
                else:
                    sentiment_dicts.append(dict(item))
            
            sentiment_data = sentiment_dicts
            
            # Get API Key
            api_key = ModelProviderFactory._get_api_key(request_body.provider_type)
            
            # Stream LLM analysis with memory context
            async for chunk in sentiment_analysis_handler.stream_sentiment_impact(
                symbol=request_body.symbol,
                sentiment_data=sentiment_dicts,
                model_name=request_body.model_name,
                provider_type=request_body.provider_type,
                api_key=api_key,
                memory_context=enhanced_history,  # Pass memory context
                user_question=request_body.question_input
            ):
                if chunk:
                    full_response.append(chunk)
                    yield f"{json.dumps({'content': chunk})}\n\n"
                    # event_data = {
                    #     "session_id": request_body.session_id,
                    #     "type": "chunk",
                    #     "data": chunk
                    # }
                    # yield f"{json.dumps(event_data, ensure_ascii=False)}\n\n"
            
            # Join full response
            market_impact_analysis = ''.join(full_response)
            
            # Analyze conversation importance
            importance_score = 0.5
            if request_body.session_id and user_id:
                try:
                    analysis_model = "gpt-4.1-nano" if request_body.provider_type == ProviderType.OPENAI else request_body.model_name
                    
                    importance_score = await analyze_conversation_importance(
                        query=request_body.question_input or f"Analyze social sentiment for {request_body.symbol}",
                        response=market_impact_analysis,
                        llm_provider=llm_generator,
                        model_name=analysis_model,
                        provider_type=request_body.provider_type
                    )
                    
                    # Boost importance for extreme sentiment
                    if sentiment_data and len(sentiment_data) > 0:
                        avg_sentiment = sum(d.get('sentimentScore', 0) for d in sentiment_data) / len(sentiment_data)
                        if abs(avg_sentiment) > 0.7:  # Very positive or negative
                            importance_score = min(1.0, importance_score + 0.15)
                    
                    logger.info(f"Sentiment analysis importance: {importance_score}")
                except Exception as e:
                    logger.error(f"Error analyzing importance: {e}")
            
            # Store in memory system
            if request_body.session_id and user_id and market_impact_analysis:
                try:
                    # Calculate sentiment metrics
                    sentiment_metrics = {}
                    if sentiment_data:
                        sentiment_scores = [d.get('sentimentScore', 0) for d in sentiment_data]
                        sentiment_metrics = {
                            "avg_sentiment": sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
                            "max_sentiment": max(sentiment_scores) if sentiment_scores else 0,
                            "min_sentiment": min(sentiment_scores) if sentiment_scores else 0,
                            "sentiment_count": len(sentiment_scores)
                        }
                    
                    metadata = {
                        "type": "social_sentiment_analysis",
                        "symbol": request_body.symbol,
                        "page": page,
                        "sentiment_metrics": sentiment_metrics,
                        "data_points": len(sentiment_data) if sentiment_data else 0
                    }
                    
                    await memory_manager.store_conversation_turn(
                        session_id=request_body.session_id,
                        user_id=user_id,
                        query=request_body.question_input or f"Analyze social sentiment for {request_body.symbol}",
                        response=market_impact_analysis,
                        metadata=metadata,
                        importance_score=importance_score
                    )
                    
                    # Save to chat history
                    if question_id:
                        chat_service.save_assistant_response(
                            session_id=request_body.session_id,
                            created_at=datetime.now(),
                            question_id=question_id,
                            content=market_impact_analysis,
                            response_time=0.1
                        )

                    trigger_summary_update_nowait(session_id=request_body.session_id, user_id=user_id)

                except Exception as save_error:
                    logger.error(f"Error saving to memory: {str(save_error)}")
            
            # Send completion event
            # completion_data = {
            #     "session_id": request_body.session_id,
            #     "type": "completion",
            #     "data": "[DONE]",
            #     # "memory_stats": memory_stats
            # }
            # yield f"{json.dumps(completion_data)}\n\n"
            yield "[DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis streaming: {str(e)}")
            yield f"{json.dumps({'error': str(e)})}\n\n"
            yield "[DONE]\n\n"
            # error_data = {
            #     "session_id": request_body.session_id,
            #     "type": "error",
            #     "data": str(e)
            # }
            # yield f"{json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        format_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )