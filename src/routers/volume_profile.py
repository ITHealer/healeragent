import json
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, Query, Depends, Request

from src.handlers.volume_profile_handler import VolumeProfileHandler
from src.helpers.stats_analysis_llm_helper import StatsAnalysisLLMHelper
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.utils.logger.custom_logging import LoggerMixin
from src.providers.provider_factory import ModelProviderFactory, ProviderType
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
from src.helpers.llm_helper import LLMGeneratorProvider
from src.agents.memory.memory_manager import MemoryManager
from src.services.background_tasks import trigger_summary_update_nowait

router = APIRouter()

api_key_auth = APIKeyAuth()
logger_mixin = LoggerMixin()
logger = logger_mixin.logger
llm_generator = LLMGeneratorProvider()
volume_handler = VolumeProfileHandler()
chat_service = ChatService()
llm_helper = StatsAnalysisLLMHelper()
memory_manager = MemoryManager()

# Schema inputs and outputs
class StatsAnalysisChatRequest(BaseModel):
    session_id: str = Field(None, description="Chat session ID")
    question_input: Optional[str] = None
    target_language: Optional[str] = None
    symbol: str = Field(..., description="Stock symbol to analyze")
    model_name: str = Field("gpt-4.1-nano-2025-04-14", description="LLM model")
    provider_type: str = Field("openai", description="Provider type")
    # lookback_days: int = Field(60, ge=20, le=252, description="Days to analyze")
    # num_bins: int = Field(10, ge=5, le=50, description="Number of price bins")

class StatsAnalysisChatResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


# Endpoints
@router.post("/volumes/chat", response_model=StatsAnalysisChatResponse)
async def stats_analysis_chat(
    request: Request,
    chat_request: StatsAnalysisChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Volume profile statistics analysis with LLM interpretation"""
    try:
        user_id = getattr(request.state, "user_id", None)
        
        # Config
        lookback_days = 252
        num_bins = 10

        # Get conversation history
        # chat_history = ""
        # if chat_request.session_id:
        #     try:
        #         chat_history = ChatMessageHistory.string_message_chat_history(
        #             session_id=chat_request.session_id
        #         )
        #     except Exception as e:
        #         logger.error(f"Error fetching chat history: {e}")
        
        # 1. Get relevant context from memory
        context = ""
        memory_stats = {}
        document_references = []
        if chat_request.session_id and user_id:
            try:
                context, memory_stats, document_references = await memory_manager.get_relevant_context(
                    session_id=chat_request.session_id,
                    user_id=user_id,
                    current_query=chat_request.question_input or f"Analyze technical indicators for {chat_request.symbol}",
                    llm_provider=llm_generator,
                    max_short_term=5,  # Top 5 relevant recent conversations
                    max_long_term=3    # Top 3 important past conversations
                )
                logger.info(f"Retrieved memory context for technical analysis: {memory_stats}")
            except Exception as e:
                logger.error(f"Error getting memory context: {e}")
        
        # Get volume profile data
        volume_profile = await volume_handler.get_volume_profile(
            symbol=chat_request.symbol,
            lookback_days=lookback_days,
            num_bins=num_bins
        )
        
        # Get API Key
        api_key = ModelProviderFactory._get_api_key(chat_request.provider_type)
        
        llm_interpretation = await llm_helper.generate_stats_analysis_with_llm(
            symbol=chat_request.symbol,
            volume_data=volume_profile,
            user_question=chat_request.question_input,
            target_language=chat_request.target_language,
            model_name=chat_request.model_name,
            provider_type=chat_request.provider_type,
            api_key=api_key,
            chat_history=context
        )
        
        # 3. Analyze conversation importance
        importance_score = 0.5  # Default score
        
        if chat_request.session_id and user_id:
            try:
                analysis_model = "gpt-4.1-nano" if chat_request.provider_type == ProviderType.OPENAI else chat_request.model_name
                
                importance_score = await analyze_conversation_importance(
                    query=chat_request.question_input or f"Analyze volumee for {chat_request.symbol}",
                    response=llm_interpretation["llm_interpretation"],
                    llm_provider=llm_generator,
                    model_name=analysis_model,
                    provider_type=chat_request.provider_type
                )
                
                logger.info(f"Volume analysis importance score: {importance_score}")
                
            except Exception as e:
                logger.error(f"Error analyzing conversation importance: {e}")
        
        # 4. Store conversation in memory system
        if chat_request.session_id and user_id:
            try:
                # Store in memory system
                await memory_manager.store_conversation_turn(
                    session_id=chat_request.session_id,
                    user_id=user_id,
                    query=chat_request.question_input or f"Analyze technical indicators for {chat_request.symbol}",
                    response=llm_interpretation["llm_interpretation"],
                    metadata={
                        "type": "volume_analysis",
                        "symbol": chat_request.symbol,
                        # "indicators": llm_interpretation.get("technical_data", {})
                    },
                    importance_score=importance_score
                )

                question_content = chat_request.question_input or f"Analyze volume profile for {chat_request.symbol}"
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
                    content=llm_interpretation,
                    response_time=0.1
                )
            except Exception as e:
                logger.error(f"Error saving to chat history: {str(e)}")
        
        return StatsAnalysisChatResponse(
            status="success",
            message="Volume profile analysis completed successfully",
            data={
                "symbol": chat_request.symbol,
                "interpretation": llm_interpretation,
                # "volume_profile_data": volume_profile,
                # "lookback_days": chat_request.lookback_days,
                # "num_bins": chat_request.num_bins
            }
        )
        
    except Exception as e:
        logger.error(f"Error in stats analysis chat: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error performing volume profile analysis: {str(e)}"
        )


@router.post("/volumes/chat/stream")
async def stats_analysis_chat_stream(
    request: Request,
    chat_request: StatsAnalysisChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Volume profile statistics analysis with LLM interpretation"""
    async def generate_stream():
        try:
            user_id = getattr(request.state, "user_id", None)
            
            # Config
            lookback_days = 252
            num_bins = 10

            # Save question first
            question_id = None
            if chat_request.session_id and user_id:
                try:
                    question_content = chat_request.question_input or f"Volume profile for {chat_request.symbol}"
                    question_id = chat_service.save_user_question(
                        session_id=chat_request.session_id,
                        created_at=datetime.now(),
                        created_by=user_id,
                        content=question_content
                    )
                except Exception as e:
                    logger.error(f"Error saving question: {e}")
            
            # # Get contexts
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
            document_references = []
            if chat_request.session_id and user_id:
                try:
                    context, memory_stats, document_references = await memory_manager.get_relevant_context(
                        session_id=chat_request.session_id,
                        user_id=user_id,
                        current_query=chat_request.question_input or f"Analyze volume for {chat_request.symbol}",
                        llm_provider=llm_generator,
                        max_short_term=5,
                        max_long_term=3
                    )
                    logger.info(f"Retrieved memory context for streaming: {memory_stats}")
                except Exception as e:
                    logger.error(f"Error getting memory: {e}")
            
            enhanced_history = ""
            if context:
                enhanced_history = f"{context}\n\n"
            if chat_history:
                enhanced_history += f"[Conversation History]\n{chat_history}"
            
            # Get volume profile data
            volume_profile = await volume_handler.get_volume_profile(
                symbol=chat_request.symbol,
                lookback_days=lookback_days,
                num_bins=num_bins
            )
            
            # Get API Key
            api_key = ModelProviderFactory._get_api_key(chat_request.provider_type)
            
            # Track full response for saving to history
            full_response = []
            
            # Stream LLM interpretation
            llm_gen = llm_helper.stream_generate_stats_analysis_with_llm(
                symbol=chat_request.symbol,
                volume_data=volume_profile,
                user_question=chat_request.question_input,
                target_language=chat_request.target_language,
                model_name=chat_request.model_name,
                provider_type=chat_request.provider_type,
                api_key=api_key,
                chat_history=enhanced_history
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

            # Save complete response
            complete_response = ''.join(full_response)
            
            # Analyze conversation importance
            importance_score = 0.5
            
            if chat_request.session_id and user_id:
                try:
                    analysis_model = "gpt-4.1-nano" if chat_request.provider_type == ProviderType.OPENAI else chat_request.model_name
                    
                    importance_score = await analyze_conversation_importance(
                        query=chat_request.question_input or f"Analyze volume for {chat_request.symbol}",
                        response=complete_response,
                        llm_provider=llm_generator,
                        model_name=analysis_model,
                        provider_type=chat_request.provider_type
                    )
                    
                    logger.info(f"Volume analysis stream importance: {importance_score}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing conversation: {e}")
            
            # Store in memory system
            if chat_request.session_id and user_id and complete_response:
                try:
                    await memory_manager.store_conversation_turn(
                        session_id=chat_request.session_id,
                        user_id=user_id,
                        query=chat_request.question_input or f"Analyze volume for {chat_request.symbol}",
                        response=complete_response,
                        metadata={
                            "type": "volume_analysis",
                            "symbol": chat_request.symbol
                        },
                        importance_score=importance_score
                    )
                    
                    chat_service.save_assistant_response(
                        session_id=chat_request.session_id,
                        created_at=datetime.now(),
                        question_id=question_id,
                        content=complete_response,
                        response_time=0.1
                    )

                    trigger_summary_update_nowait(session_id=chat_request.session_id, user_id=user_id)

                except Exception as e:
                    logger.error(f"Error saving to chat history: {str(e)}")
            
            # Completion event
            yield sse_done()

        except Exception as e:
            yield sse_error(str(e))
            yield sse_done()
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )