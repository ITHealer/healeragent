from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request, Depends

from src.handlers.relative_strength_handler import RelativeStrengthHandler
from src.helpers.relative_strength_llm_helper import RelativeStrengthLLMHelper
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.utils.logger.custom_logging import LoggerMixin
from fastapi.responses import StreamingResponse
import json
from src.handlers.llm_chat_handler import ChatMessageHistory
from src.helpers.chat_management_helper import ChatService
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.routers.llm_chat import analyze_conversation_importance
from src.agents.memory.memory_manager import get_memory_manager
from src.helpers.llm_helper import LLMGeneratorProvider
from src.helpers.language_detector import language_detector, DetectionMethod
from src.services.background_tasks import trigger_summary_update_nowait            
from src.helpers.llm_chat_helper import (
    stream_with_heartbeat,
    analyze_conversation_importance,
    SSE_HEADERS,
    DEFAULT_HEARTBEAT_SEC,
    sse_error,
    sse_done
)

router = APIRouter()

api_key_auth = APIKeyAuth()
logger_mixin = LoggerMixin()
logger = logger_mixin.logger
memory_manager = get_memory_manager()
llm_provider = LLMGeneratorProvider()
rs_handler = RelativeStrengthHandler()
llm_helper = RelativeStrengthLLMHelper()
chat_service = ChatService()

# Schema inputs and outputs
class RelativeStrengthChatRequest(BaseModel):
    session_id: str = Field(None, description="Chat session ID for context")
    question_input: Optional[str] = None
    target_language: Optional[str] = None
    symbol: str = Field(..., description="Stock symbol to analyze")
    # benchmark: str = Field("SPY", description="Benchmark symbol for comparison")
    model_name: str = Field("gpt-4.1-nano-2025-04-14", description="LLM model to use")
    provider_type: str = Field("openai", description="Provider type for the LLM")
    # lookback_periods: Optional[List[int]] = Field([21, 63, 126, 252], description="Periods to analyze")

class RelativeStrengthChatResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


# Endpoints
@router.post("/relative-strength/chat", response_model=RelativeStrengthChatResponse)
async def relative_strength_chat(
    request: Request,
    chat_request: RelativeStrengthChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Enhanced Relative Strength analysis with memory integration"""
    try:
        user_id = getattr(request.state, "user_id", None)
        
        # Config default
        benchmark = "SPY"
        lookback_periods = [21, 63, 126, 252]
        
        # # Get conversation history
        # chat_history = ""
        # if chat_request.session_id:
        #     try:
        #         chat_history = ChatMessageHistory.string_message_chat_history(
        #             session_id=chat_request.session_id
        #         )
        #     except Exception as e:
        #         logger.error(f"Error fetching chat history: {e}")
        
        # 1. Get memory context
        context = ""
        memory_stats = {}
        document_references = []
        if chat_request.session_id and user_id:
            try:
                context, memory_stats, document_references = await memory_manager.get_relevant_context(
                    session_id=chat_request.session_id,
                    user_id=user_id,
                    current_query=chat_request.question_input,
                    llm_provider=llm_provider,
                    max_short_term=5,
                    max_long_term=3
                )
            except Exception as e:
                logger.error(f"Error getting memory context: {e}")

        # Get RS data
        rs_handler = RelativeStrengthHandler()
        rs_result = await rs_handler.get_relative_strength(
            symbol=chat_request.symbol,
            benchmark=benchmark,
            lookback_periods=lookback_periods
        )
        
        # Call LLM with enhanced context
        api_key = None
        if chat_request.provider_type == "openai":
            from src.utils.config import settings
            api_key = settings.OPENAI_API_KEY

        llm_interpretation = await llm_helper.generate_rs_analysis_with_llm(
            symbol=chat_request.symbol,
            benchmark=benchmark,
            rs_data=rs_result,
            user_question=chat_request.question_input,
            target_language=chat_request.target_language,
            model_name=chat_request.model_name,
            provider_type=chat_request.provider_type,
            api_key=api_key,
            chat_history=context 
        )
        
        # 3. Analyze importance
        importance_score = 0.5
        if chat_request.session_id and user_id:
            try:
                analysis_model = "gpt-4.1-nano" if chat_request.provider_type == ProviderType.OPENAI else chat_request.model_name
                importance_score = await analyze_conversation_importance(
                    query=chat_request.question_input,
                    response=llm_interpretation, 
                    llm_provider=llm_provider,
                    model_name=analysis_model,
                    provider_type=chat_request.provider_type
                )
            except Exception as e:
                logger.error(f"Error analyzing importance: {e}")
        
        # 4. Store in memory
        if chat_request.session_id and user_id:
            try:
                await memory_manager.store_conversation_turn(
                    session_id=chat_request.session_id,
                    user_id=user_id,
                    query=chat_request.question_input,
                    response=llm_interpretation,
                    metadata={"type": "relative_strength"},
                    importance_score=importance_score
                )
                
                question_content = chat_request.question_input or f"Analyze relative strength of {chat_request.symbol} vs {benchmark}"
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
                logger.error(f"Error saving to memory/history: {str(e)}")
        
        return RelativeStrengthChatResponse(
            status="success",
            message="Relative strength analysis completed",
            data={
                "symbol": chat_request.symbol,
                "interpretation": llm_interpretation,
                "memory_stats": memory_stats 
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/relative-strength/chat/stream")
async def relative_strength_chat_stream(
    request: Request,
    chat_request: RelativeStrengthChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Relative Strength analysis with LLM"""
    user_id = getattr(request.state, "user_id", None)
    
    question_id = None
    if chat_request.session_id and user_id:
        try:
            
            question_content = chat_request.question_input or f"RS analysis for {chat_request.symbol}"
            question_id = chat_service.save_user_question(
                session_id=chat_request.session_id,
                created_at=datetime.now(),
                created_by=user_id,
                content=question_content
            )
        except Exception as e:
            logger.error(f"Error saving question: {e}")
    
    async def generate_stream():
        full_response = []
        benchmark = "SPY"
        lookback_periods = [21, 63, 126, 252]
        
        try:
            # Get contexts
            chat_history = ""
            if chat_request.session_id:
                try:
                    chat_history = ChatMessageHistory.string_message_chat_history(
                        session_id=chat_request.session_id
                    )
                except Exception as e:
                    logger.error(f"Error fetching history: {e}")
            
            # 1. Get memory context
            context = ""
            memory_stats = {}
            document_references = []
            if chat_request.session_id and user_id:
                try:
                    context, memory_stats, document_references = await memory_manager.get_relevant_context(
                        session_id=chat_request.session_id,
                        user_id=user_id,
                        current_query=chat_request.question_input,
                        llm_provider=llm_provider,
                        max_short_term=5,
                        max_long_term=3
                    )
                except Exception as e:
                    logger.error(f"Error getting memory context: {e}")
            
            enhanced_history = ""
            if context:
                enhanced_history = f"{context}\n\n"
            if chat_history:
                enhanced_history += f"[Conversation History]\n{chat_history}"

            # Get RS data
            rs_handler = RelativeStrengthHandler()
            rs_result = await rs_handler.get_relative_strength(
                symbol=chat_request.symbol,
                benchmark=benchmark,
                lookback_periods=lookback_periods
            )
            
            # Get API key
            api_key = ModelProviderFactory._get_api_key(chat_request.provider_type)
            
            # Get LLM generator
            llm_generator = llm_helper.stream_rs_analysis_with_llm(
                symbol=chat_request.symbol,
                benchmark=benchmark,
                rs_data=rs_result,
                user_question=chat_request.question_input,
                model_name=chat_request.model_name,
                provider_type=chat_request.provider_type,
                api_key=api_key,
                chat_history=enhanced_history
            )
            
            # Stream with heartbeat
            async for event in stream_with_heartbeat(llm_generator, DEFAULT_HEARTBEAT_SEC):
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
            
            # 3. Analyze importance
            importance_score = 0.5
            if chat_request.session_id and user_id:
                try:
                    analysis_model = "gpt-4.1-nano" if chat_request.provider_type == ProviderType.OPENAI else chat_request.model_name
                    importance_score = await analyze_conversation_importance(
                        query=chat_request.question_input,
                        response=complete_response,  
                        llm_provider=llm_provider,
                        model_name=analysis_model,
                        provider_type=chat_request.provider_type
                    )
                except Exception as e:
                    logger.error(f"Error analyzing importance: {e}")
            
            # 4. Store in memory
            if chat_request.session_id and user_id and complete_response:
                try:
                    await memory_manager.store_conversation_turn(
                        session_id=chat_request.session_id,
                        user_id=user_id,
                        query=chat_request.question_input,
                        response=complete_response,
                        metadata={"type": "relative_strength"},
                        importance_score=importance_score
                    )
                    
                    if question_id:
                        chat_service.save_assistant_response(
                            session_id=chat_request.session_id,
                            created_at=datetime.now(),
                            question_id=question_id,
                            content=complete_response,
                            response_time=0.1
                        )
                        
                    trigger_summary_update_nowait(session_id=chat_request.session_id, user_id=user_id)
    
                except Exception as e:
                    logger.error(f"Error saving: {e}")
            
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