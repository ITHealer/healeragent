import json
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, Request, Depends

from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.technical_analysis_llm_helper import TechnicalAnalysisLLMHelper
from src.helpers.llm_chat_helper import (
    stream_with_heartbeat,
    analyze_conversation_importance,
    SSE_HEADERS,
    DEFAULT_HEARTBEAT_SEC,
    sse_error,
    sse_done
)
from src.handlers.technical_analysis_handler import get_technical_analysis, get_technical_analysis_with_llm
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.stock.crawlers.market_data_provider import MarketData
from src.providers.provider_factory import ModelProviderFactory
from src.helpers.chat_management_helper import ChatService
from src.handlers.llm_chat_handler import ChatMessageHistory
from src.routers.llm_chat import analyze_conversation_importance
from src.agents.memory.memory_manager import MemoryManager
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ProviderType
from src.helpers.language_detector import language_detector, DetectionMethod
from src.services.background_tasks import trigger_summary_update_nowait                

router = APIRouter()

api_key_auth = APIKeyAuth()
logger_mixin = LoggerMixin()
logger = logger_mixin.logger
market_data = MarketData()
chat_service = ChatService()
memory_manager = MemoryManager()
llm_provider = LLMGeneratorProvider()
llm_helper = TechnicalAnalysisLLMHelper()


# Schema inputs and outputs
class TechnicalAnalysisChatRequest(BaseModel):
    session_id: str = Field(None, description="Chat session ID for context")
    question_input: Optional[str] = None
    target_language: Optional[str] = None
    symbol: str = Field(None, description="Stock symbol to analyze")
    model_name: str = Field("gpt-4.1-nano-2025-04-14", description="LLM model to use")
    provider_type: str = Field("openai", description="Provider type for the LLM")

class TechnicalAnalysisChatResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


# =============================================================================
# Helper Functions
# =============================================================================
def save_user_question(session_id: str, user_id: int, content: str) -> Optional[int]:
    """Save user question to chat history."""
    try:
        return chat_service.save_user_question(
            session_id=session_id,
            created_at=datetime.now(),
            created_by=user_id,
            content=content
        )
    except Exception as e:
        logger.error(f"Error saving question: {e}")
        return None


async def get_enhanced_context(session_id: str, user_id: int, question_input: str) -> str:
    """Get chat history + memory context."""
    enhanced_history = ""
    
    # Get chat history
    chat_history = ""
    if session_id:
        try:
            chat_history = ChatMessageHistory.string_message_chat_history(
                session_id=session_id
            )
        except Exception as e:
            logger.error(f"Error fetching chat history: {e}")
    
    # Get memory context
    memory_context = ""
    if session_id and user_id:
        try:
            memory_context, _, _ = await memory_manager.get_relevant_context(
                session_id=session_id,
                user_id=user_id,
                current_query=question_input,
                llm_provider=llm_provider,
                max_short_term=5,
                max_long_term=3
            )
        except Exception as e:
            logger.error(f"Error getting memory context: {e}")
    
    if memory_context:
        enhanced_history = f"[Memory Context]\n{memory_context}\n\n"
    if chat_history:
        enhanced_history += f"[Conversation History]\n{chat_history}"
    
    return enhanced_history


def get_language_instruction(detected_language: str) -> str:
    """Get language instruction for system prompt."""
    lang_map = {
        "en": "English",
        "vi": "Vietnamese",
        "zh": "Chinese",
        "zh-cn": "CHINESE (SIMPLIFIED, CHINA)",
        "zh-tw": "CHINESE (TRADITIONAL, TAIWAN)",
    }
    lang_name = lang_map.get(detected_language, "the detected language")
    
    return f"""CRITICAL LANGUAGE REQUIREMENT:
You MUST respond ENTIRELY in {lang_name} language.
- ALL text, explanations, and analysis must be in {lang_name}
- Use appropriate financial terminology for {lang_name}
- Format numbers and dates according to {lang_name} conventions"""

# # Endpoints
@router.post("/technical/chat", response_model=TechnicalAnalysisChatResponse)
async def technical_analysis_chat(
    request: Request,
    chat_request: TechnicalAnalysisChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Endpoint for technical analysis with LLM and memory integration
    
    Flow:
        1. Get relevant context from memory
        2. Get historical data from market data provider
        3. Calculate technical analysis indicators
        4. Call LLM with memory context to generate detailed analysis
        5. Analyze conversation importance
        6. Store in memory system and chat history
    """
    try:
        user_id = getattr(request.state, "user_id", None)
        
        if not chat_request.symbol:
            raise HTTPException(
                status_code=400,
                detail="Symbol is required for technical analysis"
            )
        
        # 1. Get relevant context from memory
        # Get chat history
        chat_history = ""
        if chat_request.session_id:
            try:
                chat_history = ChatMessageHistory.string_message_chat_history(
                    session_id=chat_request.session_id
                )
            except Exception as e:
                logger.error(f"Error fetching history: {e}")

        context = ""
        memory_stats = {}
        document_references = [] 
        if chat_request.session_id and user_id:
            try:
                context, memory_stats, document_references = await memory_manager.get_relevant_context( 
                    session_id=chat_request.session_id,
                    user_id=user_id,
                    current_query=chat_request.question_input or f"Analyze technical indicators for {chat_request.symbol}",
                    llm_provider=llm_provider,
                    max_short_term=5,  # Top 5 relevant recent conversations
                    max_long_term=3    # Top 3 important past conversations
                )
                logger.info(f"Retrieved memory context for technical analysis: {memory_stats}")
                if document_references:
                    logger.info(f"Found {len(document_references)} document references")
            except Exception as e:
                logger.error(f"Error getting memory context: {e}")
        
        enhanced_history = ""
        if context:
            enhanced_history = f"{context}\n\n"
        if chat_history:
            enhanced_history += f"[Conversation History]\n{chat_history}"
        
        # 2. Get technical analysis with memory context
        result = await get_technical_analysis_with_llm(
            symbol=chat_request.symbol,
            model_name=chat_request.model_name,
            provider_type=chat_request.provider_type,
            user_question=chat_request.question_input,
            system_language=chat_request.target_language,
            lookback_days=252,
            chat_history=enhanced_history 
        )
        
        # 3. Analyze conversation importance
        importance_score = 0.5  # Default score
        
        if chat_request.session_id and user_id:
            try:
                analysis_model = "gpt-4.1-nano" if chat_request.provider_type == ProviderType.OPENAI else chat_request.model_name
                
                importance_score = await analyze_conversation_importance(
                    query=chat_request.question_input or f"Analyze technical indicators for {chat_request.symbol}",
                    response=result["llm_interpretation"],
                    llm_provider=llm_provider,
                    model_name=analysis_model,
                    provider_type=chat_request.provider_type
                )
                
                logger.info(f"Technical analysis importance score: {importance_score}")
                
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
                    response=result["llm_interpretation"],
                    metadata={
                        "type": "technical_analysis",
                        "symbol": chat_request.symbol,
                        "indicators": result.get("technical_data", {})
                    },
                    importance_score=importance_score
                )

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
                    content=result["llm_interpretation"],
                    response_time=0.1
                )
            except Exception as e:
                logger.error(f"Error saving to memory/chat history: {str(e)}")
        
        return TechnicalAnalysisChatResponse(
            status="success",
            message="Technical analysis completed successfully",
            data={
                "symbol": result["symbol"],
                "interpretation": result["llm_interpretation"],
                "memory_stats": memory_stats
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in technical analysis chat: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error performing technical analysis: {str(e)}"
        )



@router.post("/technical/chat/stream")
async def technical_analysis_chat_stream(
    request: Request,
    chat_request: TechnicalAnalysisChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Stream technical analysis with heartbeat.
    """
    user_id = getattr(request.state, "user_id", None)
    
    if not chat_request.symbol:
        raise HTTPException(
            status_code=400,
            detail="Symbol is required for technical analysis"
        )
    
    # Save user question
    question_content = chat_request.question_input or f"Analyze chart patterns for {chat_request.symbol}"
    question_id = save_user_question(chat_request.session_id, user_id, question_content)
    
    # Get API Key
    api_key = ModelProviderFactory._get_api_key(chat_request.provider_type)
    
    async def generate_stream():
        full_response = []
        start_time = datetime.now()
        
        try:
            # 1. Get enhanced context (chat history + memory)
            enhanced_history = await get_enhanced_context(
                session_id=chat_request.session_id,
                user_id=user_id,
                question_input=chat_request.question_input or f"Analyze technical indicators for {chat_request.symbol}"
            )
            
            # 2. Get technical analysis data
            df = await market_data.get_historical_data_lookback_ver2(
                ticker=chat_request.symbol,
                lookback_days=252
            )
            data_dict = df.reset_index().to_dict(orient="records")
            analysis_data = get_technical_analysis(chat_request.symbol, data_dict)
            
            # 3. Language detection
            detection_method = (
                DetectionMethod.LLM if len((chat_request.question_input or "").split()) < 2 
                else DetectionMethod.LIBRARY
            )
            language_info = await language_detector.detect(
                text=chat_request.question_input or f"Analyze {chat_request.symbol}",
                method=detection_method,
                system_language=chat_request.target_language,
                model_name=chat_request.model_name,
                provider_type=chat_request.provider_type,
                api_key=api_key
            )
            detected_language = language_info["detected_language"]
            language_instruction = get_language_instruction(detected_language)
            
            # 4. Create prompt
            base_prompt = llm_helper.create_technical_analysis_prompt(
                chat_request.symbol,
                analysis_data,
                chat_request.question_input,
                language_instruction
            )
            
            if enhanced_history:
                prompt = f"Previous conversations context:\n{enhanced_history}\n\nCurrent analysis request:\n{base_prompt}"
            else:
                prompt = base_prompt
            
            messages = [
                {"role": "system", "content": "You are a professional financial analyst with memory of previous conversations."},
                {"role": "user", "content": prompt}
            ]
            
            # 5. Get LLM generator
            llm_generator = llm_provider.stream_response(
                model_name=chat_request.model_name,
                messages=messages,
                provider_type=chat_request.provider_type,
                api_key=api_key
            )
            
            # 6. Stream with heartbeat using imported function
            async for event in stream_with_heartbeat(llm_generator, DEFAULT_HEARTBEAT_SEC):
                if event["type"] == "content":
                    full_response.append(event["chunk"])
                    yield f"{json.dumps({'content': event['chunk']}, ensure_ascii=False)}\n\n"
                elif event["type"] == "heartbeat":
                    yield ": heartbeat\n\n"
                elif event["type"] == "error":
                    yield f"{json.dumps({'error': event['error']})}\n\n"
                    break
                elif event["type"] == "done":
                    break
            
            # 7. Post-processing
            complete_response = ''.join(full_response)
            response_time = (datetime.now() - start_time).total_seconds()
            
            # 8. Analyze importance
            importance_score = 0.5
            if chat_request.session_id and user_id and complete_response:
                try:
                    analysis_model = "gpt-4.1-nano" if chat_request.provider_type == ProviderType.OPENAI else chat_request.model_name
                    importance_score = await analyze_conversation_importance(
                        query=chat_request.question_input or f"Analyze technical indicators for {chat_request.symbol}",
                        response=complete_response,
                        llm_provider=llm_provider,
                        model_name=analysis_model,
                        provider_type=chat_request.provider_type
                    )
                except Exception as e:
                    logger.error(f"Error analyzing importance: {e}")
            
            # 9. Store in memory
            if chat_request.session_id and user_id and complete_response:
                try:
                    await memory_manager.store_conversation_turn(
                        session_id=chat_request.session_id,
                        user_id=user_id,
                        query=chat_request.question_input or f"Analyze technical indicators for {chat_request.symbol}",
                        response=complete_response,
                        metadata={
                            "type": "technical_analysis",
                            "symbol": chat_request.symbol
                        },
                        importance_score=importance_score
                    )
                    
                    if question_id:
                        chat_service.save_assistant_response(
                            session_id=chat_request.session_id,
                            created_at=datetime.now(),
                            question_id=question_id,
                            content=complete_response,
                            response_time=response_time
                        )
                    
                    trigger_summary_update_nowait(
                        session_id=chat_request.session_id,
                        user_id=user_id
                    )
                except Exception as e:
                    logger.error(f"Error saving to memory: {e}")
            
            yield sse_done()
            
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield sse_error(str(e))
            yield sse_done()
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )