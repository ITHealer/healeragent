import json
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, Request, Depends

from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.technical_analysis_llm_helper import TechnicalAnalysisLLMHelper
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
        if chat_request.session_id and user_id:
            try:
                context, memory_stats = await memory_manager.get_relevant_context(
                    session_id=chat_request.session_id,
                    user_id=user_id,
                    current_query=chat_request.question_input or f"Analyze technical indicators for {chat_request.symbol}",
                    llm_provider=llm_provider,
                    max_short_term=5,  # Top 5 relevant recent conversations
                    max_long_term=3    # Top 3 important past conversations
                )
                logger.info(f"Retrieved memory context for technical analysis: {memory_stats}")
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


# Endpoint stream
@router.post("/technical/chat/stream")
async def technical_analysis_chat_stream(
    request: Request,
    chat_request: TechnicalAnalysisChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Streaming version of technical analysis chat with memory integration
    """
    user_id = getattr(request.state, "user_id", None)
    
    if not chat_request.symbol:
        raise HTTPException(
            status_code=400,
            detail="Symbol is required for technical analysis"
        )
    
    # Save user question first
    question_id = None
    if chat_request.session_id and user_id:
        try:
            question_content = chat_request.question_input or f"Analyze chart patterns for {chat_request.symbol}"
            question_id = chat_service.save_user_question(
                session_id=chat_request.session_id,
                created_at=datetime.now(),
                created_by=user_id,
                content=question_content
            )
        except Exception as e:
            logger.error(f"Error saving question: {e}")
    
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
            context, memory_stats = await memory_manager.get_relevant_context(
                session_id=chat_request.session_id,
                user_id=user_id,
                current_query=chat_request.question_input or f"Analyze technical indicators for {chat_request.symbol}",
                llm_provider=llm_provider,
                max_short_term=5,
                max_long_term=3
            )
            logger.info(f"Retrieved memory context for streaming: {memory_stats}")
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
    
    # Combine contexts
    enhanced_history = ""
    if context:
        enhanced_history = f"{context}\n\n"
    if chat_history:
        enhanced_history += f"[Conversation History]\n{chat_history}"
    
    # Get API Key
    api_key = ModelProviderFactory._get_api_key(chat_request.provider_type)
            
    async def generate():
        full_response = []
        try:
            
            # Get technical analysis data
            df = await market_data.get_historical_data_lookback_ver2(
                ticker=chat_request.symbol,
                lookback_days=252
            )
            
            data_dict = df.reset_index().to_dict(orient="records")
            analysis_data = get_technical_analysis(chat_request.symbol, data_dict)
            
            # Create prompt with memory context
            detection_method = ""
            if len(chat_request.question_input.split()) < 2:
                detection_method = DetectionMethod.LLM
            else:
                detection_method = DetectionMethod.LIBRARY

            # Language detection
            language_info = await language_detector.detect(
                text=chat_request.question_input,
                method=detection_method,
                system_language=chat_request.target_language,
                model_name=chat_request.model_name,
                provider_type=chat_request.provider_type,
                api_key=api_key
            )

            detected_language = language_info["detected_language"]

            if detected_language:
                lang_name = {
                    "en": "English",
                    "vi": "Vietnamese", 
                    "zh": "Chinese",
                    "zh-cn": "CHINESE (SIMPLIFIED, CHINA)",
                    "zh-tw": "CHINESE (TRADITIONAL, TAIWAN)",
                }.get(detected_language, "the detected language")
                
            language_instruction = f"""
            CRITICAL LANGUAGE REQUIREMENT:
            You MUST respond ENTIRELY in {lang_name} language.
            - ALL text, explanations, and analysis must be in {lang_name}
            - Use appropriate financial terminology for {lang_name}
            - Format numbers and dates according to {lang_name} conventions
            """

            base_prompt = llm_helper.create_technical_analysis_prompt(
                chat_request.symbol,
                analysis_data,
                chat_request.question_input, 
                language_instruction
            )
            
            # Add memory context to prompt if available
            if context:
                prompt = f"Previous conversations context:\n{enhanced_history}\n\nCurrent analysis request:\n{base_prompt}"
            else:
                prompt = base_prompt
            
            messages = [
                {"role": "system", "content": "You are a professional financial analyst with memory of previous conversations."},
                {"role": "user", "content": prompt}
            ]
            
            # Stream response
            async for chunk in llm_provider.stream_response(
                model_name=chat_request.model_name,
                messages=messages,
                provider_type=chat_request.provider_type,
                api_key=api_key
            ):
                full_response.append(chunk)
                yield f"{json.dumps({'content': chunk})}\n\n"
                # event_data = {
                #     "session_id": chat_request.session_id,
                #     "type": "chunk",
                #     "data": chunk
                # }
                # yield f"{json.dumps(event_data, ensure_ascii=False)}\n\n"
            
            # Join the full response
            complete_response = ''.join(full_response)
            
            # Analyze conversation importance
            importance_score = 0.5
            
            if chat_request.session_id and user_id:
                try:
                    analysis_model = "gpt-4.1-nano" if chat_request.provider_type == ProviderType.OPENAI else chat_request.model_name
                    
                    importance_score = await analyze_conversation_importance(
                        query=chat_request.question_input or f"Analyze technical indicators for {chat_request.symbol}",
                        response=complete_response,
                        llm_provider=llm_provider,
                        model_name=analysis_model,
                        provider_type=chat_request.provider_type
                    )
                    
                    logger.info(f"Technical analysis stream importance: {importance_score}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing conversation: {e}")
            
            # Store in memory system
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
                    
                    chat_service.save_assistant_response(
                        session_id=chat_request.session_id,
                        created_at=datetime.now(),
                        question_id=question_id,
                        content=complete_response,
                        response_time=0.1
                    )

                    trigger_summary_update_nowait(session_id=chat_request.session_id, user_id=user_id)

                except Exception as save_error:
                    logger.error(f"Error saving assistant response: {str(save_error)}")
            
            # Send completion event with memory stats
            yield "[DONE]\n\n"
            # completion_data = {
            #     "session_id": chat_request.session_id,
            #     "type": "completion",
            #     "data": "[DONE]",
            #     # "memory_stats": memory_stats
            # }
            # yield f"{json.dumps(completion_data)}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            yield f"{json.dumps({'error': str(e)})}\n\n"
            yield "[DONE]\n\n"
            # error_data = {
            #     "session_id": chat_request.session_id,
            #     "type": "error",
            #     "data": str(e)
            # }
            # yield f"{json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
            "Connection": "keep-alive"
        }
    )