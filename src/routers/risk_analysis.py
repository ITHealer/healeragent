import json
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, Query, Depends, Body, Request

from src.handlers.risk_analysis_handler import RiskAnalysisHandler
from src.stock.crawlers.market_data_provider import MarketData
from src.helpers.risk_analysis_llm_helper import RiskAnalysisLLMHelper
from src.utils.logger.custom_logging import LoggerMixin
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.agents.memory.memory_manager import MemoryManager
from src.routers.llm_chat import analyze_conversation_importance
from src.helpers.llm_helper import LLMGeneratorProvider
from src.helpers.chat_management_helper import ChatService
from src.handlers.llm_chat_handler import ChatMessageHistory


router = APIRouter()

api_key_auth = APIKeyAuth()
logger_mixin = LoggerMixin()
logger = logger_mixin.logger
market_data = MarketData()
risk_handler = RiskAnalysisHandler(market_data)
memory_manager = MemoryManager()
llm_provider = LLMGeneratorProvider()
chat_service = ChatService()

# Schema inputs and outputs
class RiskAnalysisChatRequest(BaseModel):
    session_id: str = Field(None, description="Chat session ID for context")
    question_input: Optional[str] = None
    target_language: Optional[str] = None
    symbol: str = Field(..., description="Stock symbol to analyze")
    model_name: str = Field("gpt-4.1-nano-2025-04-14", description="LLM model to use")
    provider_type: str = Field("openai", description="Provider type for the LLM")
    # Optional parameters cho position sizing
    # entry_price: Optional[float] = Field(None, description="Entry price for position sizing")
    # stop_price: Optional[float] = Field(None, description="Stop loss price")
    # risk_amount: Optional[float] = Field(None, description="Amount willing to risk")
    # account_size: Optional[float] = Field(None, description="Total account size")
    # lookback_days: int = Field(60, ge=20, le=365, description="Days for historical analysis")


class RiskAnalysisChatResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


# Endpoints
@router.post("/risk/chat", response_model=RiskAnalysisChatResponse)
async def risk_analysis_chat(
    request: Request,
    chat_request: RiskAnalysisChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Endpoint for risk analysis with LLM and memory integration
    
    Flow:
        1. Get relevant context from memory
        2. Get historical data
        3. Calculate stop levels
        4. Calculate position sizing if enough information is available
        5. Call LLM for aggregate analysis with memory context
        6. Analyze conversation importance
        7. Store in memory system and chat history
    """
    try:
        user_id = getattr(request.state, "user_id", None)
        
        # 1. Get relevant context from memory
        context = ""
        memory_stats = {}
        
        if chat_request.session_id and user_id:
            try:
                query_text = chat_request.question_input or f"Analyze risk management for {chat_request.symbol}"
                
                context, memory_stats = await memory_manager.get_relevant_context(
                    session_id=chat_request.session_id,
                    user_id=user_id,
                    current_query=query_text,
                    llm_provider=llm_provider,
                    max_short_term=5,  # Recent risk analyses
                    max_long_term=3    # Important historical risk decisions
                )
                
                logger.info(f"Retrieved memory context for risk analysis: {memory_stats}")
            except Exception as e:
                logger.error(f"Error getting memory context: {e}")
        
        # Config
        entry_price = 0
        stop_price = 0
        risk_amount = 0
        account_size = 0
        lookback_days = 252

        # Initialize handlers
        risk_handler = RiskAnalysisHandler(market_data)
        llm_helper = RiskAnalysisLLMHelper()
        
        # 2. Calculate stop loss levels
        stop_levels_result = await risk_handler.suggest_stop_loss_levels(
            symbol=chat_request.symbol,
            lookback_days=lookback_days,
        )
        
        # 3. Calculate position sizing if enough information is available
        position_sizing_result = None
        if all([entry_price, stop_price, risk_amount, account_size]):
            position_sizing_result = await risk_handler.calculate_position_sizing(
                symbol=chat_request.symbol,
                price=entry_price,
                stop_price=stop_price,
                risk_amount=risk_amount,
                account_size=account_size,
                max_risk_percent=2.0
            )
        
        # 4. Call LLM for analysis with memory context
        api_key = ModelProviderFactory._get_api_key(chat_request.provider_type)
        
        llm_interpretation = await llm_helper.generate_risk_analysis_with_llm(
            symbol=chat_request.symbol,
            stop_levels_data=stop_levels_result,
            position_sizing_data=position_sizing_result,
            user_question=chat_request.question_input,
            target_language=chat_request.target_language,
            model_name=chat_request.model_name,
            provider_type=chat_request.provider_type,
            api_key=api_key,
            memory_context=context  # Pass memory context
        )
        
        # 5. Analyze conversation importance (higher for risk management)
        importance_score = 0.7  # Default higher for risk analysis
        
        if chat_request.session_id and user_id:
            try:
                analysis_model = "gpt-4.1-nano" if chat_request.provider_type == ProviderType.OPENAI else chat_request.model_name
                
                importance_score = await analyze_conversation_importance(
                    query=chat_request.question_input or f"Analyze risk management for {chat_request.symbol}",
                    response=llm_interpretation,
                    llm_provider=llm_provider,
                    model_name=analysis_model,
                    provider_type=chat_request.provider_type
                )
                
                # Boost importance for critical risk levels
                if stop_levels_result and stop_levels_result.get("risk_level") in ["high", "extreme"]:
                    importance_score = min(1.0, importance_score + 0.2)
                
                logger.info(f"Risk analysis importance score: {importance_score}")
                
            except Exception as e:
                logger.error(f"Error analyzing conversation importance: {e}")
        
        # 6. Store conversation in memory system
        if chat_request.session_id and user_id:
            try:
                # Extract risk metadata
                metadata = {
                    "type": "risk_analysis",
                    "symbol": chat_request.symbol,
                    "stop_levels": stop_levels_result,
                    "position_sizing": position_sizing_result,
                    "lookback_days": lookback_days,
                    "risk_level": stop_levels_result.get("risk_level") if stop_levels_result else None,
                    "atr_stop": stop_levels_result.get("atr_stop") if stop_levels_result else None,
                    "support_stop": stop_levels_result.get("support_stop") if stop_levels_result else None,
                    "percentage_stop": stop_levels_result.get("percentage_stop") if stop_levels_result else None
                }
                
                # Store in memory system
                await memory_manager.store_conversation_turn(
                    session_id=chat_request.session_id,
                    user_id=user_id,
                    query=chat_request.question_input or f"Analyze risk management for {chat_request.symbol}",
                    response=llm_interpretation,
                    metadata=metadata,
                    importance_score=importance_score
                )
                
                # Save to chat history
                question_content = chat_request.question_input or f"Analyze risk management for {chat_request.symbol}"
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
                logger.error(f"Error saving to memory/chat history: {str(e)}")
        
        # 7. Prepare response data with memory stats
        response_data = {
            "symbol": chat_request.symbol,
            "interpretation": llm_interpretation,
            "memory_stats": memory_stats
        }
        
        return RiskAnalysisChatResponse(
            status="success",
            message="Risk analysis completed successfully",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"Error in risk analysis chat: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error performing risk analysis: {str(e)}"
        )

@router.post("/risk/chat/stream")
async def risk_analysis_chat_stream(
    request: Request,
    chat_request: RiskAnalysisChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Streaming endpoint for risk analysis with memory integration
    
    Enhanced with memory context and importance scoring
    """
    user_id = getattr(request.state, "user_id", None)
    
    # Save user question first
    question_id = None
    if chat_request.session_id and user_id:
        try:
            question_content = chat_request.question_input or f"Analyze risk management for {chat_request.symbol}"
            question_id = chat_service.save_user_question(
                session_id=chat_request.session_id,
                created_at=datetime.now(),
                created_by=user_id,
                content=question_content
            )
        except Exception as save_error:
            logger.error(f"Error saving user question: {str(save_error)}")
    
    # Get memory context
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
                current_query=chat_request.question_input or f"Analyze risk management for {chat_request.symbol}",
                llm_provider=llm_provider,
                max_short_term=5,
                max_long_term=3
            )
            
            logger.info(f"Retrieved memory context for risk streaming: {memory_stats}")
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
    
    enhanced_history = ""
    if context:
        enhanced_history = f"{context}\n\n"
    if chat_history:
        enhanced_history += f"[Conversation History]\n{chat_history}"
    
    async def format_sse():
        full_response = []
        stop_levels_result = None
        position_sizing_result = None
        
        try:
            # Config
            entry_price = 0
            stop_price = 0
            risk_amount = 0
            account_size = 0
            lookback_days = 252

            # Initialize handlers
            risk_handler = RiskAnalysisHandler(market_data)
            llm_helper = RiskAnalysisLLMHelper()
            
            # 1. Calculate stop loss levels with context
            stop_levels_result = await risk_handler.suggest_stop_loss_levels(
                symbol=chat_request.symbol,
                lookback_days=lookback_days,
            )
            
            # 2. Calculate position sizing if enough information is available
            if all([entry_price, stop_price, risk_amount, account_size]):
                position_sizing_result = await risk_handler.calculate_position_sizing(
                    symbol=chat_request.symbol,
                    price=entry_price,
                    stop_price=stop_price,
                    risk_amount=risk_amount,
                    account_size=account_size,
                    max_risk_percent=2.0
                )
            
            # 3. Stream LLM analysis with context
            api_key = ModelProviderFactory._get_api_key(chat_request.provider_type)
            
            # Stream LLM interpretation
            async for chunk in llm_helper.stream_generate_risk_analysis_with_llm(
                symbol=chat_request.symbol,
                stop_levels_data=stop_levels_result,
                position_sizing_data=position_sizing_result,
                user_question=chat_request.question_input,
                model_name=chat_request.model_name,
                provider_type=chat_request.provider_type,
                api_key=api_key,
                memory_context=enhanced_history  # Pass memory context
            ):
                if chunk:
                    full_response.append(chunk)
                    yield f"{json.dumps({'content': chunk})}\n\n"
                    # event_data = {
                    #     "session_id": chat_request.session_id,
                    #     "type": "chunk",
                    #     "data": chunk
                    # }
                    # yield f"{json.dumps(event_data, ensure_ascii=False)}\n\n"
            
            # Join full response
            complete_response = ''.join(full_response)
            
            # 4. Analyze conversation importance
            importance_score = 0.7  # Default higher for risk
            
            if chat_request.session_id and user_id:
                try:
                    analysis_model = "gpt-4.1-nano" if chat_request.provider_type == ProviderType.OPENAI else chat_request.model_name
                    
                    importance_score = await analyze_conversation_importance(
                        query=chat_request.question_input or f"Analyze risk management for {chat_request.symbol}",
                        response=complete_response,
                        llm_provider=llm_provider,
                        model_name=analysis_model,
                        provider_type=chat_request.provider_type
                    )
                    
                    # Boost for critical risk levels
                    if stop_levels_result and stop_levels_result.get("risk_level") in ["high", "extreme"]:
                        importance_score = min(1.0, importance_score + 0.2)
                    
                    logger.info(f"Risk analysis stream importance: {importance_score}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing conversation: {e}")
            
            # 5. Store in memory system
            if chat_request.session_id and user_id and complete_response:
                try:
                    metadata = {
                        "type": "risk_analysis",
                        "symbol": chat_request.symbol,
                        "stop_levels": stop_levels_result,
                        "position_sizing": position_sizing_result,
                        "lookback_days": lookback_days,
                        "risk_level": stop_levels_result.get("risk_level") if stop_levels_result else None,
                        "analysis_type": "streaming"
                    }
                    
                    await memory_manager.store_conversation_turn(
                        session_id=chat_request.session_id,
                        user_id=user_id,
                        query=chat_request.question_input or f"Analyze risk management for {chat_request.symbol}",
                        response=complete_response,
                        metadata=metadata,
                        importance_score=importance_score
                    )
                    
                    # Save to chat history
                    chat_service.save_assistant_response(
                        session_id=chat_request.session_id,
                        created_at=datetime.now(),
                        question_id=question_id,
                        content=complete_response,
                        response_time=0.1
                    )
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
            logger.error(f"Error in risk analysis streaming: {str(e)}")
            yield f"{json.dumps({'error': str(e)})}\n\n"
            yield "[DONE]\n\n"
            # error_data = {
            #     "session_id": chat_request.session_id,
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