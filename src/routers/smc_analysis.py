import json
from datetime import datetime
from typing import Dict, Any, Optional, AsyncGenerator

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from src.utils.logger.custom_logging import LoggerMixin
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.providers.provider_factory import ProviderType
from src.helpers.llm_helper import LLMGeneratorProvider
from src.helpers.chat_management_helper import ChatService
from src.agents.memory.memory_manager import MemoryManager
import asyncio

from src.helpers.llm_chat_helper import (
    stream_with_heartbeat,
    analyze_conversation_importance,
    SSE_HEADERS,
    DEFAULT_HEARTBEAT_SEC,
    sse_error,
    sse_done
)

from src.services.background_tasks import trigger_summary_update_nowait

# SMC modules
from src.schemas.smc_models import (
    SMCAnalyzeRequest,
    SMCAnalyzeResponse,
    SMCAnalysisResult,
    SMCStreamChunk,
)
from src.handlers.smc_analysis_handler import SMCAnalysisHandler
from src.helpers.smc_analysis_llm_helper import SMCAnalysisLLMHelper


# ============================================================================
# ROUTER INITIALIZATION
# ============================================================================

router = APIRouter(prefix="/live-analysis")

api_key_auth = APIKeyAuth()
logger_mixin = LoggerMixin()
logger = logger_mixin.logger

smc_handler = SMCAnalysisHandler()
smc_llm_helper = SMCAnalysisLLMHelper()
llm_provider = LLMGeneratorProvider()
chat_service = ChatService()
memory_manager = MemoryManager()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post(
    "/analyze",
    response_model=SMCAnalyzeResponse,
    summary="Analyze SMC Data",
    description="""
Analyze Smart Money Concepts indicator data.

**Input Fields:**
- `session_id`: Optional Session ID
- `symbol`: Trading symbol
- `interval`: Timeframe
- `mode`: swing or internal
- `timestamp`: Data timestamp in milliseconds
- `currentPrice`: Current market price
- `smcData`: Complete SMC indicator data
- `metadata`: Summary metadata
- `tradingSignals`: Optional pre-computed signals
- `model_name`: LLM model (default: gpt-4.1-nano)
- `provider_type`: LLM provider (default: openai)
- `target_language`: Response language (default: en)
- `collection_name`: Optional RAG collection
- `question_input`: Optional additional question
    """
)
async def analyze_smc_data(
    request: Request,
    data: SMCAnalyzeRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
) -> SMCAnalyzeResponse:
    """
    Main endpoint for SMC analysis (non-streaming).
    
    Flow:
    1. Validate and parse SMC indicator data
    2. Run deterministic SMC analysis
    3. Generate LLM interpretation
    4. Update memory if session provided
    5. Return structured response
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    logger.info(f"[SMC] Analyze request for {data.symbol} @ {data.interval}")
    
    try:
        # Create session if needed
        session_id = data.session_id
        if not session_id and user_id:
            session_id = chat_service.create_chat_session(
                user_id=user_id,
                organization_id=organization_id
            )
            logger.info(f"[SMC] Created session: {session_id}")
        
        # Run SMC analysis
        analysis = smc_handler.analyze(data)
        
        # Generate LLM interpretation
        llm_interpretation = None
        try:
            llm_interpretation = await smc_llm_helper.generate_interpretation(
                analysis=analysis,
                user_question=data.question_input,
                target_language=data.target_language,
                model_name=data.model_name,
                provider_type=data.provider_type
            )
        except Exception as e:
            logger.error(f"[SMC] LLM interpretation error: {e}")
            llm_interpretation = analysis.executive_summary
        
        # Save to memory if session provided
        if session_id and user_id:
            try:
                question = data.question_input or f"SMC Analysis: {data.symbol} {data.interval}"
                response_text = llm_interpretation or analysis.executive_summary
                
                # importance = await analyze_conversation_importance(
                #     query=question,
                #     response=response_text,
                #     llm_provider=llm_provider,
                #     model_name=data.model_name,
                #     provider_type=data.provider_type
                # )
                
                # Save user question first
                question_id = chat_service.save_user_question(
                    session_id=session_id,
                    created_at=datetime.now(),
                    created_by=user_id,
                    content=question or "Predictive Analysis"
                )
                
                # Save assistant response
                if question_id:
                    chat_service.save_assistant_response(
                        session_id=session_id,
                        created_at=datetime.now(),
                        question_id=question_id,
                        content=response_text,
                        response_time=0.1
                    )
                
                # trigger_summary_update_nowait(
                #     session_id=session_id,
                #     user_id=user_id,
                #     organization_id=organization_id,
                #     model_name=data.model_name,
                #     provider_type=data.provider_type
                # )

            except Exception as e:
                logger.error(f"[SMC] Memory save error: {e}")
        
        return SMCAnalyzeResponse(
            status="success",
            message=f"Analysis complete for {data.symbol}",
            session_id=session_id,
            data=analysis,
            llm_interpretation=llm_interpretation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"[SMC] Analysis error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"SMC analysis failed: {str(e)}"
        )

@router.post(
    "/analyze/stream",
    summary="Streaming SMC Analysis",
    description="""Analysis and stream results using Server-Sent Events (SSE) with heartbeat support"""
)
async def stream_analyze_smc(
    request: Request,
    data: SMCAnalyzeRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Streaming endpoint for SMC analysis with heartbeat.
    
    Flow:
    1. Run SMC analysis (with heartbeat during processing)
    2. Stream LLM interpretation chunks (with heartbeat)
    3. Update memory if session provided
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    logger.info(f"[SMC] Stream analyze for {data.symbol} @ {data.interval}")
    
    # Create session if needed
    session_id = data.session_id
    if not session_id and user_id:
        session_id = chat_service.create_chat_session(
            user_id=user_id,
            organization_id=organization_id
        )
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            
            loop = asyncio.get_running_loop()
            analyze_task = loop.run_in_executor(None, smc_handler.analyze, data)
            
            analysis = None
            while True:
                done, _ = await asyncio.wait({analyze_task}, timeout=DEFAULT_HEARTBEAT_SEC)
                
                if done:
                    try:
                        analysis = analyze_task.result()
                    except Exception as e:
                        logger.error(f"[SMC] Analysis task error: {e}")
                        yield sse_error(str(e))
                        yield sse_done()
                        return
                    break
                else:
                    # Send heartbeat while analyzing
                    yield ": heartbeat\n\n"
           
            full_interpretation = ""
            
            llm_gen = smc_llm_helper.stream_interpretation(
                analysis=analysis,
                user_question=data.question_input,
                target_language=data.target_language,
                model_name=data.model_name,
                provider_type=data.provider_type
            )
            
            async for event in stream_with_heartbeat(llm_gen, DEFAULT_HEARTBEAT_SEC):
                if event["type"] == "content":
                    chunk = event["chunk"]
                    full_interpretation += chunk
                    yield f"{json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                elif event["type"] == "heartbeat":
                    yield ": heartbeat\n\n"
                elif event["type"] == "error":
                    yield sse_error(event["error"])
                    break
                elif event["type"] == "done":
                    break
                
            if session_id and user_id:
                try:
                    question = data.question_input or f"SMC Analysis: {data.symbol} {data.interval}"
                    response_text = full_interpretation or analysis.executive_summary
                    
                    # Save user question first
                    question_id = chat_service.save_user_question(
                        session_id=session_id,
                        created_at=datetime.now(),
                        created_by=user_id,
                        content=question
                    )
                    
                    # Save assistant response
                    if question_id:
                        chat_service.save_assistant_response(
                            session_id=session_id,
                            created_at=datetime.now(),
                            question_id=question_id,
                            content=response_text,
                            response_time=0.1
                        )
                except Exception as e:
                    logger.error(f"[SMC] Memory update error: {e}")
            
            yield sse_done()
            
        except asyncio.CancelledError:
            logger.info(f"[SMC] Stream cancelled for {data.symbol}")
            raise
        except Exception as e:
            logger.error(f"[SMC] Stream error: {e}", exc_info=True)
            yield sse_error(str(e))
            yield sse_done()
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )

# @router.post(
#     "/analyze/stream",
#     summary="Streaming SMC Analysis",
#     description="""Analysis and stream results using Server-Sent Events (SSE)"""
# )
# async def stream_analyze_smc(
#     request: Request,
#     data: SMCAnalyzeRequest,
#     api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
# ):
#     """
#     Streaming endpoint for SMC analysis.
    
#     Flow:
#     1. Run SMC analysis
#     2. Stream analysis summary
#     3. Stream trading plan
#     4. Stream LLM interpretation chunks
#     5. Update memory if session provided
#     """
#     user_id = getattr(request.state, "user_id", None)
#     organization_id = getattr(request.state, "organization_id", None)
    
#     logger.info(f"[SMC] Stream analyze for {data.symbol} @ {data.interval}")
    
#     # Create session if needed
#     session_id = data.session_id
#     if not session_id and user_id:
#         session_id = chat_service.create_chat_session(
#             user_id=user_id,
#             organization_id=organization_id
#         )
    
#     async def generate_stream() -> AsyncGenerator[str, None]:
#         try:
#             # Run SMC analysis
#             analysis = smc_handler.analyze(data)
            
#             # Yield analysis summary
#             analysis_chunk = {
#                 "type": "analysis",
#                 "content": {
#                     "symbol": analysis.symbol,
#                     "interval": analysis.interval,
#                     "current_price": analysis.current_price,
#                     "timeframe": {
#                         "type": analysis.timeframe_type or "swing_trading",
#                         "is_short_term": analysis.is_short_term if analysis.is_short_term is not None else False,
#                         "hold_duration": analysis.trading_plan.hold_duration or "unknown"
#                     },
#                     "trend": {
#                         "direction": analysis.trend_analysis.direction,
#                         "strength": analysis.trend_analysis.strength,
#                         "last_event": analysis.trend_analysis.last_structure_event,
#                         "confirmation": analysis.trend_analysis.confirmation_level
#                     },
#                     "premium_discount": {
#                         "current_zone": analysis.premium_discount_analysis.current_zone,
#                         "equilibrium": analysis.premium_discount_analysis.equilibrium_price
#                     },
#                     "executive_summary": analysis.executive_summary,
#                     "confidence": analysis.analysis_confidence,
#                     "data_quality": analysis.data_quality_score
#                 },
#                 "session_id": session_id
#             }
#             # yield f"{json.dumps({'content': analysis_chunk})}\n\n"
#             # yield f"data: {json.dumps(analysis_chunk)}\n\n"
            
#             # Yield trading plan
#             trading_plan_chunk = {
#                 "type": "trading_plan",
#                 "content": {
#                     "bias": analysis.trading_plan.bias,
#                     "signal_strength": analysis.trading_plan.signal_strength,
#                     "recommended_action": analysis.trading_plan.recommended_action,
#                     "entry_zones": [
#                         {
#                             "type": ez.zone_type,
#                             "low": ez.price_range.get("low") if ez.price_range else None,
#                             "high": ez.price_range.get("high") if ez.price_range else None,
#                             "confidence": ez.confidence
#                         }
#                         for ez in analysis.trading_plan.entry_zones
#                     ] if analysis.trading_plan.entry_zones else [],
#                     "stop_loss": analysis.trading_plan.stop_loss,
#                     "targets": [
#                         {"price": t.price, "type": t.target_type}
#                         for t in analysis.trading_plan.targets
#                     ] if analysis.trading_plan.targets else [],
#                     "risk_reward": analysis.trading_plan.risk_reward_ratio,
#                     "invalidation": analysis.trading_plan.invalidation_level,
#                     "warnings": analysis.trading_plan.key_warnings or []
#                 },
#                 "session_id": session_id
#             }
#             # yield f"data: {json.dumps(trading_plan_chunk)}\n\n"
#             # yield f"{json.dumps({'content': trading_plan_chunk})}\n\n"
            
#             # Yield order blocks
#             ob_chunk = {
#                 "type": "order_blocks",
#                 "content": {
#                     "total_active": analysis.order_block_analysis.total_active,
#                     "strongest_demand": analysis.order_block_analysis.strongest_demand_zone,
#                     "strongest_supply": analysis.order_block_analysis.strongest_supply_zone,
#                     "reasoning": analysis.order_block_analysis.reasoning
#                 },
#                 "session_id": session_id
#             }
#             # yield f"{json.dumps({'content': ob_chunk})}\n\n"
#             # yield f"data: {json.dumps(ob_chunk)}\n\n"
            
#             # Yield liquidity
#             liq_chunk = {
#                 "type": "liquidity",
#                 "content": {
#                     "nearest_eqh": analysis.liquidity_analysis.nearest_eqh,
#                     "nearest_eql": analysis.liquidity_analysis.nearest_eql,
#                     "sweep_targets": analysis.liquidity_analysis.potential_sweep_targets,
#                     "reasoning": analysis.liquidity_analysis.reasoning
#                 },
#                 "session_id": session_id
#             }
#             # yield f"{json.dumps({'content': liq_chunk})}\n\n"
#             # yield f"data: {json.dumps(liq_chunk)}\n\n"
            
#             # Yield FVG
#             fvg_chunk = {
#                 "type": "fvg",
#                 "content": {
#                     "total_active": analysis.fvg_analysis.total_active,
#                     "nearest_unfilled": analysis.fvg_analysis.nearest_unfilled_fvg,
#                     "reasoning": analysis.fvg_analysis.reasoning
#                 },
#                 "session_id": session_id
#             }
#             # yield f"{json.dumps({'content': fvg_chunk})}\n\n"
#             # yield f"data: {json.dumps(fvg_chunk)}\n\n"
            
#             # Stream LLM interpretation
#             # yield f"data: {json.dumps({'type': 'interpretation_start', 'session_id': session_id})}\n\n"
            
#             full_interpretation = ""
#             async for chunk in smc_llm_helper.stream_interpretation(
#                 analysis=analysis,
#                 user_question=data.question_input,
#                 target_language=data.target_language,
#                 model_name=data.model_name,
#                 provider_type=data.provider_type
#             ):
#                 if chunk:
#                     full_interpretation += chunk
#                     # interpretation_chunk = {
#                     #     "type": "interpretation",
#                     #     "content": chunk,
#                     #     "session_id": session_id
#                     # }
#                     yield f"{json.dumps({'content': chunk})}\n\n"
#                     # yield f"data: {json.dumps(interpretation_chunk)}\n\n"
            
#             # yield f"data: {json.dumps({'type': 'interpretation_end', 'session_id': session_id})}\n\n"
            
#             # Save to memory
#             if session_id and user_id:
#                 try:
#                     question = data.question_input or f"SMC Analysis: {data.symbol} {data.interval}"
#                     response_text = full_interpretation or analysis.executive_summary
                    
#                     # importance = await analyze_conversation_importance(
#                     #     query=question,
#                     #     response=response_text,
#                     #     llm_provider=llm_provider,
#                     #     model_name=data.model_name,
#                     #     provider_type=data.provider_type
#                     # )
                    
#                     # Save user question first
#                     question_id = chat_service.save_user_question(
#                         session_id=session_id,
#                         created_at=datetime.now(),
#                         created_by=user_id,
#                         content=question
#                     )
                    
#                     # Save assistant response
#                     if question_id:
#                         chat_service.save_assistant_response(
#                             session_id=session_id,
#                             created_at=datetime.now(),
#                             question_id=question_id,
#                             content=response_text,
#                             response_time=0.1
#                         )
                    
#                     # trigger_summary_update_nowait(
#                     #     session_id=session_id,
#                     #     user_id=user_id,
#                     #     organization_id=organization_id,
#                     #     model_name=data.model_name,
#                     #     provider_type=data.provider_type
#                     # )
#                 except Exception as e:
#                     logger.error(f"[SMC] Memory update error: {e}")
            
#             # Complete signal
#             # complete_chunk = {
#             #     "type": "complete",
#             #     "content": {
#             #         "symbol": analysis.symbol,
#             #         "interval": analysis.interval,
#             #         "confidence": analysis.analysis_confidence,
#             #         "data_quality": analysis.data_quality_score,
#             #         "timestamp": datetime.now().isoformat()
#             #     },
#             #     "session_id": session_id
#             # }
#             # yield f"data: {json.dumps(complete_chunk)}\n\n"
#             yield "[DONE]\n\n"
            
#         except Exception as e:
#             logger.error(f"[SMC] Stream error: {e}", exc_info=True)
#             yield f"{json.dumps({'error': str(e)})}\n\n"
#             yield "[DONE]\n\n"
    
#     return StreamingResponse(
#         generate_stream(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#             "X-Accel-Buffering": "no"
#         }
#     )