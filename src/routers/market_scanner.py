"""
Market Scanner Router

5 consolidated analysis API endpoints:
1. POST /scanner/technical/stream - Technical & Chart Analysis
2. POST /scanner/position/stream - Market Position (Relative Strength)
3. POST /scanner/risk/stream - Risk Analysis
4. POST /scanner/sentiment/stream - Sentiment & News
5. POST /scanner/fundamental/stream - Fundamental Analysis

All endpoints:
- Support streaming responses (SSE)
- Save to chat session for context continuity
- Use improved analysis tools
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse

from src.utils.logger.custom_logging import LoggerMixin
from src.handlers.market_scanner_handler import market_scanner_handler
from src.handlers.fundamental_analysis_handler import FundamentalAnalysisHandler
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.providers.provider_factory import ModelProviderFactory
from src.helpers.chat_management_helper import ChatService
from src.handlers.llm_chat_handler import ChatMessageHistory
from src.services.tool_call_service import ToolCallService
from src.helpers.llm_chat_helper import (
    stream_with_heartbeat,
    SSE_HEADERS,
    DEFAULT_HEARTBEAT_SEC,
    sse_error,
    sse_done
)

router = APIRouter()

api_key_auth = APIKeyAuth()
logger_mixin = LoggerMixin()
logger = logger_mixin.logger
chat_service = ChatService()
fundamental_handler = FundamentalAnalysisHandler()
tool_call_service = ToolCallService()


# =============================================================================
# REQUEST/RESPONSE SCHEMAS
# =============================================================================
class ScannerRequest(BaseModel):
    """Base request schema for scanner endpoints."""
    session_id: Optional[str] = Field(None, description="Chat session ID for context continuity")
    symbol: str = Field(..., description="Stock symbol to analyze")
    question_input: Optional[str] = Field(None, description="User's specific question")
    target_language: Optional[str] = Field(None, description="Response language (vi, en, etc.)")
    model_name: str = Field("gpt-4.1-nano-2025-04-14", description="LLM model to use")
    provider_type: str = Field("openai", description="Provider type: openai, gemini")


class TechnicalScannerRequest(ScannerRequest):
    """Request for technical analysis."""
    timeframe: str = Field("1Y", description="Timeframe: 1M, 3M, 6M, 1Y")


class PositionScannerRequest(ScannerRequest):
    """Request for market position analysis."""
    benchmark: str = Field("SPY", description="Benchmark symbol for comparison")


class RiskScannerRequest(ScannerRequest):
    """Request for risk analysis."""
    entry_price: Optional[float] = Field(None, description="Entry price for position sizing")


class ScannerResponse(BaseModel):
    """Response schema for scanner endpoints."""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


# =============================================================================
# HELPER FUNCTIONS
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


def save_assistant_response(session_id: str, question_id: int, content: str):
    """Save assistant response to chat history."""
    try:
        chat_service.save_assistant_response(
            session_id=session_id,
            created_at=datetime.now(),
            question_id=question_id,
            content=content,
            response_time=0.1
        )
    except Exception as e:
        logger.error(f"Error saving response: {e}")


def get_chat_history(session_id: str) -> str:
    """Get chat history for context."""
    try:
        if session_id:
            return ChatMessageHistory.string_message_chat_history(session_id=session_id)
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
    return ""


# =============================================================================
# STEP 1: TECHNICAL & CHART ANALYSIS
# =============================================================================
@router.post("/scanner/technical/stream", summary="Technical & Chart Analysis (Streaming)")
async def scanner_technical_stream(
    request: Request,
    scan_request: TechnicalScannerRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Comprehensive technical analysis with:
    - Trend regime (bullish/bearish/ranging)
    - Momentum indicators (RSI, MACD, Stochastic)
    - Volatility metrics (ATR, BB width, squeeze)
    - Volume confirmation
    - Support/Resistance levels
    - Trading setups (pullback/breakout)
    - Invalidation levels
    """
    user_id = getattr(request.state, "user_id", None)

    # Save question first
    question_id = None
    question_content = scan_request.question_input or f"Technical analysis for {scan_request.symbol} ({scan_request.timeframe})"
    if scan_request.session_id and user_id:
        question_id = save_user_question(scan_request.session_id, user_id, question_content)

    async def generate_stream():
        full_response = []

        try:
            # Get chat history for context
            chat_history = get_chat_history(scan_request.session_id) if scan_request.session_id else ""

            # Get API key
            api_key = ModelProviderFactory._get_api_key(scan_request.provider_type)

            # Stream analysis
            llm_gen = market_scanner_handler.stream_technical_analysis(
                symbol=scan_request.symbol,
                timeframe=scan_request.timeframe,
                model_name=scan_request.model_name,
                provider_type=scan_request.provider_type,
                api_key=api_key,
                user_question=scan_request.question_input,
                target_language=scan_request.target_language,
                chat_history=chat_history
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

            # Save complete response to chat session
            complete_response = "".join(full_response)
            if scan_request.session_id and user_id and question_id and complete_response:
                save_assistant_response(scan_request.session_id, question_id, complete_response)

            yield sse_done()

        except Exception as e:
            logger.error(f"[Scanner] Technical streaming error: {e}")
            yield sse_error(str(e))
            yield sse_done()

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )


# =============================================================================
# STEP 2: MARKET POSITION (RELATIVE STRENGTH)
# =============================================================================
@router.post("/scanner/position/stream", summary="Market Position Analysis (Streaming)")
async def scanner_position_stream(
    request: Request,
    scan_request: PositionScannerRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Relative strength analysis:
    - RS score vs benchmark (SPY default)
    - Outperforming/Underperforming status
    - Multi-timeframe RS (21d, 63d, 126d, 252d)
    - Sector context
    """
    user_id = getattr(request.state, "user_id", None)

    question_id = None
    question_content = scan_request.question_input or f"Market position analysis for {scan_request.symbol} vs {scan_request.benchmark}"
    if scan_request.session_id and user_id:
        question_id = save_user_question(scan_request.session_id, user_id, question_content)

    async def generate_stream():
        full_response = []

        try:
            chat_history = get_chat_history(scan_request.session_id) if scan_request.session_id else ""
            api_key = ModelProviderFactory._get_api_key(scan_request.provider_type)

            llm_gen = market_scanner_handler.stream_market_position(
                symbol=scan_request.symbol,
                benchmark=scan_request.benchmark,
                model_name=scan_request.model_name,
                provider_type=scan_request.provider_type,
                api_key=api_key,
                user_question=scan_request.question_input,
                target_language=scan_request.target_language,
                chat_history=chat_history
            )

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

            complete_response = "".join(full_response)
            if scan_request.session_id and user_id and question_id and complete_response:
                save_assistant_response(scan_request.session_id, question_id, complete_response)

            yield sse_done()

        except Exception as e:
            logger.error(f"[Scanner] Position streaming error: {e}")
            yield sse_error(str(e))
            yield sse_done()

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )


# =============================================================================
# STEP 3: RISK ANALYSIS
# =============================================================================
@router.post("/scanner/risk/stream", summary="Risk Analysis (Streaming)")
async def scanner_risk_stream(
    request: Request,
    scan_request: RiskScannerRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Risk analysis:
    - Stop loss levels (ATR-based, Support-based, Percentage)
    - Risk metrics (volatility, max drawdown)
    - Position sizing guidance
    - Risk/Reward assessment
    """
    user_id = getattr(request.state, "user_id", None)

    question_id = None
    question_content = scan_request.question_input or f"Risk analysis for {scan_request.symbol}"
    if scan_request.entry_price:
        question_content += f" (entry: ${scan_request.entry_price:.2f})"
    if scan_request.session_id and user_id:
        question_id = save_user_question(scan_request.session_id, user_id, question_content)

    async def generate_stream():
        full_response = []

        try:
            chat_history = get_chat_history(scan_request.session_id) if scan_request.session_id else ""
            api_key = ModelProviderFactory._get_api_key(scan_request.provider_type)

            llm_gen = market_scanner_handler.stream_risk_analysis(
                symbol=scan_request.symbol,
                model_name=scan_request.model_name,
                provider_type=scan_request.provider_type,
                api_key=api_key,
                entry_price=scan_request.entry_price,
                user_question=scan_request.question_input,
                target_language=scan_request.target_language,
                chat_history=chat_history
            )

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

            complete_response = "".join(full_response)
            if scan_request.session_id and user_id and question_id and complete_response:
                save_assistant_response(scan_request.session_id, question_id, complete_response)

            yield sse_done()

        except Exception as e:
            logger.error(f"[Scanner] Risk streaming error: {e}")
            yield sse_error(str(e))
            yield sse_done()

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )


# =============================================================================
# STEP 4: SENTIMENT & NEWS
# =============================================================================
@router.post("/scanner/sentiment/stream", summary="Sentiment & News Analysis (Streaming)")
async def scanner_sentiment_stream(
    request: Request,
    scan_request: ScannerRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Combined sentiment and news analysis:
    - Social sentiment data
    - Recent news with analysis
    - Market impact assessment
    - Trading implications
    """
    user_id = getattr(request.state, "user_id", None)

    question_id = None
    question_content = scan_request.question_input or f"Sentiment and news analysis for {scan_request.symbol}"
    if scan_request.session_id and user_id:
        question_id = save_user_question(scan_request.session_id, user_id, question_content)

    async def generate_stream():
        full_response = []

        try:
            chat_history = get_chat_history(scan_request.session_id) if scan_request.session_id else ""
            api_key = ModelProviderFactory._get_api_key(scan_request.provider_type)

            llm_gen = market_scanner_handler.stream_sentiment_news(
                symbol=scan_request.symbol,
                model_name=scan_request.model_name,
                provider_type=scan_request.provider_type,
                api_key=api_key,
                user_question=scan_request.question_input,
                target_language=scan_request.target_language,
                chat_history=chat_history
            )

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

            complete_response = "".join(full_response)
            if scan_request.session_id and user_id and question_id and complete_response:
                save_assistant_response(scan_request.session_id, question_id, complete_response)

            yield sse_done()

        except Exception as e:
            logger.error(f"[Scanner] Sentiment streaming error: {e}")
            yield sse_error(str(e))
            yield sse_done()

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )


# =============================================================================
# STEP 5: FUNDAMENTAL ANALYSIS
# =============================================================================
@router.post("/scanner/fundamental/stream", summary="Fundamental Analysis (Streaming)")
async def scanner_fundamental_stream(
    request: Request,
    scan_request: ScannerRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Comprehensive fundamental analysis:
    - Valuation metrics (P/E, P/B, P/S)
    - Growth metrics (Revenue, EPS growth)
    - Profitability (ROE, ROA, margins)
    - Financial health (D/E, current ratio)
    - Cash flow analysis
    - Investment score and rating
    """
    user_id = getattr(request.state, "user_id", None)

    question_id = None
    question_content = scan_request.question_input or f"Fundamental analysis for {scan_request.symbol}"
    if scan_request.session_id and user_id:
        question_id = save_user_question(scan_request.session_id, user_id, question_content)

    async def generate_stream():
        full_response = []

        try:
            chat_history = get_chat_history(scan_request.session_id) if scan_request.session_id else ""
            api_key = ModelProviderFactory._get_api_key(scan_request.provider_type)

            # Get fundamental data using existing handler
            comprehensive_data = await fundamental_handler.generate_comprehensive_fundamental_data(
                symbol=scan_request.symbol,
                tool_service=tool_call_service
            )

            fundamental_report = comprehensive_data.get("fundamental_report", {})
            growth_data_list = comprehensive_data.get("growth_data", [])

            # Stream AI analysis
            llm_gen = fundamental_handler.stream_comprehensive_analysis(
                symbol=scan_request.symbol,
                report=fundamental_report,
                growth_data=growth_data_list,
                model_name=scan_request.model_name,
                provider_type=scan_request.provider_type,
                api_key=api_key,
                chat_history=chat_history,
                user_question=scan_request.question_input,
                target_language=scan_request.target_language
            )

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

            complete_response = "".join(full_response)
            if scan_request.session_id and user_id and question_id and complete_response:
                save_assistant_response(scan_request.session_id, question_id, complete_response)

            yield sse_done()

        except Exception as e:
            logger.error(f"[Scanner] Fundamental streaming error: {e}")
            yield sse_error(str(e))
            yield sse_done()

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )
