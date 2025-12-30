import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import AsyncIterator
from src.handlers.trading_agents_handler import trading_agents_handler
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.chat_management_helper import ChatService
from src.handlers.llm_chat_handler import ChatMessageHistory
from src.routers.llm_chat import analyze_conversation_importance
from src.agents.memory.memory_manager import get_memory_manager
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ProviderType
from fastapi.responses import StreamingResponse                
import json
from fastapi.responses import StreamingResponse
from src.providers.provider_factory import ProviderType, ModelProviderFactory
from src.services.background_tasks import trigger_summary_update_nowait

router = APIRouter(prefix="/trading-agents")
api_key_auth = APIKeyAuth()
logger = LoggerMixin().logger
chat_service = ChatService()
memory_manager = get_memory_manager()
llm_provider = LLMGeneratorProvider()

class TradingAnalysisRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    question_input: Optional[str] = Field(
        None, 
        description="User's specific question or analysis focus"
    )
    use_memory_context: bool = Field(
        default=True, 
        description="Whether to use previous analysis context"
    )

    ticker: str = Field(..., description="Stock ticker symbol (e.g., NVDA)")
    trade_date: str = Field(..., description="Trading date in YYYY-MM-DD format")
    selected_analysts: Optional[List[str]] = Field(
        default=["market", "social", "news", "fundamentals"],
        description="List of analysts to use"
    )
    llm_provider: str = Field(default="openai", description="LLM provider")
    deep_think_llm: str = Field(default="gpt-4o-mini", description="Model for deep thinking")
    quick_think_llm: str = Field(default="gpt-4o-mini", description="Model for quick thinking")
    max_debate_rounds: int = Field(default=1, ge=1, le=5, description="Number of debate rounds")
    online_tools: bool = Field(default=True, description="Whether to use online tools")
    debug: bool = Field(default=False, description="Debug mode")
    stream_mode: str = Field(default="detailed", description="Stream mode: 'simple' or 'detailed'")


class ReflectionRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    trade_date: str = Field(..., description="Trading date")
    returns_losses: float = Field(..., description="Actual returns/losses percentage")
    selected_analysts: Optional[List[str]] = Field(default=None)


async def format_sse(
    generator: AsyncGenerator,
    ticker: str,
    heartbeat_sec: int = 15
) -> AsyncGenerator[str, None]:
    """
    SSE formatter với heartbeat an toàn:
    - KHÔNG cancel anext(generator) khi timeout (tránh làm đóng generator).
    - Khi hết heartbeat_sec mà chưa có dữ liệu -> gửi ':hb\\n\\n' nhưng giữ nguyên Task đang đợi.
    - Kết thúc chuẩn bằng 'data: [DONE]\\n\\n'.
    """
    pending_task: asyncio.Task | None = None
    try:
        while True:
            # Tạo task đọc phần tử kế tiếp nếu chưa có
            if pending_task is None:
                pending_task = asyncio.create_task(anext(generator))

            # Chờ tối đa heartbeat_sec cho kết quả
            done, _ = await asyncio.wait({pending_task}, timeout=heartbeat_sec)

            if done:
                try:
                    chunk = pending_task.result()  # lấy kết quả
                except StopAsyncIteration:
                    # Generator đã kết thúc
                    yield "data: [DONE]\n\n"
                    break
                except Exception as e:
                    # Lỗi từ generator
                    error_response = {"type": "error", "content": str(e)}
                    yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                finally:
                    # task đã xong -> reset để tạo task mới vòng sau
                    pending_task = None

                if chunk:
                    response = {
                        "ticker": ticker,
                        "timestamp": datetime.now().isoformat(),
                        "type": chunk.get("type", "update"),
                        "stage": chunk.get("stage", ""),
                        "content": chunk.get("content", "")
                    }
                    yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0)  # nhả buffer ngay
            else:
                # Hết thời gian chờ nhưng task vẫn chạy -> gửi heartbeat, KHÔNG cancel task
                yield ":hb\n\n"

    except Exception as e:
        # Lỗi ngoài ý muốn ở formatter
        error_response = {"type": "error", "content": str(e)}
        yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        # Dọn dẹp nếu còn task đang chờ
        if pending_task is not None and not pending_task.done():
            pending_task.cancel()
            # không cần await; đây là cleanup


@router.post("/analyze/stream", response_description="Stream analysis of a stock ticker")
async def analyze_ticker_stream(
    request: TradingAnalysisRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Stream real-time analysis updates using TradingAgents framework.
    
    Returns Server-Sent Events (SSE) stream with progressive updates:
    - Market analysis progress
    - Social sentiment updates  
    - News analysis results
    - Fundamentals insights
    - Bull vs Bear debate rounds
    - Final trading decision
    """
    try:
        logger.info(f"Received streaming analysis request for {request.ticker} on {request.trade_date}")
        
        # Validate date format
        try:
            datetime.strptime(request.trade_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format. Use YYYY-MM-DD"
            )
        
        # Validate analysts
        valid_analysts = ["market", "social", "news", "fundamentals"]
        for analyst in request.selected_analysts:
            if analyst not in valid_analysts:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid analyst: {analyst}. Valid options: {valid_analysts}"
                )
        
        # Create streaming generator
        stream_generator = trading_agents_handler.analyze_ticker_stream(
            ticker=request.ticker,
            trade_date=request.trade_date,
            selected_analysts=request.selected_analysts,
            llm_provider=request.llm_provider,
            deep_think_llm=request.deep_think_llm,
            quick_think_llm=request.quick_think_llm,
            max_debate_rounds=request.max_debate_rounds,
            online_tools=request.online_tools,
            stream_mode=request.stream_mode
        )

        # Return SSE streaming response
        return StreamingResponse(
            format_sse(stream_generator, request.ticker, heartbeat_sec=15),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no" 
            }
        )
        
    except Exception as e:
        logger.error(f"Error in analyze_ticker_stream: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Streaming analysis failed: {str(e)}"
        )


def create_trading_agents_summary(result: Dict) -> str:
    """
    Create a comprehensive summary from TradingAgents analysis result
    
    Args:
        result: Full analysis result from TradingAgents
    
    Returns:
        Formatted summary string for memory storage
    """
    summary_parts = []
    
    # Add header
    ticker = result.get("ticker", "N/A")
    trade_date = result.get("trade_date", "N/A")
    summary_parts.append(f"TradingAgents Analysis for {ticker} on {trade_date}")
    summary_parts.append("=" * 50)
    
    # Final decision
    final_decision = result.get("final_decision", {})
    if final_decision:
        action = final_decision.get("action", "HOLD")
        confidence = final_decision.get("confidence", 0)
        summary_parts.append(f"**Final Decision: {action} (Confidence: {confidence:.2%})**")
        
        if final_decision.get("reasoning"):
            summary_parts.append(f"Reasoning: {final_decision['reasoning']}")
    
    # Market Analysis
    if "market_analyst" in result:
        market = result["market_analyst"]
        summary_parts.append(f"**Market Analysis:**")
        summary_parts.append(f"- Trend: {market.get('trend', 'N/A')}")
        summary_parts.append(f"- Key Indicators: {market.get('key_indicators', 'N/A')}")
        if market.get("summary"):
            summary_parts.append(f"- Summary: {market['summary']}")
    
    # Social Sentiment
    if "social_analyst" in result:
        social = result["social_analyst"]
        summary_parts.append(f"**Social Sentiment:**")
        summary_parts.append(f"- Overall Sentiment: {social.get('sentiment', 'N/A')}")
        if social.get("summary"):
            summary_parts.append(f"- Summary: {social['summary']}")
    
    # News Impact
    if "news_analyst" in result:
        news = result["news_analyst"]
        summary_parts.append(f"**News Analysis:**")
        summary_parts.append(f"- Impact: {news.get('impact', 'N/A')}")
        if news.get("summary"):
            summary_parts.append(f"- Summary: {news['summary']}")
    
    # Fundamentals
    if "fundamentals_analyst" in result:
        fundamentals = result["fundamentals_analyst"]
        summary_parts.append(f"**Fundamental Analysis:**")
        summary_parts.append(f"- Valuation: {fundamentals.get('valuation', 'N/A')}")
        if fundamentals.get("summary"):
            summary_parts.append(f"- Summary: {fundamentals['summary']}")
    
    # Debate Summary
    if "debate" in result:
        debate = result["debate"]
        summary_parts.append(f"**Bull vs Bear Debate:**")
        if debate.get("bull_position"):
            summary_parts.append(f"- Bull Case: {debate['bull_position']}...")
        if debate.get("bear_position"):
            summary_parts.append(f"- Bear Case: {debate['bear_position']}...")
        if debate.get("consensus"):
            summary_parts.append(f"- Consensus: {debate['consensus']}")
    
    # Risk Assessment
    if "risk_assessment" in result:
        risk = result["risk_assessment"]
        summary_parts.append(f"**Risk Assessment:**")
        summary_parts.append(f"- Level: {risk.get('level', 'N/A')}")
        summary_parts.append(f"- Key Risks: {', '.join(risk.get('key_risks', []))}")
    
    return "".join(summary_parts)


@router.post("/analyze", response_description="Analyze a stock ticker using TradingAgents")
async def analyze_ticker(
    request2: Request,
    request: TradingAnalysisRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Analyze a stock ticker using the TradingAgents framework
    
    This endpoint triggers a comprehensive analysis using multiple AI agents:
    - Market Analyst: Technical analysis with indicators
    - Social Media Analyst: Sentiment analysis from social platforms
    - News Analyst: Global news and macroeconomic analysis
    - Fundamentals Analyst: Financial statements and insider trading
    
    The agents then debate (Bull vs Bear) to reach a trading decision.
    """
    try:
        logger.info(f"Received analysis request for {request.ticker} on {request.trade_date}")
        user_id = getattr(request2.state, "user_id", None)
        # Validate date format
        try:
            datetime.strptime(request.trade_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format. Use YYYY-MM-DD"
            )
        
        # Validate analysts
        valid_analysts = ["market", "social", "news", "fundamentals"]
        for analyst in request.selected_analysts:
            if analyst not in valid_analysts:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid analyst: {analyst}. Valid options: {valid_analysts}"
                )
        
        # 1. Get memory context for enhanced analysis
        context = ""
        memory_stats = {}
        
        if request.session_id and user_id and request.use_memory_context:
            try:
                # Get relevant context about this ticker and related analyses
                query_text = request.question_input or f"Analyze {request.ticker} for trading on {request.trade_date}"
                
                context, memory_stats, document_references = await memory_manager.get_relevant_context(
                    session_id=request.session_id,
                    user_id=user_id,
                    current_query=query_text,
                    llm_provider=llm_provider,
                    max_short_term=5,  # Recent related analyses
                    max_long_term=3    # Important historical analyses
                )
                
                logger.info(f"Retrieved memory context for TradingAgents: {memory_stats}")
                
            except Exception as e:
                logger.error(f"Error getting memory context: {e}")
        
        # Run analysis
        result = await trading_agents_handler.analyze_ticker(
            ticker=request.ticker,
            trade_date=request.trade_date,
            selected_analysts=request.selected_analysts,
            llm_provider=request.llm_provider,
            deep_think_llm=request.deep_think_llm,
            quick_think_llm=request.quick_think_llm,
            max_debate_rounds=request.max_debate_rounds,
            online_tools=request.online_tools,
            debug=request.debug
        )

        # 3. Create comprehensive summary for memory storage
        analysis_summary = create_trading_agents_summary(result)

        # 4. Analyze conversation importance (higher for complex multi-agent analysis)
        importance_score = 0.7  # Default higher for TradingAgents
        
        if request.session_id and user_id:
            try:
                # Use the quick_think_llm for importance analysis
                importance_score = await analyze_conversation_importance(
                    query=request.question_input or f"Analyze {request.ticker} for {request.trade_date}",
                    response=analysis_summary,
                    llm_provider=llm_provider,
                    model_name=request.quick_think_llm,
                    provider_type=request.llm_provider
                )
                
                # Boost importance for significant trading decisions
                if result.get("final_decision", {}).get("action") in ["BUY", "STRONG_BUY", "SELL", "STRONG_SELL"]:
                    importance_score = min(1.0, importance_score + 0.2)
                
                logger.info(f"TradingAgents analysis importance: {importance_score}")
                
            except Exception as e:
                logger.error(f"Error analyzing importance: {e}")
        
        # 5. Store comprehensive analysis in memory
        if request.session_id and user_id:
            try:
                # Extract detailed metadata from analysis
                metadata = {
                    "type": "trading_agents_analysis",
                    "ticker": request.ticker,
                    "trade_date": request.trade_date,
                    "analysts_used": request.selected_analysts,
                    "final_decision": result.get("final_decision", {}),
                    "market_analysis": result.get("market_analyst", {}).get("summary", ""),
                    "social_sentiment": result.get("social_analyst", {}).get("summary", ""),
                    "news_impact": result.get("news_analyst", {}).get("summary", ""),
                    "fundamentals": result.get("fundamentals_analyst", {}).get("summary", ""),
                    "debate_rounds": request.max_debate_rounds,
                    "bull_position": result.get("debate", {}).get("bull_position", ""),
                    "bear_position": result.get("debate", {}).get("bear_position", ""),
                    "confidence_score": result.get("final_decision", {}).get("confidence", 0)
                }
                
                # Store in memory system
                await memory_manager.store_conversation_turn(
                    session_id=request.session_id,
                    user_id=user_id,
                    query=request.question_input or f"Analyze {request.ticker} for {request.trade_date}",
                    response=analysis_summary,
                    metadata=metadata,
                    importance_score=importance_score
                )
                
                # Save to chat history
                question_id = chat_service.save_user_question(
                    session_id=request.session_id,
                    created_at=datetime.now(),
                    created_by=user_id,
                    content=request.question_input or f"TradingAgents analysis for {request.ticker} on {request.trade_date}"
                )
                
                chat_service.save_assistant_response(
                    session_id=request.session_id,
                    created_at=datetime.now(),
                    question_id=question_id,
                    content=analysis_summary,
                    response_time=0.1
                )
                
            except Exception as e:
                logger.error(f"Error saving to memory: {str(e)}")
        
        # Add memory stats to response
        result["memory_stats"] = memory_stats
        
        return JSONResponse(
            content={
                "status": "success",
                "data": result
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_ticker: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@router.post("/reflect", response_description="Reflect on a trading decision")
async def reflect_on_decision(
    request: ReflectionRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Reflect on a past trading decision with actual returns/losses
    
    This helps the agents learn from their decisions and improve future performance.
    """
    try:
        result = await trading_agents_handler.reflect_on_decision(
            ticker=request.ticker,
            trade_date=request.trade_date,
            returns_losses=request.returns_losses,
            selected_analysts=request.selected_analysts
        )
        
        return JSONResponse(
            content={
                "status": "success",
                "data": result
            }
        )
        
    except Exception as e:
        logger.error(f"Error in reflect_on_decision: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Reflection failed: {str(e)}"
        )


