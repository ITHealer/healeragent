import json
import asyncio
import aioredis
from fastapi import APIRouter, Depends, Body, Request
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings
from src.routers.equity import get_redis_client
from src.services.news_service import NewsService
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.handlers.fundamental_analysis_handler import FundamentalAnalysisHandler
from src.services.tool_call_service import ToolCallService
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from fastapi.responses import StreamingResponse
from src.helpers.llm_helper import LLMGeneratorProvider
from src.helpers.chat_management_helper import ChatService
from src.handlers.llm_chat_handler import ChatMessageHistory
from src.routers.llm_chat import analyze_conversation_importance
from src.agents.memory.memory_manager import MemoryManager
from src.services.background_tasks import trigger_summary_update_nowait

router = APIRouter()

# Initialize services
llm_generator = LLMGeneratorProvider()
api_key_auth = APIKeyAuth()
logger_mixin = LoggerMixin()
logger = logger_mixin.logger
news_service = NewsService()
FMP_API_KEY = settings.FMP_API_KEY
tool_call_service = ToolCallService()
fundamental_handler = FundamentalAnalysisHandler()
chat_service = ChatService()
memory_manager = MemoryManager()


# Define Pydantic models
class FundamentalAnalysisChatRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Chat session ID for context")
    question_input: Optional[str] = None
    target_language: Optional[str] = None
    symbol: str = Field(..., description="Stock symbol to analyze")
    # period: str = Field("annual", description="annual or quarter")
    # limit: int = Field(10, ge=1, le=100, description="Number of periods to analyze")
    model_name: str = Field("gpt-4.1-nano", description="LLM model to use")
    provider_type: str = Field("openai", description="Provider type for the LLM")


class FundamentalAnalysisChatResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


# @router.post("/fundamental/analysis",
#             response_model=FundamentalAnalysisChatResponse,
#             summary="Get financial statement growth with AI analysis")
# async def get_financial_growth_with_analysis(
#     request: Request,
#     chat_request: FundamentalAnalysisChatRequest = Body(...),
#     api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key),
#     redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
# ):
#     """
#     Get financial statement growth data with AI-powered analysis.
#     """
#     user_id = getattr(request.state, "user_id", None)

#     # Config 
#     period = "annual"
#     limit = 10

#     # Extract values from request
#     symbol = chat_request.symbol
#     period = period
#     limit = limit
#     model_name = chat_request.model_name
#     provider_type = chat_request.provider_type
    
#     # First, get the raw data using existing logic
#     cache_key = f"financial_growth_{symbol.upper()}_{period}_limit_{limit}"
    
#     # Check cache for raw data
#     growth_data_list = None
#     if redis_client:
#         try:
#             cached_data = await redis_client.get(cache_key)
#             if cached_data:
#                 cached_response = json.loads(cached_data)
#                 if cached_response.get("data") and cached_response["data"].get("data"):
#                     growth_data_list = cached_response["data"]["data"]
#                     logger.info(f"Cache HIT for raw data: {cache_key}")
#         except Exception as e:
#             logger.error(f"Redis error: {e}")
    
#     # If not cached, fetch from FMP
#     if growth_data_list is None:
#         # Get raw Pydantic models
#         growth_data_models = await tool_call_service.get_financial_statement_growth(
#             symbol=symbol, period=period, limit=limit
#         )
        
#         if growth_data_models:
#             # Convert Pydantic models to dictionaries
#             growth_data_list = [item.model_dump() if hasattr(item, 'model_dump') else item.dict() 
#                                for item in growth_data_models]
            
#             # Cache the raw data
#             if redis_client and growth_data_list:
#                 try:
#                     cache_ttl = settings.CACHE_TTL_FINANCIALS_ANNUAL if period == "annual" else settings.CACHE_TTL_FINANCIALS_QUARTERLY
#                     await redis_client.set(
#                         cache_key,
#                         json.dumps({"data": {"data": growth_data_list}}),
#                         ex=cache_ttl
#                     )
#                 except Exception as e:
#                     logger.error(f"Cache set error: {e}")
    
#     # Check if we have data to analyze
#     if not growth_data_list:
#         return FundamentalAnalysisChatResponse(
#             status="404",
#             message=f"No financial data found for {symbol}",
#             data={
#                 "symbol": symbol,
#                 # "period": period,
#                 "interpretation": None
#             }
#         )
    
#     # Now perform AI analysis
#     try:
#         # Get API Key
#         api_key = ModelProviderFactory._get_api_key(provider_type)
        
#         # Check for analysis cache
#         analysis_cache_key = f"fundamental_analysis_{symbol}_{period}_{model_name}_{provider_type}"
#         cached_analysis = None
        
#         if redis_client:
#             try:
#                 cached_analysis = await redis_client.get(analysis_cache_key)
#                 if cached_analysis:
#                     logger.info(f"Cache HIT for analysis: {analysis_cache_key}")
#                     cached_data = json.loads(cached_analysis)
#                     return FundamentalAnalysisChatResponse(
#                         status=cached_data.get("status", "200"),
#                         message=cached_data.get("message", "Financial analysis completed successfully"),
#                         data=cached_data
#                     )
#             except Exception as e:
#                 logger.error(f"Analysis cache error: {e}")
        
#         # Generate new analysis - pass dictionary data
#         analysis_result = await fundamental_handler.analyze_financial_growth(
#             symbol=symbol,
#             growth_data=growth_data_list,  # Now this is a list of dicts
#             period=period,
#             model_name=model_name,
#             provider_type=provider_type,
#             api_key=api_key
#         )
        
#         # Prepare response data
#         response_data = {
#             "symbol": symbol,
#             "interpretation": analysis_result["analysis"],
#         }
        
#         # Cache the complete response
#         if redis_client:
#             try:
#                 cache_data = {
#                     "status": "200",
#                     "message": "Financial analysis completed successfully",
#                     **response_data
#                 }
#                 # Analysis cache for shorter time (1 hour)
#                 await redis_client.set(
#                     analysis_cache_key,
#                     json.dumps(cache_data),
#                     ex=300
#                 )
#             except Exception as e:
#                 logger.error(f"Analysis cache set error: {e}")
        

#         if chat_request.session_id and user_id:
#             try:
#                 from src.helpers.chat_management_helper import ChatService
#                 chat_service = ChatService()
                
#                 question_content = chat_request.question_input or f"Analyze technical indicators for {chat_request.symbol}"
#                 question_id = chat_service.save_user_question(
#                     session_id=chat_request.session_id,
#                     created_at=datetime.now(),
#                     created_by=user_id,
#                     content=question_content
#                 )
                
#                 chat_service.save_assistant_response(
#                     session_id=chat_request.session_id,
#                     created_at=datetime.now(),
#                     question_id=question_id,
#                     content=analysis_result["analysis"],
#                     response_time=0.1 
#                 )
#             except Exception as e:
#                 logger.error(f"Error saving to chat history: {str(e)}")

#         return FundamentalAnalysisChatResponse(
#             status="200",
#             message="Financial analysis completed successfully",
#             data=response_data
#         )
        
#     except Exception as e:
#         logger.error(f"Analysis error for {symbol}: {str(e)}")
#         return FundamentalAnalysisChatResponse(
#             status="500",
#             message=f"Error analyzing financial data: {str(e)}",
#             data={
#                 "symbol": symbol,
#                 # "period": period,
#                 "interpretation": None
#             }
#         )

@router.post("/fundamental/advanced-analysis",
            response_model=FundamentalAnalysisChatResponse,
            summary="Get comprehensive fundamental analysis with AI insights")
async def get_financial_growth_with_analysis(
    request: Request,
    chat_request: FundamentalAnalysisChatRequest = Body(...),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Get COMPREHENSIVE fundamental analysis with AI-powered insights and memory integration.
    Includes full metrics: valuation, growth, profitability, leverage, cash flow, and risk.
    
    Enhanced with:
    - Memory context from previous analyses
    - Importance scoring for long-term retention
    - Tool memory integration
    """
    user_id = getattr(request.state, "user_id", None)
    symbol = chat_request.symbol
    model_name = chat_request.model_name
    provider_type = chat_request.provider_type
    
    try:
        # 1. Get memory context BEFORE cache check
        context = ""
        memory_stats = {}
        
        if chat_request.session_id and user_id:
            try:
                query_text = chat_request.question_input or f"Comprehensive fundamental analysis for {symbol}"
                
                # Get conversation memory context
                context, memory_stats = await memory_manager.get_relevant_context(
                    session_id=chat_request.session_id,
                    user_id=user_id,
                    current_query=query_text,
                    llm_provider=llm_generator,
                    max_short_term=5,  # Recent fundamental analyses
                    max_long_term=3    # Important historical fundamental insights
                )
                
                logger.info(f"Retrieved memory context for fundamental analysis: {memory_stats}")
            except Exception as e:
                logger.error(f"Error getting memory context: {e}")
        
        # 2. Check cache for complete analysis
        cache_key = f"fundamental_complete_{symbol.upper()}_{model_name}"
        
        if redis_client:
            try:
                cached_data = await redis_client.get(cache_key)
                if cached_data:
                    logger.info(f"Cache HIT for complete fundamental analysis: {cache_key}")
                    cached_response = json.loads(cached_data)
                    # Add memory stats to cached response
                    cached_response["memory_stats"] = memory_stats
                    return FundamentalAnalysisChatResponse(
                        status="200",
                        message="Comprehensive fundamental analysis completed successfully (cached)",
                        data=cached_response
                    )
            except Exception as e:
                logger.error(f"Cache error: {e}")
        
        # 3. Generate comprehensive fundamental data with context
        logger.info(f"Generating comprehensive fundamental analysis for {symbol}")
        
        comprehensive_data = await fundamental_handler.generate_comprehensive_fundamental_data(
            symbol=symbol,
            tool_service=tool_call_service,
        )
        
        fundamental_report = comprehensive_data["fundamental_report"]
        growth_data_list = comprehensive_data["growth_data"]
        
        # 4. Create AI analysis prompt with memory context
        analysis_prompt = fundamental_handler.create_comprehensive_analysis_prompt(
            symbol=symbol,
            report=fundamental_report,
            growth_data=growth_data_list,
            memory_context=context,  # Pass memory context
            user_question=chat_request.question_input,
            target_language=chat_request.target_language
        )
        
        # 5. Generate AI analysis with enhanced system prompt
        messages = [
            {
                "role": "system", 
                "content": """You are an expert fundamental analyst with 20+ years experience and memory of previous analyses.

Consider previous analyses and conversations when relevant.
Focus on:
- Data-driven analysis with specific metrics
- Clear investment recommendations based on historical patterns
- Risk/reward assessment considering past performance
- Actionable insights for investors
- Professional tone but easy to understand
- Make presentation more attractive by adding appropriate icons
- Briefly explain clearly, annotate the meaning of indicators affecting the market
- Reference relevant previous insights when applicable"""
            },
            {"role": "user", "content": analysis_prompt}
        ]
        
        # Get API Key and generate response
        api_key = ModelProviderFactory._get_api_key(provider_type)
        
        response = await fundamental_handler.llm_provider.generate_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key,
            temperature=0.3
        )
        
        analysis_text = response.get("content", "Analysis generation failed.")
        
        # 6. Calculate investment score
        investment_score = calculate_investment_score(fundamental_report)
        
        # 7. Analyze conversation importance
        importance_score = 0.6  # Default for fundamental analysis
        
        if chat_request.session_id and user_id:
            try:
                analysis_model = "gpt-4.1-nano" if provider_type == ProviderType.OPENAI else model_name
                
                importance_score = await analyze_conversation_importance(
                    query=query_text,
                    response=analysis_text,
                    llm_provider=llm_generator,
                    model_name=analysis_model,
                    provider_type=provider_type
                )
                
                # Boost importance for significant investment ratings
                rating = get_rating_from_score(investment_score["total_score"])
                if rating in ["STRONG BUY", "STRONG SELL"]:
                    importance_score = min(1.0, importance_score + 0.2)
                
                logger.info(f"Fundamental analysis importance score: {importance_score}")
                
            except Exception as e:
                logger.error(f"Error analyzing conversation importance: {e}")
        
        # 8. Store conversation in memory system
        if chat_request.session_id and user_id:
            try:
                # Extract key fundamental metrics for metadata
                metadata = {
                    "type": "fundamental_analysis_advanced",
                    "symbol": symbol,
                    "investment_score": investment_score,
                    "rating": get_rating_from_score(investment_score["total_score"]),
                    "pe_ratio": fundamental_report.get("valuation", {}).get("pe"),
                    "revenue_growth": fundamental_report.get("growth", {}).get("revenue_yoy"),
                    "profit_margin": fundamental_report.get("profitability", {}).get("profit_margin"),
                    "debt_to_equity": fundamental_report.get("leverage", {}).get("debt_to_equity"),
                    "free_cash_flow": fundamental_report.get("cash_flow", {}).get("free_cash_flow"),
                    "roe": fundamental_report.get("profitability", {}).get("roe"),
                    "current_ratio": fundamental_report.get("leverage", {}).get("current_ratio"),
                    "analysis_type": "comprehensive"
                }
                
                # Store in conversation memory
                await memory_manager.store_conversation_turn(
                    session_id=chat_request.session_id,
                    user_id=user_id,
                    query=query_text,
                    response=analysis_text,
                    metadata=metadata,
                    importance_score=importance_score
                )
                
                # Save to chat history (existing logic)
                question_id = chat_service.save_user_question(
                    session_id=chat_request.session_id,
                    created_at=datetime.now(),
                    created_by=user_id,
                    content=query_text
                )
                
                chat_service.save_assistant_response(
                    session_id=chat_request.session_id,
                    created_at=datetime.now(),
                    question_id=question_id,
                    content=analysis_text,  # Store just the analysis text
                    response_time=0.1
                )
            except Exception as e:
                logger.error(f"Error saving to memory/chat history: {str(e)}")
        
        # 9. Prepare comprehensive response with memory stats
        response_data = {
            "symbol": symbol,
            "interpretation": analysis_text,
            "quick_summary": {
                "rating": get_rating_from_score(investment_score["total_score"]),
                "key_strengths": extract_strengths(fundamental_report),
                "key_risks": extract_risks(fundamental_report),
                "action": get_action_recommendation(investment_score["total_score"])
            },
            "memory_stats": memory_stats
        }
        
        # 10. Cache the complete response
        if redis_client:
            try:
                await redis_client.set(
                    cache_key,
                    json.dumps(response_data),
                    ex=1800  # 30 minutes
                )
            except Exception as e:
                logger.error(f"Cache set error: {e}")
        
        return FundamentalAnalysisChatResponse(
            status="200",
            message="Comprehensive fundamental analysis completed successfully",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"Fundamental analysis error for {symbol}: {e}")
        return FundamentalAnalysisChatResponse(
            status="500",
            message=f"Analysis failed: {str(e)}",
            data=None
        )

@router.post("/fundamental/advanced-analysis/stream",
            summary="Get comprehensive fundamental analysis with AI insights (STREAMING)")
async def get_financial_growth_with_analysis_stream(
    request: Request,
    chat_request: FundamentalAnalysisChatRequest = Body(...),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Get COMPREHENSIVE fundamental analysis with AI-powered insights (STREAMING VERSION).
    Includes full metrics: valuation, growth, profitability, leverage, cash flow, and risk.
    """
    async def format_sse():
        try:
            user_id = getattr(request.state, "user_id", None)
            symbol = chat_request.symbol
            model_name = chat_request.model_name
            provider_type = chat_request.provider_type
            
            # Save user question first
            question_id = None
            if chat_request.session_id and user_id:
                try:
                    question_content = chat_request.question_input or f"Comprehensive fundamental analysis for {symbol}"
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
            
            # Get conversation memory context
            context = ""
            memory_stats = {}
            
            if chat_request.session_id and user_id:
                try:
                    query_text = chat_request.question_input or f"Comprehensive fundamental analysis for {symbol}"
                    
                    context, memory_stats = await memory_manager.get_relevant_context(
                        session_id=chat_request.session_id,
                        user_id=user_id,
                        current_query=query_text,
                        llm_provider=llm_generator,
                        max_short_term=5,
                        max_long_term=3
                    )
                    
                    logger.info(f"Retrieved memory context for fundamental streaming: {memory_stats}")
                except Exception as e:
                    logger.error(f"Error getting memory: {e}")

            enhanced_history = ""
            if context:
                enhanced_history = f"{context}\n\n"
            if chat_history:
                enhanced_history += f"[Conversation History]\n{chat_history}"
            
            # Generate comprehensive fundamental data (keep existing logic)
            logger.info(f"Generating comprehensive fundamental analysis for {symbol}")
            
            comprehensive_data = await fundamental_handler.generate_comprehensive_fundamental_data(
                symbol=symbol,
                tool_service=tool_call_service
            )
            
            fundamental_report = comprehensive_data["fundamental_report"]
            growth_data_list = comprehensive_data["growth_data"]
            
            # Get API Key
            api_key = ModelProviderFactory._get_api_key(provider_type)
            
            # Track full response for saving to memory
            full_response = []
            
            # Stream AI analysis with enhanced context
            async for chunk in fundamental_handler.stream_comprehensive_analysis(
                symbol=symbol,
                report=fundamental_report,
                growth_data=growth_data_list,
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key,
                chat_history=enhanced_history,  
                user_question=chat_request.question_input,
                target_language=chat_request.target_language
            ):
                if chunk:
                    full_response.append(chunk)
                    yield f"{json.dumps({'content': chunk})}\n\n"
            
            # Join full response
            analysis_text = ''.join(full_response)
            
            # Calculate investment score (keep existing logic)
            investment_score = calculate_investment_score(fundamental_report)
            
            # Prepare quick summary (keep existing logic)
            quick_summary = {
                "rating": get_rating_from_score(investment_score["total_score"]),
                "key_strengths": extract_strengths(fundamental_report),
                "key_risks": extract_risks(fundamental_report),
                "action": get_action_recommendation(investment_score["total_score"])
            }
            
            # Analyze conversation importance
            importance_score = 0.5
            
            if chat_request.session_id and user_id:
                try:
                    analysis_model = "gpt-4.1-nano" if provider_type == ProviderType.OPENAI else model_name
                    
                    importance_score = await analyze_conversation_importance(
                        query=chat_request.question_input or f"Comprehensive fundamental analysis for {symbol}",
                        response=analysis_text,
                        llm_provider=llm_generator,
                        model_name=analysis_model,
                        provider_type=provider_type
                    )
                    
                    # Boost for significant ratings
                    if quick_summary["rating"] in ["STRONG BUY", "STRONG SELL"]:
                        importance_score = min(1.0, importance_score + 0.2)
                    
                    logger.info(f"Fundamental stream importance: {importance_score}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing conversation: {e}")
            
            # Store in conversation memory
            if chat_request.session_id and user_id and analysis_text:
                try:
                    metadata = {
                        "type": "fundamental_analysis_advanced",
                        "symbol": symbol,
                        "investment_score": investment_score,
                        "rating": quick_summary["rating"],
                        "pe_ratio": fundamental_report.get("valuation", {}).get("pe"),
                        "revenue_growth": fundamental_report.get("growth", {}).get("revenue_yoy"),
                        "profit_margin": fundamental_report.get("profitability", {}).get("profit_margin"),
                        "debt_to_equity": fundamental_report.get("leverage", {}).get("debt_to_equity"),
                        "analysis_type": "comprehensive_streaming"
                    }
                    
                    await memory_manager.store_conversation_turn(
                        session_id=chat_request.session_id,
                        user_id=user_id,
                        query=chat_request.question_input or f"Comprehensive fundamental analysis for {symbol}",
                        response=analysis_text,
                        metadata=metadata,
                        importance_score=importance_score
                    )
                    
                    # Prepare response data for chat history
                    response_data = {
                        "symbol": symbol,
                        "interpretation": analysis_text,
                        "quick_summary": quick_summary
                    }
                    
                    chat_service.save_assistant_response(
                        session_id=chat_request.session_id,
                        created_at=datetime.now(),
                        question_id=question_id,
                        content=json.dumps(response_data),
                        response_time=0.1
                    )

                    trigger_summary_update_nowait(session_id=chat_request.session_id, user_id=user_id)

                except Exception as e:
                    logger.error(f"Error saving to chat history: {str(e)}")
            
            yield "[DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Fundamental analysis streaming error for {chat_request.symbol}: {e}")
            yield f"{json.dumps({'error': str(e)})}\n\n"
            yield "[DONE]\n\n"
    
    return StreamingResponse(
        format_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# Helper functions:
def safe_float(value, default=0.0):
    """Safely convert value to float, return default if None or invalid"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_get(data, key, default=None):
    """Safely get value from nested dictionary"""
    if not isinstance(data, dict):
        return default
    return data.get(key, default)

def calculate_investment_score(report: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate investment score based on fundamental metrics"""
    scores = {
        "valuation": 0,
        "growth": 0,
        "profitability": 0,
        "financial_health": 0,
        "cash_flow": 0
    }
    
    # Valuation score (lower P/E is better) - NULL SAFE
    pe = safe_float(safe_get(report.get("valuation", {}), "pe"))
    if pe > 0:  # Only calculate if we have valid P/E
        if pe < 15: 
            scores["valuation"] = 20
        elif pe < 20: 
            scores["valuation"] = 15
        elif pe < 25: 
            scores["valuation"] = 10
        elif pe < 35: 
            scores["valuation"] = 5
        else: 
            scores["valuation"] = 0
    
    # Growth score - NULL SAFE
    rev_growth = safe_float(safe_get(report.get("growth", {}), "rev_cagr_5y"))
    if rev_growth > 20: 
        scores["growth"] = 25
    elif rev_growth > 15: 
        scores["growth"] = 20
    elif rev_growth > 10: 
        scores["growth"] = 15
    elif rev_growth > 5: 
        scores["growth"] = 10
    elif rev_growth > 0:
        scores["growth"] = 5
    
    # Profitability score - NULL SAFE
    roe = safe_float(safe_get(report.get("profitability", {}), "roe"))
    if roe != 0:  # Only if we have valid ROE
        roe_percent = roe * 100 if roe < 1 else roe
        if roe_percent > 20: 
            scores["profitability"] = 20
        elif roe_percent > 15: 
            scores["profitability"] = 15
        elif roe_percent > 10: 
            scores["profitability"] = 10
        elif roe_percent > 0:
            scores["profitability"] = 5
    
    # Financial health score - NULL SAFE
    leverage_data = report.get("leverage", {})
    de_ratio = safe_float(safe_get(leverage_data, "de_ratio"))
    current_ratio = safe_float(safe_get(leverage_data, "current_ratio"))
    
    # D/E ratio scoring
    if de_ratio < 0.5: 
        scores["financial_health"] += 10
    elif de_ratio < 1: 
        scores["financial_health"] += 7
    elif de_ratio < 2: 
        scores["financial_health"] += 3
        
    # Current ratio scoring
    if current_ratio > 1.5:
        scores["financial_health"] += 10
    elif current_ratio > 1:
        scores["financial_health"] += 5
    
    # Cash flow score - NULL SAFE
    fcf_yield = safe_float(safe_get(report.get("cashflow", {}), "fcf_yield"))
    if fcf_yield > 5: 
        scores["cash_flow"] = 15
    elif fcf_yield > 3: 
        scores["cash_flow"] = 10
    elif fcf_yield > 1: 
        scores["cash_flow"] = 5
    
    total_score = sum(scores.values())
    
    return {
        "scores": scores,
        "total_score": total_score,
        "max_score": 100
    }

def get_rating_from_score(score: int) -> str:
    """Convert score to rating"""
    if score >= 80: return "STRONG BUY"
    elif score >= 65: return "BUY"
    elif score >= 50: return "HOLD"
    elif score >= 35: return "SELL"
    else: return "STRONG SELL"

def get_action_recommendation(score: int) -> str:
    """Get action recommendation based on score"""
    if score >= 80:
        return "Strong fundamental profile. Consider accumulating on any dips."
    elif score >= 65:
        return "Solid fundamentals. Good for long-term investment."
    elif score >= 50:
        return "Mixed fundamentals. Wait for better entry or hold existing position."
    elif score >= 35:
        return "Weak fundamentals. Consider reducing position or avoiding."
    else:
        return "Poor fundamentals. High risk, consider exit."

def extract_strengths(report: Dict[str, Any]) -> List[str]:
    """Extract key strengths from report - NULL SAFE"""
    strengths = []
    
    # Check growth - NULL SAFE
    rev_cagr = safe_float(safe_get(report.get("growth", {}), "rev_cagr_5y"))
    if rev_cagr > 15: 
        strengths.append(f"Strong revenue growth: {rev_cagr:.1f}% CAGR")
    
    # Check profitability - NULL SAFE
    roe = safe_float(safe_get(report.get("profitability", {}), "roe"))
    if roe != 0:
        roe_percent = roe * 100 if roe < 1 else roe
        if roe_percent > 15:
            strengths.append(f"High ROE: {roe_percent:.1f}%")
    
    # Check leverage - NULL SAFE
    de_ratio = safe_float(safe_get(report.get("leverage", {}), "de_ratio"))
    if de_ratio < 0.5 and de_ratio >= 0:
        strengths.append("Low debt levels provide financial flexibility")
    
    # Check cash flow - NULL SAFE
    fcf_yield = safe_float(safe_get(report.get("cashflow", {}), "fcf_yield"))
    if fcf_yield > 3:
        strengths.append(f"Strong FCF yield: {fcf_yield:.1f}%")
    
    # If no strengths found, add a default message
    if not strengths:
        strengths.append("Company maintains operational stability")
    
    return strengths[:3]  # Return top 3

def extract_risks(report: Dict[str, Any]) -> List[str]:
    """Extract key risks from report - NULL SAFE"""
    risks = []
    
    # Check valuation - NULL SAFE
    pe = safe_float(safe_get(report.get("valuation", {}), "pe"))
    if pe > 30:
        risks.append(f"High valuation: P/E of {pe:.1f}")
    
    # Check growth - NULL SAFE
    rev_cagr = safe_float(safe_get(report.get("growth", {}), "rev_cagr_5y"))
    if rev_cagr < 5:
        risks.append("Slow revenue growth may limit upside")
    
    # Check leverage - NULL SAFE
    de_ratio = safe_float(safe_get(report.get("leverage", {}), "de_ratio"))
    if de_ratio > 1:
        risks.append(f"Elevated debt levels: D/E ratio {de_ratio:.1f}")
    
    # Check profitability - NULL SAFE
    net_margin = safe_float(safe_get(report.get("profitability", {}), "net_margin"))
    if net_margin < 5 and net_margin >= 0:
        risks.append("Low profit margins indicate pricing pressure")
    
    # If no specific risks found, add general market risk
    if not risks:
        risks.append("Subject to general market volatility")
        risks.append("Limited financial data available for analysis")
    
    return risks[:2]  # Return top 2