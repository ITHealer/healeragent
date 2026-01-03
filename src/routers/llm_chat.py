import json
import asyncio
import datetime
from typing import AsyncGenerator, Dict, Any, List, Optional
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.handlers.openai_tool_call_handler import agentic_openai_tool_call
from src.helpers.llm_helper import LLMGeneratorProvider
from src.helpers.llm_chat_helper import (
    llm_streaming_service,
    SSE_HEADERS,
    DEFAULT_HEARTBEAT_SEC
)
from src.helpers.token_counter import TokenCounter
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.handlers.llm_chat_handler import ChatHandler, ChatService, ConversationAnalysis
# from src.agents.memory.memory_manager import MemoryManager
from src.agents.memory.memory_manager import get_memory_manager
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.utils.logger.custom_logging import LoggerMixin
from src.handlers.comprehensive_analysis_handler import ComprehensiveAnalysisHandler
from src.services.heatmap_service import SP500Service


router = APIRouter(prefix="/llm_conversation")

# =============================================================================
# Initialize Instances
# =============================================================================
api_key_auth = APIKeyAuth()
logger_mixin = LoggerMixin()
logger = logger_mixin.logger
chat_service = ChatService()
chat_handler = ChatHandler()
memory_manager = get_memory_manager()
sp500_service = SP500Service()


class PromptRequest(BaseModel):
    prompt: str
    model_name: str
    provider_type: str

class GeneralChatStreamRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    question_input: str = Field(..., description="User's question")
    target_language: Optional[str] = Field(None, description="Target response language")
    model_name: str = Field(default="gpt-4.1-nano", description="LLM model to use")
    collection_name: Optional[str] = Field(default="", description="Collection name for RAG")
    use_multi_collection: Optional[bool] = Field(default=False, description="Use multiple collections")
    provider_type: str = Field(default=ProviderType.OPENAI, description="Provider type")
    enable_thinking: Optional[bool] = Field(default=True, description="Enable extended thinking")

class MarketOverviewStreamRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    question_input: str = Field(..., description="User's question")
    target_language: Optional[str] = Field(None, description="Target response language")
    data: List[Dict[str, Any]] = Field(..., description="Market data items")
    model_name: str = Field(default="gpt-4.1-nano", description="LLM model")
    provider_type: str = Field(default=ProviderType.OPENAI, description="Provider type")

class StockAnalysisStreamRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    symbol: str = Field(..., description="Stock symbol to analyze")
    question_input: str = Field(..., description="User's question")
    target_language: Optional[str] = Field(None, description="Target response language")
    model_name: str = Field(default="gpt-4.1-nano", description="LLM model")
    provider_type: str = Field(default=ProviderType.OPENAI, description="Provider type")

class TrendingStreamRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    question_input: str = Field(..., description="User's question")
    target_language: Optional[str] = Field(None, description="Target response language")
    gainers: List[Dict[str, Any]] = Field(default_factory=list, description="Top gainers")
    losers: List[Dict[str, Any]] = Field(default_factory=list, description="Top losers")
    actives: List[Dict[str, Any]] = Field(default_factory=list, description="Most active")
    model_name: str = Field(default="gpt-4.1-nano", description="LLM model")
    provider_type: str = Field(default=ProviderType.OPENAI, description="Provider type")

class HeatmapStreamRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    question_input: str = Field(..., description="User's question")
    target_language: Optional[str] = Field(None, description="Target response language")
    model_name: str = Field(default="gpt-4.1-nano", description="LLM model")
    provider_type: str = Field(default=ProviderType.OPENAI, description="Provider type")


class ReasoningStreamRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    question_input: str = Field(..., description="User's question")
    target_language: Optional[str] = Field(None, description="Target response language")
    model_name: str = Field(default="gpt-4.1-nano", description="LLM model")
    collection_name: str = Field(default="", description="Collection name for RAG")
    use_multi_collection: bool = Field(default=False, description="Use multiple collections")
    enable_thinking: bool = Field(default=True, description="Enable extended thinking")
    provider_type: str = Field(default=ProviderType.OPENAI, description="Provider type")
    reply_to_text: Optional[str] = Field(None, description="Quoted text user is replying to")

# =============================================================================
# Helper Functions
# =============================================================================
async def analyze_conversation_importance(
    query: str,
    response: str,
    llm_provider: LLMGeneratorProvider,
    model_name: str,
    provider_type: str
) -> float:
    """
    Use LLM to analyze conversation importance and extract metadata for financial chatbot.
    
    Args:
        query: User's question
        response: Assistant's response
        llm_provider: LLM provider instance
        model_name: Model to use for analysis
        provider_type: Provider type (openai, ollama, etc.)
        
    Returns:
        Importance score between 0.0 and 1.0
    """
    try:
        # Normalize provider type
        if isinstance(provider_type, str):
            provider_type_lower = provider_type.lower()
            if provider_type_lower == "openai":
                provider_enum = ProviderType.OPENAI
            elif provider_type_lower == "ollama":
                provider_enum = ProviderType.OLLAMA
            else:
                provider_enum = ProviderType.OPENAI  # Default fallback
        else:
            provider_enum = provider_type
        
        # Build analysis prompt with explicit JSON instruction
        analysis_prompt = f"""Analyze this financial conversation and provide importance score (0.0-1.0):

User Query: {query}
Assistant Response: {response[:1000]}...

Score based on:
- Financial relevance and complexity
- Actionable insights provided
- Educational/strategic value
- Whether it's follow-up or references context

Focus on: trading, investment, market analysis, portfolio, risk management.

IMPORTANT: Respond with ONLY a valid JSON object in this exact format (no markdown, no explanation):
{{"importance_score": 0.8, "reasoning": "Brief explanation"}}"""

        messages = [
            {
                "role": "system", 
                "content": "You are a financial conversation analyst. Always respond with valid JSON only."
            },
            {
                "role": "user", 
                "content": analysis_prompt
            }
        ]
        
        # Get API key
        api_key = ModelProviderFactory._get_api_key(provider_enum)
        
        # Prepare structured output format (only for OpenAI)
        structured_format = None
        if provider_enum == ProviderType.OPENAI:
            structured_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "importance_analysis",
                    "schema": ConversationAnalysis.model_json_schema(),
                    "strict": False 
                }
            }
        
        # Call LLM
        llm_response = await llm_provider.generate_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_enum,
            api_key=api_key,
            temperature=0.1,
            response_format=structured_format
        )
        
        # Get content from response
        content = llm_response.get("content", "")
        
        # Check if content is empty
        if not content or not content.strip():
            logger.warning("Empty response from LLM for importance analysis")
            return 0.5  # Default medium importance
        
        # Clean content (remove markdown code blocks if present)
        content_cleaned = content.strip()
        if content_cleaned.startswith("```json"):
            content_cleaned = content_cleaned.replace("```json", "").replace("```", "").strip()
        elif content_cleaned.startswith("```"):
            content_cleaned = content_cleaned.replace("```", "").strip()
        
        # Try to parse as JSON
        try:
            parsed_data = json.loads(content_cleaned)
            
            # Validate with Pydantic if possible
            try:
                analysis = ConversationAnalysis.model_validate(parsed_data)
                importance_score = analysis.importance_score
            except Exception:
                # Fallback to direct dict access
                importance_score = float(parsed_data.get("importance_score", 0.5))
            
            # Validate score range
            importance_score = max(0.0, min(1.0, importance_score))
            
            logger.info(f"Analyzed importance: {importance_score}")
            return importance_score
            
        except json.JSONDecodeError as je:
            # JSON parsing failed - try regex fallback
            logger.warning(f"JSON decode failed for content: {content_cleaned[:100]}... Error: {je}")
            
            # Try to extract score using regex
            import re
            score_match = re.search(r'"importance_score"\s*:\s*([0-9.]+)', content_cleaned)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    score = max(0.0, min(1.0, score))
                    logger.info(f"Extracted importance score via regex: {score}")
                    return score
                except ValueError:
                    pass
            
            # Final fallback based on response length and keywords
            score = 0.5  # Default medium
            
            # Simple heuristic: check for financial keywords
            financial_keywords = ['trade', 'invest', 'portfolio', 'stock', 'market', 'risk', 'analysis', 'buy', 'sell']
            keyword_count = sum(1 for keyword in financial_keywords if keyword.lower() in response.lower())
            
            if keyword_count >= 3:
                score = 0.7
            elif keyword_count >= 1:
                score = 0.6
            
            logger.info(f"Using fallback importance score: {score}")
            return score
        
    except Exception as e:
        logger.error(f"Error analyzing conversation importance with LLM: {e}")
        # Return default score instead of failing
        return 0.5


def save_user_question(session_id: str, user_id: int, content: str) -> Optional[int]:
    """
    Save user question to chat history.
    """
    try:
        return chat_service.save_user_question(
            session_id=session_id,
            created_at=datetime.datetime.now(),
            created_by=user_id,
            content=content
        )
    except Exception as e:
        logger.error(f"Error saving question: {e}")
        return None

async def ensure_memory_collections(session_id: str, user_id: int) -> None:
    """Ensure memory collections exist for session."""
    if session_id and user_id:
        try:
            await memory_manager.ensure_session_memory_collections(
                session_id=session_id,
                user_id=user_id
            )
        except Exception as e:
            logger.warning(f"Error ensuring memory collections: {e}")

# =============================================================================
# TOOLS CALL ENDPOINT
# =============================================================================
@router.post("/tools/openai_tool_call")
async def openai_tool_call(request: PromptRequest):
    """Tool calling endpoint."""
    try:
        result = await agentic_openai_tool_call(request.prompt, request.model_name, request.provider_type)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI tool call failed: {str(e)}")


@router.post("/chat/provider/reasoning/stream", response_description="ReAct/CoT Reasoning Streaming")
async def reasoning_stream(
    request: Request,
    data: ReasoningStreamRequest,
    _api_key: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Stream ReAct/CoT reasoning responses."""
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    clean_thinking = not data.enable_thinking
    
    async def stream_with_heartbeat():
        
        pending_task: Optional[asyncio.Task] = None
        base_generator = None
        heartbeat_sec = DEFAULT_HEARTBEAT_SEC
        
        try:
            base_generator = chat_handler.handle_chat_provider_reasoning_reply_text_stream(
                session_id=data.session_id,
                question_input=data.question_input,
                system_language=data.target_language,
                model_name=data.model_name,
                collection_name=data.collection_name,
                provider_type=data.provider_type,
                user_id=user_id,
                organization_id=organization_id,
                use_multi_collection=data.use_multi_collection,
                clean_thinking=clean_thinking,
                enable_thinking=data.enable_thinking,
                reply_to_text=data.reply_to_text  
            )
            
            # Stream with heartbeat
            while True:
                if pending_task is None:
                    pending_task = asyncio.create_task(anext(base_generator))
                
                # Wait with timeout - SAFE: don't cancel on timeout
                done, _ = await asyncio.wait({pending_task}, timeout=heartbeat_sec)
                
                if done:
                    try:
                        chunk = pending_task.result()
                    except StopAsyncIteration:
                        break
                    except Exception as e:
                        logger.error(f"Reasoning stream error: {e}")
                        yield f"{json.dumps({'error': str(e)})}\n\n"
                        break
                    finally:
                        pending_task = None
                    
                    # Format chunk as SSE
                    if chunk:
                        response = {"content": chunk}
                        yield f"{json.dumps(response)}\n\n"
                        await asyncio.sleep(0)  # Yield control for flush
                else:
                    # Timeout - send heartbeat comment (SSE spec)
                    yield ": heartbeat\n\n"
            
            yield "[DONE]\n\n"
            
        except asyncio.CancelledError:
            logger.info(f"Reasoning stream cancelled: session={data.session_id}")
            raise
        except Exception as e:
            logger.error(f"Error in reasoning stream: {e}")
            yield f"{json.dumps({'error': str(e)})}\n\n"
            yield "[DONE]\n\n"
        finally:
            # Clean up pending task
            if pending_task is not None and not pending_task.done():
                pending_task.cancel()
                try:
                    await pending_task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
            
            # Close base generator
            if base_generator is not None and hasattr(base_generator, 'aclose'):
                try:
                    await base_generator.aclose()
                except Exception:
                    pass
    
    return StreamingResponse(
        stream_with_heartbeat(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )


@router.post("/general/stream")
async def general_stream_api(
    request: Request,
    data: GeneralChatStreamRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Stream general chat response."""
    user_id = getattr(request.state, "user_id", None)
    
    # Save question
    question_id = save_user_question(data.session_id, user_id, data.question_input)
    
    return StreamingResponse(
        llm_streaming_service.stream_general_chat(
            content=data.question_input,
            session_id=data.session_id,
            user_id=user_id,
            model_name=data.model_name,
            provider_type=data.provider_type,
            target_language=data.target_language,
            enable_thinking=data.enable_thinking,
            heartbeat_sec=DEFAULT_HEARTBEAT_SEC,
            question_id=question_id,
            store_to_memory=True
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )

# =============================================================================
# MARKET OVERVIEW ENDPOINTS
# =============================================================================
@router.post("/market/overview/stream")
async def market_overview_stream_api(
    request: Request,
    data: MarketOverviewStreamRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Stream market overview analysis."""
    user_id = getattr(request.state, "user_id", None)
 
    question_content = data.question_input or f"Market overview for {len(data.data)} symbols"
    question_id = save_user_question(data.session_id, user_id, question_content)
    
    return StreamingResponse(
        llm_streaming_service.stream_market_overview(
            market_data=data.data,
            question_input=data.question_input,
            session_id=data.session_id,
            user_id=user_id,
            model_name=data.model_name,
            provider_type=data.provider_type,
            target_language=data.target_language,
            heartbeat_sec=DEFAULT_HEARTBEAT_SEC,
            question_id=question_id,
            store_to_memory=True
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )


# =============================================================================
# TRENDING ENDPOINTS
# =============================================================================
@router.post("/market/trending/stream")
async def trending_analysis_stream_api(
    request: Request, 
    data: TrendingStreamRequest, 
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Stream trending stocks analysis."""
    user_id = getattr(request.state, "user_id", None)
    
    # Ensure memory collections exist
    # if data.session_id and user_id:
    #     await memory_manager.ensure_session_memory_collections(
    #         session_id=data.session_id,
    #         user_id=user_id
    #     )
    
    question_content = data.question_input or "Trending stocks analysis"
    question_id = save_user_question(data.session_id, user_id, question_content)
    
    return StreamingResponse(
        llm_streaming_service.stream_trending_analysis(
            gainers=data.gainers or [], 
            losers=data.losers or [],
            actives=data.actives or [],
            question_input=data.question_input,
            session_id=data.session_id,
            user_id=user_id,
            model_name=data.model_name,
            provider_type=data.provider_type,
            target_language=data.target_language,
            heartbeat_sec=DEFAULT_HEARTBEAT_SEC,
            question_id=question_id,
            store_to_memory=True
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )


# =============================================================================
# STOCK ANALYSIS ENDPOINTS
# =============================================================================
@router.post("/analysis/stream")
async def stock_analysis_stream_api(
    request: Request,
    data: StockAnalysisStreamRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Stream stock technical analysis."""
    user_id = getattr(request.state, "user_id", None)
    
    # Check memory collections exist
    # if data.session_id and user_id:
    #     await memory_manager.ensure_session_memory_collections(
    #         session_id=data.session_id,
    #         user_id=user_id
    #     )

    # Save question
    question_content = data.question_input or f"Analysis for {data.symbol}"
    question_id = save_user_question(data.session_id, user_id, question_content)

    analysis_handler = ComprehensiveAnalysisHandler()
    try:
        analysis_data = await analysis_handler.perform_comprehensive_analysis(
            symbol=data.symbol
        )
    except Exception as e:
        logger.error(f"Analysis failed for {data.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    return StreamingResponse(
        llm_streaming_service.stream_stock_analysis(
            analysis_data=analysis_data,
            user_query=data.question_input,
            session_id=data.session_id,
            user_id=user_id,
            model_name=data.model_name,
            provider_type=data.provider_type,
            target_language=data.target_language,
            heartbeat_sec=DEFAULT_HEARTBEAT_SEC,
            question_id=question_id,
            store_to_memory=True
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )


# =============================================================================
# HEATMAP ENDPOINTS
# =============================================================================
@router.post("/market/heatmap/stream")
async def heatmap_analysis_stream_api(
    request: Request, 
    data: HeatmapStreamRequest, 
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Stream S&P 500 heatmap analysis."""
    user_id = getattr(request.state, "user_id", None)
    
    # Check memory collections exist
    # if data.session_id and user_id:
    #     await memory_manager.ensure_session_memory_collections(
    #         session_id=data.session_id,
    #         user_id=user_id
    #     )
    
    question_content = data.question_input or "S&P 500 heatmap analysis"
    question_id = save_user_question(data.session_id, user_id, question_content)
    
    # Fetch S&P 500 heatmap data
    try:
        heatmap_data = await sp500_service.get_sp500_constituents_with_quotes()
        heatmap_data_dicts = []
        if heatmap_data:
            heatmap_data_dicts = [
                item.model_dump() if hasattr(item, 'model_dump') else item 
                for item in heatmap_data
            ]
    except Exception as e:
        logger.error(f"Failed to fetch heatmap data: {e}")
        raise HTTPException(status_code=500, detail=f"Heatmap data fetch failed: {str(e)}")
    
    return StreamingResponse(
        llm_streaming_service.stream_heatmap_analysis(
            heatmap_data=heatmap_data_dicts,
            question_input=data.question_input,
            session_id=data.session_id,
            user_id=user_id,
            model_name=data.model_name,
            provider_type=data.provider_type,
            target_language=data.target_language,
            heartbeat_sec=DEFAULT_HEARTBEAT_SEC,
            question_id=question_id,
            store_to_memory=True
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )
