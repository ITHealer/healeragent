import json
import asyncio
import datetime
from pydantic import BaseModel, validator, Field
from typing import List, Dict, Any, Tuple, Optional
from collections.abc import AsyncGenerator
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends, Request, HTTPException

from src.schemas.response import (
    GeneralChatBot, 
    AnalysisResponsePayload, 
    MarketOverviewPayload, 
    StockStringPayload, 
    PromptRequest
)
from src.schemas.response import StockHeatmapPayload
from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.llm_chat_helper import (
    analyze_stock_heatmap, 
    analyze_stock_heatmap_stream, 
    analyze_stock,
    analyze_stock_stream,
    analyze_stock_trending,
    analyze_stock_trending_stream,
    analyze_market_overview, 
    analyze_market_overview_stream,
    general_chat_bot, 
    general_chat_bot_stream
)
from src.handlers.comprehensive_analysis_handler import ComprehensiveAnalysisHandler
from src.handlers.openai_tool_call_handler import agentic_openai_tool_call
from src.handlers.llm_chat_handler import ChatHandler, ChatMessageHistory, ChatService
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.helpers.llm_helper import LLMGeneratorProvider
from src.helpers.streaming_tool_helper import StreamingToolHelper
from src.providers.provider_factory import ProviderType
from src.services.heatmap_service import SP500Service
from src.agents.memory.memory_manager import MemoryManager
from src.providers.provider_factory import ModelProviderFactory

from src.services.background_tasks import trigger_summary_update_nowait


from src.agents.memory.core_memory import CoreMemory
from src.agents.memory.memory_update_agent import MemoryUpdateAgent
from src.helpers.context_assembler import ContextAssembler
from src.helpers.token_counter import TokenCounter
from src.helpers.system_prompts import get_system_message_general_chat


router = APIRouter(prefix="/llm_conversation")

# Initialize Instance
api_key_auth = APIKeyAuth()
logger_mixin = LoggerMixin()
logger = logger_mixin.logger
chat_service = ChatService()
chat_handler = ChatHandler()
llm_provider = LLMGeneratorProvider()
streaming_helper = StreamingToolHelper()
memory_manager = MemoryManager()
sp500_service = SP500Service()
analysis_handler = ComprehensiveAnalysisHandler()


# Initialize Core Memory components
core_memory_manager = CoreMemory()
context_assembler = ContextAssembler()
token_counter = TokenCounter()
memory_update_agent = MemoryUpdateAgent()


class ConversationAnalysis(BaseModel):
    importance_score: float = Field(ge=0.0, le=1.0, description="Importance score from 0.0 to 1.0")

# API /chat/provider
class ChatRequestWithProvider(BaseModel):
    session_id: str
    question_input: str
    target_language: str = None
    model_name: str = 'gpt-4.1-nano'
    collection_name: str = ''
    use_multi_collection: bool = False
    enable_thinking: bool = True
    provider_type: str = Field(ProviderType.OPENAI, description="Provider type: ollama, openai, gemini")

class ChatRequestResoning(BaseModel):
    session_id: str
    question_input: str
    target_language: str = None
    model_name: str = 'gpt-4.1-nano'
    collection_name: str = ''
    use_multi_collection: bool = False
    enable_thinking: bool = True
    provider_type: str = Field(ProviderType.OPENAI, description="Provider type: ollama, openai, gemini")
    reply_to_text: Optional[str] = Field(None, description="Quoted text that user is replying to")


# Helper function to format SSE response
async def format_sse(generator, session_id) -> AsyncGenerator[str, None]:
    async for chunk in generator:
        if chunk:
            response = { 
                # "id": session_id,
                # "role": "assistant", 
                "content": chunk
            }
            yield f"{json.dumps(response)}\n\n" # f"data: {json.dumps({'content': chunk})}\n\n"
    yield "[DONE]\n\n"

# ======================= DEFINE API ENDPOINTS =======================

# TOOLS CALL
@router.post("/tools/openai_tool_call")
async def openai_tool_call(request: PromptRequest):
    """
    Tool calling.
    """
    try:
        result = await agentic_openai_tool_call(request.prompt, request.model_name, request.provider_type)
        return result
    except ValueError as ve: # Catch specific value errors (e.g., missing API key)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI tool call failed: {str(e)}")


# ReAct+CoT
# @router.post("/chat/provider/reasoning/stream", response_description="Stream chat with ReAct and Chain of Thought")
# async def chat_provider_reasoning_stream(
#     request: Request,
#     chat_request: ChatRequestWithProvider,
#     _api_key: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
# ):
#     """Stream ReAct/CoT responses via Server-Sent Events (SSE)."""

#     # Extract authenticated context from request scope
#     user_id = getattr(request.state, "user_id", None)
#     organization_id = getattr(request.state, "organization_id", None)
    
#     # Provider selection comes from the client payload
#     provider_type = chat_request.provider_type
    
#     # Clean thinking is the opposite of enable_thinking
#     clean_thinking = not chat_request.enable_thinking
    
#     return StreamingResponse(
#         format_sse(
#             chat_handler.handle_chat_provider_reasoning_stream(
#                 session_id=chat_request.session_id,
#                 question_input=chat_request.question_input,
#                 system_language=chat_request.target_language,
#                 model_name=chat_request.model_name,
#                 collection_name=chat_request.collection_name,
#                 provider_type=provider_type,
#                 user_id=user_id,
#                 organization_id=organization_id,
#                 use_multi_collection=chat_request.use_multi_collection,
#                 clean_thinking=clean_thinking,
#                 enable_thinking=chat_request.enable_thinking
#             ),
#             session_id=chat_request.session_id
#         ),
#         media_type="text/event-stream"
#     )


@router.post("/chat/provider/reasoning/stream", response_description="Stream chat with ReAct and Chain of Thought")
async def chat_provider_reasoning_with_reply_stream(
    request: Request,
    chat_request: ChatRequestResoning,
    _api_key: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Stream ReAct/CoT responses via Server-Sent Events (SSE)."""

    # Extract authenticated context from request scope
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    # Provider selection comes from the client payload
    provider_type = chat_request.provider_type
    
    # Clean thinking is the opposite of enable_thinking
    clean_thinking = not chat_request.enable_thinking
    
    return StreamingResponse(
        format_sse(
            chat_handler.handle_chat_provider_reasoning_reply_text_stream(
                session_id=chat_request.session_id,
                question_input=chat_request.question_input,
                system_language=chat_request.target_language,
                model_name=chat_request.model_name,
                collection_name=chat_request.collection_name,
                provider_type=provider_type,
                user_id=user_id,
                organization_id=organization_id,
                use_multi_collection=chat_request.use_multi_collection,
                clean_thinking=clean_thinking,
                enable_thinking=chat_request.enable_thinking,
                reply_to_text=chat_request.reply_to_text  
            ),
            session_id=chat_request.session_id
        ),
        media_type="text/event-stream"
    )


async def analyze_conversation_importance(
    query: str,
    response: str,
    llm_provider: LLMGeneratorProvider,
    model_name: str,
    provider_type: str
) -> float:
    """
    Use LLM to analyze conversation importance and extract metadata for financial chatbot
    Supports all languages and providers (OpenAI, Ollama, etc.)
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
Assistant Response: {response[:1200]}...

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
    
# =============================================================================
# region General Chat APIs
# =============================================================================  
@router.post("/general")
async def general_api(
    request: Request, 
    data: GeneralChatBot, 
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    try:
        user_id = getattr(request.state, "user_id", None)
        
        # 1. Get relevant context from memory
        context = ""
        memory_stats = {}
        if data.session_id and user_id:
            try:
                context, memory_stats = await memory_manager.get_relevant_context(
                    session_id=data.session_id,
                    user_id=user_id,
                    current_query=data.question_input,
                    llm_provider=llm_provider,
                    max_short_term=5,  # Top 5 relevant recent conversations
                    max_long_term=3    # Top 3 important past conversations
                )
                logger.info(f"Retrieved memory context: {memory_stats}")
            except Exception as e:
                logger.error(f"Error getting memory context: {e}")
        
        # 2. Call LLM with memory context
        response_text = await general_chat_bot(
            data.question_input,
            data.target_language,
            context,
            data.model_name,
            enable_thinking=data.enable_thinking,
            provider_type=data.provider_type
        )
        
        # 3. Analyze conversation with LLM
        importance_score = 0.5
        
        if data.session_id and user_id:
            try:
                analysis_model = "gpt-4.1-nano" if data.provider_type == ProviderType.OPENAI else data.model_name
                
                importance_score = await analyze_conversation_importance(
                    query=data.question_input,
                    response=response_text,
                    llm_provider=llm_provider,
                    model_name=analysis_model,
                    provider_type=data.provider_type
                )
                
                logger.info(f"LLM analysis - Importance: {importance_score}")
                
            except Exception as e:
                logger.error(f"Error analyzing conversation: {e}")

         # 4. Store conversation in memory system
        if data.session_id and user_id:
            try:
                await memory_manager.store_conversation_turn(
                    session_id=data.session_id,
                    user_id=user_id,
                    query=data.question_input,
                    response=response_text,
                    metadata=None,
                    importance_score=importance_score
                )

                question_id = chat_service.save_user_question(
                    session_id=data.session_id,
                    created_at=datetime.datetime.now(),
                    created_by=user_id,
                    content=data.question_input
                )
                
                chat_service.save_assistant_response(
                    session_id=data.session_id,
                    created_at=datetime.datetime.now(),
                    question_id=question_id,
                    content=response_text,
                    response_time=0.1
                )

                # Trigger summary update - NO AWAIT, NO BLOCKING!
                trigger_summary_update_nowait(
                    session_id=data.session_id,
                    user_id=user_id
                )

            except Exception as save_error:
                logger.error(f"Error saving to memory: {str(save_error)}")
        
        return {
            "content": response_text,
            "memory_stats": memory_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/general/stream")
async def general_stream_api(
    request: Request,
    data: GeneralChatBot,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    General chat streaming with enhanced memory integration
    """
    user_id = getattr(request.state, "user_id", None)
    
    question_id = None
    if data.session_id and user_id:
        try:
            question_id = chat_service.save_user_question(
                session_id=data.session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id,
                content=data.question_input
            )
        except Exception as save_error:
            logger.error(f"Error saving user question: {str(save_error)}")
    

    # Get chat history
    chat_history = ""
    if data.session_id:
        try:
            chat_history = ChatMessageHistory.string_message_chat_history(
                session_id=data.session_id
            )
        except Exception as e:
            logger.error(f"Error fetching history: {e}")

    context = ""
    memory_stats = {}
    if data.session_id and user_id:
        try:
            context, memory_stats, document_references = await memory_manager.get_relevant_context(
                session_id=data.session_id,
                user_id=user_id,
                current_query=data.question_input,
                llm_provider=llm_provider,
                max_short_term=5,
                max_long_term=3
            )
            logger.info(f"Retrieved memory context: {memory_stats}")
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
    
    enhanced_history = ""
    if context:
        enhanced_history = f"{context}\n\n"
    if chat_history:
        enhanced_history += f"[Conversation History]\n{chat_history}"
    
    async def format_sse():
        full_response = []
        try:
            # Stream with enhanced context
            async for chunk in general_chat_bot_stream(
                content=data.question_input,
                system_language=data.target_language,
                chat_history=enhanced_history,
                model_name=data.model_name,
                enable_thinking=data.enable_thinking,
                provider_type=data.provider_type
            ):
                if chunk:
                    full_response.append(chunk)
                    yield f"{json.dumps({'content': chunk})}\n\n"
            
            # Join the full response
            complete_response = ''.join(full_response)
            
            # 3. Analyze conversation with LLM
            importance_score = 0.5
            
            if data.session_id and user_id:
                try:
                    analysis_model = "gpt-4.1-nano" if data.provider_type == ProviderType.OPENAI else data.model_name
                    
                    importance_score = await analyze_conversation_importance(
                        query=data.question_input,
                        response=complete_response,
                        llm_provider=llm_provider,
                        model_name=analysis_model,
                        provider_type=data.provider_type
                    )
                    
                    logger.info(f"LLM analysis - Importance: {importance_score}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing conversation: {e}")

            # 4. Store conversation in memory system
            if data.session_id and user_id and complete_response:
                try:
                    await memory_manager.store_conversation_turn(
                        session_id=data.session_id,
                        user_id=user_id,
                        query=data.question_input,
                        response=complete_response,
                        metadata=None,
                        importance_score=importance_score
                    )
                    
                    chat_service.save_assistant_response(
                        session_id=data.session_id,
                        created_at=datetime.datetime.now(),
                        question_id=question_id,
                        content=complete_response,
                        response_time=0.1
                    )

                    trigger_summary_update_nowait(session_id=data.session_id, user_id=user_id)

                except Exception as save_error:
                    logger.error(f"Error saving assistant response: {str(save_error)}")
            
            yield "[DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            yield f"{json.dumps({'error': str(e)})}\n\n"
            yield "[DONE]\n\n"
    
    return StreamingResponse(
        format_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
            "Connection": "keep-alive"
        }
    )

# =============================================================================
# region Market Overview APIs
# =============================================================================
@router.post("/market/overview")
async def analyze_market_overview_api(
    request: Request, 
    data: MarketOverviewPayload, 
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    try:
        user_id = getattr(request.state, "user_id", None)

        # 1. Get relevant context from memory
        # Get chat history
        chat_history = ""
        if data.session_id:
            try:
                chat_history = ChatMessageHistory.string_message_chat_history(
                    session_id=data.session_id
                )
            except Exception as e:
                logger.error(f"Error fetching history: {e}")

        context = ""
        memory_stats = {}
        if data.session_id and user_id:
            try:
                context, memory_stats, document_references = await memory_manager.get_relevant_context(
                    session_id=data.session_id,
                    user_id=user_id,
                    current_query=data.question_input,
                    llm_provider=llm_provider,
                    max_short_term=5,  # Top 5 relevant recent conversations
                    max_long_term=3    # Top 3 important past conversations
                )
                logger.info(f"Retrieved memory context: {memory_stats}")
            except Exception as e:
                logger.error(f"Error getting memory context: {e}")
        
        enhanced_history = ""
        if context:
            enhanced_history = f"{context}\n\n"
        if chat_history:
            enhanced_history += f"[Conversation History]\n{chat_history}"
        
        response_text = await analyze_market_overview(
            data,
            enhanced_history, 
            data.model_name,
            provider_type=data.provider_type
        )

        # 3. Analyze conversation with LLM
        importance_score = 0.5
        
        if data.session_id and user_id:
            try:
                analysis_model = "gpt-4.1-nano" if data.provider_type == ProviderType.OPENAI else data.model_name
                
                importance_score = await analyze_conversation_importance(
                    query=data.question_input,
                    response=response_text,
                    llm_provider=llm_provider,
                    model_name=analysis_model,
                    provider_type=data.provider_type
                )
                
                logger.info(f"LLM analysis - Importance: {importance_score}")
                
            except Exception as e:
                logger.error(f"Error analyzing conversation: {e}")

         # 4. Store conversation in memory system
        if data.session_id and user_id:
            try:
                await memory_manager.store_conversation_turn(
                    session_id=data.session_id,
                    user_id=user_id,
                    query=data.question_input,
                    response=response_text,
                    metadata=None,
                    importance_score=importance_score
                )

                question_id = chat_service.save_user_question(
                    session_id=data.session_id,
                    created_at=datetime.datetime.now(),
                    created_by=user_id,
                    content=data.question_input
                )
                
                chat_service.save_assistant_response(
                    session_id=data.session_id,
                    created_at=datetime.datetime.now(),
                    question_id=question_id,
                    content=response_text,
                    response_time=0.1
                )

                trigger_summary_update_nowait(session_id=data.session_id, user_id=user_id)

            except Exception as save_error:
                logger.error(f"Error saving to memory: {str(save_error)}")
        
        return {
            "interpretation": response_text,
            "memory_stats": memory_stats  
        }
    except Exception as e:
        logger.error(f"Error in market_overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/market/overview/stream")
async def market_overview_stream_api(
    request: Request,
    data: MarketOverviewPayload,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Streaming version of market overview with memory integration
    """
    user_id = getattr(request.state, "user_id", None)
    
    # Save user question
    question_id = None
    if data.session_id and user_id:
        try:
            question_id = chat_service.save_user_question(
                session_id=data.session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id,
                content=data.question_input or f"Market overview for {len(data.data)} symbols"
            )
        except Exception as save_error:
            logger.error(f"Error saving user question: {str(save_error)}")
    
        chat_history = ""
        if data.session_id:
            try:
                chat_history = ChatMessageHistory.string_message_chat_history(
                    session_id=data.session_id
                )
            except Exception as e:
                logger.error(f"Error fetching history: {e}")

        context = ""
        memory_stats = {}
        if data.session_id and user_id:
            try:
                context, memory_stats, document_references = await memory_manager.get_relevant_context(
                    session_id=data.session_id,
                    user_id=user_id,
                    current_query=data.question_input,
                    llm_provider=llm_provider,
                    max_short_term=5,  # Top 5 relevant recent conversations
                    max_long_term=3    # Top 3 important past conversations
                )
                logger.info(f"Retrieved memory context: {memory_stats}")
            except Exception as e:
                logger.error(f"Error getting memory context: {e}")
        
        enhanced_history = ""
        if context:
            enhanced_history = f"{context}\n\n"
        if chat_history:
            enhanced_history += f"[Conversation History]\n{chat_history}"
    
    async def stream_with_memory():
        full_response = []
        
        try:
            async for chunk in analyze_market_overview_stream(
                data,
                enhanced_history,
                data.model_name,
                provider_type=data.provider_type
            ):
                if chunk:
                    full_response.append(chunk)
                    yield f"{json.dumps({'content': chunk})}\n\n"
            
            # After streaming completes, store in memory
            complete_response = ''.join(full_response)
            
            # 3. Analyze conversation with LLM
            importance_score = 0.5
            
            if data.session_id and user_id:
                try:
                    analysis_model = "gpt-4.1-nano" if data.provider_type == ProviderType.OPENAI else data.model_name
                    
                    importance_score = await analyze_conversation_importance(
                        query=data.question_input,
                        response=complete_response,
                        llm_provider=llm_provider,
                        model_name=analysis_model,
                        provider_type=data.provider_type
                    )
                    
                    logger.info(f"LLM analysis - Importance: {importance_score}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing conversation: {e}")

            # 4. Store conversation in memory system
            if data.session_id and user_id and complete_response:
                try:
                    await memory_manager.store_conversation_turn(
                        session_id=data.session_id,
                        user_id=user_id,
                        query=data.question_input,
                        response=complete_response,
                        metadata=None,
                        importance_score=importance_score
                    )
                    
                    # Save assistant response to chat history
                    if question_id:
                        chat_service.save_assistant_response(
                            session_id=data.session_id,
                            created_at=datetime.datetime.now(),
                            question_id=question_id,
                            content=complete_response,
                            response_time=0.1
                        )

                    trigger_summary_update_nowait(session_id=data.session_id, user_id=user_id)

                except Exception as save_error:
                    logger.error(f"Error saving to memory/history: {str(save_error)}")
            
            # Send completion event
            yield "[DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            yield f"{json.dumps({'error': str(e)})}\n\n"
            yield "[DONE]\n\n"
    
    return StreamingResponse(
        stream_with_memory(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
            "Connection": "keep-alive"
        }
    )


# =============================================================================
# region Trending APIs
# =============================================================================
@router.post("/market/trending")
async def analyze_stock_trending_api(
    request: Request, 
    data: StockStringPayload, 
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    try:
        user_id = getattr(request.state, "user_id", None)

        # Ensure memory collections exist
        if data.session_id and user_id:
            await memory_manager.ensure_session_memory_collections(
                session_id=data.session_id,
                user_id=user_id
            )

        # 1. Get relevant context from memory
        # Get chat history
        chat_history = ""
        if data.session_id:
            try:
                chat_history = ChatMessageHistory.string_message_chat_history(
                    session_id=data.session_id
                )
            except Exception as e:
                logger.error(f"Error fetching history: {e}")

        context = ""
        memory_stats = {}
        if data.session_id and user_id:
            try:
                context, memory_stats, document_references = await memory_manager.get_relevant_context(
                    session_id=data.session_id,
                    user_id=user_id,
                    current_query=data.question_input or "Analyze trending: gainers, losers, actives",
                    llm_provider=llm_provider,
                    max_short_term=5,  # Top 5 relevant recent conversations
                    max_long_term=3    # Top 3 important past conversations
                )
                logger.info(f"Retrieved memory context: {memory_stats}")
            except Exception as e:
                logger.error(f"Error getting memory context: {e}")
        
        enhanced_history = ""
        if context:
            enhanced_history = f"{context}\n\n"
        if chat_history:
            enhanced_history += f"[Conversation History]\n{chat_history}"
        
        response_text = await analyze_stock_trending(
            data, 
            enhanced_history, 
            data.model_name,
            provider_type=data.provider_type
        )

        # 3. Analyze conversation with LLM
        importance_score = 0.5
        
        if data.session_id and user_id:
            try:
                analysis_model = "gpt-4.1-nano" if data.provider_type == ProviderType.OPENAI else data.model_name
                
                importance_score = await analyze_conversation_importance(
                    query=data.question_input,
                    response=response_text,
                    llm_provider=llm_provider,
                    model_name=analysis_model,
                    provider_type=data.provider_type
                )
                
                logger.info(f"LLM analysis - Importance: {importance_score}")
                
            except Exception as e:
                logger.error(f"Error analyzing conversation: {e}")

        # 4. Store conversation in memory system
        if data.session_id and user_id:
            try:
                # Prepare metadata
                metadata = {
                    "type": "trending_analysis",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "symbols": [item['symbol'] for category in [data.gainers, data.losers, data.actives] 
                              for item in category if category],
                    "categories": ["gainers", "losers", "actives"],
                    "importance_score": importance_score
                }
                
                await memory_manager.store_conversation_turn(
                    session_id=data.session_id,
                    user_id=user_id,
                    query=data.question_input or "Trending analysis request",
                    response=response_text,
                    metadata=metadata,
                    importance_score=importance_score
                )

                question_id = chat_service.save_user_question(
                    session_id=data.session_id,
                    created_at=datetime.datetime.now(),
                    created_by=user_id,
                    content=data.question_input
                )
                
                chat_service.save_assistant_response(
                    session_id=data.session_id,
                    created_at=datetime.datetime.now(),
                    question_id=question_id,
                    content=response_text,
                    response_time=0.1
                )
                
            except Exception as e:
                logger.error(f"Error saving to memory: {e}")
        
        return {
            "interpretation": response_text,
            "memory_stats": memory_stats,
            "metadata": {
                "importance_score": importance_score,
                "context_used": bool(context),
                "timestamp": datetime.datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error in trending analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/market/trending/stream")
async def analyze_stock_trending_stream_api(
    request: Request, 
    data: StockStringPayload, 
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Stock market trending analysis endpoint with streaming response
    
    This endpoint:
    1. Analyzes gainers, losers, and most active stocks
    2. Uses LLM to provide market insights (STREAMING)
    3. Integrates with memory system for context-aware responses
    4. Saves the conversation to chat history
    """
    user_id = getattr(request.state, "user_id", None)
    
    # Ensure memory collections exist
    if data.session_id and user_id:
        await memory_manager.ensure_session_memory_collections(
            session_id=data.session_id,
            user_id=user_id
        )
    
    # Save user question
    question_id = None
    if data.session_id and user_id:
        try:
            # Create more descriptive question content
            question_content = data.question_input or f"Trending analysis at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            question_id = chat_service.save_user_question(
                session_id=data.session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id,
                content=question_content
            )
        except Exception as save_error:
            logger.error(f"Error saving user question: {str(save_error)}")
    
    # Get chat history
    chat_history = ""
    if data.session_id:
        try:
            chat_history = ChatMessageHistory.string_message_chat_history(
                session_id=data.session_id
            )
        except Exception as e:
            logger.error(f"Error fetching history: {e}")

    context = ""
    memory_stats = {}
    if data.session_id and user_id:
        try:
            context, memory_stats, document_references = await memory_manager.get_relevant_context(
                session_id=data.session_id,
                user_id=user_id,
                current_query=data.question_input or "Analyze trending: gainers, losers, actives",
                llm_provider=llm_provider,
                max_short_term=5,
                max_long_term=3
            )
            logger.info(f"Retrieved memory context: {memory_stats}")
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
    
    enhanced_history = ""
    if context:
        enhanced_history = f"{context}\n\n"
    if chat_history:
        enhanced_history += f"[Conversation History]\n{chat_history}"
    
    async def stream_with_memory():
        full_response = []
        start_time = datetime.datetime.now()
        
        try:
            # Stream the analysis
            async for chunk in analyze_stock_trending_stream(
                data,
                enhanced_history,
                data.model_name,
                provider_type=data.provider_type
            ):
                if chunk:
                    full_response.append(chunk)
                    yield f"{json.dumps({'content': chunk})}\n\n"
            
            # Process complete response
            complete_response = ''.join(full_response)
            response_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Analyze conversation importance
            importance_score = 0.5
            
            if data.session_id and user_id and complete_response:
                try:
                    analysis_model = "gpt-4.1-nano" if data.provider_type == ProviderType.OPENAI else data.model_name
                    
                    importance_score = await analyze_conversation_importance(
                        query=data.question_input or "Trending analysis",
                        response=complete_response,
                        llm_provider=llm_provider,
                        model_name=analysis_model,
                        provider_type=data.provider_type
                    )
                    
                    logger.info(f"Conversation importance: {importance_score}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing importance: {e}")

            # Store conversation
            if data.session_id and user_id and complete_response:
                try:
                    # Extract symbols from data
                    all_symbols = []
                    for category in [data.gainers, data.losers, data.actives]:
                        if category:
                            all_symbols.extend([item['symbol'] for item in category])
                    
                    # Prepare metadata
                    metadata = {
                        "type": "trending_analysis_stream",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "symbols": list(set(all_symbols)),  # Unique symbols
                        "categories": ["gainers", "losers", "actives"],
                        "importance_score": importance_score,
                        "response_time": response_time,
                        "model": data.model_name,
                        "provider": data.provider_type
                    }
                    
                    await memory_manager.store_conversation_turn(
                        session_id=data.session_id,
                        user_id=user_id,
                        query=data.question_input or "Trending analysis request",
                        response=complete_response,
                        metadata=metadata,
                        importance_score=importance_score
                    )
                    
                    # Save to chat history
                    if question_id:
                        chat_service.save_assistant_response(
                            session_id=data.session_id,
                            created_at=datetime.datetime.now(),
                            question_id=question_id,
                            content=complete_response,
                            response_time=response_time
                        )

                    trigger_summary_update_nowait(session_id=data.session_id, user_id=user_id)

                except Exception as e:
                    logger.error(f"Error saving conversation: {e}")
            
            # Send completion event
            yield "[DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            yield f"{json.dumps({'error': str(e)})}\n\n"
            yield "[DONE]\n\n"
    
    return StreamingResponse(
        stream_with_memory(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# =============================================================================
# region Stock Analysis APIs
# =============================================================================    
@router.post("/analysis")
async def stock_analysis_api(
    request: Request,
    data: AnalysisResponsePayload,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Stock analysis with memory integration
    """
    user_id = getattr(request.state, "user_id", None)
    
    try:
        # Perform analysis
        analysis_data = await analysis_handler.perform_comprehensive_analysis(
            symbol=data.symbol,
        )
        
        # 1. Get relevant context from memory
        context = ""
        memory_stats = {}
        if data.session_id and user_id:
            try:
                context, memory_stats, document_references = await memory_manager.get_relevant_context(
                    session_id=data.session_id,
                    user_id=user_id,
                    current_query=data.question_input,
                    llm_provider=llm_provider,
                    max_short_term=5,  # Top 5 relevant recent conversations
                    max_long_term=3    # Top 3 important past conversations
                )
                logger.info(f"Retrieved memory context: {memory_stats}")
            except Exception as e:
                logger.error(f"Error getting memory context: {e}")
            
        # Call LLM
        response_text = await analyze_stock(
            analysis_data, 
            user_query=data.question_input, 
            system_language=data.target_language,
            model_name=data.model_name, 
            chat_history=context,
            provider_type=data.provider_type
        )
        
        # 3. Analyze conversation with LLM
        importance_score = 0.5
        
        if data.session_id and user_id:
            try:
                analysis_model = "gpt-4.1-nano" if data.provider_type == ProviderType.OPENAI else data.model_name
                
                importance_score = await analyze_conversation_importance(
                    query=data.question_input,
                    response=response_text,
                    llm_provider=llm_provider,
                    model_name=analysis_model,
                    provider_type=data.provider_type
                )
                
                logger.info(f"LLM analysis - Importance: {importance_score}")
                
            except Exception as e:
                logger.error(f"Error analyzing conversation: {e}")

        # 4. Store conversation in memory system
        if data.session_id and user_id:
            try:
                await memory_manager.store_conversation_turn(
                    session_id=data.session_id,
                    user_id=user_id,
                    query=data.question_input,
                    response=response_text,
                    metadata=None,
                    importance_score=importance_score
                )
                
                question_id = chat_service.save_user_question(
                    session_id=data.session_id,
                    created_at=datetime.datetime.now(),
                    created_by=user_id,
                    content=data.question_input or f"Analysis for {data.symbol}"
                )
                
                chat_service.save_assistant_response(
                    session_id=data.session_id,
                    created_at=datetime.datetime.now(),
                    question_id=question_id,
                    content=response_text,
                    response_time=0.1
                )

                trigger_summary_update_nowait(session_id=data.session_id, user_id=user_id)

            except Exception as save_error:
                logger.error(f"Error saving to memory/history: {str(save_error)}")
        
        return {
            "symbol": data.symbol,
            "interpretation": response_text,
            # "analysis_data": analysis_data,
            "memory_stats": memory_stats
        }
        
    except Exception as e:
        logger.error(f"Error in stock analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analysis/stream")
async def stock_analysis_stream_api(
    request: Request,
    data: AnalysisResponsePayload,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Stock analysis endpoint with streaming response and memory integration
    
    This endpoint:
    1. Performs comprehensive technical analysis on the stock
    2. Fetches real-time news
    3. Retrieves relevant tool memory context
    4. Uses LLM to provide investment recommendations (STREAMING)
    5. Saves the analysis to tool memory and chat history
    """
    user_id = getattr(request.state, "user_id", None)
    
    # Check memory collections exist
    if data.session_id and user_id:
        await memory_manager.ensure_session_memory_collections(
            session_id=data.session_id,
            user_id=user_id
        )

    # Save user question first
    question_id = None
    if data.session_id and user_id:
        try:
            question_id = chat_service.save_user_question(
                session_id=data.session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id,
                content=data.question_input or f"Analysis for {data.symbol}"
            )
        except Exception as e:
            logger.error(f"Error saving question: {e}")
    
    # Get chat history
    chat_history = ""
    if data.session_id:
        try:
            chat_history = ChatMessageHistory.string_message_chat_history(
                session_id=data.session_id
            )
        except Exception as e:
            logger.error(f"Error fetching history: {e}")

    context = ""
    memory_stats = {}
    if data.session_id and user_id:
        try:
            context, memory_stats, document_references = await memory_manager.get_relevant_context(
                session_id=data.session_id,
                user_id=user_id,
                current_query=data.question_input,
                llm_provider=llm_provider,
                max_short_term=5,
                max_long_term=3
            )
            logger.info(f"Retrieved memory context: {memory_stats}")
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
    
    enhanced_history = ""
    if context:
        enhanced_history = f"{context}\n\n"
    if chat_history:
        enhanced_history += f"[Conversation History]\n{chat_history}"
    
    async def format_sse_with_memory():
        full_response = []
        analysis_data = {}
        start_time = datetime.datetime.now()

        try:
            analysis_handler = ComprehensiveAnalysisHandler()
            
            # Perform comprehensive analysis
            analysis_data = await analysis_handler.perform_comprehensive_analysis(
                symbol=data.symbol,
            )
            
            # Stream LLM analysis
            async for chunk in analyze_stock_stream(
                analysis_data, 
                user_query=data.question_input,
                system_language=data.target_language,
                model_name=data.model_name, 
                chat_history=enhanced_history,
                provider_type=data.provider_type
            ):
                if chunk:
                    full_response.append(chunk)
                    yield f"{json.dumps({'content': chunk})}\n\n"
            
            # Join full response for saving
            complete_response = ''.join(full_response)
            response_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Analyze conversation importance
            importance_score = 0.5
            
            if data.session_id and user_id and complete_response:
                try:
                    analysis_model = "gpt-4.1-nano" if data.provider_type == ProviderType.OPENAI else data.model_name
                    
                    importance_score = await analyze_conversation_importance(
                        query=data.question_input or "Stock Analysis",
                        response=complete_response,
                        llm_provider=llm_provider,
                        model_name=analysis_model,
                        provider_type=data.provider_type
                    )
                    
                    logger.info(f"Conversation importance: {importance_score}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing importance: {e}")

            # Save to memory after streaming completes
            if data.session_id and user_id and complete_response:
                try:
                    # Prepare metadata
                    metadata = {
                        "type": "stock_analysis_stream",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "symbols": data.symbol, 
                        "importance_score": importance_score,
                        "response_time": response_time,
                        "model": data.model_name,
                        "provider": data.provider_type
                    }

                    await memory_manager.store_conversation_turn(
                        session_id=data.session_id,
                        user_id=user_id,
                        query=data.question_input or "Stock analysis request",
                        response=complete_response,
                        metadata=metadata,
                        importance_score=importance_score
                    )
                    
                    # Save to chat history
                    if question_id:
                        chat_service.save_assistant_response(
                            session_id=data.session_id,
                            created_at=datetime.datetime.now(),
                            question_id=question_id,
                            content=complete_response,
                            response_time=response_time
                        )

                    trigger_summary_update_nowait(session_id=data.session_id, user_id=user_id)

                except Exception as save_error:
                    logger.error(f"Error saving to memory/history: {str(save_error)}")
            
            # Send completion event
            yield "[DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in stock analysis streaming: {str(e)}")
            yield f"{json.dumps({'error': str(e)})}\n\n"
            yield "[DONE]\n\n"
    
    return StreamingResponse(
        format_sse_with_memory(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# =============================================================================
# region HEATMAP APIs
# =============================================================================
@router.post("/market/heatmap")
async def analyze_stock_heatmap_api(
    request: Request, 
    data: StockHeatmapPayload, 
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Stock market heatmap analysis endpoint
    
    This endpoint:
    1. Analyzes market sectors and individual stocks performance
    2. Uses LLM to provide sector rotation insights
    3. Integrates with memory system for context-aware responses
    4. Saves the conversation to chat history
    """
    try:
        user_id = getattr(request.state, "user_id", None)
        
        # Check memory collections exist
        if data.session_id and user_id:
            await memory_manager.ensure_session_memory_collections(
                session_id=data.session_id,
                user_id=user_id
            )

        # 1. Get relevant context from memory
        context = ""
        memory_stats = {}
        if data.session_id and user_id:
            try:
                context, memory_stats, document_references = await memory_manager.get_relevant_context(
                    session_id=data.session_id,
                    user_id=user_id,
                    current_query=data.question_input or "Analyze market heatmap and sectors",
                    llm_provider=llm_provider,
                    max_short_term=5,
                    max_long_term=3
                )
                logger.info(f"Retrieved memory context: {memory_stats}")
            except Exception as e:
                logger.error(f"Error getting memory context: {e}")
        
        # Fetch S&P 500 data from service
        logger.info("Fetching S&P 500 heatmap data...")
        sp500_service = SP500Service()
        heatmap_data = await sp500_service.get_sp500_constituents_with_quotes()
        
        if not heatmap_data:
            logger.warning("No heatmap data available")
            response_text = "Sorry, I'm unable to fetch market heatmap data at the moment. Please try again later."
        else:
            # Convert Pydantic objects to dictionaries for LLM processing
            heatmap_data_dicts = [item.model_dump() for item in heatmap_data]
            
            # 2. Analyze heatmap
            response_text = await analyze_stock_heatmap(
                data=data,
                heatmap_data=heatmap_data_dicts,
                chat_history=context,
                model_name=data.model_name,
                provider_type=data.provider_type,
                user_query=data.question_input 
            )

        # 3. Analyze conversation importance
        importance_score = 0.5
        
        if data.session_id and user_id:
            try:
                analysis_model = "gpt-4.1-nano" if data.provider_type == ProviderType.OPENAI else data.model_name
                
                importance_score = await analyze_conversation_importance(
                    query=data.question_input or "Heatmap analysis",
                    response=response_text,
                    llm_provider=llm_provider,
                    model_name=analysis_model,
                    provider_type=data.provider_type
                )
                
                logger.info(f"Conversation importance: {importance_score}")
                
            except Exception as e:
                logger.error(f"Error analyzing importance: {e}")

        # 4. Store conversation
        if data.session_id and user_id:
            try:
                # Extract unique sectors and symbols from data
                sectors = set()
                symbols = []
                
                if hasattr(data, 'data') and isinstance(data.data, list):
                    for item in data.data:
                        if 'sector' in item:
                            sectors.add(item['sector'])
                        if 'symbol' in item:
                            symbols.append(item['symbol'])
                
                # Prepare metadata
                metadata = {
                    "type": "heatmap_analysis",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "symbols": symbols[:20],  # Top 20 symbols
                    "sectors": list(sectors),
                    "total_stocks": len(symbols),
                    "importance_score": importance_score
                }
                
                await memory_manager.store_conversation_turn(
                    session_id=data.session_id,
                    user_id=user_id,
                    query=data.question_input or "Heatmap analysis request",
                    response=response_text,
                    metadata=metadata,
                    importance_score=importance_score
                )
                
                # Save to chat history
                question_id = chat_service.save_user_question(
                    session_id=data.session_id,
                    created_at=datetime.datetime.now(),
                    created_by=user_id,
                    content=data.question_input or f"Market heatmap analysis at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
                
                chat_service.save_assistant_response(
                    session_id=data.session_id,
                    created_at=datetime.datetime.now(),
                    question_id=question_id,
                    content=response_text,
                    response_time=0.1
                )

                trigger_summary_update_nowait(session_id=data.session_id, user_id=user_id)
                
            except Exception as e:
                logger.error(f"Error saving to memory: {e}")
        
        return {
            "interpretation": response_text,
            "memory_stats": memory_stats,
            "metadata": {
                "importance_score": importance_score,
                "context_used": bool(context),
                "timestamp": datetime.datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error in heatmap analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/market/heatmap/stream")
async def analyze_stock_heatmap_stream_api(
    request: Request, 
    data: StockHeatmapPayload, 
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Stock market heatmap analysis endpoint with streaming response
    
    This endpoint:
    1. Analyzes market sectors and individual stocks performance (STREAMING)
    2. Uses LLM to provide real-time sector rotation insights
    3. Integrates with memory system for context-aware responses
    4. Saves the conversation to chat history
    """
    user_id = getattr(request.state, "user_id", None)
    
    # Check memory collections exist
    if data.session_id and user_id:
        await memory_manager.ensure_session_memory_collections(
            session_id=data.session_id,
            user_id=user_id
        )
    
    # Save user question
    question_id = None
    if data.session_id and user_id:
        try:
            question_content = data.question_input or f"Heatmap analysis at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            question_id = chat_service.save_user_question(
                session_id=data.session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id,
                content=question_content
            )
        except Exception as save_error:
            logger.error(f"Error saving user question: {str(save_error)}")
    
    # Get memory context
    chat_history = ""
    if data.session_id:
        try:
            chat_history = ChatMessageHistory.string_message_chat_history(
                session_id=data.session_id
            )
        except Exception as e:
            logger.error(f"Error fetching history: {e}")

    context = ""
    memory_stats = {}
    if data.session_id and user_id:
        try:
            context, memory_stats, document_references = await memory_manager.get_relevant_context(
                session_id=data.session_id,
                user_id=user_id,
                current_query=data.question_input or "Analyze market heatmap and sectors",
                llm_provider=llm_provider,
                max_short_term=5,
                max_long_term=3
            )
            logger.info(f"Retrieved memory context: {memory_stats}")
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
    
    enhanced_history = ""
    if context:
        enhanced_history = f"{context}\n\n"
    if chat_history:
        enhanced_history += f"[Conversation History]\n{chat_history}"
    
    # Fetch S&P 500 data from service
    logger.info("Fetching S&P 500 heatmap data...")
    sp500_service = SP500Service()
    heatmap_data = await sp500_service.get_sp500_constituents_with_quotes()

    heatmap_data_dicts = []
    if heatmap_data:
        heatmap_data_dicts = [
            item.model_dump() if hasattr(item, 'model_dump') else item 
            for item in heatmap_data
        ]
    else:
        logger.warning("No heatmap data available from S&P 500 service")
    
    async def stream_with_memory():
        full_response = []
        start_time = datetime.datetime.now()
        
        try:
            # Stream the analysis
            async for chunk in analyze_stock_heatmap_stream(
                data,
                heatmap_data_dicts,
                enhanced_history,
                data.model_name,
                provider_type=data.provider_type,
                user_query=data.question_input 
            ):
                if chunk:
                    full_response.append(chunk)
                    yield f"{json.dumps({'content': chunk})}\n\n"
            
            # Process complete response
            complete_response = ''.join(full_response)
            response_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Analyze conversation importance
            importance_score = 0.5
            
            if data.session_id and user_id and complete_response:
                try:
                    analysis_model = "gpt-4.1-nano" if data.provider_type == ProviderType.OPENAI else data.model_name
                    
                    importance_score = await analyze_conversation_importance(
                        query=data.question_input or "Heatmap analysis",
                        response=complete_response,
                        llm_provider=llm_provider,
                        model_name=analysis_model,
                        provider_type=data.provider_type
                    )
                    
                    logger.info(f"Conversation importance: {importance_score}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing importance: {e}")

            # Store conversation
            if data.session_id and user_id and complete_response:
                try:
                    # Extract sectors and symbols
                    sectors = set()
                    symbols = []
                    
                    if hasattr(data, 'data') and isinstance(data.data, list):
                        for item in data.data:
                            if 'sector' in item:
                                sectors.add(item['sector'])
                            if 'symbol' in item:
                                symbols.append(item['symbol'])
                    
                    # Prepare metadata
                    metadata = {
                        "type": "heatmap_analysis_stream",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "symbols": symbols[:20],  # Top 20
                        "sectors": list(sectors),
                        "total_stocks": len(symbols),
                        "importance_score": importance_score,
                        "response_time": response_time,
                        "model": data.model_name,
                        "provider": data.provider_type
                    }
                    
                    await memory_manager.store_conversation_turn(
                        session_id=data.session_id,
                        user_id=user_id,
                        query=data.question_input or "Heatmap analysis request",
                        response=complete_response,
                        metadata=metadata,
                        importance_score=importance_score
                    )
                    
                    # Save to chat history
                    if question_id:
                        chat_service.save_assistant_response(
                            session_id=data.session_id,
                            created_at=datetime.datetime.now(),
                            question_id=question_id,
                            content=complete_response,
                            response_time=response_time
                        )

                    trigger_summary_update_nowait(session_id=data.session_id, user_id=user_id)

                except Exception as e:
                    logger.error(f"Error saving conversation: {e}")
            
            # Send completion event
            yield "[DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            yield f"{json.dumps({'error': str(e)})}\n\n"
            yield "[DONE]\n\n"

    return StreamingResponse(
        stream_with_memory(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )