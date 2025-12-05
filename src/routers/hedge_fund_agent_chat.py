import asyncio
import json
import datetime
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.handlers.hedge_fund_agent_chat_handler import AgentChatHandler
from src.agents.memory.memory_manager import MemoryManager
from src.handlers.llm_chat_handler import ChatHandler, ChatMessageHistory, ChatService
from src.helpers.llm_helper import LLMGeneratorProvider
from src.hedge_fund.llm.models import ModelProvider
from src.providers.provider_factory import ProviderType
import logging
from src.handlers.openai_tool_call_handler import agentic_openai_tool_call

router = APIRouter(prefix="/agents")

# Initialize services
api_key_auth = APIKeyAuth()
agent_handler = AgentChatHandler()
memory_manager = MemoryManager()
chat_service = ChatService()
llm_provider = LLMGeneratorProvider()
logger = logging.getLogger(__name__)

# Request models with collection support
class AgentChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model_name: str = "gpt-5-nano-2025-08-07"
    collection_name: Optional[str] = None  # Support for uploaded documents
    provider_type: ProviderType = ProviderType.OPENAI
    enable_thinking: bool = False
    include_market_data: bool = True
    
class AgentStreamRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model_name: str = "gpt-5-nano-2025-08-07"
    collection_name: Optional[str] = None  # Support for uploaded documents
    provider_type: ProviderType = ProviderType.OPENAI
    enable_thinking: bool = False
    include_market_data: bool = True
    tickers: Optional[List[str]] = None

# Available agents configuration
AVAILABLE_AGENTS = {
    "warren_buffett": {
        "name": "Warren Buffett",
        "description": "Value investing master, focuses on fundamentals and long-term growth",
        "avatar": "/assets/agents/buffett.png",
        "specialties": ["value investing", "fundamental analysis", "long-term growth"],
        "personality": """You are Warren Buffett, the Oracle of Omaha and CEO of Berkshire Hathaway.
        
        Your communication style:
        - Use folksy metaphors and simple language to explain complex concepts
        - Often reference your experiences at Berkshire Hathaway
        - Quote your mentor Benjamin Graham and partner Charlie Munger
        - Show humor and humility
        
        Your investment philosophy:
        - Buy wonderful companies at fair prices
        - Your favorite holding period is forever
        - Be fearful when others are greedy, greedy when others are fearful
        - Focus on business fundamentals, not market noise
        - Invest within your circle of competence
        
        Personal traits:
        - Love Cherry Coke and McDonald's
        - Live modestly in Omaha despite immense wealth
        - Write famous annual letters to shareholders
        - Play bridge and ukulele in spare time"""
    }
}

@router.get("/list")
async def list_agents(
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """List all available trading agent personalities"""
    return [
        {
            "agent_id": agent_id,
            "name": info["name"],
            "description": info["description"],
            "avatar": info["avatar"],
            "specialties": info["specialties"]
        }
        for agent_id, info in AVAILABLE_AGENTS.items()
    ]

@router.post("/{agent_id}/chat")
async def chat_with_agent(
    request: Request,
    agent_id: str,
    data: AgentChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Chat with a specific trading agent (non-streaming)"""
    if agent_id not in AVAILABLE_AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    user_id = getattr(request.state, "user_id", None)
    
    # Save user question
    question_id = None
    if data.session_id and user_id:
        try:
            question_id = chat_service.save_user_question(
                session_id=data.session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id,
                content=data.message
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
    
    # Get memory context (including collection if provided)
    context = ""
    memory_stats = {}
    document_references = []
    
    if data.session_id and user_id:
        try:
            # Get memory with collection support
            if data.collection_name:
                context, memory_stats, document_references = await memory_manager.get_relevant_context_with_collection(
                    session_id=data.session_id,
                    user_id=user_id,
                    current_query=data.message,
                    collection_name=data.collection_name,
                    llm_provider=llm_provider,
                    max_short_term=5,
                    max_long_term=3
                )
            else:
                context, memory_stats, document_references = await memory_manager.get_relevant_context(
                    session_id=data.session_id,
                    user_id=user_id,
                    current_query=data.message,
                    llm_provider=llm_provider,
                    max_short_term=5,
                    max_long_term=3
                )
            logger.info(f"Memory context for agent {agent_id}: {memory_stats}")
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
    
    # Extract tickers using dual approach
    identified_tickers = await extract_tickers_smart(
        message=data.message,
        agent_handler=agent_handler,
        agent_id=agent_id
    )
    
    # Process with agent
    try:
        response = await agent_handler.process_message(
            agent_id=agent_id,
            message=data.message,
            session_id=data.session_id,
            chat_history=chat_history,
            memory_context=context,
            personality=AVAILABLE_AGENTS[agent_id]["personality"],
            model_name=data.model_name,
            provider_type=data.provider_type, 
            include_market_data=data.include_market_data,
            identified_tickers=identified_tickers,
            user_id=user_id
        )
        
        # Analyze importance
        importance_score = 0.5
        if data.session_id and user_id:
            try:
                importance_score = await agent_handler.analyze_importance(
                    query=data.message,
                    response=response["content"],
                    agent_id=agent_id,
                    llm_provider=llm_provider,
                    model_name=data.model_name,
                    provider_type=data.provider_type
                )
            except Exception as e:
                logger.error(f"Error analyzing importance: {e}")
        
        # Store in memory
        if data.session_id and user_id:
            try:
                await memory_manager.store_conversation_turn(
                    session_id=data.session_id,
                    user_id=user_id,
                    query=data.message,
                    response=response["content"],
                    metadata={
                        "agent_id": agent_id,
                        "agent_name": AVAILABLE_AGENTS[agent_id]["name"],
                        "tickers_analyzed": response.get("tickers_analyzed", [])
                    },
                    importance_score=importance_score
                )
                
                # Save assistant response
                chat_service.save_assistant_response(
                    session_id=data.session_id,
                    created_at=datetime.datetime.now(),
                    question_id=question_id,
                    content=response["content"],
                    response_time=response.get("response_time", 0.1)
                )
            except Exception as e:
                logger.error(f"Error storing response: {e}")
        
        return {
            "agent_id": agent_id,
            "agent_name": AVAILABLE_AGENTS[agent_id]["name"],
            "content": response["content"],
            "tickers_analyzed": response.get("tickers_analyzed", []),
            "sources": response.get("sources", []),
            "memory_stats": memory_stats
        }
        
    except Exception as e:
        logger.error(f"Error in agent chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{agent_id}/stream")
async def stream_agent_chat(
    request: Request,
    agent_id: str,
    data: AgentStreamRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Stream chat response from trading agent"""
    if agent_id not in AVAILABLE_AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    user_id = getattr(request.state, "user_id", None)
    
    # Save user question
    question_id = None
    if data.session_id and user_id:
        try:
            question_id = chat_service.save_user_question(
                session_id=data.session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id,
                content=data.message
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
    
    # Get memory context with collection support
    context = ""
    memory_stats = {}
    if data.session_id and user_id:
        try:
            if data.collection_name:
                context, memory_stats, _ = await memory_manager.get_relevant_context_with_collection(
                    session_id=data.session_id,
                    user_id=user_id,
                    current_query=data.message,
                    collection_name=data.collection_name,
                    llm_provider=llm_provider,
                    max_short_term=5,
                    max_long_term=3
                )
            else:
                context, memory_stats, _ = await memory_manager.get_relevant_context(
                    session_id=data.session_id,
                    user_id=user_id,
                    current_query=data.message,
                    llm_provider=llm_provider,
                    max_short_term=5,
                    max_long_term=3
                )
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
    
    # Enhance history with memory
    enhanced_history = ""
    if context:
        enhanced_history = f"[Relevant Context from Memory]\n{context}\n\n"
    if chat_history:
        enhanced_history += f"[Recent Conversation]\n{chat_history}"
    
    # Extract tickers if not provided
    identified_tickers = data.tickers
    if not identified_tickers:
        identified_tickers = await extract_tickers_smart(
            message=data.message,
            agent_handler=agent_handler,
            agent_id=agent_id
        )
    
    async def stream_sse():
        full_response = []
        tickers_analyzed = []
        
        try:
            # Start streaming from agent
            async for chunk in agent_handler.stream_message(
                agent_id=agent_id,
                message=data.message,
                session_id=data.session_id,
                enhanced_history=enhanced_history,
                personality=AVAILABLE_AGENTS[agent_id]["personality"],
                model_name=data.model_name,
                provider_type=data.provider_type,
                include_market_data=data.include_market_data,
                identified_tickers=identified_tickers
            ):
                if chunk.get("type") == "content":
                    content = chunk.get("content", "")
                    full_response.append(content)
                    yield f"{json.dumps({'content': content})}\n\n"
                
                elif chunk.get("type") == "metadata":
                    tickers_analyzed = chunk.get("tickers_analyzed", [])
                    yield f"{json.dumps({'metadata': chunk})}\n\n"
                
                elif chunk.get("type") == "thinking" and data.enable_thinking:
                    yield f"{json.dumps({'thinking': chunk.get('content')})}\n\n"
            
            # Complete response
            complete_response = ''.join(full_response)
            
            # Analyze importance
            importance_score = 0.5
            if data.session_id and user_id and complete_response:
                try:
                    importance_score = await agent_handler.analyze_importance(
                        query=data.message,
                        response=complete_response,
                        agent_id=agent_id,
                        llm_provider=llm_provider,
                        model_name=data.model_name,
                        provider_type=data.provider_type
                    )
                except Exception as e:
                    logger.error(f"Error analyzing importance: {e}")
            
            # Store in memory
            if data.session_id and user_id and complete_response:
                try:
                    await memory_manager.store_conversation_turn(
                        session_id=data.session_id,
                        user_id=user_id,
                        query=data.message,
                        response=complete_response,
                        metadata={
                            "agent_id": agent_id,
                            "agent_name": AVAILABLE_AGENTS[agent_id]["name"],
                            "tickers_analyzed": tickers_analyzed
                        },
                        importance_score=importance_score
                    )
                    
                    chat_service.save_assistant_response(
                        session_id=data.session_id,
                        created_at=datetime.datetime.now(),
                        question_id=question_id,
                        content=complete_response,
                        response_time=0.1
                    )
                except Exception as e:
                    logger.error(f"Error saving response: {e}")
            
            yield "[DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in agent streaming: {str(e)}")
            yield f"{json.dumps({'error': str(e)})}\n\n"
            yield "[DONE]\n\n"
    
    return StreamingResponse(
        stream_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
            "Connection": "keep-alive"
        }
    )

async def extract_tickers_smart(
    message: str,
    agent_handler: AgentChatHandler,
    agent_id: str
) -> List[str]:
    """
    Smart ticker extraction using dual approach:
    1. Traditional regex-based extraction
    2. OpenAI tool calling for better accuracy
    
    Priority: OpenAI tool call > regex extraction
    """
    tickers_regex = []
    tickers_tool_call = []
    
    # Method 1: Extract using regex (existing method)
    try:
        agent = agent_handler.agents.get(agent_id)
        if agent:
            tickers_regex = agent.extract_tickers(message)
    except Exception as e:
        logger.error(f"Error in regex ticker extraction: {e}")
    
    # Method 2: Extract using OpenAI tool call
    try:
        tool_prompt = f"""Extract stock ticker symbols from this message. 
        Return only valid stock tickers as a JSON list.
        Message: {message}
        
        Example response: {{"tickers": ["AAPL", "GOOGL", "TSLA"]}}
        If no tickers found, return: {{"tickers": []}}"""
        
        result = await agentic_openai_tool_call(tool_prompt)
        if result and isinstance(result, dict):
            tickers_tool_call = result.get("tickers", [])
    except Exception as e:
        logger.error(f"Error in tool call ticker extraction: {e}")
    
    # Combine and prioritize results
    if tickers_tool_call and tickers_regex:
        # Both methods found tickers - find common ones first
        common_tickers = list(set(tickers_tool_call) & set(tickers_regex))
        if common_tickers:
            return common_tickers[:3]  # Most confident results
        else:
            # Prioritize tool call results
            return tickers_tool_call[:3]
    elif tickers_tool_call:
        # Only tool call found tickers
        return tickers_tool_call[:3]
    elif tickers_regex:
        # Only regex found tickers
        return tickers_regex[:3]
    else:
        # No tickers found
        return []