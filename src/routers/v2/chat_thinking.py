import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field, field_validator, model_validator
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse

from src.utils.logger.custom_logging import LoggerMixin
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.handlers.llm_chat_handler import ChatService
from src.providers.provider_factory import ProviderType
from src.utils.constants import APIModelName
from src.utils.config import settings
from src.agents.streaming.streaming_chat_handler import CancellationToken
from src.helpers.llm_chat_helper import SSE_HEADERS, DEFAULT_HEARTBEAT_SEC

# ============================================================================
# ROUTER INITIALIZATION
# ============================================================================

router = APIRouter(prefix="/assistant")
api_key_auth = APIKeyAuth()
logger = LoggerMixin().logger
chat_service = ChatService()


# ============================================================================
# OPENROUTER MODEL PRESETS
# ============================================================================

OPENROUTER_MODEL_PRESETS = {
    # High Performance
    "gpt-4o": "openai/gpt-4o",
    
    # Cost Effective
    "gpt-5-nano": "openai/gpt-5-nano",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4.1-nano": "openai/gpt-4.1-nano",
    
    # Open Source
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "gemma-3-27b-it": "google/gemma-3-27b-it:free",
}

# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class UIContextRequest(BaseModel):
    """
    UI Context for Soft Context Inheritance.

    Frontend sends this to indicate the current UI state,
    enabling smart disambiguation of ambiguous symbols.

    Example:
        User is on Crypto tab, asks "giá BTC"
        → BTC resolved as Bitcoin (crypto) not BTC Digital (stock)
    """

    current_tab: str = Field(
        default="auto",
        description="Current UI tab: 'crypto', 'stock', or 'auto'",
        examples=["crypto", "stock", "auto"]
    )
    recent_symbols: List[str] = Field(
        default_factory=list,
        description="Recently viewed symbols (for context reinforcement)",
        examples=[["BTC", "ETH", "SOL"]]
    )
    watchlist_type: Optional[str] = Field(
        default=None,
        description="Type of watchlist currently displayed"
    )
    language: str = Field(
        default="vi",
        description="User's preferred language"
    )


class ThinkingChatRequest(BaseModel):
    """Request schema for thinking chat endpoints"""

    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation continuity"
    )
    question_input: str = Field(
        ...,
        description="User question/query",
        min_length=1,
        max_length=10000
    )
    chart_displayed: bool = Field(
        default=False,
        description="Whether a chart is currently displayed to the user"
    )
    model_name: str = Field(
        default=APIModelName.GPT41Nano,
        description="LLM model name",
        examples=["gpt-4.1-nano", "gpt-4o-mini", "gpt-5-nano", "gpt-oss:20b", "gemma-3-27b-it", "llama-3.1-70b"]
    )
    provider_type: str = Field(
        default=ProviderType.OPENAI,
        description="LLM provider type (openai, openrouter, gemini, ollama)"
    )
    enable_thinking: bool = Field(
        default=True,
        description="Enable thinking display in stream"
    )

    # UI Context for Soft Context Inheritance
    ui_context: Optional[UIContextRequest] = Field(
        default=None,
        description=(
            "UI context from frontend for soft symbol disambiguation. "
            "If provided, ambiguous symbols will be resolved based on current tab."
        )
    )
    enable_tools: bool = Field(
        default=True, 
        description="Enable tool execution"
    )
    enable_think_tool: bool = Field(
        default=False,
        description="Enable Think Tool for step-by-step reasoning before planning"
    )
    enable_compaction: bool = Field(
        default=True,
        description=(
            "Enable automatic context compaction when approaching token limits. "
            "Uses smart summary to preserve important context while reducing token usage."
        )
    )
    enable_llm_events: bool = Field(
        default=True,
        description="Enable LLM decision events"
    )
    enable_agent_tree: bool = Field(
        default=True,
        description="Enable agent tree tracking"
    )
    stream_config: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Custom stream configuration"
    )

    @field_validator('provider_type')
    @classmethod
    def validate_provider_type(cls, v: str) -> str:
        """Validate provider type is supported."""
        valid_providers = ProviderType.list()
        if v.lower() not in valid_providers:
            raise ValueError(
                f"Invalid provider_type '{v}'. "
                f"Must be one of: {valid_providers}"
            )
        return v.lower()
    
    @field_validator('model_name')
    @classmethod
    def expand_model_preset(cls, v: str, info) -> str:
        """Expand OpenRouter model presets to full format."""
        return v

    @model_validator(mode='after')
    def normalize_model_for_provider(self) -> 'ThinkingChatRequest':
        """Normalize model name based on provider type"""
        if self.provider_type == ProviderType.OPENROUTER:
            # Expand preset for OpenRouter
            if self.model_name in OPENROUTER_MODEL_PRESETS:
                self.model_name = OPENROUTER_MODEL_PRESETS[self.model_name]
        elif self.provider_type == ProviderType.OPENAI:
            # Strip prefix if present for OpenAI
            if "/" in self.model_name:
                self.model_name = self.model_name.split("/")[-1]
        
        return self

class ThinkingChatResponse(BaseModel):
    """Response schema for non-streaming thinking chat"""
    
    status: str
    session_id: str
    response: str
    thinking_summary: Optional[Dict[str, Any]] = None
    tools_executed: Optional[List[Dict[str, Any]]] = None
    stats: Optional[Dict[str, Any]] = None,
    agent_tree: Optional[Dict[str, Any]] = None


class ProviderInfoResponse(BaseModel):
    """Response schema for provider information."""
    
    provider: str
    name: str
    description: str
    api_key_configured: bool
    example_models: List[str]

# ============================================================================
# STREAMING HANDLER FACTORY
# ============================================================================

class StreamingHandlerFactory:
    """
    Factory for creating StreamingChatHandler instances
    
    Uses lazy initialization to avoid import overhead on module load.
    Caches components that can be reused across requests.
    """
    
    _instance = None
    _components_initialized = False
    
    # Cached components (initialized once)
    _planning_agent = None
    _task_executor = None
    _core_memory = None
    _summary_manager = None
    _session_repo = None
    _llm_provider = None
    _tool_execution_service = None
    _chat_repo = None
    _validation_agent = None
    
    @classmethod
    def _initialize_components(cls):
        """Initialize shared components (called once)"""
        if cls._components_initialized:
            return
        
        try:
            # Import components
            from src.agents.planning.planning_agent import PlanningAgent
            from src.agents.action.task_executor import TaskExecutor
            from src.agents.memory.core_memory import CoreMemory
            from src.agents.memory.recursive_summary import RecursiveSummaryManager
            from src.database.repository.sessions import SessionRepository
            from src.database.repository.chat import ChatRepository
            from src.helpers.llm_helper import LLMGeneratorProvider
            from src.services.v2.tool_execution_service import ToolExecutionService
            from src.agents.validation.validation_agent import ValidationAgent
            from src.utils.config import settings
            
            # Initialize components
            cls._tool_execution_service = ToolExecutionService()
            cls._validation_agent = ValidationAgent()
            
            cls._task_executor = TaskExecutor(
                tool_execution_service=cls._tool_execution_service,
                validation_agent=cls._validation_agent,
                max_retries=2
            )
            
            cls._core_memory = CoreMemory()
            cls._summary_manager = RecursiveSummaryManager()
            cls._session_repo = SessionRepository()
            cls._chat_repo = ChatRepository()
            cls._llm_provider = LLMGeneratorProvider()
            
            cls._components_initialized = True
            logger.info("[THINKING:FACTORY] Shared components initialized")
            
        except Exception as e:
            logger.error(f"[THINKING:FACTORY] Failed to initialize: {e}")
            raise
    
    @classmethod
    async def get_handler(
        cls,
        model_name: str,
        provider_type: str,
        stream_config: Optional[Dict[str, Any]] = None,
        enable_llm_events: bool = True,
        enable_agent_tree: bool = True
    ):
        """
        Get a configured StreamingChatHandler
        
        Args:
            model_name: LLM model name
            provider_type: LLM provider
            stream_config: Optional custom streaming config
            
        Returns:
            Configured StreamingChatHandler instance
        """
        # Ensure components are initialized
        cls._initialize_components()
        
        # Import streaming components
        from src.agents.streaming import StreamingChatHandler, StreamingConfig
        from src.agents.planning.planning_agent import PlanningAgent
        
        # Create planning agent with specified model
        planning_agent = PlanningAgent(
            model_name=settings.MODEL_DEFAULT,
            provider_type=settings.PROVIDER_DEFAULT 
        )
        
        # Configure streaming
        config = StreamingConfig(
            enable_thinking_display=True,
            enable_tool_progress=True,
            enable_context_events=True,
            enable_memory_events=False,
            save_messages=True,
            enable_llm_decision_events=enable_llm_events,
            enable_agent_tree=enable_agent_tree
        )
        
        # Apply custom config if provided
        if stream_config:
            for key, value in stream_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Create handler
        handler = StreamingChatHandler(
            planning_agent=planning_agent,
            task_executor=cls._task_executor,
            core_memory=cls._core_memory,
            summary_manager=cls._summary_manager,
            session_repo=cls._session_repo,
            llm_provider=cls._llm_provider,
            tool_execution_service=cls._tool_execution_service,
            chat_repo=cls._chat_repo,
            config=config
        )
        
        return handler


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_session_id(user_id: int, organization_id: Optional[int] = None) -> str:
    """Create a new session using ChatService"""
    return chat_service.create_chat_session(
        user_id=user_id,
        organization_id=organization_id
    )


def format_sse_done() -> str:
    """Format SSE done marker"""
    return "data: [DONE]\n\n"

def format_sse_heartbeat() -> str:
    """Format SSE heartbeat comment"""
    return ": heartbeat\n\n"

def get_model_display_name(model_name: str, provider_type: str) -> str:
    """Get human-readable model name for logging."""
    if provider_type == ProviderType.OPENROUTER and "/" in model_name:
        # Extract just the model part for display
        return model_name.split("/")[-1]
    return model_name

# ============================================================================
# STREAMING ENDPOINT
# ============================================================================

@router.post(
    "/stream",
    summary="Streaming Chat with Thinking Display",
    description="""
Stream chat response with real-time thinking process display.

Returns Server-Sent Events (SSE) with progressive updates.
    """,
    response_class=StreamingResponse,
)
async def chat_thinking_stream(
    request: Request,
    data: ThinkingChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Streaming chat with real-time thinking display
    
    Returns SSE stream with:
    - Thinking process events
    - Tool execution progress
    - Response text streaming
    """
    # Get user context from request state
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")
    
    # Create session if needed
    session_id = data.session_id
    if not session_id:
        session_id = create_session_id(user_id, organization_id)
        logger.info(f"[THINKING:STREAM] Created new session: {session_id}")
    
    # Get display name for logging
    model_display = get_model_display_name(data.model_name, data.provider_type)
    
    logger.info(
        f"[THINKING:STREAM] Starting stream: session={session_id}, "
        f"user={user_id}, provider={data.provider_type}, model={model_display}, "
        f"compaction={'ON' if data.enable_compaction else 'OFF'}"
    )
    
    # Create cancellation token for this request
    cancellation_token = CancellationToken()

    async def event_generator():
        """Generate SSE events with heartbeat support"""

        try:
            handler = await StreamingHandlerFactory.get_handler(
                model_name=data.model_name,
                provider_type=data.provider_type,
                stream_config=data.stream_config,
                enable_llm_events=data.enable_llm_events,
                enable_agent_tree=data.enable_agent_tree
            )
            
            # ============= HEARTBEAT PATTERN =============
            stream_gen = handler.handle_chat_stream(
                query=data.question_input,
                session_id=session_id,
                user_id=user_id,
                model_name=data.model_name,
                provider_type=data.provider_type,
                organization_id=organization_id,
                enable_thinking=data.enable_thinking,
                chart_displayed=data.chart_displayed,
                enable_compaction=data.enable_compaction,
                enable_think_tool=data.enable_think_tool,
                cancellation_token=cancellation_token
            )
            
            pending_task = None
            
            try:
                while True:
                    # Check client disconnect
                    if await request.is_disconnected():
                        logger.info(f"[THINKING:STREAM] Client disconnected")
                        cancellation_token.cancel()
                        break
                    
                    if pending_task is None:
                        pending_task = asyncio.create_task(anext(stream_gen))
                    
                    done, _ = await asyncio.wait(
                        {pending_task}, 
                        timeout=DEFAULT_HEARTBEAT_SEC
                    )
                    
                    if done:
                        try:
                            event = pending_task.result()
                        except StopAsyncIteration:
                            break
                        except Exception as e:
                            logger.error(f"[THINKING:STREAM] Event error: {e}")
                            error_event = {
                                "type": "error",
                                "error": str(e),
                                "timestamp": datetime.utcnow().isoformat()
                            }
                            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
                            break
                        finally:
                            pending_task = None
                        
                        # Yield the event
                        yield event.to_sse()
                    else:
                        # Timeout - send heartbeat
                        yield ": heartbeat\n\n"
                        
            finally:
                # Cleanup
                if pending_task is not None and not pending_task.done():
                    pending_task.cancel()
                    try:
                        await pending_task
                    except (asyncio.CancelledError, StopAsyncIteration):
                        pass
                
                if hasattr(stream_gen, 'aclose'):
                    try:
                        await stream_gen.aclose()
                    except Exception:
                        pass
            # ============= END HEARTBEAT PATTERN =============
            
            # Final done marker
            if not cancellation_token.is_cancelled:
                yield format_sse_done()

        except asyncio.CancelledError:
            logger.info(f"[THINKING:STREAM] Stream cancelled (asyncio)")
            cancellation_token.cancel()
            
            error_event = {
                "type": "error",
                "error": "Stream cancelled",
                "phase": "streaming",
                "recoverable": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
            yield format_sse_done()

        except Exception as e:
            logger.error(f"[THINKING:STREAM] Error: {e}", exc_info=True)
            
            error_event = {
                "type": "error",
                "error": str(e),
                "phase": "unknown",
                "recoverable": False,
                "timestamp": datetime.utcnow().isoformat()
            }
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
            yield format_sse_done()

        finally:
            if not cancellation_token.is_cancelled:
                cancellation_token.cancel()
            
            logger.info(f"[THINKING:STREAM] Stream ended for session {session_id}")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )


# @router.post(
#     "/stream/v2",
#     summary="Streaming Chat v2 - Cancellation",
#     description="""
# Enhanced streaming with more aggressive cancellation detection.

# Checks for client disconnect more frequently and cleans up resources properly.
#     """,
#     response_class=StreamingResponse,
# )
# async def chat_thinking_stream_v2(
#     request: Request,
#     data: ThinkingChatRequest,
#     api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
# ):
#     """
#     Streaming chat v2 with enhanced cancellation
    
#     Task 1: More aggressive disconnect detection
#     """
#     user_id = getattr(request.state, "user_id", None)
#     organization_id = getattr(request.state, "organization_id", None)
    
#     if not user_id:
#         raise HTTPException(status_code=400, detail="User ID required")
    
#     session_id = data.session_id or create_session_id(user_id, organization_id)
    
#     logger.info(f"[THINKING:STREAM:V2] Starting enhanced stream: session={session_id}")
    
#     cancellation_token = CancellationToken()
    
#     async def event_generator_with_heartbeat():
#         event_count = 0
        
#         try:
#             handler = await StreamingHandlerFactory.get_handler(
#                 model_name=data.model_name,
#                 provider_type=data.provider_type,
#                 stream_config=data.stream_config,
#                 enable_llm_events=data.enable_llm_events,
#                 enable_agent_tree=data.enable_agent_tree
#             )
            
#             stream_gen = handler.handle_chat_stream(
#                 query=data.question_input,
#                 session_id=session_id,
#                 user_id=user_id,
#                 model_name=data.model_name,
#                 provider_type=data.provider_type,
#                 organization_id=organization_id,
#                 enable_thinking=data.enable_thinking,
#                 chart_displayed=data.chart_displayed,
#                 enable_compaction=data.enable_compaction,
#                 enable_think_tool=data.enable_think_tool,
#                 cancellation_token=cancellation_token
#             )
            
#             pending_task = None
            
#             try:
#                 while True:
#                     # Check disconnect
#                     if await request.is_disconnected():
#                         logger.info(f"[THINKING:STREAM:V2] Client disconnected at event {event_count}")
#                         cancellation_token.cancel()
#                         break
                    
#                     if pending_task is None:
#                         pending_task = asyncio.create_task(anext(stream_gen))
                    
#                     done, _ = await asyncio.wait(
#                         {pending_task}, 
#                         timeout=DEFAULT_HEARTBEAT_SEC
#                     )
                    
#                     if done:
#                         try:
#                             event = pending_task.result()
#                         except StopAsyncIteration:
#                             break
#                         except Exception as e:
#                             logger.error(f"[THINKING:STREAM:V2] Error: {e}")
#                             error_event = {"type": "error", "error": str(e)}
#                             yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
#                             break
#                         finally:
#                             pending_task = None
                        
#                         event_count += 1
#                         yield event.to_sse()
                        
#                         # Allow other tasks to run
#                         await asyncio.sleep(0)
#                     else:
#                         # Heartbeat
#                         yield ": heartbeat\n\n"
                        
#             finally:
#                 if pending_task is not None and not pending_task.done():
#                     pending_task.cancel()
#                     try:
#                         await pending_task
#                     except (asyncio.CancelledError, StopAsyncIteration):
#                         pass
                
#                 if hasattr(stream_gen, 'aclose'):
#                     try:
#                         await stream_gen.aclose()
#                     except Exception:
#                         pass
            
#             if not cancellation_token.is_cancelled:
#                 yield format_sse_done()
            
#         except asyncio.CancelledError:
#             logger.info(f"[THINKING:STREAM:V2] Cancelled at event {event_count}")
#             cancellation_token.cancel()
#             yield format_sse_done()
            
#         except Exception as e:
#             logger.error(f"[THINKING:STREAM:V2] Error at event {event_count}: {e}")
#             error_event = {"type": "error", "error": str(e)}
#             yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
#             yield format_sse_done()
        
#         finally:
#             cancellation_token.cancel()
#             logger.info(f"[THINKING:STREAM:V2] Completed: {event_count} events")
    
#     return StreamingResponse(
#         event_generator_with_heartbeat(),
#         media_type="text/event-stream",
#         headers=SSE_HEADERS
#     )


# ============================================================================
# NON-STREAMING ENDPOINT
# ============================================================================
@router.post(
    "/complete",
    summary="Complete Chat with Thinking Summary",
    description="Non-streaming chat with thinking summary and agent tree.",
    response_model=ThinkingChatResponse
)
async def chat_thinking_complete(
    request: Request,
    data: ThinkingChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Non-streaming chat with thinking summary
    
    Includes agent tree in response
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")
    
    session_id = data.session_id or create_session_id(user_id, organization_id)
    
    logger.info(f"[THINKING:COMPLETE] Starting: session={session_id[:16]}...")
    
    try:
        handler = await StreamingHandlerFactory.get_handler(
            model_name=data.model_name,
            provider_type=data.provider_type,
            stream_config=data.stream_config,
            enable_llm_events=data.enable_llm_events,
            enable_agent_tree=data.enable_agent_tree
        )
        
        from src.agents.streaming import StreamEventType
        
        response_text = ""
        thinking_events = []
        tools_executed = []
        final_stats = {}
        agent_tree = None
        phases_seen = set()
        
        async for event in handler.handle_chat_stream(
            query=data.question_input,
            session_id=session_id,
            user_id=user_id,
            model_name=data.model_name,
            provider_type=data.provider_type,
            organization_id=organization_id,
            enable_thinking=data.enable_thinking,
            chart_displayed=data.chart_displayed,
            enable_compaction=data.enable_compaction,
            enable_think_tool=data.enable_think_tool
        ):
            event_dict = event.to_dict()
            
            if 'phase' in event_dict.get('data', {}):
                phases_seen.add(event_dict['data']['phase'])
            
            # Collect thinking events (including Task 5 LLM events)
            if event.event_type in [
                StreamEventType.THINKING_START,
                StreamEventType.THINKING_DELTA,
                StreamEventType.THINKING_END,
                StreamEventType.PLANNING_PROGRESS,
                StreamEventType.PLANNING_COMPLETE,
                StreamEventType.LLM_THOUGHT,
                StreamEventType.LLM_DECISION,
                StreamEventType.LLM_ACTION
            ]:
                thinking_events.append(event_dict)
            
            # Collect tool events (with call_id - Task 3)
            elif event.event_type in [
                StreamEventType.TOOL_START,
                StreamEventType.TOOL_COMPLETE
            ]:
                tools_executed.append(event_dict)
            
            # Collect response text
            elif event.event_type == StreamEventType.TEXT_DELTA:
                chunk = event_dict.get('data', {}).get('chunk', '')
                response_text += chunk
            
            # Get final stats and agent tree (Task 6)
            elif event.event_type == StreamEventType.DONE:
                final_stats = event_dict.get('data', {}).get('stats', {})
                agent_tree = event_dict.get('data', {}).get('agent_tree')
        
        # Build thinking summary
        thinking_summary = {
            "total_steps": len(thinking_events),
            "phases": list(phases_seen),
            "planning_events": len([
                e for e in thinking_events 
                if 'planning' in e.get('event_type', '').lower()
            ]),
            "llm_events": len([
                e for e in thinking_events
                if 'llm_' in e.get('event_type', '').lower()
            ]),
            "recent_events": thinking_events[-10:]
        }
        
        # Build tools summary (with call_id - Task 3)
        tools_summary = []
        for tool_event in tools_executed:
            if tool_event.get('event_type') == 'tool_complete':
                tools_summary.append({
                    'tool_name': tool_event.get('data', {}).get('tool_name', 'unknown'),
                    'call_id': tool_event.get('data', {}).get('call_id'),  # Task 3
                    'status': tool_event.get('data', {}).get('status', 'unknown'),
                    'duration_ms': tool_event.get('data', {}).get('duration_ms', 0)
                })
        
        logger.info(
            f"[THINKING:COMPLETE] Done: {len(response_text)} chars, "
            f"{len(tools_summary)} tools, {len(thinking_events)} events"
        )
        
        return ThinkingChatResponse(
            status="success",
            session_id=session_id,
            response=response_text,
            thinking_summary=thinking_summary,
            tools_executed=tools_summary,
            stats=final_stats,
            agent_tree=agent_tree  # Task 6
        )
        
    except Exception as e:
        logger.error(f"[THINKING:COMPLETE] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AGENT TREE ENDPOINT 
# ============================================================================

# @router.get(
#     "/debug/tree/{flow_id}",
#     summary="Get Agent Tree for Flow",
#     description="Get the agent execution tree for debugging (Task 6)"
# )
# async def get_agent_tree(
#     flow_id: str,
#     api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
# ):
#     """
#     Get agent tree for a specific flow
    
#     Useful for debugging complex queries
#     """
#     # In production, this would fetch from a cache or database
#     # For now, return a placeholder
#     return JSONResponse({
#         "status": "info",
#         "message": "Agent tree is returned in the stream/complete response",
#         "hint": "Check the 'agent_tree' field in DoneEvent or ThinkingChatResponse"
#     })