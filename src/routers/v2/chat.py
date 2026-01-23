import json
import asyncio
import datetime
from typing import Dict, Any, Optional
from collections.abc import AsyncGenerator

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from src.utils.logger.custom_logging import LoggerMixin
from src.handlers.llm_chat_handler import ChatService
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.providers.provider_factory import ProviderType

from src.agents.memory.core_memory import CoreMemory
from src.agents.memory.memory_update_agent import MemoryUpdateAgent
from src.agents.memory.recursive_summary import RecursiveSummaryManager
from src.handlers.v2.chat_handler import ChatHandler
from src.utils.constants import APIModelName
from src.config.mode_config import ResponseMode


# ============================================================================
# ROUTER INITIALIZATION
# ============================================================================

router = APIRouter()

# Initialize services
api_key_auth = APIKeyAuth()
logger_mixin = LoggerMixin()
logger = logger_mixin.logger

chat_service = ChatService()
chat_handler = ChatHandler()

core_memory_manager = CoreMemory()
summary_manager = RecursiveSummaryManager()
memory_update_agent = MemoryUpdateAgent()


# ============================================================================
# REQUEST SCHEMAS - Defined inline for clarity
# ============================================================================

class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation")
    question_input: str = Field(
        ...,
        description="User's question",
        min_length=1,
        max_length=10000
    )
    chart_displayed: bool = Field(
        default=False,
        description=(
            "Is chart displayed?"
        )
    )
    target_language: Optional[str] = Field(
        default=None,
        description="Target response language",
        examples=["vi", "en", "auto"]
    )
    model_name: str = Field(
        default=APIModelName.GPT41Nano,
        description="LLM model name",
        examples=["gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini", "gpt-oss:20b"]
    )
    provider_type: str = Field(
        default=ProviderType.OPENAI,
        description="LLM provider type",
        examples=["openai", "ollama"]
    )
    collection_name: str = Field(
        default='',
        description="Collection name for RAG"
    )
    use_multi_collection: bool = Field(
        default=False,
        description="Multi-collection"
    )
    enable_thinking: bool = Field(
        default=False,
        description=(
            "Enable extended thinking mode for supported models."
        )
    )
    enable_think_tool: bool = Field(
        default=False,
        description=(
            "Enable Think Tool for step-by-step reasoning before planning."
        )
    )
    enable_compaction: bool = Field(
        default=True,
        description=(
            "Enable automatic context compaction when approaching token limits. "
            "Uses smart summary to preserve important context while reducing token usage."
        )
    )
    response_mode: str = Field(
        default=ResponseMode.AUTO,
        description=(
            "Response mode for controlling response quality vs speed tradeoff. "
            "Options: 'auto' (LLM decides), 'fast' (quick responses), 'expert' (thorough analysis)"
        ),
        examples=["auto", "fast", "expert"]
    )
    

class CoreMemoryUpdateRequest(BaseModel):
    persona: Optional[str] = Field(
        default=None, 
        description="New content for persona block (replaces existing)"
    )
    human: Optional[str] = Field(
        default=None, 
        description="New content for human block (replaces existing)"
    )
    append_to_human: Optional[str] = Field(
        default=None, 
        description="Content to append to human block"
    )
    section: Optional[str] = Field(
        default=None, 
        description="Section name for append operation (e.g., 'Portfolio', 'Preferences')"
    )


class ChatResponse(BaseModel):
    status: str
    data: Dict[str, Any]

@router.post(
    "/chat/complete",
    summary="Complete Chat (Non-Streaming)",
    description="""
Main chat API với full agent capabilities.

**Features:**
- Task-based planning & execution
- Multi-tool orchestration (32 tools across 10 categories)
- Memory system (Core Memory + Recursive Summary)
- Validation with retry on failure
- Optional Think Tool for complex reasoning
- **Response Modes**: Fast/Auto/Expert for speed vs quality control

**Response Modes:**
- `fast`: Quick responses (3-6s), filtered tools, smaller model
- `auto`: LLM decides based on query complexity (default)
- `expert`: Thorough analysis (15-45s), all tools, larger model

**Flow:**
1. Phase 1: Load Context (Core Memory + Summary + Chat History)
2. Phase 2: Planning (Classification → Tool Selection → Task Plan)
3. Phase 4: Tool Execution (Parallel/Sequential with validation)
4. Phase 5: Context Assembly (Organize tool results)
5. Phase 6: LLM Response Generation
6. Phase 7: Background Tasks (Memory updates, Summary creation)

**Use case**: Chat interfaces cần response một lần duy nhất
    """
)
async def chat_complete(
    request: Request,
    data: ChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Complete chat response - Returns full answer after processing.
    """
    # Get user info from request state
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")
    
    # Create session if not provided
    if not data.session_id:
        data.session_id = chat_service.create_chat_session(
            user_id=user_id,
            organization_id=organization_id
        )
        logger.info(f"[CHAT] Created new session: {data.session_id}")
    
    try:
        # Collect response chunks
        response_chunks = []
        
        # Process chat
        async for chunk in chat_handler.handle_chat_with_reasoning(
            query=data.question_input,
            session_id=data.session_id,
            user_id=user_id,
            model_name=data.model_name,
            provider_type=data.provider_type,
            organization_id=organization_id,
            enable_thinking=data.enable_thinking,
            enable_think_tool=data.enable_think_tool,
            enable_compaction=data.enable_compaction,
            chart_displayed=data.chart_displayed,
            stream=False
        ):
            if chunk:
                response_chunks.append(chunk)
        
        complete_response = ''.join(response_chunks)
        
        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "response": complete_response,
                    "session_id": data.session_id,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            }
        )
        
    except Exception as e:
        logger.error(f"[CHAT] Complete failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Chat processing failed: {str(e)}"
        )


@router.post(
    "/chat/stream",
    summary="Streaming Chat",
    description="""
Streaming version of chat API - Returns response as Server-Sent Events.

**Features:** Same as /chat/complete + Response Mode streaming events

**Response Modes:**
- `fast` (⚡): Quick responses (3-6s), gpt-4o-mini, 8 tools, 2 turns max
- `auto` (🔄): LLM classifies complexity, routes to fast/expert (default)
- `expert` (🧠): Deep analysis (15-45s), gpt-4o, 31 tools, 6 turns max

**SSE Events for Response Modes:**
- `mode_selecting`: Started classifying query complexity (AUTO mode)
- `mode_selected`: Mode determined with reason and model info

**Response format:**
- Content-Type: text/event-stream
- Each chunk: `{"content": "text"}\\n\\n`
- End marker: `[DONE]\\n\\n`

**Use case**: Chat interfaces cần real-time progressive display
    """
)
async def chat_stream(
    request: Request,
    data: ChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Streaming chat response - Returns answer progressively via SSE.
    """
    # Get user info from request state
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")
    
    # Create session if not provided
    if not data.session_id:
        data.session_id = chat_service.create_chat_session(
            user_id=user_id,
            organization_id=organization_id
        )
        logger.info(f"[CHAT] Created new session: {data.session_id}")
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generate streaming response with SSE format"""
        try:
            async for chunk in chat_handler.handle_chat_with_reasoning(
                query=data.question_input,
                session_id=data.session_id,
                user_id=user_id,
                model_name=data.model_name,
                provider_type=data.provider_type,
                organization_id=organization_id,
                enable_thinking=data.enable_thinking,
                enable_think_tool=data.enable_think_tool,
                enable_compaction=data.enable_compaction,
                chart_displayed=data.chart_displayed,
                stream=True
            ):
                if chunk:
                    payload = json.dumps({"content": chunk})
                    yield f"{payload}\n\n"
            
            yield "[DONE]\n\n"
            
        except Exception as e:
            logger.error(f"[CHAT] Stream error: {e}", exc_info=True)
            error_payload = json.dumps({"error": str(e)})
            yield f"{error_payload}\n\n"
            yield "[DONE]\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================================
# CORE MEMORY APIs
# ============================================================================

@router.get(
    "/core-memory/view",
    summary="View Core Memory",
    description="View current core memory blocks (Persona + Human)",
)
async def view_core_memory(
    request: Request,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """View current core memory for user"""
    user_id = getattr(request.state, "user_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")
    
    try:
        core_memory = await core_memory_manager.load_core_memory(user_id)
        stats = await core_memory_manager.get_memory_stats(user_id)
        
        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "persona": core_memory['persona'],
                    "human": core_memory['human'],
                    "stats": stats
                }
            }
        )
    except Exception as e:
        logger.error(f"[MEMORY] Error viewing core memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/core-memory/update",
    summary="Update Core Memory",
    description="Update core memory blocks (persona, human, or append)",
)
async def update_core_memory(
    request: Request,
    update_data: CoreMemoryUpdateRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Update core memory blocks"""
    user_id = getattr(request.state, "user_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")
    
    try:
        if update_data.append_to_human:
            success = await core_memory_manager.append_to_human(
                user_id=user_id,
                new_info=update_data.append_to_human,
                section=update_data.section
            )
            operation = "append"
        elif update_data.persona and update_data.human:
            success = await core_memory_manager.save_core_memory(
                user_id=user_id,
                persona=update_data.persona,
                human=update_data.human
            )
            operation = "full_update"
        elif update_data.persona:
            success = await core_memory_manager.update_persona(
                user_id=user_id,
                new_persona=update_data.persona
            )
            operation = "update_persona"
        elif update_data.human:
            success = await core_memory_manager.update_human(
                user_id=user_id,
                new_human=update_data.human
            )
            operation = "update_human"
        else:
            raise HTTPException(
                status_code=400, 
                detail="Must provide persona, human, or append_to_human"
            )
        
        if success:
            stats = await core_memory_manager.get_memory_stats(user_id)
            return JSONResponse(
                content={
                    "status": "success",
                    "message": f"Core memory {operation} completed",
                    "data": stats
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Update failed")
            
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"[MEMORY] Error updating core memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/core-memory/stats",
    summary="Core Memory Statistics",
    description="Get core memory usage statistics",
)
async def get_core_memory_stats(
    request: Request,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Get core memory statistics"""
    user_id = getattr(request.state, "user_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")
    
    try:
        stats = await core_memory_manager.get_memory_stats(user_id)
        return JSONResponse(
            content={
                "status": "success",
                "data": stats
            }
        )
    except Exception as e:
        logger.error(f"[MEMORY] Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SUMMARY APIs
# ============================================================================

@router.get(
    "/summary/view",
    summary="View Session Summary",
    description="View current active summary for a session",
)
async def view_summary(
    request: Request,
    session_id: str,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """View active summary for session"""
    try:
        summary_text = await summary_manager.get_active_summary(session_id)
        stats = await summary_manager.get_summary_stats(session_id)
        
        if not summary_text:
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "No summary exists for this session yet",
                    "session_id": session_id,
                    "data": None
                }
            )
        
        return JSONResponse(
            content={
                "status": "success",
                "session_id": session_id,
                "data": {
                    "summary_text": summary_text,
                    "stats": stats
                }
            }
        )
    except Exception as e:
        logger.error(f"[SUMMARY] Error viewing summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/summary/stats",
    summary="Summary Statistics",
    description="Get summary statistics for a session",
)
async def get_summary_stats(
    request: Request,
    session_id: str,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Get summary statistics"""
    try:
        stats = await summary_manager.get_summary_stats(session_id)
        return JSONResponse(
            content={
                "status": "success",
                "session_id": session_id,
                "data": stats
            }
        )
    except Exception as e:
        logger.error(f"[SUMMARY] Error getting summary stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/summary/create",
    summary="Force Create Summary",
    description="Manually trigger summary creation for a session",
)
async def force_create_summary(
    request: Request,
    session_id: str,
    model_name: str = APIModelName.GPT41Nano,
    provider_type: str = ProviderType.OPENAI,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Force create summary for session"""
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")
    
    try:
        result = await summary_manager.check_and_create_summary(
            session_id=session_id,
            user_id=user_id,
            organization_id=organization_id,
            model_name=model_name,
            provider_type=provider_type
        )
        
        return JSONResponse(
            content={
                "status": "success",
                "session_id": session_id,
                "data": result
            }
        )
    except Exception as e:
        logger.error(f"[SUMMARY] Error creating summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/summary/delete",
    summary="Delete Session Summaries",
    description="Delete all summaries for a session",
)
async def delete_summaries(
    request: Request,
    session_id: str,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Delete all summaries for session"""
    user_id = getattr(request.state, "user_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")
    
    try:
        success = await summary_manager.delete_session_summaries(session_id)
        
        return JSONResponse(
            content={
                "status": "success",
                "message": "Summaries deleted" if success else "No summaries to delete",
                "session_id": session_id,
                "deleted": success
            }
        )
    except Exception as e:
        logger.error(f"[SUMMARY] Error deleting summaries: {e}")
        raise HTTPException(status_code=500, detail=str(e))
