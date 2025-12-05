import json
import asyncio
from datetime import datetime
from pydantic import BaseModel, validator, Field
from typing import List, Dict, Any, Tuple, Optional
from collections.abc import AsyncGenerator
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import APIRouter, Depends, Request, HTTPException

from src.schemas.response import (
    GeneralChatBot,
    PromptRequest
)
from src.utils.logger.custom_logging import LoggerMixin
from src.handlers.llm_chat_handler import ChatHandler, ChatMessageHistory, ChatService
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.helpers.llm_helper import LLMGeneratorProvider
from src.helpers.streaming_tool_helper import StreamingToolHelper
from src.providers.provider_factory import ProviderType
from src.agents.memory.memory_manager import MemoryManager
from src.providers.provider_factory import ModelProviderFactory

from src.services.background_tasks import trigger_summary_update_nowait


from src.agents.memory.core_memory import CoreMemory
from src.agents.memory.memory_update_agent import MemoryUpdateAgent
from src.helpers.context_assembler import ContextAssembler
from src.helpers.token_counter import TokenCounter
from src.helpers.system_prompts import get_system_message_general_chat
from src.agents.memory.recursive_summary import RecursiveSummaryManager
from src.utils.constants import LocalModelName
from src.handlers.v2.chat_handler import ChatHandler
from src.agents.reasoning.inner_thoughts_agent import InnerThoughtsAgent
from src.services.memory_search_service import MemorySearchService

router = APIRouter()

# Initialize Instance
api_key_auth = APIKeyAuth()
logger_mixin = LoggerMixin()
logger = logger_mixin.logger

chat_service = ChatService()
chat_handler = ChatHandler()
llm_provider = LLMGeneratorProvider()

token_counter = TokenCounter()
core_memory_manager = CoreMemory()
summary_manager = RecursiveSummaryManager(
    model_name=LocalModelName.GPTOSS,
    provider_type=ProviderType.OLLAMA
)
context_assembler = ContextAssembler()
memory_update_agent = MemoryUpdateAgent()


class ChatRequest(GeneralChatBot):
    enable_reasoning: bool = Field(default=True, description="Use Inner Thoughts reasoning")
    enable_memory_search: bool = Field(default=True, description="Allow memory searches")
    enable_tools: bool = Field(default=True, description="Allow tool execution")
    stream: bool = Field(default=True, description="Stream response")


class InnerThoughtsDebugRequest(BaseModel):
    query: str = Field(..., description="Query to analyze")
    session_id: str = Field(..., description="Session ID")
    include_context: bool = Field(default=True, description="Include full context")
    include_decision: bool = Field(default=True, description="Include reasoning decision")


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

@router.post("/chat/general/stream")
async def chat_general_stream_with_core_memory(
    request: Request,
    data: GeneralChatBot,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    General chat endpoint with MemGPT-style Core Memory
    
    Flow:
        1. Load Core Memory (Persona + Human blocks)
        2. Load Active Summary (if exists)
        3. Get Recent Chat History (FIFO queue, last 20 messages)
        3. Assemble Context with proper priority
        4. Generate Response with full context
        5. Save conversation to database
        6. Auto-update Core Memory if user shares profile info
        7. CHECK & CREATE Summary if > threshold
    """
    
    # Extract user info from request
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found")
    
    # Step 0: Create session if not exists
    if not data.session_id:
        try:
            data.session_id = chat_service.create_chat_session(
                user_id=user_id,
                organization_id=organization_id
            )
            logger.info(f"Created new session: {data.session_id}")
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise HTTPException(status_code=500, detail="Failed to create session")
    
    # Step 1: Assemble context (with Core Memory + Summary + History)
    try:
        system_prompt = get_system_message_general_chat(
            enable_thinking=data.enable_thinking,
            model_name=data.model_name,
            detected_language=data.target_language or "auto"
        )
        
        # Context assembler AUTO load summary if any
        messages, context_metadata = await context_assembler.prepare_messages_for_llm(
            user_id=user_id,
            session_id=data.session_id,
            current_query=data.question_input,
            system_prompt=system_prompt,
            enable_thinking=data.enable_thinking,
            model_name=data.model_name,
            max_history_messages=20  # FIFO queue size
        )
        
        logger.info(
            f"[CONTEXT ASSEMBLED] Total tokens: {context_metadata['total_tokens']}, "
            f"Usage: {context_metadata['usage_percent']}%, "
            f"Core Memory: {context_metadata['token_breakdown']['core_memory']} tokens"
        )
        
    except Exception as e:
        logger.error(f"Error assembling context: {e}")
        raise HTTPException(status_code=500, detail=f"Context assembly failed: {str(e)}")
    
    # Step 2: Save user question
    question_id = None
    try:
        question_id = chat_service.save_user_question(
            session_id=data.session_id,
            created_at=datetime.datetime.now(),
            created_by=user_id,
            content=data.question_input
        )
        logger.info(f"Saved user question: {question_id}")
    except Exception as save_error:
        logger.error(f"Error saving user question: {save_error}")
    
    # Step 3: Stream response
    async def format_sse():
        full_response = []
        
        try:
            async for chunk in llm_provider.stream_response(
                model_name=data.model_name,
                messages=messages,
                provider_type=data.provider_type,
                api_key=ModelProviderFactory._get_api_key(data.provider_type),
                clean_thinking=True,
                enable_thinking=data.enable_thinking
            ):
                if chunk:
                    full_response.append(chunk)
                    yield f"{json.dumps({'content': chunk})}\n\n"
            
            # Join complete response
            complete_response = ''.join(full_response)
            
            # Step 4: Save assistant response
            if complete_response:
                try:
                    chat_service.save_assistant_response(
                        session_id=data.session_id,
                        created_at=datetime.datetime.now(),
                        question_id=question_id,
                        content=complete_response,
                        response_time=0.1
                    )
                    logger.info(f"Saved assistant response for session {data.session_id}")
                except Exception as save_error:
                    logger.error(f"Error saving response: {save_error}")
            
            # Step 5: Auto-update Core Memory (background - non-blocking)
            if data.session_id and user_id and complete_response:
                try:
                    # Analyze conversation for memory updates
                    asyncio.create_task(
                        memory_update_agent.analyze_for_updates(
                            user_id=user_id,
                            user_message=data.question_input,
                            assistant_message=complete_response,
                            model_name=data.model_name,
                            provider_type=data.provider_type
                        )
                    )
                    logger.info("Triggered memory update analysis")
                except Exception as mem_error:
                    logger.warning(f"Memory update trigger failed: {mem_error}")
            
            # Step 6: CHECK & CREATE SUMMARY (Background - Non-blocking)
            if data.session_id and user_id and complete_response:
                try:
                    asyncio.create_task(
                        summary_manager.check_and_create_summary(
                            session_id=data.session_id,
                            user_id=user_id,
                            organization_id=organization_id,
                            model_name=data.model_name,
                            provider_type=data.provider_type
                        )
                    )
                    logger.info("Triggered summary check/creation")
                except Exception as summary_error:
                    logger.warning(f"Summary trigger failed: {summary_error}")
            
            # Step 7: Send context metadata as final event
            # metadata_event = {
            #     "type": "metadata",
            #     "context_stats": {
            #         "total_tokens": context_metadata['total_tokens'],
            #         "usage_percent": context_metadata['usage_percent'],
            #         "core_memory_tokens": context_metadata['token_breakdown']['core_memory'],
            #         "history_tokens": context_metadata['token_breakdown']['chat_history'],
            #         "has_core_memory": True
            #     }
            # }
            # yield f"{json.dumps(metadata_event)}\n\n"
            
            # Final done signal
            yield "[DONE]\n\n"
            # yield f"{json.dumps({'type': 'done'})}\n\n"
            
        except Exception as stream_error:
            yield f"{json.dumps({'error': str(e)})}\n\n"
            yield "[DONE]\n\n"
            # logger.error(f"Streaming error: {stream_error}")
            # error_event = {
            #     "type": "error",
            #     "message": str(stream_error)
            # }
            # yield f"{json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        format_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================================
# Non-streaming endpoint
# ============================================================================
@router.post("/chat/general")
async def chat_general_non_stream(
    request: Request,
    data: GeneralChatBot,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Non-streaming version of general chat with Core Memory + Recursive Summary
    """
    
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found")
    
    # Create session if needed
    if not data.session_id:
        data.session_id = chat_service.create_chat_session(
            user_id=user_id,
            organization_id=organization_id
        )
    
    # Save user question
    question_id = chat_service.save_user_question(
        session_id=data.session_id,
        created_at=datetime.datetime.now(),
        created_by=user_id,
        content=data.question_input
    )
    
    try:
        # Assemble context (with Summary)
        system_prompt = get_system_message_general_chat(
            enable_thinking=data.enable_thinking,
            model_name=data.model_name,
            detected_language=data.target_language or "auto"
        )
        
        messages, context_metadata = await context_assembler.prepare_messages_for_llm(
            user_id=user_id,
            session_id=data.session_id,
            current_query=data.question_input,
            system_prompt=system_prompt,
            enable_thinking=data.enable_thinking,
            model_name=data.model_name,
            max_history_messages=20
        )
        
        response = await llm_provider.generate_response(
            model_name=data.model_name,
            messages=messages,
            provider_type=data.provider_type,
            api_key=ModelProviderFactory._get_api_key(data.provider_type),
            enable_thinking=data.enable_thinking
        )
        
        complete_response = response.get('content', '')
        
        # Save assistant response
        chat_service.save_assistant_response(
            session_id=data.session_id,
            created_at=datetime.datetime.now(),
            question_id=question_id,
            content=complete_response,
            response_time=0.1
        )
        
        # CHECK & CREATE SUMMARY (Background)
        task_name = f"summary-task-{data.session_id}-{user_id}"

        asyncio.create_task(
            summary_manager.check_and_create_summary(
                session_id=data.session_id,
                user_id=user_id,
                organization_id=organization_id,
                model_name=data.model_name,
                provider_type=data.provider_type
            ),
            name=task_name
        )

        # Trigger memory updates (background)
        asyncio.create_task(
            memory_update_agent.analyze_for_updates(
                user_id=user_id,
                user_message=data.question_input,
                assistant_message=complete_response,
                model_name=data.model_name,
                provider_type=data.provider_type
            )
        )
        
        return {
            "status": "success",
            "data": {
                "response": complete_response,
                "session_id": data.session_id,
                "context_stats": context_metadata
            }
        }
        
    except Exception as e:
        logger.error(f"Error in non-streaming chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# CORE MEMORY MANAGEMENT ENDPOINTS
# ============================================================================

class CoreMemoryUpdateRequest(BaseModel):
    persona: Optional[str] = None
    human: Optional[str] = None
    append_to_human: Optional[str] = None
    section: Optional[str] = None


@router.get("/core-memory/stats")
async def get_core_memory_stats(
    request: Request,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Get core memory statistics for current user"""
    user_id = getattr(request.state, "user_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found")
    
    try:
        stats = await core_memory_manager.get_memory_stats(user_id)
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/core-memory/view")
async def view_core_memory(
    request: Request,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """View current core memory blocks"""
    user_id = getattr(request.state, "user_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found")
    
    try:
        core_memory = await core_memory_manager.load_core_memory(user_id)
        stats = await core_memory_manager.get_memory_stats(user_id)
        
        return {
            "status": "success",
            "data": {
                "persona": core_memory['persona'],
                "human": core_memory['human'],
                "stats": stats
            }
        }
    except Exception as e:
        logger.error(f"Error viewing core memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/core-memory/update")
async def update_core_memory(
    request: Request,
    update_data: CoreMemoryUpdateRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Update core memory blocks
    
    Options:
    - Update persona only: provide persona field
    - Update human only: provide human field  
    - Update both: provide both fields
    - Append to human: provide append_to_human and optional section
    """
    user_id = getattr(request.state, "user_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found")
    
    try:
        # Determine operation type
        if update_data.append_to_human:
            # Append operation
            success = await core_memory_manager.append_to_human(
                user_id=user_id,
                new_info=update_data.append_to_human,
                section=update_data.section
            )
            operation = "append"
            
        elif update_data.persona and update_data.human:
            # Full update
            success = await core_memory_manager.save_core_memory(
                user_id=user_id,
                persona=update_data.persona,
                human=update_data.human
            )
            operation = "full_update"
            
        elif update_data.persona:
            # Persona only
            success = await core_memory_manager.update_persona(
                user_id=user_id,
                new_persona=update_data.persona
            )
            operation = "update_persona"
            
        elif update_data.human:
            # Human only
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
            # Get updated stats
            stats = await core_memory_manager.get_memory_stats(user_id)
            
            return {
                "status": "success",
                "message": f"Core memory {operation} completed",
                "data": stats
            }
        else:
            raise HTTPException(status_code=500, detail="Update failed")
            
    except ValueError as ve:
        # Token limit exceeded
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error updating core memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/core-memory/reset")
async def reset_core_memory(
    request: Request,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """Reset core memory to default for current user"""
    user_id = getattr(request.state, "user_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found")
    
    try:
        # Delete user config file to force recreation
        user_config_path = core_memory_manager._get_user_config_path(user_id)
        
        if user_config_path.exists():
            user_config_path.unlink()
            logger.info(f"Deleted core memory config for user {user_id}")
        
        # Create new default
        core_memory = await core_memory_manager.load_core_memory(user_id)
        stats = await core_memory_manager.get_memory_stats(user_id)
        
        return {
            "status": "success",
            "message": "Core memory reset to default",
            "data": {
                "persona": core_memory['persona'],
                "human": core_memory['human'],
                "stats": stats
            }
        }
        
    except Exception as e:
        logger.error(f"Error resetting core memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/core-memory/manual-update")
async def manual_update_memory(
    request: Request,
    category: str,
    information: str,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Manually add specific information to user's memory profile
    
    Args:
        category: Category like "Portfolio", "Watchlist", "Risk Tolerance"
        information: Information to add
    """
    user_id = getattr(request.state, "user_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found")
    
    try:
        success = await memory_update_agent.manual_update(
            user_id=user_id,
            category=category,
            information=information
        )
        
        if success:
            stats = await core_memory_manager.get_memory_stats(user_id)
            return {
                "status": "success",
                "message": f"Added {category} to memory",
                "data": stats
            }
        else:
            raise HTTPException(status_code=500, detail="Update failed")
            
    except Exception as e:
        logger.error(f"Error in manual memory update: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/summary/stats")
async def get_summary_stats(
    request: Request,
    session_id: str,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Get recursive summary statistics for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        Summary statistics including version, tokens, message count
    """
    try:
        stats = await summary_manager.get_summary_stats(session_id)
        
        return {
            "status": "success",
            "session_id": session_id,
            "data": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting summary stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary/view")
async def view_summary(
    request: Request,
    session_id: str,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    View current active summary for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        Active summary text and metadata
    """
    try:
        summary_text = await summary_manager.get_active_summary(session_id)
        stats = await summary_manager.get_summary_stats(session_id)
        
        if not summary_text:
            return {
                "status": "success",
                "message": "No summary exists for this session yet",
                "session_id": session_id,
                "data": None
            }
        
        return {
            "status": "success",
            "session_id": session_id,
            "data": {
                "summary_text": summary_text,
                "stats": stats
            }
        }
        
    except Exception as e:
        logger.error(f"Error viewing summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summary/create")
async def force_create_summary(
    request: Request,
    session_id: str,
    model_name: str,
    provider_type: str,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Manually trigger summary creation for a session
    Useful for testing or forcing summarization
    
    Args:
        session_id: Session identifier
        
    Returns:
        Summary creation result
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found")
    
    try:
        result = await summary_manager.check_and_create_summary(
            session_id=session_id,
            user_id=user_id,
            organization_id=organization_id,
            model_name=model_name,
            provider_type=provider_type
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error creating summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/summary/delete")
async def delete_summaries(
    request: Request,
    session_id: str,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Delete all summaries for a session
    Useful for cleanup or resetting conversation state
    
    Query Parameters:
        session_id: Session identifier
        
    Returns:
        {
            "status": "success",
            "message": "All summaries deleted successfully",
            "session_id": "xxx",
            "deleted": true
        }
    """
    user_id = getattr(request.state, "user_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found")
    
    try:
        logger.info(
            f"Delete summaries requested for session {session_id} "
            f"by user {user_id}"
        )
        
        success = await summary_manager.delete_session_summaries(session_id)
        
        if success:
            logger.info(f"Successfully deleted summaries for session {session_id}")
            return {
                "status": "success",
                "message": "All summaries deleted successfully",
                "session_id": session_id,
                "deleted": True
            }
        else:
            logger.warning(f"No summaries found or failed to delete for session {session_id}")
            return {
                "status": "success",
                "message": "No summaries to delete or deletion failed",
                "session_id": session_id,
                "deleted": False
            }
        
    except Exception as e:
        logger.error(f"Error deleting summaries: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete summaries: {str(e)}"
        )


# ============================================================================
# 5. GET SUMMARY CONFIGURATION
# ============================================================================
@router.get("/summary/config")
async def get_summary_config(
    request: Request,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Get current summary configuration settings
    
    Returns:
        {
            "status": "success",
            "data": {
                "message_threshold": 20,
                "max_summary_tokens": 1000,
                "model_name": "gpt-4o-mini",
                "provider_type": "openai"
            }
        }
    """
    try:
        config = {
            "message_threshold": RecursiveSummaryManager.MESSAGE_THRESHOLD,
            "max_summary_tokens": RecursiveSummaryManager.MAX_SUMMARY_TOKENS,
            "model_name": summary_manager.summary_service.model_name,
            "provider_type": summary_manager.provider_type
        }
        
        return {
            "status": "success",
            "data": config
        }
        
    except Exception as e:
        logger.error(f"Error getting summary config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get config: {str(e)}"
        )


# ============================================================================
# BULK SUMMARY OPERATIONS
# ============================================================================
@router.post("/summary/bulk-create")
async def bulk_create_summaries(
    request: Request,
    session_ids: List[str],
    model_name: str,
    provider_type: str,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Create summaries for multiple sessions at once
    Useful for batch processing or migration
    
    Body:
        {
            "session_ids": ["session1", "session2", "session3"]
        }
        
    Returns:
        {
            "status": "success",
            "data": {
                "total_sessions": 3,
                "successful": 2,
                "failed": 1,
                "results": [
                    {
                        "session_id": "session1",
                        "success": true,
                        "version": 2
                    },
                    ...
                ]
            }
        }
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found")
    
    if not session_ids or len(session_ids) == 0:
        raise HTTPException(status_code=400, detail="No session IDs provided")
    
    try:
        results = []
        successful = 0
        failed = 0
        
        for session_id in session_ids:
            try:
                result = await summary_manager.check_and_create_summary(
                    session_id=session_id,
                    user_id=user_id,
                    organization_id=organization_id,
                    model_name=model_name,
                    provider_type=provider_type
                )
                
                if result.get('created'):
                    successful += 1
                    results.append({
                        "session_id": session_id,
                        "success": True,
                        "version": result.get('version'),
                        "token_count": result.get('token_count')
                    })
                else:
                    results.append({
                        "session_id": session_id,
                        "success": False,
                        "reason": result.get('reason', result.get('error'))
                    })
                    
            except Exception as e:
                failed += 1
                results.append({
                    "session_id": session_id,
                    "success": False,
                    "error": str(e)
                })
        
        logger.info(
            f"Bulk summary creation completed: "
            f"{successful} successful, {failed} failed out of {len(session_ids)}"
        )
        
        return {
            "status": "success",
            "data": {
                "total_sessions": len(session_ids),
                "successful": successful,
                "failed": failed,
                "results": results
            }
        }
        
    except Exception as e:
        logger.error(f"Error in bulk summary creation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Bulk operation failed: {str(e)}"
        )



# ============================================================================
# Main Chat Endpoints
# ============================================================================

# @router.post("/chat/v2/stream")
# async def chat_v2_stream(
#     request: Request,
#     data: ChatRequest,
#     api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
# ):
#     """
#     Advanced chat with Inner Thoughts reasoning (Streaming)
    
#     Flow:
#     1. Load context (Core Memory + Summary + History)
#     2. Inner Thoughts reasoning
#     3. Conditional memory search
#     4. Tool execution if needed
#     5. Stream response with full context
#     6. Auto-update memory in background
    
#     Features:
#     - Multi-language support (semantic understanding)
#     - Smart memory search (only when needed)
#     - Tool execution based on reasoning
#     - Recursive summary integration
#     - Core memory auto-updates
#     """
    
#     user_id = getattr(request.state, "user_id", None)
#     organization_id = getattr(request.state, "organization_id", None)
    
#     if not user_id:
#         raise HTTPException(status_code=400, detail="User ID required")
    
#     # Create session if not exists
#     if not data.session_id:
#         data.session_id = chat_service.create_chat_session(
#             user_id=user_id,
#             organization_id=organization_id
#         )
    
#     async def stream_sse():
#         try:
#             # Start event
#             yield f"data: {json.dumps({'type': 'start', 'session_id': data.session_id})}\n\n"
            
#             # Stream response with reasoning
#             async for chunk in chat_handler.handle_chat_with_reasoning(
#                 query=data.question_input,
#                 session_id=data.session_id,
#                 user_id=user_id,
#                 model_name=data.model_name,
#                 provider_type=data.provider_type,
#                 organization_id=organization_id,
#                 enable_thinking=data.enable_thinking,
#                 stream=data.stream
#             ):
#                 if chunk:
#                     yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
            
#             # Done event
#             yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
#         except Exception as e:
#             error_msg = f"Error: {str(e)}"
#             yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
    
#     return StreamingResponse(
#         stream_sse(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#             "X-Accel-Buffering": "no"
#         }
#     )


# @router.post("/complete")
# async def chat_v2_complete(
#     request: Request,
#     data: ChatRequest,
#     api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
# ):
#     # Get user ID and organization ID
#     user_id = getattr(request.state, "user_id", None)
#     organization_id = getattr(request.state, "organization_id", None)
    
#     # Validate user ID
#     if not user_id:
#         raise HTTPException(status_code=400, detail="User ID required")
    
#     # Create session if not exists
#     if not data.session_id:
#         data.session_id = chat_service.create_chat_session(
#             user_id=user_id,
#             organization_id=organization_id
#         )
    
#     try:
#         # Stream full response 
#         response_chunks = []
        
#         # Stream response
#         async for chunk in chat_handler.handle_chat_with_reasoning(
#             query=data.question_input,
#             session_id=data.session_id,
#             user_id=user_id,
#             model_name=data.model_name,
#             provider_type=data.provider_type,
#             organization_id=organization_id,
#             enable_thinking=data.enable_thinking,
#             stream=False
#         ):
#             if chunk:
#                 response_chunks.append(chunk)
        
#         # Join the full response
#         complete_response = ''.join(response_chunks)
        
#         return JSONResponse(
#             content={
#                 "status": "success",
#                 "data": {
#                     "response": complete_response,
#                     "session_id": data.session_id,
#                     "timestamp": datetime.now().isoformat()
#                 }
#             }
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Memory Management Endpoints
# ============================================================================

@router.post("/memory/search")
async def search_memory_v2(
    request: Request,
    session_id: str,
    query: str,
    search_type: str = "hybrid",
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Direct memory search endpoint
    
    Args:
        session_id: Session ID
        query: Search query
        search_type: recall/archival/hybrid
    """
    
    user_id = getattr(request.state, "user_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")
    
    try:
        memory_search = MemorySearchService()
        
        results = {
            "recall": [],
            "archival": []
        }
        
        if search_type in ["recall", "hybrid"]:
            results["recall"] = await memory_search.search_recall_memory(
                session_id=session_id,
                strategy="topic",
                params={"topic": query, "limit": 10},
                user_id=user_id
            )
        
        if search_type in ["archival", "hybrid"]:
            results["archival"] = await memory_search.search_archival_memory(
                query=query,
                user_id=user_id,
                limit=5
            )
        
        return JSONResponse(
            content={
                "status": "success",
                "query": query,
                "search_type": search_type,
                "results": {
                    "recall_count": len(results["recall"]),
                    "archival_count": len(results["archival"]),
                    "recall_results": results["recall"][:5],  # First 5
                    "archival_results": results["archival"][:3]  # First 3
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# System Status Endpoints
# ============================================================================

@router.get("/stats")
async def get_system_stats(
    request: Request,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Get system statistics and usage metrics
    """
    
    user_id = getattr(request.state, "user_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")
    
    try:
        # Get user's core memory stats
        memory_stats = await core_memory_manager.get_memory_stats(user_id)
        
        # Get recent sessions
        from src.database.repository.sessions import SessionRepository
        session_repo = SessionRepository()
        recent_sessions = await session_repo.get_recent_sessions(
            user_id=user_id,
            limit=5
        )
        
        return JSONResponse(
            content={
                "status": "success",
                "user_id": user_id,
                "stats": {
                    "core_memory": memory_stats,
                    "recent_sessions": recent_sessions,
                    "total_sessions": len(recent_sessions)
                },
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post(
    "/complete",
    summary="Complete Chat (Non-Streaming)",
    description="""
    Main chat API với full agent capabilities:
    - Task-based planning & execution
    - Multi-tool orchestration
    - Memory system (Core + Recursive Summary)
    - Validation & adaptive replanning
    
    **Use case**: Chat interfaces cần response một lần duy nhất
    """
)
async def chat_v2_complete(
    request: Request,
    data: ChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Complete chat response - Returns full answer after processing
    
    Flow:
    1. Load context (Core Memory + Summary + Chat History)
    2. Task Planning
    3. Tool Execution with Validation
    4. Context Assembly
    5. LLM Generation
    6. Memory Updates (background)
    
    Returns:
        JSON response with complete answer
    """
    # Get user ID and organization ID
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    # Validate user ID
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")
    
    # Create session if not exists
    if not data.session_id:
        data.session_id = chat_service.create_chat_session(
            user_id=user_id,
            organization_id=organization_id
        )
    
    try:
        # Collect streaming chunks
        response_chunks = []
        
        # Stream response internally (non-streaming to user)
        async for chunk in chat_handler.handle_chat_with_reasoning(
            query=data.question_input,
            session_id=data.session_id,
            user_id=user_id,
            model_name=data.model_name,
            provider_type=data.provider_type,
            organization_id=organization_id,
            enable_thinking=data.enable_thinking,
            stream=False  # Internal streaming only
        ):
            if chunk:
                response_chunks.append(chunk)
        
        # Join the full response
        complete_response = ''.join(response_chunks)
        
        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "response": complete_response,
                    "session_id": data.session_id,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Chat processing failed: {str(e)}"
        )


@router.post(
    "/stream",
    summary="Streaming Chat",
    description="""
    Streaming version of chat API - Returns response as Server-Sent Events
    
    **Use case**: Chat interfaces cần real-time progressive display
    
    **Response format**: 
    - Content-Type: text/event-stream
    - Each chunk: "data: {text}\n\n"
    - End marker: "data: [DONE]\n\n"
    """
)
async def chat_v2_stream(
    request: Request,
    data: ChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Streaming chat response - Returns answer progressively
    
    Same flow as /complete but streams to user in real-time
    
    Returns:
        StreamingResponse with Server-Sent Events
    """
    # Get user ID and organization ID
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    # Validate user ID
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")
    
    # Create session if not exists
    if not data.session_id:
        data.session_id = chat_service.create_chat_session(
            user_id=user_id,
            organization_id=organization_id
        )
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        try:
            async for chunk in chat_handler.handle_chat_with_reasoning(
                query=data.question_input,
                session_id=data.session_id,
                user_id=user_id,
                model_name=data.model_name,
                provider_type=data.provider_type,
                organization_id=organization_id,
                enable_thinking=data.enable_thinking,
                stream=True
            ):
                if chunk:
                    # Format as Server-Sent Event
                    yield f"data: {chunk}\n\n"
            
            # Send completion marker
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield f"data: {error_msg}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


# ════════════════════════════════════════════════════════════════════════════
# DEBUG/TEST APIs - Component Testing Endpoints
# ════════════════════════════════════════════════════════════════════════════

@router.post(
    "/debug/planning",
    summary="[DEBUG] Test Planning Agent",
    description="""
    Test only the Planning Agent component
    
    **Use case**: 
    - Debug task planning logic
    - Verify query intent detection
    - Check tool selection
    
    **Returns**: Task plan without execution
    """,
    tags=["Debug & Testing"]
)
async def debug_planning(
    request: Request,
    data: ChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Test Planning Agent in isolation
    
    Returns:
        Task plan with intent, strategy, and task breakdown
    """
    user_id = getattr(request.state, "user_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")
    
    try:
        from src.agents.planning.planning_agent import PlanningAgent
        from src.providers.provider_factory import ProviderType
        
        planning_agent = PlanningAgent(
            model_name=data.model_name,
            provider_type=ProviderType.OPENAI
        )
        
        # Get task plan
        task_plan = await planning_agent.think_and_plan(
            query=data.question_input,
            recent_chat=[],
            core_memory={},
            summary="",
            available_tools=chat_handler.AVAILABLE_TOOLS
        )
        
        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "query": data.question_input,
                    "intent": task_plan.query_intent,
                    "strategy": task_plan.strategy,
                    "complexity": task_plan.estimated_complexity,
                    "tasks": [
                        {
                            "id": task.id,
                            "description": task.description,
                            "priority": task.priority.value,
                            "tools": [
                                {
                                    "tool_name": tool.tool_name,
                                    "params": tool.params
                                }
                                for tool in task.tools_needed
                            ],
                            "dependencies": task.dependencies
                        }
                        for task in task_plan.tasks
                    ]
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Planning test failed: {str(e)}"
        )


@router.post(
    "/debug/tool-execution",
    summary="[DEBUG] Test Tool Execution",
    description="""
    Test individual tool execution without full agent flow
    
    **Use case**:
    - Debug tool implementation
    - Verify API responses
    - Test data formatting
    
    **Parameters**:
    - tool_name: Name of tool to test
    - symbols: List of stock symbols
    - Additional tool-specific parameters
    """,
    tags=["Debug & Testing"]
)
async def debug_tool_execution(
    request: Request,
    tool_name: str,
    symbols: list[str],
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Execute single tool for testing
    
    Args:
        tool_name: Tool to execute (e.g., "showStockPrice")
        symbols: List of symbols (e.g., ["AAPL", "NVDA"])
        
    Returns:
        Raw tool execution result
    """
    try:
        from src.services.v2.tool_execution_service import ToolExecutionService
        
        tool_service = ToolExecutionService()
        
        # Execute tool
        result = await tool_service.execute_single_tool(
            tool_name=tool_name,
            tool_params={
                "symbols": symbols,
                "additional_params": {}
            },
            query=f"Debug test for {symbols}",
            chat_history="",
            system_language="en",
            provider_type="openai",
            model_name="gpt-4o-mini"
        )
        
        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "tool_name": tool_name,
                    "symbols": symbols,
                    "result": result
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tool execution test failed: {str(e)}"
        )


@router.post(
    "/debug/validation",
    summary="[DEBUG] Test Validation Agent",
    description="""
    Test validation logic on sample tool results
    
    **Use case**:
    - Debug validation criteria
    - Test deterministic vs LLM validation
    - Verify missing data detection
    
    **Parameters**:
    - tool_name: Tool that was executed
    - tool_results: Simulated or actual tool results
    - query: Original user query
    """,
    tags=["Debug & Testing"]
)
async def debug_validation(
    request: Request,
    tool_name: str,
    tool_results: Dict[str, Any],
    query: str,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Test validation agent on sample results
    
    Args:
        tool_name: Name of tool being validated
        tool_results: Tool execution results
        query: User's original query
        
    Returns:
        Validation result with sufficiency assessment
    """
    try:
        from src.agents.validation.validation_agent import ValidationAgent
        from src.providers.provider_factory import ProviderType
        
        validator = ValidationAgent(
            provider_type=ProviderType.OPENAI,
            model_name="gpt-4o-mini"
        )
        
        # Validate results
        validation = await validator.validate_tool_results(
            original_query=query,
            tool_name=tool_name,
            tool_params={},
            tool_results=tool_results,
            query_intent="Debug test",
            symbols=[]
        )
        
        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "query": query,
                    "tool_name": tool_name,
                    "validation": {
                        "is_sufficient": validation.is_sufficient,
                        "confidence": validation.confidence,
                        "missing_data": validation.missing_data,
                        "next_action": validation.next_action,
                        "reasoning": validation.reasoning
                    }
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Validation test failed: {str(e)}"
        )


@router.post(
    "/debug/memory",
    summary="[DEBUG] Test Memory System",
    description="""
    Test memory loading and searching
    
    **Use case**:
    - Debug Core Memory loading
    - Test Recursive Summary generation
    - Verify memory search functionality
    
    **Returns**: Current memory state for user/session
    """,
    tags=["Debug & Testing"]
)
async def debug_memory(
    request: Request,
    session_id: str,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Inspect memory state for debugging
    
    Args:
        session_id: Session to inspect
        
    Returns:
        Core memory, summary, and recent chat history
    """
    user_id = getattr(request.state, "user_id", None)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")
    
    try:
        from src.agents.memory.core_memory import CoreMemory
        from src.agents.memory.recursive_summary import RecursiveSummaryManager
        from src.database.repository.sessions import SessionRepository
        
        # Load memory components
        core_memory = CoreMemory()
        summary_manager = RecursiveSummaryManager()
        session_repo = SessionRepository()
        
        # Get core memory
        core_mem = await core_memory.load_core_memory(user_id)
        
        # Get summary
        summary = await summary_manager.get_active_summary(session_id)
        
        # Get recent chat
        recent_chat = await session_repo.get_session_messages(
            session_id=session_id,
            limit=10
        )
        
        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "core_memory": {
                        "persona": core_mem.get('persona', ''),
                        "human": core_mem.get('human', ''),
                        "total_chars": len(core_mem.get('persona', '')) + len(core_mem.get('human', ''))
                    },
                    "summary": {
                        "content": summary,
                        "length": len(summary) if summary else 0
                    },
                    "recent_chat": {
                        "message_count": len(recent_chat),
                        "messages": recent_chat
                    }
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Memory debug failed: {str(e)}"
        )


@router.get(
    "/debug/health",
    summary="[DEBUG] System Health Check",
    description="""
    Check health of all agent components
    
    **Returns**:
    - Component availability
    - Configuration status
    - Quick diagnostic info
    """,
    tags=["Debug & Testing"]
)
async def debug_health():
    """
    System health check for debugging
    
    Returns:
        Status of all major components
    """
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    try:
        # Check Planning Agent
        try:
            from src.agents.planning.planning_agent import PlanningAgent
            health_status["components"]["planning_agent"] = {
                "status": "available",
                "task_decomposition": chat_handler.ENABLE_TASK_DECOMPOSITION
            }
        except Exception as e:
            health_status["components"]["planning_agent"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check Validation Agent
        try:
            from src.agents.validation.validation_agent import ValidationAgent
            health_status["components"]["validation_agent"] = {
                "status": "available",
                "enabled": chat_handler.ENABLE_VALIDATION
            }
        except Exception as e:
            health_status["components"]["validation_agent"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check Tool Execution Service
        try:
            from src.services.v2.tool_execution_service import ToolExecutionService
            health_status["components"]["tool_execution"] = {
                "status": "available",
                "available_tools": chat_handler.AVAILABLE_TOOLS
            }
        except Exception as e:
            health_status["components"]["tool_execution"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check Memory System
        try:
            from src.agents.memory.core_memory import CoreMemory
            health_status["components"]["memory_system"] = {
                "status": "available",
                "core_memory": "ok",
                "recursive_summary": "ok"
            }
        except Exception as e:
            health_status["components"]["memory_system"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Overall status
        all_ok = all(
            comp.get("status") == "available" 
            for comp in health_status["components"].values()
        )
        
        health_status["overall_status"] = "healthy" if all_ok else "degraded"
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        return JSONResponse(
            content={
                "overall_status": "error",
                "error": str(e)
            },
            status_code=500
        )