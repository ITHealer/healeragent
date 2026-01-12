"""
Character Agent Chat API

Provides API endpoints for chatting with investment character agents.
Each character has a unique personality, investment philosophy, and metric focus.

The system injects character personas into the existing Agent Loop,
leveraging all existing tools and memory systems.

Endpoints:
- GET  /character-agents/list              - List all available characters
- GET  /character-agents/{id}              - Get character details
- POST /character-agents/{id}/chat         - Non-streaming chat
- POST /character-agents/{id}/stream       - Streaming chat (SSE)
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections.abc import AsyncGenerator

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from src.utils.logger.custom_logging import LoggerMixin
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.handlers.llm_chat_handler import ChatService
from src.providers.provider_factory import ProviderType
from src.utils.constants import APIModelName

# Character system
from src.agents.characters import (
    CharacterRouter,
    get_character_router,
    CharacterPersona,
)

# Agent Loop + Intent Classifier (reuse V4 architecture)
from src.agents.classification import (
    IntentClassifier,
    IntentResult,
    get_intent_classifier,
)
from src.agents.unified import (
    UnifiedAgent,
    get_unified_agent,
)

# Streaming
from src.services.streaming_event_service import (
    StreamEventEmitter,
    StreamEventType,
    format_done_marker,
    with_heartbeat,
)

# SSE Cancellation handling
from src.utils.sse_cancellation import (
    with_cancellation,
)

# Memory
from src.agents.memory import (
    WorkingMemoryIntegration,
    setup_working_memory_for_request,
)
from src.agents.memory.memory_manager import get_memory_manager

# Context Builder
from src.services.context_builder import (
    get_context_builder,
    ContextBuilder,
    ContextPhase,
    AssembledContext,
)

# Conversation Compaction
from src.services.conversation_compactor import (
    check_and_compact_if_needed,
    CompactionResult,
)


# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter(prefix="/character-agents")

_api_key_auth = APIKeyAuth()
_logger = LoggerMixin().logger
_chat_service = ChatService()


# =============================================================================
# Singletons (ALL lazy-loaded to prevent blocking on import)
# =============================================================================

_character_router: Optional[CharacterRouter] = None
_intent_classifier: Optional[IntentClassifier] = None
_unified_agent: Optional[UnifiedAgent] = None
_memory_manager = None  # Lazy-loaded to prevent blocking


def _get_memory_manager():
    """Get or create MemoryManager singleton (lazy-loaded)."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = get_memory_manager()
    return _memory_manager


def _get_character_router() -> CharacterRouter:
    """Get or create CharacterRouter singleton."""
    global _character_router
    if _character_router is None:
        _character_router = get_character_router()
    return _character_router


def _get_intent_classifier() -> IntentClassifier:
    """Get or create IntentClassifier singleton."""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = get_intent_classifier()
    return _intent_classifier


def _get_unified_agent() -> UnifiedAgent:
    """Get or create UnifiedAgent singleton."""
    global _unified_agent
    if _unified_agent is None:
        _unified_agent = get_unified_agent()
    return _unified_agent


# =============================================================================
# Session Repository (for conversation history)
# =============================================================================

_session_repo = None
_chat_repo = None


def _get_session_repo():
    """Get or create session repository."""
    global _session_repo
    if _session_repo is None:
        from src.database.repository.sessions import SessionRepository
        _session_repo = SessionRepository()
    return _session_repo


def _get_chat_repo():
    """Get or create chat repository."""
    global _chat_repo
    if _chat_repo is None:
        from src.database.repository.chat import ChatRepository
        _chat_repo = ChatRepository()
    return _chat_repo


async def _load_conversation_history(session_id: str, limit: int = 10) -> List[Dict[str, str]]:
    """Load recent conversation history from session."""
    try:
        session_repo = _get_session_repo()
        recent_chat = await session_repo.get_session_messages(
            session_id=session_id,
            limit=limit,
        )
        recent_chat.reverse()  # Oldest first for LLM
        return [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in recent_chat
            if msg.get("content")
        ]
    except Exception as e:
        _logger.warning(f"[CHARACTER_CHAT] Failed to load conversation history: {e}")
        return []


async def _save_conversation_turn(
    session_id: str,
    user_id: str,
    user_query: str,
    assistant_response: str,
    character_id: str,
    character_name: str,
    response_time_ms: float,
) -> None:
    """Save conversation turn with character metadata."""
    try:
        chat_repo = _get_chat_repo()

        # Save user question
        question_id = chat_repo.save_user_question(
            session_id=session_id,
            created_at=datetime.now(),
            created_by=user_id,
            content=user_query,
        )

        # Save assistant response
        chat_repo.save_assistant_response(
            session_id=session_id,
            created_at=datetime.now(),
            question_id=question_id,
            content=assistant_response,
            response_time=response_time_ms / 1000.0,
        )

        # Store in memory with character metadata
        memory_manager = _get_memory_manager()
        await memory_manager.store_conversation_turn(
            session_id=session_id,
            user_id=user_id,
            query=user_query,
            response=assistant_response,
            metadata={
                "character_id": character_id,
                "character_name": character_name,
                "conversation_type": "character_chat",
            },
            importance_score=0.7,  # Character conversations are typically important
        )

        _logger.info(
            f"[CHARACTER_CHAT] ✅ Saved conversation: "
            f"character={character_name}, query={len(user_query)}chars"
        )

    except Exception as e:
        _logger.warning(f"[CHARACTER_CHAT] Failed to save conversation: {e}")


# =============================================================================
# Request/Response Models
# =============================================================================

class CharacterChatRequest(BaseModel):
    """Request for character chat."""
    query: str = Field(
        ...,
        alias="question_input",
        description="User's question or message",
        max_length=10000,
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation continuity",
    )
    model_name: str = Field(
        default=APIModelName.GPT41Nano,
        description="LLM model name",
    )
    provider_type: str = Field(
        default=ProviderType.OPENAI,
        description="LLM provider type",
    )
    enable_thinking: bool = Field(
        default=True,
        description="Enable thinking/reasoning display",
    )
    enable_tools: bool = Field(
        default=True,
        description="Enable tool execution for data fetching",
    )

    class Config:
        populate_by_name = True


class CharacterInfoResponse(BaseModel):
    """Character information response."""
    character_id: str
    name: str
    title: str
    description: str
    avatar_url: str
    investment_style: str
    specialties: List[str]
    metric_focus: List[str]
    time_horizon: str
    risk_tolerance: str


class CharacterListResponse(BaseModel):
    """List of characters response."""
    characters: List[CharacterInfoResponse]
    total: int


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/list", response_model=CharacterListResponse)
async def list_characters(
    api_key_data: Dict[str, Any] = Depends(_api_key_auth.author_with_api_key),
):
    """
    List all available investment character agents.

    Returns character info including name, title, investment style,
    specialties, and metric focus areas.
    """
    char_router = _get_character_router()
    characters = char_router.list_characters()

    return CharacterListResponse(
        characters=[
            CharacterInfoResponse(
                character_id=c.character_id,
                name=c.name,
                title=c.title,
                description=c.description,
                avatar_url=c.avatar_url,
                investment_style=c.investment_style,
                specialties=c.specialties,
                metric_focus=c.metric_focus,
                time_horizon=c.time_horizon,
                risk_tolerance=c.risk_tolerance,
            )
            for c in characters
        ],
        total=len(characters),
    )


@router.get("/{character_id}", response_model=CharacterInfoResponse)
async def get_character(
    character_id: str,
    api_key_data: Dict[str, Any] = Depends(_api_key_auth.author_with_api_key),
):
    """
    Get detailed information about a specific character.

    Args:
        character_id: Character identifier (e.g., "warren_buffett")
    """
    char_router = _get_character_router()
    character = char_router.get_character_info(character_id)

    if not character:
        raise HTTPException(
            status_code=404,
            detail=f"Character '{character_id}' not found. "
            f"Available: {char_router.list_character_ids()}",
        )

    return CharacterInfoResponse(
        character_id=character.character_id,
        name=character.name,
        title=character.title,
        description=character.description,
        avatar_url=character.avatar_url,
        investment_style=character.investment_style,
        specialties=character.specialties,
        metric_focus=character.metric_focus,
        time_horizon=character.time_horizon,
        risk_tolerance=character.risk_tolerance,
    )


@router.post("/{character_id}/stream")
async def stream_character_chat(
    request: Request,
    character_id: str,
    data: CharacterChatRequest,
    api_key_data: Dict[str, Any] = Depends(_api_key_auth.author_with_api_key),
):
    """
    Stream chat with a character agent via SSE.

    The character's personality and investment philosophy are injected
    into the Agent Loop, which handles tool calling and response generation.

    Args:
        character_id: Character to chat with (e.g., "warren_buffett")
        data: Chat request with query and configuration

    Returns:
        SSE stream with chat events
    """
    # Validate character exists
    char_router = _get_character_router()
    persona = char_router.get_character(character_id)

    if not persona:
        raise HTTPException(
            status_code=404,
            detail=f"Character '{character_id}' not found",
        )

    # Get user info
    user_id = getattr(request.state, "user_id", None)
    org_id = getattr(request.state, "organization_id", None)

    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")

    # Create/get session
    session_id = data.session_id
    if not session_id:
        session_id = _chat_service.create_chat_session(
            user_id=user_id,
            organization_id=org_id,
        )

    emitter = StreamEventEmitter(session_id=session_id)

    async def _generate_stream() -> AsyncGenerator[str, None]:
        """Generate SSE events for character chat."""
        start_time = datetime.now()
        full_response = []

        try:
            # =================================================================
            # Phase 1: Setup
            # =================================================================
            yield emitter.emit_session_start({
                "mode": "character_chat",
                "character_id": character_id,
                "character_name": persona.name,
            })

            # Setup working memory
            flow_id = f"char_{character_id}_{session_id[:8]}_{int(datetime.now().timestamp())}"
            wm_integration = setup_working_memory_for_request(
                session_id=session_id,
                user_id=user_id,
                flow_id=flow_id,
            )

            # =================================================================
            # Phase 2: Load Context
            # =================================================================
            context_builder = get_context_builder()
            assembled_context = await context_builder.build_context(
                session_id=session_id,
                user_id=user_id,
                phase=ContextPhase.INTENT_CLASSIFICATION.value,
            )

            core_memory_context = assembled_context.core_memory
            conversation_summary = assembled_context.summary
            wm_symbols = assembled_context.wm_symbols

            # Load conversation history
            conversation_history = await _load_conversation_history(
                session_id=session_id,
                limit=10,
            )

            # Compact if needed
            if conversation_history:
                conversation_history, _ = await check_and_compact_if_needed(
                    messages=conversation_history,
                    system_prompt="",
                    symbols=[],
                    additional_context="",
                )

            # =================================================================
            # Phase 3: Intent Classification
            # =================================================================
            yield emitter.emit_classifying()

            intent_classifier = _get_intent_classifier()
            intent_result = await intent_classifier.classify(
                query=data.query,
                conversation_history=conversation_history,
                working_memory_symbols=wm_symbols,
                core_memory_context=core_memory_context,
                conversation_summary=conversation_summary,
            )

            yield emitter.emit_classified(
                query_type=intent_result.query_type,
                requires_tools=intent_result.requires_tools,
                symbols=intent_result.validated_symbols,
                categories=[],
                confidence=intent_result.confidence,
                language=intent_result.response_language,
                reasoning=intent_result.reasoning,
            )

            # Save classification to working memory
            wm_integration.save_classification(
                query_type=intent_result.query_type,
                categories=[],
                symbols=intent_result.validated_symbols,
                language=intent_result.response_language,
                reasoning=intent_result.reasoning,
            )

            # =================================================================
            # Phase 4: Build Character System Prompt
            # =================================================================
            # Inject character persona into system prompt
            character_system_prompt = char_router.build_system_prompt(
                character_id=character_id,
                additional_context=f"""
## CURRENT CONVERSATION CONTEXT
- User is asking about: {intent_result.intent_summary or data.query[:100]}
- Detected symbols: {', '.join(intent_result.validated_symbols) if intent_result.validated_symbols else 'None'}
- Response language: {intent_result.response_language}

Remember to stay in character as {persona.name} throughout the conversation.
Focus on the metrics most important to your investment philosophy: {', '.join(persona.metric_focus)}.
""",
            )

            # =================================================================
            # Phase 5: Agent Execution with Character Prompt
            # =================================================================
            unified_agent = _get_unified_agent()

            _logger.info(
                f"[CHARACTER_CHAT] 🚀 Starting {persona.name} agent loop "
                f"with {len(unified_agent.catalog.get_tool_names())} tools"
            )

            async for event in unified_agent.run_stream_with_all_tools(
                query=data.query,
                intent_result=intent_result,
                conversation_history=conversation_history,
                conversation_summary=conversation_summary,
                system_language=intent_result.response_language,
                user_id=int(user_id) if user_id else None,
                session_id=session_id,
                core_memory=core_memory_context,
                enable_reasoning=data.enable_thinking,
                model_name=data.model_name,
                provider_type=data.provider_type,
                max_turns=6,
                working_memory_symbols=wm_symbols,
                # CRITICAL: Inject character persona as system prompt override
                system_prompt_override=character_system_prompt,
            ):
                event_type = event.get("type", "unknown")

                if event_type == "turn_start":
                    turn = event.get("turn", 1)
                    yield emitter.emit_turn_start(turn_number=turn, max_turns=6)

                elif event_type == "tool_calls":
                    tools = event.get("tools", [])
                    yield emitter.emit_tool_calls(tools)

                elif event_type == "tool_results":
                    results = event.get("results", [])
                    yield emitter.emit_tool_results(results)

                elif event_type == "content":
                    content = event.get("content", "")
                    full_response.append(content)
                    yield emitter.emit_content(content)

                elif event_type == "thinking":
                    if data.enable_thinking:
                        yield emitter.emit_thinking(event.get("content", ""))

                elif event_type == "done":
                    break

            # =================================================================
            # Phase 6: Save Conversation
            # =================================================================
            complete_response = "".join(full_response)
            response_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            await _save_conversation_turn(
                session_id=session_id,
                user_id=user_id,
                user_query=data.query,
                assistant_response=complete_response,
                character_id=character_id,
                character_name=persona.name,
                response_time_ms=response_time_ms,
            )

            # Emit completion
            yield emitter.emit_done(
                metadata={
                    "character_id": character_id,
                    "character_name": persona.name,
                    "symbols_analyzed": intent_result.validated_symbols,
                    "response_time_ms": response_time_ms,
                }
            )
            yield format_done_marker()

        except Exception as e:
            _logger.error(f"[CHARACTER_CHAT] Error: {e}", exc_info=True)
            yield emitter.emit_error(str(e))
            yield format_done_marker()

    # Cleanup function for SSE cancellation
    async def cleanup_on_disconnect():
        _logger.info(f"[CHARACTER_CHAT] Client disconnected - session {session_id}")

    # Wrap with heartbeat (15s interval) then cancellation handling
    generator_with_heartbeat = with_heartbeat(
        event_generator=_generate_stream(),
        emitter=emitter,
        heartbeat_interval=15.0,
    )

    return StreamingResponse(
        with_cancellation(
            request=request,
            generator=generator_with_heartbeat,
            cleanup_fn=cleanup_on_disconnect,
            check_interval=0.5,
            emit_cancelled_event=True,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{character_id}/chat")
async def chat_with_character(
    request: Request,
    character_id: str,
    data: CharacterChatRequest,
    api_key_data: Dict[str, Any] = Depends(_api_key_auth.author_with_api_key),
):
    """
    Non-streaming chat with a character agent.

    Args:
        character_id: Character to chat with (e.g., "warren_buffett")
        data: Chat request with query and configuration

    Returns:
        Complete chat response
    """
    # Validate character exists
    char_router = _get_character_router()
    persona = char_router.get_character(character_id)

    if not persona:
        raise HTTPException(
            status_code=404,
            detail=f"Character '{character_id}' not found",
        )

    # Get user info
    user_id = getattr(request.state, "user_id", None)
    org_id = getattr(request.state, "organization_id", None)

    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")

    # Create/get session
    session_id = data.session_id
    if not session_id:
        session_id = _chat_service.create_chat_session(
            user_id=user_id,
            organization_id=org_id,
        )

    start_time = datetime.now()

    try:
        # Load context
        context_builder = get_context_builder()
        assembled_context = await context_builder.build_context(
            session_id=session_id,
            user_id=user_id,
            phase=ContextPhase.INTENT_CLASSIFICATION.value,
        )

        conversation_history = await _load_conversation_history(session_id, limit=10)
        if conversation_history:
            conversation_history, _ = await check_and_compact_if_needed(
                messages=conversation_history,
                system_prompt="",
                symbols=[],
                additional_context="",
            )

        # Intent classification
        intent_classifier = _get_intent_classifier()
        intent_result = await intent_classifier.classify(
            query=data.query,
            conversation_history=conversation_history,
            working_memory_symbols=assembled_context.wm_symbols,
            core_memory_context=assembled_context.core_memory,
            conversation_summary=assembled_context.summary,
        )

        # Build character prompt
        character_system_prompt = char_router.build_system_prompt(
            character_id=character_id,
            additional_context=f"""
## CURRENT CONVERSATION CONTEXT
- User is asking about: {intent_result.intent_summary or data.query[:100]}
- Detected symbols: {', '.join(intent_result.validated_symbols) if intent_result.validated_symbols else 'None'}
- Response language: {intent_result.response_language}

Remember to stay in character as {persona.name} throughout the conversation.
""",
        )

        # Run agent (non-streaming)
        unified_agent = _get_unified_agent()
        result = await unified_agent.run_with_all_tools(
            query=data.query,
            intent_result=intent_result,
            conversation_history=conversation_history,
            conversation_summary=assembled_context.summary,
            system_language=intent_result.response_language,
            user_id=int(user_id) if user_id else None,
            session_id=session_id,
            core_memory=assembled_context.core_memory,
            model_name=data.model_name,
            provider_type=data.provider_type,
            max_turns=6,
            working_memory_symbols=assembled_context.wm_symbols,
            system_prompt_override=character_system_prompt,
        )

        response_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Save conversation
        await _save_conversation_turn(
            session_id=session_id,
            user_id=user_id,
            user_query=data.query,
            assistant_response=result.response,
            character_id=character_id,
            character_name=persona.name,
            response_time_ms=response_time_ms,
        )

        return {
            "success": True,
            "session_id": session_id,
            "character": {
                "id": character_id,
                "name": persona.name,
                "title": persona.title,
            },
            "content": result.response,
            "symbols_analyzed": intent_result.validated_symbols,
            "total_turns": result.total_turns,
            "total_tool_calls": result.total_tool_calls,
            "response_time_ms": response_time_ms,
        }

    except Exception as e:
        _logger.error(f"[CHARACTER_CHAT] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
