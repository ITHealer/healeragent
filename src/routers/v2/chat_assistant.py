from datetime import datetime
from typing import Dict, Any, Optional, List
from collections.abc import AsyncGenerator

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from src.utils.logger.custom_logging import LoggerMixin
from src.utils.image import (
    ImageProcessor,
    ProcessedImage,
    process_image_input,
    ImageContent,
    ImageSource,
)
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.handlers.llm_chat_handler import ChatService
from src.providers.provider_factory import ProviderType
from src.utils.constants import APIModelName
# Classification
from src.agents.classification import (
    IntentClassifier,
    IntentResult,
    get_intent_classifier,
)
# Charts
from src.agents.charts import (
    charts_to_dict_list,
    get_chart_resolver,
)
# Unified Agent (ChatGPT-style tool execution)
from src.agents.unified import (
    UnifiedAgent,
    get_unified_agent,
)
# Streaming
from src.services.streaming_event_service import (
    StreamEventEmitter,
    format_done_marker,
    with_heartbeat,
)
# Thinking Timeline (ChatGPT-style "Thought for Xs" display)
from src.agents.streaming import (
    ThinkingTimeline,
    ThinkingPhase,
)
# SSE Cancellation handling
from src.utils.sse_cancellation import with_cancellation
# LEARN Phase - Memory updates after execution
from src.agents.hooks import LearnHook
# Working Memory Integration (session-scoped scratchpad)
from src.agents.memory import (
    WorkingMemoryIntegration,
    setup_working_memory_for_request,
)
# Context Management Service (auto-compaction)
from src.services.context_management_service import (
    ContextManagementService,
    get_context_manager,
)
# Conversation Compactor (auto-compress long conversations)
from src.services.conversation_compactor import (
    check_and_compact_if_needed,
    get_conversation_compactor,
    CompactionResult,
)
# Recursive Summary Manager (for long conversation summarization)
from src.agents.memory.recursive_summary import (
    get_recursive_summary_manager,
    RecursiveSummaryManager,
)
# ContextBuilder - Centralized context assembly
from src.services.context_builder import (
    get_context_builder,
    ContextBuilder,
    ContextPhase,
    AssembledContext,
)


# Router instance
router = APIRouter(prefix="/chat-assistant")

# Core services
_api_key_auth = APIKeyAuth()
_logger = LoggerMixin().logger
_chat_service = ChatService()

# --- Agent Singletons ---

_unified_agent: Optional[UnifiedAgent] = None


def _get_unified_agent() -> UnifiedAgent:
    """Get or create Unified Agent singleton."""
    global _unified_agent
    if _unified_agent is None:
        _unified_agent = get_unified_agent()
    return _unified_agent


# --- Context Management (auto-compaction) ---

_context_manager: Optional[ContextManagementService] = None


def _get_context_manager() -> ContextManagementService:
    """Get or create Context Manager singleton for auto-compaction."""
    global _context_manager
    if _context_manager is None:
        _context_manager = get_context_manager(
            enable_compaction=True,
            max_context_tokens=180000,
            trigger_percent=80.0,  # Auto-compact at 80% usage
            strategy="smart_summary",
        )
    return _context_manager


# --- Image Processing ---

_image_processor: Optional[ImageProcessor] = None


def _get_image_processor() -> ImageProcessor:
    """Get or create ImageProcessor singleton."""
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessor()
    return _image_processor


async def _process_images(
    images: Optional[List["ImageInput"]],
) -> Optional[List[ProcessedImage]]:
    """
    Process image inputs from request into ProcessedImages.

    Handles flexible input formats:
    - Standard: source="url", data="https://..."
    - Flexible: source="https://..." (auto-detect URL in source field)
    - Base64: source="base64", data="iVBORw0..."
    - Data URL: source="data_url", data="data:image/png;base64,..."

    Args:
        images: List of ImageInput from request

    Returns:
        List of ProcessedImage or None if no images
    """
    if not images:
        return None

    processor = _get_image_processor()
    processed = []

    for img_input in images:
        try:
            source_value = img_input.source
            data_value = img_input.data or ""  # Handle None

            # ========================================================
            # VALIDATION: Skip invalid/placeholder image data
            # ========================================================
            # Check for invalid placeholder values
            invalid_values = {"string", "null", "undefined", "none", ""}
            if source_value.lower() in invalid_values and not data_value:
                _logger.debug(f"[IMAGE] Skipping invalid image source: '{source_value}'")
                continue

            # Skip if both source and data are empty/invalid
            if not source_value or source_value.lower() in invalid_values:
                if not data_value or data_value.lower() in invalid_values:
                    _logger.debug(f"[IMAGE] Skipping empty image data")
                    continue

            # Auto-detect: if source looks like a URL, treat it as the data
            if source_value.startswith(("http://", "https://")):
                # User put URL in source field instead of data
                data_value = source_value
                source_type = ImageSource.URL
                _logger.debug(f"[IMAGE] Auto-detected URL in source field")

            # Auto-detect: if source looks like a data URL
            elif source_value.startswith("data:"):
                data_value = source_value
                source_type = ImageSource.BASE64
                # Extract base64 from data URL
                parts = data_value.split(",", 1)
                if len(parts) == 2:
                    data_value = parts[1]
                _logger.debug(f"[IMAGE] Auto-detected data URL in source field")

            # Standard source types
            elif source_value == "url":
                # Validate that data contains a valid URL
                if not data_value or not data_value.startswith(("http://", "https://")):
                    _logger.debug(f"[IMAGE] Skipping: source='url' but data is invalid: '{data_value[:50] if data_value else 'empty'}'")
                    continue
                source_type = ImageSource.URL
            elif source_value == "base64":
                # Validate that data looks like base64 (min length)
                if not data_value or len(data_value) < 50:
                    _logger.debug(f"[IMAGE] Skipping: source='base64' but data too short or empty")
                    continue
                source_type = ImageSource.BASE64
            elif source_value == "data_url":
                source_type = ImageSource.BASE64
                # Extract base64 from data URL format
                if data_value.startswith("data:"):
                    parts = data_value.split(",", 1)
                    if len(parts) == 2:
                        data_value = parts[1]
            else:
                # Try to auto-detect from data field
                if data_value.startswith(("http://", "https://")):
                    source_type = ImageSource.URL
                elif data_value.startswith("data:"):
                    source_type = ImageSource.BASE64
                    parts = data_value.split(",", 1)
                    if len(parts) == 2:
                        data_value = parts[1]
                elif len(data_value) > 100:
                    # Assume base64 only if data is long enough
                    source_type = ImageSource.BASE64
                else:
                    # Invalid or placeholder data, skip
                    _logger.debug(f"[IMAGE] Skipping unrecognized format: source='{source_value}', data='{data_value[:30] if data_value else 'empty'}'")
                    continue

            image_content = ImageContent(
                source=source_type,
                data=data_value,
                media_type=img_input.media_type,
            )

            processed_img = await processor.process(image_content)
            processed.append(processed_img)

        except Exception as e:
            _logger.warning(f"[IMAGE] Failed to process image: {e}")
            # Continue with other images

    return processed if processed else None


# --- Request/Response Schemas ---

class ImageInput(BaseModel):
    """
    Image input for multimodal requests.

    Flexible input formats (auto-detected):
    - URL in source: source="https://example.com/image.png" (data ignored)
    - Standard URL: source="url", data="https://example.com/image.png"
    - Base64: source="base64", data="iVBORw0KGgo..."
    - Data URL: source="data:image/png;base64,..." (data ignored)
    """
    source: str = Field(
        default="url",
        description="Image source type (url/base64/data_url) OR the actual URL/data URL directly",
        examples=["url", "https://example.com/image.png", "data:image/png;base64,..."]
    )
    data: Optional[str] = Field(
        default=None,
        description="Image data: URL or base64 string. Can be omitted if source contains the URL/data directly."
    )
    media_type: Optional[str] = Field(
        default=None,
        description="MIME type (image/png, image/jpeg, etc.). Auto-detected if not provided.",
        examples=["image/png", "image/jpeg", "image/webp"]
    )
    alt_text: Optional[str] = Field(
        default=None,
        description="Alt text description for the image"
    )


class UIContextRequest(BaseModel):
    """UI context for Soft Context Inheritance."""
    active_tab: str = Field(
        default="none",
        description="Active UI tab: stock, crypto, forex, commodity, or none",
        examples=["stock", "crypto", "none"]
    )
    recent_symbols: List[str] = Field(
        default_factory=list,
        description="Recently viewed symbols in UI"
    )
    preferred_quote_currency: str = Field(
        default="USD",
        description="Preferred quote currency for crypto (USD, USDT, etc.)"
    )


class ChatRequest(BaseModel):
    """Chat request schema."""

    query: Optional[str] = Field(
        default=None,
        alias="question_input",
        description="User's question or message. Can be empty if images are provided.",
        max_length=10000
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation continuity"
    )
    mode: str = Field(
        default="auto",
        description="Processing mode: auto, normal, or deep_research",
        examples=["auto", "normal", "deep_research"]
    )
    model_name: str = Field(
        default=APIModelName.GPT41Nano,
        description="LLM model name"
    )
    provider_type: str = Field(
        default=ProviderType.OPENAI,
        description="LLM provider type"
    )
    chart_displayed: bool = Field(
        default=False,
        description="Whether chart is displayed in UI"
    )
    enable_thinking: bool = Field(
        default=True,
        description="Enable thinking/reasoning display in stream"
    )
    enable_tools: bool = Field(
        default=True,
        description="Enable tool execution"
    )
    enable_think_tool: bool = Field(
        default=False,
        description="Enable Think Tool for step-by-step reasoning before planning"
    )
    enable_web_search: bool = Field(
        default=False,
        description="Enable web search for additional information beyond financial data tools. When True, forces web search category."
    )
    enable_compaction: bool = Field(
        default=True,
        description="Enable automatic context compaction when approaching token limits"
    )
    enable_llm_events: bool = Field(
        default=True,
        description="Enable LLM decision events (thought, decision, action)"
    )
    enable_agent_tree: bool = Field(
        default=False,
        description="Enable agent tree tracking for debugging"
    )
    enable_tool_search_mode: bool = Field(
        default=False,
        description="Enable Tool Search Mode: Start with only tool_search for 85% token savings. Agent discovers tools dynamically via semantic search."
    )
    stream_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom stream configuration"
    )
    # Soft Context Inheritance
    ui_context: Optional[UIContextRequest] = Field(
        default=None,
        description="UI context for symbol resolution (active tab, recent symbols)"
    )
    # Multimodal: Image inputs
    images: Optional[List[ImageInput]] = Field(
        default=None,
        description="Optional list of images for multimodal analysis (charts, screenshots, financial reports)"
    )

    class Config:
        populate_by_name = True

class ModeInfo(BaseModel):
    """Processing mode metadata."""
    name: str
    description: str
    typical_llm_calls: str
    typical_duration: str
    use_case: str


class ModesResponse(BaseModel):
    """Available modes response."""
    modes: List[ModeInfo]
    default_mode: str


# --- Database Repositories ---

_session_repo = None
_chat_repo = None


def _get_session_repo():
    """Get or create session repository singleton."""
    global _session_repo
    if _session_repo is None:
        from src.database.repository.sessions import SessionRepository
        _session_repo = SessionRepository()
    return _session_repo


def _get_chat_repo():
    """Get or create chat repository singleton."""
    global _chat_repo
    if _chat_repo is None:
        from src.database.repository.chat import ChatRepository
        _chat_repo = ChatRepository()
    return _chat_repo


async def _save_conversation_turn(
    session_id: str,
    user_id: str,
    user_query: str,
    assistant_response: str,
    response_time_ms: float,
) -> None:
    """
    Save a conversation turn (user question + assistant response) to database.

    This is CRITICAL for conversation memory - without saving,
    follow-up questions have no context.

    Args:
        session_id: Session UUID
        user_id: User ID
        user_query: User's question
        assistant_response: Assistant's response
        response_time_ms: Time taken to generate response
    """
    try:
        chat_repo = _get_chat_repo()

        # Save user question
        question_id = chat_repo.save_user_question(
            session_id=session_id,
            created_at=datetime.now(),
            created_by=user_id,
            content=user_query,
        )
        _logger.debug(f"[CHAT] Saved user question: {question_id[:8]}...")

        # Save assistant response
        response_id = chat_repo.save_assistant_response(
            session_id=session_id,
            created_at=datetime.now(),
            question_id=question_id,
            content=assistant_response,
            response_time=response_time_ms / 1000.0,  # Convert ms to seconds
        )
        _logger.debug(f"[CHAT] Saved assistant response: {response_id[:8]}...")

        _logger.info(
            f"[CHAT] ✅ Conversation saved: "
            f"query={len(user_query)} chars, response={len(assistant_response)} chars"
        )

    except Exception as e:
        _logger.warning(f"[CHAT] Failed to save conversation (non-fatal): {e}")


async def _load_conversation_history(session_id: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    Load recent conversation history from session.

    This is CRITICAL for memory - without loading history,
    the agent has no context about previous messages.

    Args:
        session_id: Session identifier
        limit: Maximum number of messages to load

    Returns:
        List of message dicts with role and content
    """
    try:
        session_repo = _get_session_repo()
        recent_chat = await session_repo.get_session_messages(
            session_id=session_id,
            limit=limit,
        )
        # CRITICAL: Reverse to chronological order (oldest first)
        # Repository returns newest first (desc), but LLM needs oldest first
        recent_chat.reverse()

        # Format for agent
        return [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in recent_chat
            if msg.get("content")
        ]
    except Exception as e:
        _logger.warning(f"[CHAT] Failed to load conversation history: {e}")
        return []


# =============================================================================
# Main Chat API: IntentClassifier + Agent with ALL Tools (Simplified Architecture)
# =============================================================================

# IntentClassifier singleton
_intent_classifier: Optional[IntentClassifier] = None


def _get_intent_classifier() -> IntentClassifier:
    """Get or create IntentClassifier singleton."""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = get_intent_classifier()
    return _intent_classifier


@router.post(
    "/chat",
    summary="Streaming Chat (Intent Classifier + All Tools)",
    description="""
    Chat with simplified architecture:
    - Single LLM call for classification + symbol normalization
    - Agent sees ALL tools and decides which to call (ChatGPT-style)
    - No separate Router - Agent has full autonomy

    New Flow (2 LLM calls total):
    1. IntentClassifier: Extract intent, normalize symbols, determine complexity
    2. UnifiedAgent: See ALL tools, decide & execute (iterative loop)

    Benefits:
    - Simpler architecture (no Router)
    - Agent has full tool access (no category blindness)
    - Symbol normalization in classification (GOOGLE → GOOGL)
    - Faster overall response (fewer LLM calls)
    """,
    response_class=StreamingResponse,
)
async def stream_chat(
    request: Request,
    data: ChatRequest,
    api_key_data: Dict[str, Any] = Depends(_api_key_auth.author_with_api_key),
):
    """
    Stream chat responses via SSE using IntentClassifier + Agent with ALL tools.

    This is the simplified ChatGPT-style architecture:
    - Phase 1: IntentClassifier (single LLM) → symbols, complexity, market_type
    - Phase 2: Agent sees ALL tools → decides which to call → iterative loop
    """
    user_id = getattr(request.state, "user_id", None)
    org_id = getattr(request.state, "organization_id", None)

    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")

    # Validate: at least query or images must be provided
    if not data.query and not data.images:
        raise HTTPException(
            status_code=400,
            detail="Either question_input or images must be provided"
        )

    # Default query for image-only requests
    query = data.query or "Analyze this image"

    session_id = data.session_id
    if not session_id:
        session_id = _chat_service.create_chat_session(
            user_id=user_id,
            organization_id=org_id
        )

    emitter = StreamEventEmitter(session_id=session_id)

    async def _generate() -> AsyncGenerator[str, None]:
        """Generate SSE events using IntentClassifier + Agent with ALL tools."""
        start_time = datetime.now()

        # Stats tracking
        stats = {
            "total_turns": 0,
            "total_tool_calls": 0,
            "charts": None,
            "complexity": None,
            # For LEARN phase
            "tool_results": [],
            "final_content": "",
            # Compaction tracking
            "compaction_result": None,
        }

        # =================================================================
        # THINKING TIMELINE: Track thought process for UI display
        # ChatGPT-style "Thought for Xs" with timeline of steps
        # =================================================================
        thinking_timeline = ThinkingTimeline()

        try:
            # =================================================================
            # Phase 0: Process images (if any)
            # =================================================================
            processed_images: Optional[List[ProcessedImage]] = None
            if data.images:
                processed_images = await _process_images(data.images)
                if processed_images:
                    _logger.info(
                        f"[CHAT] Processed {len(processed_images)} images"
                    )

            # =================================================================
            # Phase 1: Session Start + Working Memory Setup
            # =================================================================
            yield emitter.emit_session_start({
                "mode_requested": "intent_classifier",
                "version": "v4",
            })

            # Setup WorkingMemoryIntegration (session-scoped scratchpad)
            flow_id = f"v4_{session_id[:8]}_{int(datetime.now().timestamp())}"
            wm_integration = setup_working_memory_for_request(
                session_id=session_id,
                user_id=user_id,
                flow_id=flow_id,
            )
            _logger.debug(f"[CHAT] WorkingMemory initialized: {flow_id}")

            # =================================================================
            # Phase 1.5: UNIFIED CONTEXT LOADING via ContextBuilder
            # Centralizes: Core Memory + Summary + Recent Messages + WM Symbols
            # Replaces manual loading for consistency across all phases
            # =================================================================
            context_builder = get_context_builder()
            assembled_context: AssembledContext = await context_builder.build_context(
                session_id=session_id,
                user_id=user_id,
                phase=ContextPhase.INTENT_CLASSIFICATION.value,
            )

            # Extract components for backward compatibility
            core_memory_context: Optional[str] = assembled_context.core_memory
            conversation_summary: Optional[str] = assembled_context.summary
            wm_symbols: List[str] = assembled_context.wm_symbols

            # Log context summary
            ctx_summary = assembled_context.get_context_summary()
            _logger.info(
                f"[CHAT] ✅ ContextBuilder loaded: "
                f"core_memory={ctx_summary['core_memory_chars']}chars, "
                f"summary={ctx_summary['summary_chars']}chars, "
                f"wm_symbols={ctx_summary['wm_symbols']}"
            )

            if wm_symbols:
                _logger.info(f"[CHAT] Working Memory symbols from previous turns: {wm_symbols}")

            # Load conversation history (separate from ContextBuilder for compaction)
            conversation_history = await _load_conversation_history(
                session_id=session_id,
                limit=10,  # Last 10 messages for context
            )
            _logger.debug(
                f"[CHAT] Loaded {len(conversation_history)} messages from history"
            )

            # =================================================================
            # Phase 1.6: Context Compaction Check
            # Auto-compress conversation when token count exceeds threshold
            # =================================================================
            compaction_result: Optional[CompactionResult] = None
            if conversation_history:
                conversation_history, compaction_result = await check_and_compact_if_needed(
                    messages=conversation_history,
                    system_prompt="",  # Will be built after classification
                    symbols=[],  # Symbols extracted during classification
                    additional_context="",
                )
                if compaction_result:
                    stats["compaction_result"] = compaction_result
                    _logger.info(
                        f"[CHAT] ✅ Context compacted: "
                        f"{compaction_result.original_tokens:,} → {compaction_result.final_tokens:,} tokens "
                        f"(saved {compaction_result.tokens_saved:,}, {compaction_result.compression_ratio:.1%})"
                    )
                    yield emitter.emit_progress(
                        phase="context_compaction",
                        progress_percent=15,
                        message=f"Compressed context: saved {compaction_result.tokens_saved:,} tokens",
                    )

            # =================================================================
            # Phase 2: Intent Classification (SINGLE LLM call)
            # =================================================================
            yield emitter.emit_classifying()

            # Timeline: Start classification
            thinking_timeline.add_step(
                phase=ThinkingPhase.CLASSIFICATION.value,
                action="Analyzing query...",
            )

            # Emit timeline event for UI
            for timeline_event in thinking_timeline.get_pending_events():
                yield timeline_event.to_sse()

            intent_classifier = _get_intent_classifier()

            # Timeline: LLM call for classification
            thinking_timeline.add_step(
                phase=ThinkingPhase.CLASSIFICATION.value,
                action="LLM Call: Intent Classification",
                is_llm_call=True,
            )
            for timeline_event in thinking_timeline.get_pending_events():
                yield timeline_event.to_sse()

            intent_result = await intent_classifier.classify(
                query=query,
                ui_context=data.ui_context.model_dump() if data.ui_context else None,
                conversation_history=conversation_history,  # Recent K messages (raw)
                working_memory_symbols=wm_symbols,  # Symbols from previous turns!
                core_memory_context=core_memory_context,  # User profile context!
                conversation_summary=conversation_summary,  # Older messages (compressed)
            )

            # Timeline: Classification complete with results
            symbols_display = ", ".join(intent_result.validated_symbols) if intent_result.validated_symbols else "none"
            thinking_timeline.add_step(
                phase=ThinkingPhase.SYMBOL_DETECTION.value,
                action="Detected symbols",
                details=symbols_display,
                success=True,
            )
            for timeline_event in thinking_timeline.get_pending_events():
                yield timeline_event.to_sse()

            # Emit classification result (using intent result)
            yield emitter.emit_classified(
                query_type=intent_result.query_type,
                requires_tools=intent_result.requires_tools,
                symbols=intent_result.validated_symbols,  # Already normalized!
                categories=[],  # Agent sees ALL tools - no categories
                confidence=intent_result.confidence,
                language=intent_result.response_language,
                reasoning=intent_result.reasoning,
                intent_summary=intent_result.intent_summary,
                classification_method=intent_result.classification_method,
            )

            stats["complexity"] = intent_result.complexity.value

            # Save classification to working memory (for cross-turn continuity)
            wm_integration.save_classification(
                query_type=intent_result.query_type,
                categories=[],  # V4 doesn't use categories
                symbols=intent_result.validated_symbols,
                language=intent_result.response_language,
                reasoning=intent_result.reasoning,
            )

            # =================================================================
            # Phase 3: Agent Execution with ALL Tools
            # =================================================================
            unified_agent = _get_unified_agent()

            # Timeline: Agent starting with tools
            thinking_timeline.add_step(
                phase=ThinkingPhase.TOOL_SELECTION.value,
                action="Agent starting with all tools",
                details=f"{len(unified_agent.catalog.get_tool_names())} tools available",
            )
            for timeline_event in thinking_timeline.get_pending_events():
                yield timeline_event.to_sse()

            # Stream events from Agent with ALL tools
            # CRITICAL: Pass conversation_history for memory/context!
            # CRITICAL: Pass wm_symbols for cross-turn symbol continuity!
            # CRITICAL: Pass core_memory for user personalization!
            # CRITICAL: Pass conversation_summary for long-term context!
            _logger.info(f"[CHAT] 🚀 Starting agent loop with {len(unified_agent.catalog.get_tool_names())} tools")
            agent_event_count = 0
            async for event in unified_agent.run_stream_with_all_tools(
                query=query,
                intent_result=intent_result,
                conversation_history=conversation_history,  # Recent K messages (raw)
                conversation_summary=conversation_summary,  # Old messages (compressed)
                system_language=intent_result.response_language,
                user_id=int(user_id) if user_id else None,
                session_id=session_id,
                core_memory=core_memory_context,  # User profile & preferences
                enable_reasoning=data.enable_thinking,
                images=processed_images,
                model_name=data.model_name,
                provider_type=data.provider_type,
                max_turns=6,
                enable_tool_search_mode=data.enable_tool_search_mode,  # Token savings
                working_memory_symbols=wm_symbols,  # Symbols from previous turns
                enable_think_tool=data.enable_think_tool,  # STRONG think tool instruction
                enable_web_search=data.enable_web_search,  # FORCE inject web search tools
            ):
                event_type = event.get("type", "unknown")
                agent_event_count += 1

                if event_type == "turn_start":
                    turn = event.get("turn", 1)
                    stats["total_turns"] = turn
                    _logger.info(f"[CHAT] 📍 Turn {turn} started")
                    yield emitter.emit_turn_start(
                        turn_number=turn,
                        max_turns=6,
                    )

                elif event_type == "tool_calls":
                    tools = event.get("tools", [])
                    stats["total_tool_calls"] += len(tools)

                    # Timeline: Add step for each tool call
                    for tool in tools:
                        tool_name = tool.get("name", "unknown")
                        tool_args = tool.get("arguments", {})
                        symbol = tool_args.get("symbol", "")
                        thinking_timeline.add_step(
                            phase=ThinkingPhase.TOOL_EXECUTION.value,
                            action=f"Tool: {tool_name}",
                            is_tool_call=True,
                            details=f"({symbol})" if symbol else None,
                        )
                    for timeline_event in thinking_timeline.get_pending_events():
                        yield timeline_event.to_sse()

                    yield emitter.emit_tool_calls(tools)

                elif event_type == "tool_results":
                    results = event.get("results", [])
                    stats["tool_results"].extend(results)

                    # Timeline: Update tool results with success/failure
                    for result in results:
                        tool_name = result.get("tool", "unknown")
                        success = result.get("success", False)
                        thinking_timeline.add_step(
                            phase=ThinkingPhase.DATA_GATHERING.value,
                            action=f"Result: {tool_name}",
                            is_tool_call=True,
                            details="success" if success else "failed",
                            success=success,
                        )
                    for timeline_event in thinking_timeline.get_pending_events():
                        yield timeline_event.to_sse()

                    yield emitter.emit_tool_results(results)

                elif event_type == "content":
                    content = event.get("content", "")
                    if content:
                        stats["final_content"] += content
                        # Log first content chunk
                        if len(stats["final_content"]) == len(content):
                            _logger.info(f"[CHAT] 📝 First content chunk received ({len(content)} chars)")
                        yield emitter.emit_content(content)

                elif event_type == "thinking":
                    # Handle think tool calls - emit as thinking step for frontend
                    phase = event.get("phase", "analyzing")
                    thought_content = event.get("content", "")

                    if thought_content:
                        # Add to thinking timeline
                        thinking_timeline.add_step(
                            phase=ThinkingPhase.DATA_GATHERING.value,
                            action=f"💭 Think: {phase.title()}",
                            details=thought_content[:100] + "..." if len(thought_content) > 100 else thought_content,
                        )
                        for timeline_event in thinking_timeline.get_pending_events():
                            yield timeline_event.to_sse()

                        # Emit as separate thinking_step event for detailed display
                        yield emitter.emit_thinking_step(
                            title=f"Thinking ({phase})",
                            content=thought_content,
                            phase=phase,
                        )

                        _logger.info(f"[CHAT] 💭 Think [{phase}]: {thought_content[:80]}...")

                elif event_type == "max_turns_reached":
                    yield emitter.emit_progress(
                        phase="max_turns",
                        progress_percent=90,
                        message=f"Max turns ({event.get('turns', 0)}) reached",
                    )

                elif event_type == "done":
                    stats["total_turns"] = event.get("total_turns", stats["total_turns"])
                    stats["total_tool_calls"] = event.get("total_tool_calls", stats["total_tool_calls"])
                    _logger.info(
                        f"[CHAT] ✅ Agent done: {stats['total_turns']} turns, "
                        f"{stats['total_tool_calls']} tools, {len(stats['final_content'])} chars content"
                    )

                elif event_type == "error":
                    _logger.error(f"[CHAT] ❌ Agent error: {event.get('error', 'Unknown')}")
                    yield emitter.emit_error(
                        error_message=event.get("error", "Unknown error"),
                        error_code="AGENT_ERROR",
                    )

            # Log agent loop completion
            _logger.info(
                f"[CHAT] 🏁 Agent loop finished: {agent_event_count} events, "
                f"content_length={len(stats['final_content'])}"
            )

            # Warn if no content was generated
            if not stats["final_content"]:
                _logger.warning(
                    f"[CHAT] ⚠️ No content generated! "
                    f"turns={stats['total_turns']}, tools={stats['total_tool_calls']}"
                )

            # =================================================================
            # Phase 4: Done
            # =================================================================
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Timeline: Synthesis step
            thinking_timeline.add_step(
                phase=ThinkingPhase.SYNTHESIS.value,
                action="Response generation complete",
                details=f"{stats['total_turns']} turns, {stats['total_tool_calls']} tools",
                success=True,
            )

            # Emit final thinking summary (ChatGPT-style "Thought for Xs")
            thinking_summary_event = thinking_timeline.get_summary_event()
            yield thinking_summary_event.to_sse()

            # =================================================================
            # LEARN Phase: Update memory after successful execution
            # =================================================================
            try:
                learn_hook = LearnHook()
                await learn_hook.on_execution_complete(
                    query=query,
                    classification=intent_result,  # IntentResult has compatible interface
                    tool_results=stats["tool_results"],
                    response=stats["final_content"],
                    user_id=int(user_id) if user_id else None,
                )
            except Exception as learn_err:
                _logger.warning(f"[LEARN] Memory update failed (non-fatal): {learn_err}")

            # =================================================================
            # SAVE Phase: Persist conversation to database (CRITICAL FOR MEMORY!)
            # Without this, conversation history will be empty on next turn
            # =================================================================
            await _save_conversation_turn(
                session_id=session_id,
                user_id=str(user_id),
                user_query=query,
                assistant_response=stats["final_content"],
                response_time_ms=elapsed_ms,
            )

            # =================================================================
            # SUMMARY Phase: Create/update recursive summary if threshold reached
            # This compresses old messages for future turns, enabling long
            # conversations without context overflow.
            # =================================================================
            try:
                summary_manager = get_recursive_summary_manager()
                summary_result = await summary_manager.check_and_create_summary(
                    session_id=session_id,
                    user_id=str(user_id),
                    organization_id=org_id,
                )
                if summary_result.get("created"):
                    _logger.info(
                        f"[CHAT] ✅ Summary created: v{summary_result.get('version')} "
                        f"({summary_result.get('token_count')} tokens, "
                        f"summarized {summary_result.get('messages_summarized')} messages)"
                    )
                else:
                    _logger.debug(
                        f"[CHAT] Summary skipped: {summary_result.get('reason', 'unknown')}"
                    )
            except Exception as sum_err:
                _logger.warning(f"[CHAT] Summary creation failed (non-fatal): {sum_err}")

            # Complete WorkingMemory request (cleanup task-specific data, preserve symbols)
            wm_integration.complete_request()

            # =================================================================
            # Chart Resolution: Map tool results to frontend charts
            # This is critical for FE to show relevant charts with symbols
            # IMPORTANT: Only use symbols from current query (intent_result),
            # NOT from working memory (wm_symbols) - wm_symbols are for LLM
            # context, not for chart display
            # =================================================================
            try:
                # Only use symbols from current query intent - NOT from working memory
                chart_symbols = list(intent_result.validated_symbols) if intent_result.validated_symbols else []

                # Use chart resolver to map tool results to charts
                chart_resolver = get_chart_resolver()
                charts = chart_resolver.resolve_from_tool_results(
                    tool_results=stats["tool_results"],
                    symbols=chart_symbols,
                    query=query,
                    max_charts=3,
                )
                if charts:
                    stats["charts"] = charts_to_dict_list(charts)
                    _logger.info(
                        f"[CHAT] Charts resolved: {[c.type for c in charts]} "
                        f"with symbols={chart_symbols}"
                    )
            except Exception as chart_err:
                _logger.warning(f"[CHAT] Chart resolution failed (non-fatal): {chart_err}")

            yield emitter.emit_done(
                total_turns=stats["total_turns"],
                total_tool_calls=stats["total_tool_calls"],
                total_time_ms=elapsed_ms,
                charts=stats.get("charts"),
            )
            yield format_done_marker()

        except Exception as e:
            _logger.error(f"[CHAT] Error: {e}", exc_info=True)
            # Attempt to complete working memory even on error
            try:
                wm_integration.complete_request()
            except Exception:
                pass
            yield emitter.emit_error(str(e), "CHAT_ERROR")
            yield format_done_marker()

    # Wrap generator with SSE cancellation handling
    # This detects client disconnection and runs cleanup
    async def cleanup_on_disconnect():
        """Cleanup resources when client disconnects"""
        _logger.info(f"[CHAT] Client disconnected - cleaning up session {session_id}")
        # Any additional cleanup can be added here

    # Wrap with heartbeat (15s interval) then cancellation handling
    # Order: _generate() -> with_heartbeat() -> with_cancellation()
    generator_with_heartbeat = with_heartbeat(
        event_generator=_generate(),
        emitter=emitter,
        heartbeat_interval=15.0,  # Emit heartbeat every 15s if no activity
    )

    return StreamingResponse(
        with_cancellation(
            request=request,
            generator=generator_with_heartbeat,
            cleanup_fn=cleanup_on_disconnect,
            check_interval=0.5,  # Check every 500ms
            emit_cancelled_event=True,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )