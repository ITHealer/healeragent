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
from src.handlers.v2.mode_router import (
    ModeRouter,
    QueryMode,
    ModeDecision,
    get_mode_router,
)
from src.handlers.v2.normal_mode_chat_handler import (
    NormalModeChatHandler,
    get_normal_mode_chat_handler,
)
from src.handlers.v2.chat_handler import ChatHandler as DeepResearchHandler
from src.services.streaming_event_service import (
    StreamEventEmitter,
    format_done_marker,
)
from src.agents.classification import (
    UnifiedClassifier,
    ClassifierContext,
    get_unified_classifier,
    # New architecture
    IntentClassifier,
    IntentResult,
    IntentComplexity,
    get_intent_classifier,
)
from src.agents.charts import (
    resolve_charts_from_classification,
    charts_to_dict_list,
)
# LLM Router + Unified Agent (ChatGPT-style 2-phase tool selection)
from src.agents.router import (
    LLMToolRouter,
    RouterDecision,
    Complexity,
    ExecutionStrategy,
    get_tool_router,
)
from src.agents.unified import (
    UnifiedAgent,
    get_unified_agent,
)
# Streaming
from src.services.streaming_event_service import (
    StreamEventEmitter,
    StreamEventType,
    format_sse_event,
    format_done_marker,
    create_error_event,
)
from src.agents.streaming import (
    StreamingChatHandler,
    StreamingConfig,
    CancellationToken,
    StreamState,
    # Events
    StartEvent,
    ThinkingStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    PlanningProgressEvent,
    PlanningCompleteEvent,
    ToolStartEvent,
    ToolProgressEvent,
    ToolCompleteEvent,
    ContextLoadedEvent,
    TextDeltaEvent,
    DoneEvent,
    ErrorEvent,
    LLMThoughtEvent,
    LLMDecisionEvent,
)
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


# Router instance
router = APIRouter(prefix="/chat-assistant")

# Core services
_api_key_auth = APIKeyAuth()
_logger = LoggerMixin().logger
_chat_service = ChatService()

# Handler singletons (lazy initialized)
_normal_handler: Optional[NormalModeChatHandler] = None
_deep_handler: Optional[DeepResearchHandler] = None
_mode_router: Optional[ModeRouter] = None
_classifier: Optional[UnifiedClassifier] = None


def _get_normal_handler() -> NormalModeChatHandler:
    """Get or create Normal Mode handler singleton."""
    global _normal_handler, _deep_handler
    if _normal_handler is None:
        if _deep_handler is None:
            _deep_handler = DeepResearchHandler()
        _normal_handler = get_normal_mode_chat_handler(fallback_handler=_deep_handler)
    return _normal_handler


def _get_deep_handler() -> DeepResearchHandler:
    """Get or create Deep Research handler singleton."""
    global _deep_handler
    if _deep_handler is None:
        _deep_handler = DeepResearchHandler()
    return _deep_handler


def _get_mode_router() -> ModeRouter:
    """Get or create Mode Router singleton."""
    global _mode_router
    if _mode_router is None:
        _mode_router = get_mode_router()
    return _mode_router


def _get_classifier() -> UnifiedClassifier:
    """Get or create Classifier singleton."""
    global _classifier
    if _classifier is None:
        _classifier = get_unified_classifier()
    return _classifier


# --- LLM Router + Unified Agent Singletons (ChatGPT-style) ---

_tool_router: Optional[LLMToolRouter] = None
_unified_agent: Optional[UnifiedAgent] = None


def _get_tool_router() -> LLMToolRouter:
    """Get or create LLM Tool Router singleton."""
    global _tool_router
    if _tool_router is None:
        _tool_router = get_tool_router()
    return _tool_router


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


# Cached components for StreamingChatHandler
_streaming_components_initialized = False
_planning_agent = None
_task_executor = None
_core_memory = None
_summary_manager = None
_session_repo = None
_llm_provider = None
_tool_execution_service = None
_chat_repo = None
_validation_agent = None

# Session repository for loading conversation history (V4)
_v4_session_repo = None


def _get_v4_session_repo():
    """Get or create session repository for V4."""
    global _v4_session_repo
    if _v4_session_repo is None:
        from src.database.repository.sessions import SessionRepository
        _v4_session_repo = SessionRepository()
    return _v4_session_repo


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
        session_repo = _get_v4_session_repo()
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
        _logger.warning(f"[V4] Failed to load conversation history: {e}")
        return []

def _init_streaming_components():
    """Initialize shared components for StreamingChatHandler (called once)"""
    global _streaming_components_initialized
    global _planning_agent, _task_executor, _core_memory, _summary_manager
    global _session_repo, _llm_provider, _tool_execution_service, _chat_repo
    global _validation_agent

    if _streaming_components_initialized:
        return

    from src.agents.planning.planning_agent import PlanningAgent
    from src.agents.action.task_executor import TaskExecutor
    from src.agents.memory.core_memory import CoreMemory
    from src.agents.memory.recursive_summary import RecursiveSummaryManager
    from src.database.repository.sessions import SessionRepository
    from src.database.repository.chat import ChatRepository
    from src.helpers.llm_helper import LLMGeneratorProvider
    from src.services.v2.tool_execution_service import ToolExecutionService
    from src.agents.validation.validation_agent import ValidationAgent

    _tool_execution_service = ToolExecutionService()
    _validation_agent = ValidationAgent()
    _task_executor = TaskExecutor(
        tool_execution_service=_tool_execution_service,
        validation_agent=_validation_agent,
        max_retries=2
    )
    _core_memory = CoreMemory()
    _summary_manager = RecursiveSummaryManager()
    _session_repo = SessionRepository()
    _chat_repo = ChatRepository()
    _llm_provider = LLMGeneratorProvider()

    _streaming_components_initialized = True
    _logger.info("[UNIFIED_CHAT] StreamingChatHandler components initialized")


async def get_streaming_handler(
    enable_llm_events: bool = True,
    enable_agent_tree: bool = False,
) -> StreamingChatHandler:
    """
    Get a configured StreamingChatHandler for Deep Research Mode.

    Uses the 7-phase upfront planning pipeline with true streaming:
    - Phase 1: Context Loading (with compaction)
    - Phase 2: Planning (3-stage semantic)
    - Phase 3: Memory Search
    - Phase 4: Tool Execution (with progress)
    - Phase 5: Context Assembly
    - Phase 6: Response Generation (streaming)
    - Phase 7: Post-Processing
    """
    _init_streaming_components()

    from src.agents.planning.planning_agent import PlanningAgent
    from src.utils.config import settings

    # Create planning agent
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
        enable_agent_tree=enable_agent_tree,
    )

    handler = StreamingChatHandler(
        planning_agent=planning_agent,
        task_executor=_task_executor,
        core_memory=_core_memory,
        summary_manager=_summary_manager,
        session_repo=_session_repo,
        llm_provider=_llm_provider,
        tool_execution_service=_tool_execution_service,
        chat_repo=_chat_repo,
        config=config
    )

    return handler


# --- Streaming Chat Endpoint ---

@router.post(
    "/chat",
    summary="Streaming Chat",
    description="""Chat with SSE streaming.""",
    response_class=StreamingResponse,
)
async def stream_chat(
    request: Request,
    data: ChatRequest,
    api_key_data: Dict[str, Any] = Depends(_api_key_auth.author_with_api_key),
):
    """Stream chat responses via SSE with automatic mode selection."""
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

    # Process images outside of generator to avoid async issues
    processed_images: Optional[List[ProcessedImage]] = None

    async def _generate() -> AsyncGenerator[str, None]:
        nonlocal processed_images
        """Generate SSE events."""
        start_time = datetime.now()
        mode_decision: Optional[ModeDecision] = None

        # Stats tracking for accurate done event
        stats = {
            "total_turns": 0,
            "total_tool_calls": 0,
            "last_event_type": None,
            "charts": None,
        }

        try:
            # Process images if provided
            if data.images:
                processed_images = await _process_images(data.images)
                if processed_images:
                    _logger.info(f"[CHAT] Processed {len(processed_images)} images for multimodal request")

            # Session start
            yield emitter.emit_session_start({"mode_requested": data.mode})

            # Classify query
            yield emitter.emit_classifying()

            classifier = _get_classifier()

            # Build classification context WITH images for multimodal analysis
            ctx = ClassifierContext(
                query=query,  # Use validated query
                conversation_history=[],
                ui_context=data.ui_context.model_dump() if data.ui_context else None,
                images=processed_images,  # Include images for vision-based classification
            )
            classification = await classifier.classify(ctx)

            # Stream classification thinking/reasoning (NEW - #16 Stream thinking process)
            if classification.reasoning and data.enable_thinking:
                yield emitter.emit_thinking(
                    content=f"🎯 Classification reasoning:\n{classification.reasoning[:500]}",
                    phase="classification_reasoning",
                )

            yield emitter.emit_classified(
                query_type=classification.query_type.value,
                requires_tools=classification.requires_tools,
                symbols=classification.symbols,
                categories=classification.tool_categories,
                confidence=classification.confidence,
                language=classification.response_language,
                reasoning=classification.reasoning,  # AI thought process
                intent_summary=classification.intent_summary,
                classification_method=classification.classification_method,  # llm, llm_vision, fallback
            )

            # Resolve charts from classification (for frontend display)
            charts = resolve_charts_from_classification(
                classification=classification,
                query=query,
                max_charts=3,
            )
            if charts:
                stats["charts"] = charts_to_dict_list(charts)
                _logger.info(f"[UNIFIED_CHAT] Charts resolved: {[c.type for c in charts]}")

            # Determine mode
            mode_router = _get_mode_router()
            mode_decision = mode_router.determine_mode(
                query=query,
                explicit_mode=data.mode,
                classification=classification,
            )

            _logger.info(
                f"[CHAT:ROUTE] mode={mode_decision.mode.value} | "
                f"method={mode_decision.detection_method}"
            )

            # Collect events to identify final content chunk
            pending_events: List[str] = []

            # Route to appropriate handler
            if mode_decision.mode == QueryMode.DEEP_RESEARCH:
                # Pass classification to avoid duplicate LLM call in PlanningAgent
                classification_dict = classification.to_dict() if classification else None
                async for event in _stream_deep_research(
                    query=query,
                    session_id=session_id,
                    user_id=user_id,
                    org_id=org_id,
                    model_name=data.model_name,
                    provider_type=data.provider_type,
                    chart_displayed=data.chart_displayed,
                    enable_thinking=data.enable_thinking,
                    emitter=emitter,
                    stats=stats,
                    enable_llm_events=data.enable_llm_events,
                    enable_agent_tree=data.enable_agent_tree,
                    enable_think_tool=data.enable_think_tool,
                    enable_compaction=data.enable_compaction,
                    classification=classification_dict,
                    images=processed_images,
                ):
                    yield event
            else:
                async for event in _stream_normal_mode(
                    query=query,
                    session_id=session_id,
                    user_id=user_id,
                    org_id=org_id,
                    model_name=data.model_name,
                    provider_type=data.provider_type,
                    chart_displayed=data.chart_displayed,
                    enable_thinking=data.enable_thinking,
                    emitter=emitter,
                    classification=classification,
                    stats=stats,
                    enable_tools=data.enable_tools,
                    enable_think_tool=data.enable_think_tool,
                    enable_llm_events=data.enable_llm_events,
                    enable_compaction=data.enable_compaction,
                    enable_web_search=data.enable_web_search,
                    images=processed_images,
                ):
                    pending_events.append(event)

            # Process events with is_final fix for last content chunk
            last_content_idx = -1
            for i in range(len(pending_events) - 1, -1, -1):
                if '"type": "content"' in pending_events[i] or '"type":"content"' in pending_events[i]:
                    last_content_idx = i
                    break

            # Yield all events, fixing is_final on the last content chunk
            for i, event in enumerate(pending_events):
                if i == last_content_idx:
                    # Replace is_final: false with is_final: true
                    event = event.replace('"is_final": false', '"is_final": true')
                    event = event.replace('"is_final":false', '"is_final":true')
                yield event

            # Done with accurate stats
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            yield emitter.emit_done(
                total_turns=stats["total_turns"],
                total_tool_calls=stats["total_tool_calls"],
                total_time_ms=elapsed_ms,
                charts=stats.get("charts"),
            )
            yield format_done_marker()

        except Exception as e:
            _logger.error(f"[UNIFIED_CHAT] Error: {e}", exc_info=True)
            yield emitter.emit_error(str(e), "PROCESSING_ERROR")
            yield format_done_marker()

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# --- Stream Handlers ---

async def _stream_normal_mode(
    query: str,
    session_id: str,
    user_id: str,
    org_id: Optional[str],
    model_name: str,
    provider_type: str,
    chart_displayed: bool,
    enable_thinking: bool,
    emitter: StreamEventEmitter,
    classification,
    stats: Dict[str, Any],
    enable_tools: bool = True,
    enable_think_tool: bool = False,
    enable_llm_events: bool = True,
    enable_compaction: bool = True,
    enable_web_search: bool = False,
    images: Optional[List[ProcessedImage]] = None,
    ) -> AsyncGenerator[str, None]:
    """Stream Normal Mode responses as SSE events."""

    from datetime import datetime as dt

    handler = _get_normal_handler()

    # Force web search category when enabled by user
    if enable_web_search:
        if "web" not in classification.tool_categories:
            classification.tool_categories.append("web")
        if not classification.requires_tools:
            classification.requires_tools = True

    # Emit tools loading
    if enable_tools and classification.tool_categories:
        yield emitter.emit_tools_loading(classification.tool_categories)

    turn_count = 0
    tool_count = 0
    thinking_start_time = None

    # Emit reasoning events for classification phase
    if enable_thinking:
        thinking_start_time = dt.now()

        # Classification start
        yield emitter.emit_reasoning(
            phase="classification",
            content=f"Analyzing query: {query[:100]}..." if len(query) > 100 else f"Analyzing query: {query}",
            action="start",
        )

        # Classification reasoning
        if classification.reasoning:
            yield emitter.emit_reasoning(
                phase="classification",
                content=classification.reasoning,
                action="thought",
            )

        # Intent summary
        if classification.intent_summary:
            yield emitter.emit_reasoning(
                phase="classification",
                content=f"Intent: {classification.intent_summary}",
                action="thought",
            )

        # Classification complete
        yield emitter.emit_reasoning(
            phase="classification",
            content=f"Query type: {classification.query_type.value}, Symbols: {classification.symbols}",
            action="complete",
            metadata={
                "query_type": classification.query_type.value,
                "symbols": classification.symbols,
                "categories": classification.tool_categories,
                "requires_tools": classification.requires_tools,
            }
        )

    # Build kwargs for handler (images is optional for backward compatibility)
    handler_kwargs = {
        "query": query,
        "session_id": session_id,
        "user_id": user_id,
        "model_name": model_name,
        "provider_type": provider_type,
        "chart_displayed": chart_displayed,
        "organization_id": org_id,
        "enable_thinking": enable_thinking,
        "enable_llm_events": enable_llm_events,
        "stream": True,
        "enable_web_search": enable_web_search,
        "classification": classification,
    }
    # Only add images if present (backward compatible with old handlers)
    if images:
        handler_kwargs["images"] = images

    async for chunk in handler.handle_chat(**handler_kwargs):
        # Convert handler output to SSE events
        if isinstance(chunk, dict):
            event_type = chunk.get("type", "content")

            if event_type == "turn_start":
                turn_count = chunk.get("turn", turn_count + 1)
                stats["total_turns"] = turn_count  # Update stats
                yield emitter.emit_turn_start(turn_count, 10)

                # Emit reasoning for new turn
                if enable_thinking:
                    yield emitter.emit_reasoning(
                        phase="tool_selection",
                        content=f"Turn {turn_count}: Analyzing query and selecting appropriate tools...",
                        action="start",
                        metadata={"turn": turn_count}
                    )

            elif event_type == "tool_calls":
                tools = chunk.get("tools", [])
                tool_count += len(tools)
                stats["total_tool_calls"] = tool_count  # Update stats
                yield emitter.emit_tool_calls(tools)

                # Emit reasoning about tool decisions
                if enable_thinking or enable_llm_events:
                    for tool in tools:
                        tool_name = tool.get("name", "unknown")
                        yield emitter.emit_reasoning(
                            phase="tool_selection",
                            content=f"Using {tool_name} to gather required information",
                            action="decision",
                            metadata={
                                "tool": tool_name,
                                "params": tool.get("arguments", {}),
                            }
                        )

            elif event_type == "tool_results":
                results = chunk.get("results", [])
                yield emitter.emit_tool_results(results)

                # Emit reasoning about tool results
                if enable_thinking:
                    success_count = sum(1 for r in results if r.get("success", True))
                    yield emitter.emit_reasoning(
                        phase="tool_analysis",
                        content=f"Received {len(results)} tool results ({success_count} successful)",
                        action="progress",
                        metadata={"total": len(results), "success": success_count}
                    )

            elif event_type == "content":
                content = chunk.get("content", "")
                if content:
                    yield emitter.emit_content(content)

            elif event_type == "thinking":
                # Convert legacy thinking events to unified reasoning
                if enable_thinking:
                    yield emitter.emit_reasoning(
                        phase=chunk.get("phase", "reasoning"),
                        content=chunk.get("content", ""),
                        action="thought",
                    )

            elif event_type == "llm_thought":
                # Convert legacy llm_thought to unified reasoning
                if enable_thinking or enable_llm_events:
                    yield emitter.emit_reasoning(
                        phase="reasoning",
                        content=chunk.get("thought", ""),
                        action="thought",
                        metadata={"context": chunk.get("context", "")}
                    )

            elif event_type == "done":
                # Extract charts from done event if present
                if chunk.get("charts"):
                    stats["charts"] = chunk.get("charts")

                # Emit synthesis complete
                if enable_thinking and thinking_start_time:
                    duration_ms = (dt.now() - thinking_start_time).total_seconds() * 1000
                    yield emitter.emit_reasoning(
                        phase="synthesis",
                        content=f"Response complete: {turn_count} turns, {tool_count} tool calls",
                        action="complete",
                        progress=1.0,
                        metadata={
                            "turns": turn_count,
                            "tool_calls": tool_count,
                            "duration_ms": round(duration_ms, 0),
                        }
                    )

            else:
                # Unknown event type, emit as content
                yield emitter.emit_content(str(chunk))
        else:
            # Plain string chunk
            yield emitter.emit_content(str(chunk))


async def _stream_deep_research(
    query: str,
    session_id: str,
    user_id: str,
    org_id: Optional[str],
    model_name: str,
    provider_type: str,
    chart_displayed: bool,
    enable_thinking: bool,
    emitter: StreamEventEmitter,
    stats: Dict[str, Any],
    enable_llm_events: bool = True,
    enable_agent_tree: bool = False,
    enable_think_tool: bool = False,
    enable_compaction: bool = True,
    classification: Dict[str, Any] = None,
    images: Optional[List[ProcessedImage]] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream Deep Research Mode responses as SSE events.

    Args:
        classification: Pre-computed classification from UnifiedClassifier.
                       Passed to PlanningAgent to avoid duplicate classification.
        images: Optional list of processed images for multimodal analysis.
    """
    from datetime import datetime as dt

    # Get StreamingChatHandler
    handler = await get_streaming_handler(
        enable_llm_events=enable_llm_events,
        enable_agent_tree=enable_agent_tree,
    )

    yield emitter.emit_progress(
        phase="initialization",
        progress_percent=5,
        message="Starting Deep Research Mode (Upfront Planning)..."
    )

    thinking_start = None
    current_phase = "initialization"
    tool_count = 0  # Track tool calls for stats
    phase_progress = {
        "context": 10,
        "planning": 30,
        "execution": 60,
        "assembly": 80,
        "generation": 90,
    }

    # Stream events from StreamingChatHandler
    # Pass pre-computed classification to avoid duplicate LLM call in PlanningAgent
    # Build kwargs (images is optional for backward compatibility)
    stream_kwargs = {
        "query": query,
        "session_id": session_id,
        "user_id": int(user_id) if user_id else 0,
        "model_name": model_name,
        "provider_type": provider_type,
        "organization_id": int(org_id) if org_id else None,
        "enable_thinking": enable_thinking,
        "chart_displayed": chart_displayed,
        "enable_compaction": enable_compaction,
        "enable_think_tool": enable_think_tool,
        "classification": classification,
    }
    if images:
        stream_kwargs["images"] = images

    async for event in handler.handle_chat_stream(**stream_kwargs):
        # Map StreamEvent types to StreamEventEmitter

        if isinstance(event, StartEvent):
            yield emitter.emit_session(
                session_id=session_id,
                mode="deep_research",
                model=model_name,
            )

        elif isinstance(event, ThinkingStartEvent):
            current_phase = event.data.get("phase", "thinking")
            thinking_start = dt.now()
            # Emit unified reasoning event
            yield emitter.emit_reasoning(
                phase=current_phase,
                content=event.data.get("message", f"Starting {current_phase}..."),
                action="start",
            )
            # Also emit progress for UI progress bar
            progress = phase_progress.get(current_phase, 50)
            yield emitter.emit_progress(
                phase=current_phase,
                progress_percent=progress,
                message=event.data.get("message", f"Processing {current_phase}..."),
            )

        elif isinstance(event, ThinkingDeltaEvent):
            content = event.data.get("chunk", event.data.get("content", ""))
            if content and enable_thinking:
                yield emitter.emit_reasoning(
                    phase=current_phase,
                    content=content,
                    action="progress",
                )

        elif isinstance(event, ThinkingEndEvent):
            duration = 0.0
            if thinking_start:
                duration = (dt.now() - thinking_start).total_seconds() * 1000
            yield emitter.emit_reasoning(
                phase=current_phase,
                content=event.data.get("summary", f"Completed {current_phase}"),
                action="complete",
                metadata={"duration_ms": round(duration, 0)}
            )

        elif isinstance(event, PlanningProgressEvent):
            yield emitter.emit_progress(
                phase="planning",
                progress_percent=phase_progress["planning"],
                message=event.data.get("message", "Creating research plan..."),
            )

        elif isinstance(event, PlanningCompleteEvent):
            task_count = event.data.get("task_count", 0)
            yield emitter.emit_progress(
                phase="planning_complete",
                progress_percent=40,
                message=f"Plan created with {task_count} tasks",
            )

        elif isinstance(event, ToolStartEvent):
            tool_name = event.data.get("tool_name", "unknown")
            tool_count += 1
            stats["total_tool_calls"] = tool_count  # Update stats
            stats["total_turns"] = 1  # Deep Research is 1 comprehensive turn

            yield emitter.emit_tool_calls([
                {"name": tool_name, "arguments": event.data.get("arguments", {})}
            ])
            # Emit reasoning about tool execution
            if enable_thinking or enable_llm_events:
                yield emitter.emit_reasoning(
                    phase="tool_execution",
                    content=f"Executing {tool_name}",
                    action="decision",
                    metadata={"tool": tool_name, "arguments": event.data.get("arguments", {})}
                )

        elif isinstance(event, ToolProgressEvent):
            yield emitter.emit_progress(
                phase="tool_execution",
                progress_percent=phase_progress["execution"],
                message=event.data.get("message", "Executing tools..."),
            )

        elif isinstance(event, ToolCompleteEvent):
            tool_name = event.data.get("tool_name", "unknown")
            success = event.data.get("success", True)
            yield emitter.emit_tool_results([
                {"tool": tool_name, "success": success}
            ])

        elif isinstance(event, ContextLoadedEvent):
            yield emitter.emit_progress(
                phase="context_loaded",
                progress_percent=15,
                message="Context loaded successfully",
            )

        elif isinstance(event, TextDeltaEvent):
            content = event.data.get("chunk", "")
            if content:
                yield emitter.emit_content(content)

        elif isinstance(event, LLMThoughtEvent):
            # Convert to unified reasoning event
            if enable_thinking or enable_llm_events:
                yield emitter.emit_reasoning(
                    phase="reasoning",
                    content=event.data.get("thought", ""),
                    action="thought",
                    metadata={"context": event.data.get("context", "")}
                )

        elif isinstance(event, LLMDecisionEvent):
            # Convert to unified reasoning event
            if enable_thinking or enable_llm_events:
                yield emitter.emit_reasoning(
                    phase="reasoning",
                    content=event.data.get("decision", ""),
                    action="decision",
                    metadata={
                        "action_type": event.data.get("action", ""),
                        "confidence": event.data.get("confidence", 0.0),
                    }
                )

        elif isinstance(event, DoneEvent):
            # Emit synthesis complete reasoning
            if enable_thinking:
                yield emitter.emit_reasoning(
                    phase="synthesis",
                    content=f"Deep Research complete: {tool_count} tools executed",
                    action="complete",
                    progress=1.0,
                    metadata={"tool_count": tool_count}
                )

            # Final progress
            yield emitter.emit_progress(
                phase="complete",
                progress_percent=100,
                message="Deep Research completed",
            )

        elif isinstance(event, ErrorEvent):
            yield create_error_event(
                error_code=event.data.get("error_type", "DEEP_RESEARCH_ERROR"),
                error_message=event.data.get("error_message", str(event.data)),
            )

        else:
            # Unknown event - try to extract content
            if hasattr(event, 'data'):
                content = event.data.get("chunk", event.data.get("content", ""))
                if content:
                    yield emitter.emit_content(str(content))


@router.post(
    "/chat/complete",
    summary="Complete Chat",
    description="Non-streaming chat. Returns complete response as JSON.",
)
async def complete_chat(
    request: Request,
    data: ChatRequest,
    api_key_data: Dict[str, Any] = Depends(_api_key_auth.author_with_api_key),
):
    """Process chat request and return complete response."""
    user_id = getattr(request.state, "user_id", None)
    org_id = getattr(request.state, "organization_id", None)

    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")

    session_id = data.session_id or _chat_service.create_chat_session(
        user_id=user_id, organization_id=org_id
    )

    start_time = datetime.now()

    try:
        # Classify query
        classifier = _get_classifier()
        ctx = ClassifierContext(
            query=query,
            conversation_history=[],
            ui_context=data.ui_context.model_dump() if data.ui_context else None,
        )
        classification = await classifier.classify(ctx)

        # Determine processing mode
        mode_router = _get_mode_router()
        mode_decision = mode_router.determine_mode(
            query=query,
            explicit_mode=data.mode,
            classification=classification,
        )

        # Route to handler and collect response
        response_chunks = []

        if mode_decision.mode == QueryMode.NORMAL:
            handler = _get_normal_handler()
            async for chunk in handler.handle_chat(
                query=query,
                session_id=session_id,
                user_id=user_id,
                model_name=data.model_name,
                provider_type=data.provider_type,
                chart_displayed=data.chart_displayed,
                organization_id=org_id,
                enable_thinking=data.enable_thinking,
                stream=False,
            ):
                if isinstance(chunk, str):
                    response_chunks.append(chunk)
                elif isinstance(chunk, dict) and "content" in chunk:
                    response_chunks.append(chunk["content"])
        else:
            handler = _get_deep_handler()
            async for chunk in handler.handle_chat_with_reasoning(
                query=query,
                chart_displayed=data.chart_displayed,
                session_id=session_id,
                user_id=user_id,
                model_name=data.model_name,
                provider_type=data.provider_type,
                organization_id=org_id,
                enable_thinking=data.enable_thinking,
                stream=False,
            ):
                if isinstance(chunk, str):
                    response_chunks.append(chunk)

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "response": "".join(response_chunks),
                    "session_id": session_id,
                    "mode": mode_decision.mode.value,
                    "mode_reason": mode_decision.reason,
                    "classification": {
                        "query_type": classification.query_type.value,
                        "symbols": classification.symbols,
                        "tool_categories": classification.tool_categories,
                    },
                    "processing_time_ms": elapsed_ms,
                    "timestamp": datetime.now().isoformat(),
                }
            }
        )

    except Exception as e:
        _logger.error(f"[CHAT:ERROR] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# =============================================================================
# V3 API: LLM Router + Unified Agent (ChatGPT-style 2-phase tool selection)
# =============================================================================

@router.post(
    "/chat/v3",
    summary="Streaming Chat V3 (LLM Router)",
    description="""
    Chat with LLM-as-Router architecture for optimal tool selection.

    New Flow:
    1. Classify query (extract symbols, language, intent)
    2. LLM Router sees ALL tools → selects relevant tools + complexity
    3. Unified Agent executes with complexity-based strategy

    Benefits:
    - No category blindness (Router sees all 38+ tools)
    - ~60% token reduction (2-level tool loading)
    - Adaptive execution (simple/medium/complex strategies)
    """,
    response_class=StreamingResponse,
)
async def stream_chat_v3(
    request: Request,
    data: ChatRequest,
    api_key_data: Dict[str, Any] = Depends(_api_key_auth.author_with_api_key),
):
    """
    Stream chat responses via SSE using LLM Router + Unified Agent.

    This is the ChatGPT-style 2-phase tool selection:
    - Phase 1: Router LLM sees all tool summaries → selects tools
    - Phase 2: Agent LLM gets full schemas for selected tools only
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

    async def _generate_v3() -> AsyncGenerator[str, None]:
        """Generate SSE events using LLM Router + Unified Agent."""
        start_time = datetime.now()

        # Stats tracking
        stats = {
            "total_turns": 0,
            "total_tool_calls": 0,
            "charts": None,
            "complexity": None,
            "selected_tools": [],
            # For LEARN phase
            "tool_results": [],
            "final_content": "",
            # Compaction tracking
            "compaction_result": None,
        }

        try:
            # =================================================================
            # Phase 0: Process images (if any)
            # =================================================================
            processed_images: Optional[List[ProcessedImage]] = None
            if data.images:
                processed_images = await _process_images(data.images)
                if processed_images:
                    _logger.info(
                        f"[CHAT_V3] Processed {len(processed_images)} images"
                    )

            # =================================================================
            # Phase 1: Session Start + Working Memory Setup
            # =================================================================
            yield emitter.emit_session_start({
                "mode_requested": "llm_router",
                "version": "v3",
            })

            # Setup WorkingMemoryIntegration (session-scoped scratchpad)
            flow_id = f"v3_{session_id[:8]}_{int(datetime.now().timestamp())}"
            wm_integration = setup_working_memory_for_request(
                session_id=session_id,
                user_id=user_id,
                flow_id=flow_id,
            )
            _logger.debug(f"[CHAT_V3] WorkingMemory initialized: {flow_id}")

            # Get Working Memory symbols from previous turns (CRITICAL FOR CONTEXT)
            wm_symbols = wm_integration.get_current_symbols()
            wm_summary = ""
            if wm_symbols:
                wm_summary = f"SYMBOLS FROM RECENT TURNS: {', '.join(wm_symbols)}\nWhen user refers to 'nó', 'this stock', 'công ty này', 'symbol đó' etc., they likely refer to these symbols."
                _logger.info(f"[CHAT_V3] Working Memory symbols from previous turns: {wm_symbols}")

            # =================================================================
            # Phase 1.5: Load Conversation History (CRITICAL FOR MEMORY!)
            # =================================================================
            conversation_history = await _load_conversation_history(
                session_id=session_id,
                limit=10,
            )
            _logger.debug(
                f"[CHAT_V3] Loaded {len(conversation_history)} messages from history"
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
                        f"[CHAT_V3] ✅ Context compacted: "
                        f"{compaction_result.original_tokens:,} → {compaction_result.final_tokens:,} tokens "
                        f"(saved {compaction_result.tokens_saved:,}, {compaction_result.compression_ratio:.1%})"
                    )
                    yield emitter.emit_progress(
                        phase="context_compaction",
                        progress_percent=15,
                        message=f"Compressed context: saved {compaction_result.tokens_saved:,} tokens",
                    )

            # =================================================================
            # Phase 2: Classification (reuse existing classifier)
            # =================================================================
            yield emitter.emit_classifying()

            classifier = _get_classifier()
            classification_context = ClassifierContext(
                query=query,
                conversation_history=conversation_history,  # NOW WITH HISTORY!
                ui_context=data.ui_context.model_dump() if data.ui_context else None,
                images=processed_images,
                working_memory_summary=wm_summary,  # Symbols from previous turns!
            )
            classification = await classifier.classify(classification_context)

            # Emit classification result
            yield emitter.emit_classified(
                query_type=classification.query_type.value,
                requires_tools=classification.requires_tools,
                symbols=classification.symbols,
                categories=classification.tool_categories,
                confidence=classification.confidence,
                language=classification.response_language,
                reasoning=classification.reasoning,
                intent_summary=classification.intent_summary,
                classification_method=classification.classification_method,
            )

            # Emit classification reasoning
            if classification.reasoning and data.enable_thinking:
                yield emitter.emit_reasoning(
                    phase="classification",
                    content=classification.reasoning,
                    action="complete",
                    metadata={
                        "query_type": classification.query_type.value,
                        "symbols": classification.symbols,
                    },
                )

            # Save classification to working memory (for cross-turn continuity)
            wm_integration.save_classification(
                query_type=classification.query_type.value,
                categories=classification.tool_categories,
                symbols=classification.symbols,
                language=classification.response_language,
                reasoning=classification.reasoning,
            )

            # Resolve charts for frontend
            charts = resolve_charts_from_classification(
                classification=classification,
                query=query,
                max_charts=3,
            )
            if charts:
                stats["charts"] = charts_to_dict_list(charts)

            # =================================================================
            # Phase 3: LLM Tool Router (2-phase tool selection)
            # =================================================================
            if data.enable_thinking:
                yield emitter.emit_reasoning(
                    phase="tool_routing",
                    content="Analyzing query to select optimal tools from catalog...",
                    action="start",
                )

            tool_router = _get_tool_router()
            router_decision = await tool_router.route(
                query=query,
                symbols=classification.symbols,
                context=classification_context,
                classification=classification,  # Pass classification for disambiguation
                enable_web_search=data.enable_web_search,  # Force webSearch when enabled
            )

            stats["complexity"] = router_decision.complexity.value
            stats["selected_tools"] = router_decision.selected_tools

            # Emit routing decision
            yield emitter.emit_reasoning(
                phase="tool_routing",
                content=router_decision.reasoning,
                action="decision",
                metadata={
                    "selected_tools": router_decision.selected_tools,
                    "complexity": router_decision.complexity.value,
                    "strategy": router_decision.execution_strategy.value,
                    "confidence": router_decision.confidence,
                    "max_turns": router_decision.suggested_max_turns,
                },
            )

            _logger.info(
                f"[CHAT_V3:ROUTE] tools={router_decision.selected_tools} | "
                f"complexity={router_decision.complexity.value} | "
                f"strategy={router_decision.execution_strategy.value}"
            )

            # =================================================================
            # Phase 4: Unified Agent Execution
            # =================================================================
            unified_agent = _get_unified_agent()

            # Stream events from Unified Agent
            # Pass user-provided model_name for final response generation
            # CRITICAL: Pass conversation_history for memory/context!
            async for event in unified_agent.run_stream(
                query=query,
                router_decision=router_decision,
                classification=classification,
                conversation_history=conversation_history,  # NOW WITH HISTORY!
                system_language=classification.response_language,
                user_id=int(user_id) if user_id else None,
                session_id=session_id,
                enable_reasoning=data.enable_thinking,
                images=processed_images,
                model_name=data.model_name,
                provider_type=data.provider_type,
            ):
                event_type = event.get("type", "unknown")

                if event_type == "reasoning":
                    # Forward reasoning events
                    yield emitter.emit_reasoning(
                        phase=event.get("phase", "execution"),
                        content=event.get("content", ""),
                        action=event.get("action", "thought"),
                        metadata=event.get("metadata"),
                    )

                elif event_type == "turn_start":
                    turn = event.get("turn", 1)
                    stats["total_turns"] = turn
                    yield emitter.emit_turn_start(
                        turn_number=turn,
                        max_turns=router_decision.suggested_max_turns,
                    )

                elif event_type == "tool_calls":
                    tools = event.get("tools", [])
                    stats["total_tool_calls"] += len(tools)
                    yield emitter.emit_tool_calls(tools)

                elif event_type == "tool_results":
                    results = event.get("results", [])
                    stats["tool_results"].extend(results)  # Capture for LEARN
                    yield emitter.emit_tool_results(results)

                elif event_type == "content":
                    content = event.get("content", "")
                    if content:
                        stats["final_content"] += content  # Capture for LEARN
                        yield emitter.emit_content(content)

                elif event_type == "max_turns_reached":
                    yield emitter.emit_progress(
                        phase="max_turns",
                        progress_percent=90,
                        message=f"Max turns ({event.get('turns', 0)}) reached",
                    )

                elif event_type == "done":
                    stats["total_turns"] = event.get("total_turns", stats["total_turns"])
                    stats["total_tool_calls"] = event.get("total_tool_calls", stats["total_tool_calls"])

                elif event_type == "error":
                    yield emitter.emit_error(
                        error_message=event.get("error", "Unknown error"),
                        error_code="UNIFIED_AGENT_ERROR",
                    )

            # =================================================================
            # Phase 5: Done
            # =================================================================
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Final reasoning summary
            if data.enable_thinking:
                yield emitter.emit_reasoning(
                    phase="synthesis",
                    content=(
                        f"Complete: {stats['total_turns']} turns, "
                        f"{stats['total_tool_calls']} tool calls, "
                        f"complexity={stats['complexity']}"
                    ),
                    action="complete",
                    progress=1.0,
                    metadata={
                        "turns": stats["total_turns"],
                        "tool_calls": stats["total_tool_calls"],
                        "complexity": stats["complexity"],
                        "selected_tools": stats["selected_tools"],
                        "duration_ms": round(elapsed_ms, 0),
                    },
                )

            # =================================================================
            # LEARN Phase: Update memory after successful execution
            # =================================================================
            try:
                learn_hook = LearnHook()
                learn_result = await learn_hook.on_execution_complete(
                    query=query,
                    classification=classification,
                    tool_results=stats["tool_results"],
                    response=stats["final_content"],
                    user_id=int(user_id) if user_id else None,
                )
                updates = learn_result.get("updates", [])
                if updates:
                    _logger.debug(
                        f"[LEARN] Memory updated: {len(updates)} updates"
                    )
            except Exception as learn_err:
                # LEARN errors should not fail the request
                _logger.warning(f"[LEARN] Memory update failed (non-fatal): {learn_err}")

            # Complete WorkingMemory request (cleanup task-specific data, preserve symbols)
            wm_integration.complete_request()

            yield emitter.emit_done(
                total_turns=stats["total_turns"],
                total_tool_calls=stats["total_tool_calls"],
                total_time_ms=elapsed_ms,
                charts=stats.get("charts"),
            )
            yield format_done_marker()

        except Exception as e:
            _logger.error(f"[CHAT_V3] Error: {e}", exc_info=True)
            # Attempt to complete working memory even on error
            try:
                wm_integration.complete_request()
            except Exception:
                pass
            yield emitter.emit_error(str(e), "CHAT_V3_ERROR")
            yield format_done_marker()

    return StreamingResponse(
        _generate_v3(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/chat/v3/complete",
    summary="Complete Chat V3 (LLM Router)",
    description="Non-streaming chat using LLM Router + Unified Agent.",
)
async def complete_chat_v3(
    request: Request,
    data: ChatRequest,
    api_key_data: Dict[str, Any] = Depends(_api_key_auth.author_with_api_key),
):
    """Process chat request using LLM Router and return complete response."""
    user_id = getattr(request.state, "user_id", None)
    org_id = getattr(request.state, "organization_id", None)

    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")

    query = data.query or "Analyze this image"
    session_id = data.session_id or _chat_service.create_chat_session(
        user_id=user_id, organization_id=org_id
    )

    start_time = datetime.now()

    try:
        # Process images
        processed_images = await _process_images(data.images) if data.images else None

        # Classify
        classifier = _get_classifier()
        classification_context = ClassifierContext(
            query=query,
            conversation_history=[],
            ui_context=data.ui_context.model_dump() if data.ui_context else None,
            images=processed_images,
        )
        classification = await classifier.classify(classification_context)

        # Route with LLM Router
        tool_router = _get_tool_router()
        router_decision = await tool_router.route(
            query=query,
            symbols=classification.symbols,
            context=classification_context,
            classification=classification,  # Pass classification for disambiguation
            enable_web_search=data.enable_web_search,  # Force webSearch when enabled
        )

        # Execute with Unified Agent
        unified_agent = _get_unified_agent()
        result = await unified_agent.run(
            query=query,
            router_decision=router_decision,
            classification=classification,
            conversation_history=[],
            system_language=classification.response_language,
            user_id=int(user_id) if user_id else None,
            session_id=session_id,
        )

        # LEARN Phase: Update memory after successful execution
        if result.success:
            try:
                learn_hook = LearnHook()
                await learn_hook.on_execution_complete(
                    query=query,
                    classification=classification,
                    tool_results=getattr(result, "tool_results", []),
                    response=result.response or "",
                    user_id=int(user_id) if user_id else None,
                )
            except Exception as learn_err:
                _logger.warning(f"[LEARN] Memory update failed (non-fatal): {learn_err}")

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        return JSONResponse(
            content={
                "status": "success" if result.success else "error",
                "data": {
                    "response": result.response,
                    "session_id": session_id,
                    "version": "v3",
                    "router_decision": {
                        "selected_tools": router_decision.selected_tools,
                        "complexity": router_decision.complexity.value,
                        "strategy": router_decision.execution_strategy.value,
                        "reasoning": router_decision.reasoning,
                        "confidence": router_decision.confidence,
                    },
                    "classification": {
                        "query_type": classification.query_type.value,
                        "symbols": classification.symbols,
                        "tool_categories": classification.tool_categories,
                    },
                    "execution": {
                        "total_turns": result.total_turns,
                        "total_tool_calls": result.total_tool_calls,
                        "execution_time_ms": result.total_execution_time_ms,
                    },
                    "processing_time_ms": elapsed_ms,
                    "timestamp": datetime.now().isoformat(),
                },
                "error": result.error if not result.success else None,
            }
        )

    except Exception as e:
        _logger.error(f"[CHAT_V3:ERROR] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat V3 failed: {str(e)}")


# =============================================================================
# V4 API: IntentClassifier + Agent with ALL Tools (Simplified Architecture)
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
    "/chat/v4",
    summary="Streaming Chat V4 (Intent Classifier + All Tools)",
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
async def stream_chat_v4(
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

    async def _generate_v4() -> AsyncGenerator[str, None]:
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

        try:
            # =================================================================
            # Phase 0: Process images (if any)
            # =================================================================
            processed_images: Optional[List[ProcessedImage]] = None
            if data.images:
                processed_images = await _process_images(data.images)
                if processed_images:
                    _logger.info(
                        f"[CHAT_V4] Processed {len(processed_images)} images"
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
            _logger.debug(f"[CHAT_V4] WorkingMemory initialized: {flow_id}")

            # Get Working Memory symbols from previous turns (CRITICAL FOR CONTEXT)
            wm_symbols = wm_integration.get_current_symbols()
            if wm_symbols:
                _logger.info(f"[CHAT_V4] Working Memory symbols from previous turns: {wm_symbols}")

            # Load Core Memory for user profile context (portfolio, preferences)
            core_memory_context: Optional[str] = None
            try:
                from src.agents.memory.core_memory import get_core_memory
                core_memory = get_core_memory()
                cm_data = await core_memory.load_core_memory(str(user_id))
                human_block = cm_data.get("human", "")
                if human_block and len(human_block) > 50:  # Only if meaningful content
                    core_memory_context = human_block
                    _logger.debug(f"[CHAT_V4] Loaded Core Memory context: {len(human_block)} chars")
            except Exception as cm_err:
                _logger.warning(f"[CHAT_V4] Core Memory load failed (non-fatal): {cm_err}")

            # =================================================================
            # Phase 1.5: Load Conversation History (CRITICAL FOR MEMORY!)
            # This was missing in V4 - V3 has it via NormalModeChatHandler._load_context()
            # =================================================================
            conversation_history = await _load_conversation_history(
                session_id=session_id,
                limit=10,  # Last 10 messages for context
            )
            _logger.debug(
                f"[CHAT_V4] Loaded {len(conversation_history)} messages from history"
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
                        f"[CHAT_V4] ✅ Context compacted: "
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

            intent_classifier = _get_intent_classifier()
            intent_result = await intent_classifier.classify(
                query=query,
                ui_context=data.ui_context.model_dump() if data.ui_context else None,
                conversation_history=conversation_history,  # Pass history for better context!
                working_memory_symbols=wm_symbols,  # Symbols from previous turns!
                core_memory_context=core_memory_context,  # User profile context!
            )

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

            # Emit classification reasoning
            if intent_result.reasoning and data.enable_thinking:
                yield emitter.emit_reasoning(
                    phase="intent_classification",
                    content=intent_result.reasoning,
                    action="complete",
                    metadata={
                        "intent": intent_result.intent_summary,
                        "symbols": intent_result.validated_symbols,
                        "complexity": intent_result.complexity.value,
                        "market_type": intent_result.market_type.value,
                    },
                )

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

            if data.enable_thinking:
                yield emitter.emit_reasoning(
                    phase="agent_start",
                    content=f"Starting agent with ALL {len(unified_agent.catalog.get_tool_names())} tools available",
                    action="start",
                )

            # Stream events from Agent with ALL tools
            # CRITICAL: Pass conversation_history for memory/context!
            async for event in unified_agent.run_stream_with_all_tools(
                query=query,
                intent_result=intent_result,
                conversation_history=conversation_history,  # NOW WITH HISTORY!
                system_language=intent_result.response_language,
                user_id=int(user_id) if user_id else None,
                session_id=session_id,
                enable_reasoning=data.enable_thinking,
                images=processed_images,
                model_name=data.model_name,
                provider_type=data.provider_type,
                max_turns=6,
                enable_tool_search_mode=data.enable_tool_search_mode,  # Tool Search Mode for token savings
            ):
                event_type = event.get("type", "unknown")

                if event_type == "reasoning":
                    yield emitter.emit_reasoning(
                        phase=event.get("phase", "execution"),
                        content=event.get("content", ""),
                        action=event.get("action", "thought"),
                        metadata=event.get("metadata"),
                    )

                elif event_type == "turn_start":
                    turn = event.get("turn", 1)
                    stats["total_turns"] = turn
                    yield emitter.emit_turn_start(
                        turn_number=turn,
                        max_turns=6,
                    )

                elif event_type == "tool_calls":
                    tools = event.get("tools", [])
                    stats["total_tool_calls"] += len(tools)
                    yield emitter.emit_tool_calls(tools)

                elif event_type == "tool_results":
                    results = event.get("results", [])
                    stats["tool_results"].extend(results)
                    yield emitter.emit_tool_results(results)

                elif event_type == "content":
                    content = event.get("content", "")
                    if content:
                        stats["final_content"] += content
                        yield emitter.emit_content(content)

                elif event_type == "max_turns_reached":
                    yield emitter.emit_progress(
                        phase="max_turns",
                        progress_percent=90,
                        message=f"Max turns ({event.get('turns', 0)}) reached",
                    )

                elif event_type == "done":
                    stats["total_turns"] = event.get("total_turns", stats["total_turns"])
                    stats["total_tool_calls"] = event.get("total_tool_calls", stats["total_tool_calls"])

                elif event_type == "error":
                    yield emitter.emit_error(
                        error_message=event.get("error", "Unknown error"),
                        error_code="AGENT_ERROR",
                    )

            # =================================================================
            # Phase 4: Done
            # =================================================================
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Final reasoning summary
            if data.enable_thinking:
                yield emitter.emit_reasoning(
                    phase="synthesis",
                    content=(
                        f"Complete: {stats['total_turns']} turns, "
                        f"{stats['total_tool_calls']} tool calls, "
                        f"complexity={stats['complexity']}"
                    ),
                    action="complete",
                    progress=1.0,
                    metadata={
                        "turns": stats["total_turns"],
                        "tool_calls": stats["total_tool_calls"],
                        "complexity": stats["complexity"],
                        "symbols": intent_result.validated_symbols,
                        "duration_ms": round(elapsed_ms, 0),
                    },
                )

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

            # Complete WorkingMemory request (cleanup task-specific data, preserve symbols)
            wm_integration.complete_request()

            yield emitter.emit_done(
                total_turns=stats["total_turns"],
                total_tool_calls=stats["total_tool_calls"],
                total_time_ms=elapsed_ms,
                charts=stats.get("charts"),
            )
            yield format_done_marker()

        except Exception as e:
            _logger.error(f"[CHAT_V4] Error: {e}", exc_info=True)
            # Attempt to complete working memory even on error
            try:
                wm_integration.complete_request()
            except Exception:
                pass
            yield emitter.emit_error(str(e), "CHAT_V4_ERROR")
            yield format_done_marker()

    return StreamingResponse(
        _generate_v4(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )