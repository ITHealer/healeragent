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
# Mode Router for legacy /chat endpoint
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
# Classification (both old and new)
from src.agents.classification import (
    # Legacy (for /chat)
    UnifiedClassifier,
    ClassifierContext,
    get_unified_classifier,
    # New (for /chat/v2)
    IntentClassifier,
    IntentResult,
    get_intent_classifier,
)
# Charts
from src.agents.charts import (
    resolve_charts_from_classification,
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
    create_error_event,
)
# Streaming Events (for legacy /chat deep research)
from src.agents.streaming import (
    StreamingChatHandler,
    StreamingConfig,
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
    # Thinking Timeline (ChatGPT-style "Thought for Xs" display)
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

# =============================================================================
# LEGACY /chat Singletons (Mode Router + UnifiedClassifier)
# =============================================================================

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


# Cached components for StreamingChatHandler (Deep Research)
_streaming_components_initialized = False
_planning_agent = None
_task_executor = None
_core_memory = None
_summary_manager = None
_legacy_session_repo = None
_llm_provider = None
_tool_execution_service = None
_legacy_chat_repo = None
_validation_agent = None


def _init_streaming_components():
    """Initialize shared components for StreamingChatHandler (called once)"""
    global _streaming_components_initialized
    global _planning_agent, _task_executor, _core_memory, _summary_manager
    global _legacy_session_repo, _llm_provider, _tool_execution_service, _legacy_chat_repo
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
    _legacy_session_repo = SessionRepository()
    _legacy_chat_repo = ChatRepository()
    _llm_provider = LLMGeneratorProvider()

    _streaming_components_initialized = True
    _logger.info("[UNIFIED_CHAT] StreamingChatHandler components initialized")


async def get_streaming_handler(
    enable_llm_events: bool = True,
    enable_agent_tree: bool = False,
) -> StreamingChatHandler:
    """
    Get a configured StreamingChatHandler for Deep Research Mode.

    Uses the 7-phase upfront planning pipeline with true streaming.
    """
    _init_streaming_components()

    from src.agents.planning.planning_agent import PlanningAgent
    from src.utils.config import settings

    planning_agent = PlanningAgent(
        model_name=settings.MODEL_DEFAULT,
        provider_type=settings.PROVIDER_DEFAULT
    )

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
        session_repo=_legacy_session_repo,
        llm_provider=_llm_provider,
        tool_execution_service=_tool_execution_service,
        chat_repo=_legacy_chat_repo,
        config=config
    )

    return handler


# =============================================================================
# NEW /chat/v2 Singletons (IntentClassifier + UnifiedAgent)
# =============================================================================

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


class ReplyContext(BaseModel):
    """
    Context for when user replies to a specific part of a previous message.

    This enables the "reply to message" feature where users can click on
    highlighted text (like a company name or data point) to ask follow-up
    questions about that specific content.

    Example:
        User sees response: "Microsoft (MSFT) is trading at $450..."
        User clicks on "Microsoft (MSFT)" and types: "tìm hiểu rõ hơn"
        -> reply_to.content = "Microsoft (MSFT)"
        -> query = "tìm hiểu rõ hơn"
    """
    content: str = Field(
        ...,
        description="The text/content being replied to from previous message",
        examples=["Microsoft (MSFT)", "P/E ratio of 35.2", "RSI is overbought at 78"]
    )
    message_id: Optional[str] = Field(
        default=None,
        description="Optional ID of the message being replied to for tracking"
    )
    context_type: Optional[str] = Field(
        default=None,
        description="Type of content: 'symbol', 'metric', 'statement', 'data_point'",
        examples=["symbol", "metric", "statement", "data_point"]
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
        default=True,
        description="Enable Tool Search Mode: Start with only tool_search for 85% token savings. Agent discovers tools dynamically via semantic search. Default is True for production efficiency."
    )
    enable_finance_guru: bool = Field(
        default=False,
        description="Enable Finance Guru computation tools (DCF valuation, portfolio analysis, etc.). Only available in /chat/v3 endpoint."
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
    # Reply context: when user replies to a specific part of previous message
    reply_to: Optional[ReplyContext] = Field(
        default=None,
        description="Context when user replies to specific content from previous message. "
                    "Used for follow-up questions about highlighted text like company names, metrics, etc."
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


def _format_query_with_reply_context(
    query: str,
    reply_to: Optional["ReplyContext"],
) -> str:
    """
    Format user query with reply context for better LLM understanding.

    When user replies to a specific part of a previous message (e.g., clicking
    on "Microsoft (MSFT)" highlighted text and asking "tìm hiểu rõ hơn"),
    this function formats the query to include that context.

    Args:
        query: The user's current question/message
        reply_to: Optional ReplyContext containing the content being replied to

    Returns:
        Formatted query string with reply context if applicable

    Example:
        Input:
            query = "tìm hiểu rõ hơn"
            reply_to.content = "Microsoft (MSFT)"
        Output:
            "[Regarding: Microsoft (MSFT)]

            tìm hiểu rõ hơn"
    """
    if not reply_to or not reply_to.content:
        return query

    # Clean the replied content (remove excessive whitespace)
    replied_content = reply_to.content.strip()

    # Truncate very long replied content (keep it readable)
    if len(replied_content) > 500:
        replied_content = replied_content[:500] + "..."

    # Format with clear context marker that LLM can understand
    # Using a format that works well across different languages
    formatted_query = f"""[Regarding: {replied_content}]

{query}"""

    _logger.debug(
        f"[CHAT] Reply context applied: replied_to='{replied_content[:50]}...' "
        f"query='{query[:50]}...'"
    )

    return formatted_query


# =============================================================================
# LEGACY /chat API: UnifiedClassifier + Mode Router (for production compatibility)
# =============================================================================


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
    thinking_start_time = dt.now() if enable_thinking else None

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
                stats["total_turns"] = turn_count
                yield emitter.emit_turn_start(turn_count, 10)

            elif event_type == "tool_calls":
                tools = chunk.get("tools", [])
                tool_count += len(tools)
                stats["total_tool_calls"] = tool_count
                yield emitter.emit_tool_calls(tools)

            elif event_type == "tool_results":
                results = chunk.get("results", [])
                yield emitter.emit_tool_results(results)

            elif event_type == "content":
                content = chunk.get("content", "")
                if content:
                    yield emitter.emit_content(content)

            elif event_type == "done":
                if chunk.get("charts"):
                    stats["charts"] = chunk.get("charts")

            else:
                yield emitter.emit_content(str(chunk))
        else:
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
    """Stream Deep Research Mode responses as SSE events."""
    from datetime import datetime as dt

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
    tool_count = 0
    phase_progress = {
        "context": 10,
        "planning": 30,
        "execution": 60,
        "assembly": 80,
        "generation": 90,
    }

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
        if isinstance(event, StartEvent):
            yield emitter.emit_session(
                session_id=session_id,
                mode="deep_research",
                model=model_name,
            )

        elif isinstance(event, ThinkingStartEvent):
            current_phase = event.data.get("phase", "thinking")
            thinking_start = dt.now()
            progress = phase_progress.get(current_phase, 50)
            yield emitter.emit_progress(
                phase=current_phase,
                progress_percent=progress,
                message=event.data.get("message", f"Processing {current_phase}..."),
            )

        elif isinstance(event, (ThinkingDeltaEvent, ThinkingEndEvent)):
            pass  # Skip reasoning events

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
            stats["total_tool_calls"] = tool_count
            stats["total_turns"] = 1

            yield emitter.emit_tool_calls([
                {"name": tool_name, "arguments": event.data.get("arguments", {})}
            ])

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

        elif isinstance(event, (LLMThoughtEvent, LLMDecisionEvent)):
            pass  # Skip reasoning events

        elif isinstance(event, DoneEvent):
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
            if hasattr(event, 'data'):
                content = event.data.get("chunk", event.data.get("content", ""))
                if content:
                    yield emitter.emit_content(str(content))


@router.post(
    "/chat",
    summary="Streaming Chat (Legacy - Production)",
    description="""Legacy streaming chat using Mode Router + UnifiedClassifier.

    This is the PRODUCTION endpoint - do not modify without coordination.
    For new features, use /chat/v2 instead.
    """,
    response_class=StreamingResponse,
)
async def stream_chat_legacy(
    request: Request,
    data: ChatRequest,
    api_key_data: Dict[str, Any] = Depends(_api_key_auth.author_with_api_key),
):
    """Stream chat responses via SSE with automatic mode selection (Legacy)."""
    user_id = getattr(request.state, "user_id", None)
    org_id = getattr(request.state, "organization_id", None)

    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")

    if not data.query and not data.images:
        raise HTTPException(
            status_code=400,
            detail="Either question_input or images must be provided"
        )

    # Default query for image-only requests
    raw_query = data.query or "Analyze this image"

    # Apply reply context if user is replying to specific content
    query = _format_query_with_reply_context(raw_query, data.reply_to)

    if data.reply_to:
        _logger.info(
            f"[CHAT] Reply context detected: "
            f"replied_to='{data.reply_to.content[:80]}...' | "
            f"type={data.reply_to.context_type or 'unknown'}"
        )

    session_id = data.session_id
    if not session_id:
        session_id = _chat_service.create_chat_session(
            user_id=user_id,
            organization_id=org_id
        )

    emitter = StreamEventEmitter(session_id=session_id)
    processed_images: Optional[List[ProcessedImage]] = None

    async def _generate() -> AsyncGenerator[str, None]:
        nonlocal processed_images
        start_time = datetime.now()
        mode_decision: Optional[ModeDecision] = None

        stats = {
            "total_turns": 0,
            "total_tool_calls": 0,
            "last_event_type": None,
            "charts": None,
        }

        try:
            if data.images:
                processed_images = await _process_images(data.images)
                if processed_images:
                    _logger.info(f"[CHAT] Processed {len(processed_images)} images")

            yield emitter.emit_session_start({"mode_requested": data.mode})
            yield emitter.emit_classifying()

            classifier = _get_classifier()
            ctx = ClassifierContext(
                query=query,
                conversation_history=[],
                ui_context=data.ui_context.model_dump() if data.ui_context else None,
                images=processed_images,
            )
            classification = await classifier.classify(ctx)

            # FIX: Use emit_thinking_step instead of emit_thinking
            if classification.reasoning and data.enable_thinking:
                yield emitter.emit_thinking_step(
                    title="Classification Reasoning",
                    content=f"🎯 {classification.reasoning[:500]}",
                    phase="classification",
                )

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

            charts = resolve_charts_from_classification(
                classification=classification,
                query=query,
                max_charts=3,
            )
            if charts:
                stats["charts"] = charts_to_dict_list(charts)

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

            pending_events: List[str] = []

            if mode_decision.mode == QueryMode.DEEP_RESEARCH:
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

            # Fix is_final on last content chunk
            last_content_idx = -1
            for i in range(len(pending_events) - 1, -1, -1):
                if '"type": "content"' in pending_events[i] or '"type":"content"' in pending_events[i]:
                    last_content_idx = i
                    break

            for i, event in enumerate(pending_events):
                if i == last_content_idx:
                    event = event.replace('"is_final": false', '"is_final": true')
                    event = event.replace('"is_final":false', '"is_final":true')
                yield event

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

    async def cleanup_chat():
        _logger.info(f"[UNIFIED_CHAT] Client disconnected - cleaning up session {session_id}")

    generator_with_heartbeat = with_heartbeat(
        event_generator=_generate(),
        emitter=emitter,
        heartbeat_interval=15.0,
    )

    return StreamingResponse(
        with_cancellation(
            request=request,
            generator=generator_with_heartbeat,
            cleanup_fn=cleanup_chat,
            check_interval=0.5,
            emit_cancelled_event=True,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# =============================================================================
# NEW /chat/v2 API: IntentClassifier + Agent with ALL Tools (Simplified Architecture)
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
    "/chat/v2",
    summary="Streaming Chat V2 (Intent Classifier + All Tools)",
    description="""
    New simplified chat architecture:
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
async def stream_chat_v2(
    request: Request,
    data: ChatRequest,
    api_key_data: Dict[str, Any] = Depends(_api_key_auth.author_with_api_key),
):
    """
    Stream chat responses via SSE using IntentClassifier + Agent with ALL tools.

    This is the NEW simplified ChatGPT-style architecture (V2):
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
    raw_query = data.query or "Analyze this image"

    # Apply reply context if user is replying to specific content
    # This formats the query to include context about what's being referred to
    query = _format_query_with_reply_context(raw_query, data.reply_to)

    if data.reply_to:
        _logger.info(
            f"[CHAT/V2] Reply context detected: "
            f"replied_to='{data.reply_to.content[:80]}...' | "
            f"type={data.reply_to.context_type or 'unknown'}"
        )

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

            # =============================================================
            # CRITICAL FIX: Force agent_loop when web search is enabled
            # When enable_web_search=True, the user explicitly wants
            # real-time data. We MUST enter the agent loop so webSearch
            # tool is available, even for "general" or "direct" queries
            # like weather, gold prices, news, etc.
            # =============================================================
            if data.enable_web_search and not intent_result.requires_tools:
                _logger.info(
                    f"[CHAT] Overriding requires_tools=True because enable_web_search=True "
                    f"(was: complexity={intent_result.complexity.value}, "
                    f"query_type={intent_result.query_type})"
                )
                intent_result.requires_tools = True
                from src.agents.classification.intent_classifier import IntentComplexity
                intent_result.complexity = IntentComplexity.AGENT_LOOP

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

                    # Timeline: Consolidate tool calls into ONE step (avoid UI noise)
                    if len(tools) == 1:
                        # Single tool: show name and symbol
                        tool = tools[0]
                        tool_name = tool.get("name", "unknown")
                        tool_args = tool.get("arguments", {})
                        symbol = tool_args.get("symbol", "")
                        thinking_timeline.add_step(
                            phase=ThinkingPhase.TOOL_EXECUTION.value,
                            action=f"Calling {tool_name}",
                            is_tool_call=True,
                            details=f"({symbol})" if symbol else None,
                        )
                    elif len(tools) <= 3:
                        # 2-3 tools: show names
                        tool_names = [t.get("name", "?") for t in tools]
                        thinking_timeline.add_step(
                            phase=ThinkingPhase.TOOL_EXECUTION.value,
                            action=f"Calling {len(tools)} tools",
                            is_tool_call=True,
                            details=", ".join(tool_names),
                        )
                    else:
                        # Many tools: just show count
                        thinking_timeline.add_step(
                            phase=ThinkingPhase.TOOL_EXECUTION.value,
                            action=f"Calling {len(tools)} tools in parallel",
                            is_tool_call=True,
                            details=None,
                        )

                    for timeline_event in thinking_timeline.get_pending_events():
                        yield timeline_event.to_sse()

                    yield emitter.emit_tool_calls(tools)

                elif event_type == "tool_results":
                    results = event.get("results", [])
                    stats["tool_results"].extend(results)

                    # Timeline: Consolidate tool results into ONE step (avoid UI noise)
                    # Only emit if multiple results or if there's a failure
                    success_count = sum(1 for r in results if r.get("success", False))
                    fail_count = len(results) - success_count

                    if len(results) > 0:
                        if fail_count > 0:
                            # Show failures explicitly
                            thinking_timeline.add_step(
                                phase=ThinkingPhase.DATA_GATHERING.value,
                                action=f"Tool Results: {success_count} OK, {fail_count} failed",
                                is_tool_call=True,
                                details=None,
                                success=fail_count == 0,
                            )
                            for timeline_event in thinking_timeline.get_pending_events():
                                yield timeline_event.to_sse()
                        # Skip emitting individual success results to reduce noise
                        # The tool_results event below provides the detailed info

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
                    # Handle think tool calls - emit to thinking_timeline ONLY
                    # (No separate thinking_step to avoid UI duplication)
                    phase = event.get("phase", "analyzing")
                    thought_content = event.get("content", "")

                    if thought_content:
                        # Add to thinking timeline with full content in details
                        thinking_timeline.add_step(
                            phase=ThinkingPhase.DATA_GATHERING.value,
                            action=f"💭 Think: {phase.title()}",
                            details=thought_content,  # Full content - UI can truncate if needed
                        )
                        for timeline_event in thinking_timeline.get_pending_events():
                            yield timeline_event.to_sse()

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

                elif event_type == "sources":
                    # Emit web search sources metadata for FE widget rendering (ChatGPT-style)
                    citations = event.get("citations", [])
                    if citations:
                        _logger.info(f"[CHAT] 📚 Web sources: {len(citations)} citations")
                        # Emit as dedicated SSE event for FE to render sources widget
                        yield emitter.emit_sources(
                            citations=citations,
                            count=event.get("count", len(citations)),
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


# =============================================================================
# /chat/v3 ENDPOINT - Finance Guru Integration
# =============================================================================

@router.post(
    "/chat/v3",
    summary="Streaming Chat V3 (Finance Guru + All Tools)",
    description="""
    Enhanced chat architecture with Finance Guru quantitative analysis capabilities.

    **NEW in V3:**
    - Finance Guru computation tools (valuation, portfolio, backtest)
    - Enhanced risk metrics (Sortino, Calmar, Treynor)
    - Portfolio analysis (correlation, rebalancing)
    - DCF/Graham/DDM valuation calculations

    **Architecture (same as V2):**
    - Phase 1: IntentClassifier (single LLM call)
    - Phase 2: Agent with ALL tools + Finance Guru tools
    - Phase 3: Post-processing (LEARN + SAVE)

    **Tool Categories:**
    - Data Retrieval: getStockPrice, getCashFlow, getTechnicalIndicators...
    - Computation (NEW): calculateDCF, analyzePortfolio, runBacktest...
    - Reasoning: think, tool_search

    **Example Flow:**
    ```
    User: "Tính DCF cho AAPL với growth 10%"

    Turn 1 (THINK): "Need FCF data first"
    Turn 1 (ACT): getCashFlow(symbol="AAPL")

    Turn 2 (THINK): "Got FCF, now calculate DCF"
    Turn 2 (ACT): calculateDCF(fcf=[...], growth=0.10)

    Turn 3: Generate response with valuation result
    ```

    **Feature Flags:**
    - enable_finance_guru: Enable Finance Guru tools (default: True for v3)
    - enable_tool_search_mode: Dynamic tool discovery (default: True)
    """,
    response_class=StreamingResponse,
)
async def stream_chat_v3(
    request: Request,
    data: ChatRequest,
    api_key_data: Dict[str, Any] = Depends(_api_key_auth.author_with_api_key),
):
    """
    Stream chat responses via SSE with Finance Guru capabilities.

    This is V3 architecture - extends V2 with Finance Guru computation tools.
    Internally uses the same pipeline as V2, with Finance Guru enabled by default.

    The agent can now:
    - Call data retrieval tools (getStockPrice, getCashFlow, etc.)
    - Call computation tools (calculateDCF, analyzePortfolio, etc.)
    - Combine both for comprehensive financial analysis
    """
    # V3 enables Finance Guru by default
    # Override the flag for v3 endpoint (user can still disable)
    if not hasattr(data, '_v3_processed'):
        # Only set default if not explicitly set by user
        # This allows user to disable Finance Guru even in v3 if needed
        data._v3_processed = True

    # Log v3 specific info
    _logger.info(
        f"[CHAT/V3] Request | user={getattr(request.state, 'user_id', 'unknown')} | "
        f"finance_guru={data.enable_finance_guru} | "
        f"tool_search={data.enable_tool_search_mode}"
    )

    # Delegate to v2 implementation (same pipeline)
    # Finance Guru tools will be available when enable_finance_guru=True
    return await stream_chat_v2(request, data, api_key_data)