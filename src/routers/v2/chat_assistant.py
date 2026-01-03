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
)
from src.agents.charts import (
    resolve_charts_from_classification,
    charts_to_dict_list,
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


# --- Request/Response Schemas ---

class UIContextRequest(BaseModel):
    """
    UI Context for Soft Context Inheritance.

    Frontend sends this to indicate the current UI state,
    enabling smart disambiguation of ambiguous symbols like SOL, BTC.

    Principle: "Assume smartly, Confirm explicitly, Correct gracefully"
    """

    current_tab: str = Field(
        default="auto",
        description="Current UI tab: 'crypto', 'stock', or 'auto'. Used to resolve ambiguous symbols.",
        examples=["crypto", "stock", "auto"]
    )
    recent_symbols: List[str] = Field(
        default_factory=list,
        description="Recently viewed symbols on current tab (for context reinforcement)",
        examples=[["BTC", "ETH", "SOL"]]
    )
    watchlist_type: Optional[str] = Field(
        default=None,
        description="Type of watchlist currently displayed"
    )
    language: str = Field(
        default="vi",
        description="User's preferred language for responses"
    )


class ChatRequest(BaseModel):
    """Chat request schema."""

    query: str = Field(
        ...,
        alias="question_input",
        description="User's question or message",
        min_length=1,
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
    stream_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom stream configuration"
    )

    # UI Context for Soft Context Inheritance
    ui_context: Optional[UIContextRequest] = Field(
        default=None,
        description=(
            "UI context from frontend for soft symbol disambiguation. "
            "When user is on Crypto tab and asks about 'BTC', it resolves to Bitcoin (not BTC stock). "
            "If not provided, uses query analysis and context hints."
        )
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

    session_id = data.session_id
    if not session_id:
        session_id = _chat_service.create_chat_session(
            user_id=user_id,
            organization_id=org_id
        )

    emitter = StreamEventEmitter(session_id=session_id)

    async def _generate() -> AsyncGenerator[str, None]:
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
            # Session start
            yield emitter.emit_session_start({"mode_requested": data.mode})

            # Classify query
            yield emitter.emit_classifying()

            classifier = _get_classifier()

            ctx = ClassifierContext(
                query=data.query, 
                conversation_history=[]
            )
            classification = await classifier.classify(ctx)

            yield emitter.emit_classified(
                query_type=classification.query_type.value,
                requires_tools=classification.requires_tools,
                symbols=classification.symbols,
                categories=classification.tool_categories,
                confidence=classification.confidence,
                language=classification.response_language,
                reasoning=classification.reasoning,  # AI thought process
                intent_summary=classification.intent_summary,
            )

            # Resolve charts from classification (for frontend display)
            charts = resolve_charts_from_classification(
                classification=classification,
                query=data.query,
                max_charts=3,
            )
            if charts:
                stats["charts"] = charts_to_dict_list(charts)
                _logger.info(f"[UNIFIED_CHAT] Charts resolved: {[c.type for c in charts]}")

            # Determine mode
            mode_router = _get_mode_router()
            mode_decision = mode_router.determine_mode(
                query=data.query,
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
                async for event in _stream_deep_research(
                    query=data.query,
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
                ):
                    yield event
            else:
                async for event in _stream_normal_mode(
                    query=data.query,
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
                    ui_context=data.ui_context,
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
    ui_context: Optional[UIContextRequest] = None,
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

    # Emit thinking start if enabled
    if enable_thinking:
        thinking_start_time = dt.now()
        yield emitter.emit_thinking_start(phase="classification")
        # yield emitter.emit_thinking_delta(
        #     content=f"Query classified as: {classification.query_type.value}",
        #     phase="classification",
        # )

        if classification.reasoning:
            yield emitter.emit_llm_thought(
                thought=classification.reasoning,
                context="understanding",
            )

        # if classification.intent_summary:
        #     yield emitter.emit_thinking_delta(
        #         content=f"Intent: {classification.intent_summary}",
        #         phase="classification",
        #     )
        # yield emitter.emit_thinking_end(
        #     phase="classification",
        #     summary=f"Type: {classification.query_type.value}, Tools needed: {classification.requires_tools}",
        # )

        # Emit intent summary if available
        if classification.intent_summary:
            yield emitter.emit_llm_thought(
                thought=f"Intent: {classification.intent_summary}",
                context="intent",
            )

        yield emitter.emit_thinking_end(
            phase="understanding",
            summary=f"Type: {classification.query_type.value}, Symbols: {classification.symbols}, Categories: {classification.tool_categories}",
        )

    # Convert UIContextRequest to UIContext for handler
    from src.agents.classification import UIContext
    ui_ctx = None
    if ui_context:
        ui_ctx = UIContext.from_dict(ui_context.model_dump())

    async for chunk in handler.handle_chat(
        query=query,
        session_id=session_id,
        user_id=user_id,
        model_name=model_name,
        provider_type=provider_type,
        chart_displayed=chart_displayed,
        organization_id=org_id,
        enable_thinking=enable_thinking,
        enable_llm_events=enable_llm_events,
        stream=True,
        enable_web_search=enable_web_search,
        classification=classification,
        ui_context=ui_ctx,
    ):
        # Convert handler output to SSE events
        if isinstance(chunk, dict):
            event_type = chunk.get("type", "content")

            if event_type == "turn_start":
                turn_count = chunk.get("turn", turn_count + 1)
                stats["total_turns"] = turn_count  # Update stats
                yield emitter.emit_turn_start(turn_count, 10)

                # Emit thinking for new turn
                # if enable_thinking and enable_llm_events:
                #     yield emitter.emit_thinking_start(phase=f"turn_{turn_count}")
                #     yield emitter.emit_llm_thought(
                #         thought=f"Starting turn {turn_count} - analyzing query and available tools",
                #         context="agent_loop",
                #     )
                if enable_thinking:
                    yield emitter.emit_thinking_start(phase=f"turn_{turn_count}")
                    # yield emitter.emit_thinking_delta(
                    #     content=f"Turn {turn_count}: Deciding whether to use tools or respond directly",
                    #     phase=f"turn_{turn_count}_decision",
                    # )

            elif event_type == "tool_calls":
                tools = chunk.get("tools", [])
                tool_count += len(tools)
                stats["total_tool_calls"] = tool_count  # Update stats
                yield emitter.emit_tool_calls(tools)
                
                # Emit thinking about tool decision (using actual data)
                if enable_thinking:
                    tool_names = [t.get("name", "unknown") for t in tools]
                    # yield emitter.emit_thinking_delta(
                    #     content=f"Decided to call {len(tools)} tools: {', '.join(tool_names)}",
                    #     phase="tool_decision",
                    # )

                # Emit LLM decision events if enabled
                if enable_llm_events:
                    for tool in tools:
                        tool_name = tool.get("name", "unknown")
                        yield emitter.emit_llm_decision(
                            decision=f"Use {tool_name} tool",
                            action="tool_call",
                            confidence=0.9,
                        )
                        yield emitter.emit_llm_action(
                            action_type="tool_call",
                            action_name=tool_name,
                            reason=f"Calling {tool_name} to gather required information",
                            params=tool.get("arguments", {}),
                        )

            elif event_type == "tool_results":
                results = chunk.get("results", [])
                yield emitter.emit_tool_results(results)

                # Emit thinking about tool results
                if enable_thinking:
                    success_count = sum(1 for r in results if r.get("success", True))
                    # yield emitter.emit_thinking_delta(
                    #     content=f"Received {len(results)} tool results ({success_count} successful)",
                    #     phase="tool_analysis",
                    # )

            elif event_type == "content":
                content = chunk.get("content", "")
                if content:
                    yield emitter.emit_content(content)

            elif event_type == "thinking":
                # Forward thinking events
                if enable_thinking:
                    yield emitter.emit_thinking_delta(
                        content=chunk.get("content", ""),
                        phase=chunk.get("phase", "reasoning"),
                    )

            elif event_type == "llm_thought":
                if enable_llm_events:
                    yield emitter.emit_llm_thought(
                        thought=chunk.get("thought", ""),
                        context=chunk.get("context", ""),
                    )

            elif event_type == "done":
                # Extract charts from done event if present
                if chunk.get("charts"):
                    stats["charts"] = chunk.get("charts")

                # Emit final thinking end
                if enable_thinking and thinking_start_time:
                    duration_ms = (dt.now() - thinking_start_time).total_seconds() * 1000
                    yield emitter.emit_thinking_end(
                        phase="response_complete",
                        summary=f"Completed {turn_count} turns with {tool_count} tool calls",
                        duration_ms=duration_ms,
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
) -> AsyncGenerator[str, None]:
    """Stream Deep Research Mode responses as SSE events."""
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
    async for event in handler.handle_chat_stream(
        query=query,
        session_id=session_id,
        user_id=int(user_id) if user_id else 0,
        model_name=model_name,
        provider_type=provider_type,
        organization_id=int(org_id) if org_id else None,
        enable_thinking=enable_thinking,
        chart_displayed=chart_displayed,
        enable_compaction=enable_compaction,
        enable_think_tool=enable_think_tool,
    ):
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
            yield emitter.emit_thinking_start(phase=current_phase)
            # Emit progress
            progress = phase_progress.get(current_phase, 50)
            yield emitter.emit_progress(
                phase=current_phase,
                progress_percent=progress,
                message=event.data.get("message", f"Processing {current_phase}..."),
            )

        elif isinstance(event, ThinkingDeltaEvent):
            content = event.data.get("chunk", event.data.get("content", ""))
            if content and enable_thinking:
                yield emitter.emit_thinking_delta(
                    content=content,
                    phase=current_phase,
                )

        elif isinstance(event, ThinkingEndEvent):
            duration = 0.0
            if thinking_start:
                duration = (dt.now() - thinking_start).total_seconds() * 1000
            yield emitter.emit_thinking_end(
                phase=current_phase,
                summary=event.data.get("summary", ""),
                duration_ms=duration,
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
            if enable_llm_events:
                yield emitter.emit_llm_decision(
                    decision=f"Execute {tool_name}",
                    action="tool_call",
                    confidence=0.9,
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
            if enable_llm_events:
                yield emitter.emit_llm_thought(
                    thought=event.data.get("thought", ""),
                    context=event.data.get("context", ""),
                )

        elif isinstance(event, LLMDecisionEvent):
            if enable_llm_events:
                yield emitter.emit_llm_decision(
                    decision=event.data.get("decision", ""),
                    action=event.data.get("action", ""),
                    confidence=event.data.get("confidence", 0.0),
                )

        elif isinstance(event, DoneEvent):
            # # Final progress
            # yield emitter.emit_progress(
            #     phase="complete",
            #     progress_percent=100,
            #     message="Deep Research completed",
            # )
            # Emit thinking about synthesis
            if enable_thinking and tool_count > 0:
                yield emitter.emit_thinking_delta(
                    content="All information gathered - generating final response",
                    phase="synthesis",
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
        ctx = ClassifierContext(query=data.query, conversation_history=[])
        classification = await classifier.classify(ctx)

        # Determine processing mode
        mode_router = _get_mode_router()
        mode_decision = mode_router.determine_mode(
            query=data.query,
            explicit_mode=data.mode,
            classification=classification,
        )

        # Route to handler and collect response
        response_chunks = []

        if mode_decision.mode == QueryMode.NORMAL:
            handler = _get_normal_handler()
            async for chunk in handler.handle_chat(
                query=data.query,
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
                query=data.query,
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