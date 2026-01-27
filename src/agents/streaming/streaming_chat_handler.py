import asyncio
import time
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Any, Optional
from dataclasses import dataclass, field

from src.utils.logger.custom_logging import LoggerMixin
from src.database.repository.chat import ChatRepository
from src.agents.streaming.stream_events import (
    StreamEvent,
    StreamEventType,
    StreamState,
    StartEvent,
    ThinkingStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    PlanningProgressEvent,
    PlanningCompleteEvent,
    ToolStartEvent,
    ToolCompleteEvent,
    ContextLoadingEvent,
    ContextLoadedEvent,
    TextDeltaEvent,
    TextCompleteEvent,
    MemoryUpdateEvent,
    DoneEvent,
    ErrorEvent,
    HeartbeatEvent,
    # LLM Decision Events
    LLMThoughtEvent,
    LLMDecisionEvent,
    LLMActionEvent,
    # Agent Tree Events
    AgentNodeEvent,
    # Utilities
    generate_call_id,
    generate_node_id,
)

from src.agents.streaming.agent_tree import (
    AgentTree,
    TreeNodeContext,
    NodeType,
    NodeStatus,
    create_tree_for_request,
)
from src.services.context_management_service import ContextManagementService


@dataclass
class StreamingConfig:
    """Configuration for streaming chat behavior."""

    # Feature toggles
    enable_thinking_display: bool = True
    enable_tool_progress: bool = True
    enable_context_events: bool = True
    enable_memory_events: bool = False

    # Thinking cutoff
    max_thinking_length: int = 2000

    # LLM decision events
    enable_llm_decision_events: bool = True

    # Agent tree tracking
    enable_agent_tree: bool = True

    # Content limits
    max_tool_preview_length: int = 500

    # Buffer settings
    text_chunk_min_size: int = 5  # Minimum chars before emitting text chunk

    # Persistence
    save_messages: bool = True

    # Context compaction (matching ChatHandler)
    enable_compaction: bool = True
    max_context_tokens: int = 80000  # tokens

    # SSE cancellation and heartbeat
    cancellation_check_interval: float = 0.1
    enable_heartbeat: bool = True
    heartbeat_interval: int = 30

    # LLM generation settings
    max_output_tokens: int = 8192  # Max tokens for response generation
    temperature: float = 0.7  # Default temperature for response


class CancellationToken:
    """Token to track client disconnection state."""

    def __init__(self):
        self._cancelled = False
        self._cancel_event = asyncio.Event()
    
    def cancel(self):
        """Mark request as cancelled."""
        self._cancelled = True
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        """Check if request is cancelled."""
        return self._cancelled

    async def wait_for_cancel(self, timeout: float = None) -> bool:
        """Wait for cancellation with optional timeout. Returns True if cancelled."""
        try:
            await asyncio.wait_for(self._cancel_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


class StreamingChatHandler(LoggerMixin):
    """
    Streaming chat handler with 7-phase pipeline.

    Phases: Context -> Planning -> Execution -> Assembly -> Generation -> Completion
    Features: Context compaction, working memory, think tool, atomic tools
    """

    def __init__(
        self,
        planning_agent,
        task_executor,
        core_memory,
        summary_manager,
        session_repo,
        llm_provider,
        tool_execution_service=None,
        chat_repo=None,
        config: StreamingConfig = None
    ):
        super().__init__()
        
        self.planning_agent = planning_agent
        self.task_executor = task_executor
        self.core_memory = core_memory
        self.summary_manager = summary_manager
        self.session_repo = session_repo
        self.llm_provider = llm_provider
        self.tool_execution_service = tool_execution_service
        self.config = config or StreamingConfig()
        self.chat_repo = chat_repo or ChatRepository()
        
        # Instance state (reset per request)
        self._current_task_plan = None
        self._execution_results = {}
        self._accumulated_response = ""
        self._current_question_id = None
        
        # Cancellation token
        self._cancellation_token: Optional[CancellationToken] = None

        # Agent tree
        self._agent_tree: Optional[AgentTree] = None

        # Context management service (lazy init)
        self._context_manager = None
        
        # Think tool service (lazy init per request)
        self._think_service = None
        
        # Working memory integration (lazy init per request)
        self._wm_integration = None
    
        self.context_manager = ContextManagementService(
            enable_compaction=True,
            max_context_tokens=100000,
            trigger_percent=85.0,
            compaction_strategy="smart_summary"
        )

        self.logger.info("[STREAM:INIT] Handler initialized")

    def _get_context_manager(self):
        """Lazy initialization of context manager."""
        if self._context_manager is None:
            try:
                from src.services.context_management_service import ContextManagementService
                self._context_manager = ContextManagementService(
                    enable_compaction=self.config.enable_compaction,
                    max_context_tokens=self.config.max_context_tokens,
                    trigger_percent=80.0,
                    compaction_strategy="smart_summary"
                )
            except ImportError:
                self.logger.warning("[STREAM:INIT] ContextManagementService not available")
        return self._context_manager

    async def _check_cancellation(self) -> bool:
        """Check if client has disconnected. Returns True if cancelled."""
        if self._cancellation_token and self._cancellation_token.is_cancelled:
            self.logger.info("[STREAM] Client disconnected - cancelling request")
            return True
        return False

    async def _cleanup_on_cancel(self, flow_id: str):
        """Cleanup resources when request is cancelled."""
        self.logger.info(f"[{flow_id}] Cleaning up cancelled request")

        if self._wm_integration:
            self._wm_integration.save_error(
                error_type="client_disconnected",
                message="Client disconnected before completion",
                recoverable=True
            )
            self._wm_integration.complete_request(clear_task_data=False)

        if self._agent_tree:
            for node_id, node in self._agent_tree.nodes.items():
                if node.status in [NodeStatus.STARTED, NodeStatus.RUNNING]:
                    self._agent_tree.end_node(
                        node_id=node_id,
                        success=False,
                        error="cancelled"
                    )

        self.logger.debug(f"[{flow_id}] Cleanup complete")

    async def handle_chat_stream(
        self,
        query: str,
        session_id: str,
        user_id: int,
        model_name: str = "gpt-4.1-nano",
        provider_type: str = "openai",
        organization_id: Optional[int] = None,
        enable_thinking: bool = True,
        chart_displayed: bool = False,
        enable_compaction: bool = True,
        enable_think_tool: bool = False,
        cancellation_token: CancellationToken = None,
        **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Main streaming entry point. Yields StreamEvent objects for SSE.

        Phases: Context -> Planning -> Execution -> Assembly -> Generation -> Done
        """
        flow_id = f"STREAM-{datetime.now().strftime('%H%M%S')}-{session_id[:8]}"
        flow_start = time.time()

        # Reset instance state
        self._current_task_plan = None
        self._execution_results = {}
        self._accumulated_response = ""
        self._current_question_id = None
        self._cancellation_token = cancellation_token

        if self.config.enable_agent_tree:
            self._agent_tree = create_tree_for_request(flow_id)

        state = StreamState(session_id=session_id, flow_id=flow_id)

        # Log request start
        query_preview = query[:80] + "..." if len(query) > 80 else query
        self.logger.info(
            f"[{flow_id}] START | model={model_name} | "
            f"compaction={'ON' if enable_compaction else 'OFF'} | "
            f"think_tool={'ON' if enable_think_tool else 'OFF'}"
        )
        self.logger.debug(f"[{flow_id}] Query: {query_preview}")
        
        # Setup Working Memory for this request
        wm_context = ""
        try:
            from src.agents.memory.working_memory_integration import (
                setup_working_memory_for_request
            )
            self._wm_integration = setup_working_memory_for_request(
                session_id=session_id,
                user_id=str(user_id),
                flow_id=flow_id
            )
            wm_context = self._wm_integration.get_context_for_planning(max_tokens=1000)
        except ImportError:
            self.logger.warning(f"[{flow_id}] Working Memory not available")
            self._wm_integration = None
        
        # Setup Think Tool if enabled
        if enable_think_tool:
            try:
                from src.services.think_tool_service import ThinkToolService
                self._think_service = ThinkToolService(
                    enabled=True,
                    model_name=model_name
                )
            except ImportError:
                self.logger.warning(f"[{flow_id}] ThinkToolService not available")
                self._think_service = None
        
        try:
            # Check for early cancellation
            if await self._check_cancellation():
                yield ErrorEvent(
                    error_message="Request cancelled",
                    error_type="Cancelled",
                    phase="initialization",
                    recoverable=True
                )
                return
            
            # Save user message
            if self.config.save_messages:
                await self._save_user_message(
                    flow_id=flow_id,
                    session_id=session_id,
                    user_id=user_id,
                    query=query
                )
            
            root_node_id = self._agent_tree.root_id if self._agent_tree else None

            # Phase 1: Start event
            yield StartEvent(
                session_id=session_id,
                flow_id=flow_id,
                model_name=model_name,
                node_id=root_node_id
            )
            
            # Phase 2: Context loading
            if await self._check_cancellation():
                await self._cleanup_on_cancel(flow_id)
                return

            context_node_id = None
            if self._agent_tree:
                context_node_id = self._agent_tree.start_node(
                    node_type=NodeType.PHASE,
                    name="context_loading",
                    parent_id=root_node_id
                )

            state.start_phase("context_loading")
            self.logger.info(f"[{flow_id}] PHASE:CONTEXT | Loading context...")
            
            if enable_thinking and self.config.enable_context_events:
                yield ThinkingStartEvent(
                    phase="context",
                    message="Loading conversation context...",
                    estimated_steps=3,
                    node_id=context_node_id,
                    parent_id=root_node_id
                )
            
            context_data = await self._phase1_load_context_stream(
                flow_id=flow_id,
                user_id=user_id,
                session_id=session_id,
                state=state,
                enable_events=self.config.enable_context_events,
                enable_compaction=enable_compaction,
                wm_context=wm_context,
                parent_node_id=context_node_id
            )
            
            # Emit context events
            for event in context_data.get('events', []):
                if isinstance(event, StreamEvent):
                    yield event
            
            if self.config.enable_context_events:
                yield ContextLoadedEvent(
                    total_tokens=context_data.get('total_tokens'),
                    context_usage_percent=context_data.get('usage_percent'),
                    components=context_data.get('components'),
                    node_id=context_node_id
                )
            
            context_duration = state.end_phase()

            # End context node in tree
            if self._agent_tree and context_node_id:
                self._agent_tree.end_node(context_node_id, success=True)

            if enable_thinking and self.config.enable_context_events:
                yield ThinkingEndEvent(
                    phase="context",
                    summary=self._build_context_summary(context_data),
                    duration_ms=int(context_duration),
                    node_id=context_node_id
                )

            self.logger.info(f"[{flow_id}] PHASE:CONTEXT | Complete | {context_duration:.0f}ms")

            # Phase 3: Planning
            if await self._check_cancellation():
                await self._cleanup_on_cancel(flow_id)
                return

            planning_node_id = None
            if self._agent_tree:
                planning_node_id = self._agent_tree.start_node(
                    node_type=NodeType.PLANNING,
                    name="planning",
                    parent_id=root_node_id
                )

            state.start_phase("planning")
            self.logger.info(f"[{flow_id}] PHASE:PLANNING | Analyzing query...")
            
            plan_data = None
            async for event in self._phase2_planning_stream(
                flow_id=flow_id,
                query=query,
                context_data=context_data,
                state=state,
                enable_thinking=enable_thinking,
                wm_context=wm_context,
                model_name=model_name,
                provider_type=provider_type,
                parent_node_id=planning_node_id
            ):
                # Check cancellation during planning
                if await self._check_cancellation():
                    await self._cleanup_on_cancel(flow_id)
                    return
                
                if isinstance(event, dict) and 'task_plan' in event:
                    plan_data = event
                else:
                    yield event
            
            planning_duration = state.end_phase()

            if self._agent_tree and planning_node_id:
                task_count = len(plan_data['task_plan'].tasks) if plan_data and plan_data.get('task_plan') else 0
                self._agent_tree.end_node(
                    planning_node_id,
                    success=plan_data is not None,
                    metadata={"task_count": task_count}
                )

            self.logger.info(f"[{flow_id}] PHASE:PLANNING | Complete | {planning_duration:.0f}ms")

            # Check if tools needed
            if not plan_data or not plan_data.get('task_plan') or len(plan_data['task_plan'].tasks) == 0:
                self.logger.info(f"[{flow_id}] No tools needed - direct response")
                execution_data = {'all_tool_results': {}, 'tools_executed': [], 'mode': None}
                execution_duration = 0
            else:
                # Phase 4: Tool execution
                if await self._check_cancellation():
                    await self._cleanup_on_cancel(flow_id)
                    return

                execution_node_id = None
                if self._agent_tree:
                    execution_node_id = self._agent_tree.start_node(
                        node_type=NodeType.PHASE,
                        name="execution",
                        parent_id=root_node_id
                    )

                state.start_phase("execution", execution_node_id)
                task_count = len(plan_data['task_plan'].tasks)
                self.logger.info(f"[{flow_id}] PHASE:EXECUTION | Executing {task_count} tasks...")
                
                if enable_thinking and self.config.enable_thinking_display:
                    yield ThinkingStartEvent(
                        phase="execution",
                        message=f"Executing {task_count} analysis tasks...",
                        estimated_steps=task_count,
                        node_id=execution_node_id,
                        parent_id=root_node_id
                    )

                execution_data = {'all_tool_results': {}, 'tools_executed': [], 'mode': 'task_based'}

                async for event in self._phase4_tool_execution_stream_fixed(
                    flow_id=flow_id,
                    query=query,
                    plan_data=plan_data,
                    context_data=context_data,
                    provider_type=provider_type,
                    model_name=model_name,
                    user_id=user_id,
                    session_id=session_id,
                    state=state,
                    parent_node_id=execution_node_id
                ):
                    # Check cancellation during execution
                    if await self._check_cancellation():
                        await self._cleanup_on_cancel(flow_id)
                        return
                    
                    if isinstance(event, dict) and 'all_tool_results' in event:
                        execution_data = event
                    else:
                        yield event
                
                execution_duration = state.end_phase()

                # End execution node
                if self._agent_tree and execution_node_id:
                    self._agent_tree.end_node(
                        execution_node_id,
                        success=True,
                        metadata={"tools_executed": len(execution_data.get('tools_executed', []))}
                    )

                if enable_thinking and self.config.enable_thinking_display:
                    tools_count = len(execution_data.get('tools_executed', []))
                    yield ThinkingEndEvent(
                        phase="execution",
                        summary=f"Completed {tools_count} tool executions",
                        duration_ms=int(execution_duration),
                        node_id=execution_node_id
                    )

                self.logger.info(f"[{flow_id}] PHASE:EXECUTION | Complete | {execution_duration:.0f}ms")

            # Phase 5: Context assembly
            if await self._check_cancellation():
                await self._cleanup_on_cancel(flow_id)
                return

            state.start_phase("assembly")
            self.logger.debug(f"[{flow_id}] PHASE:ASSEMBLY | Assembling context...")
            
            assembled_context = await self._phase5_context_assembly(
                flow_id=flow_id,
                query=query,
                chart_displayed=chart_displayed,
                plan_data=plan_data or {},
                execution_data=execution_data,
                model_name=model_name,
                enable_thinking=enable_thinking,
                context_data=context_data
            )
            
            assembly_duration = state.end_phase()
            self.logger.debug(f"[{flow_id}] PHASE:ASSEMBLY | Complete | {assembly_duration:.0f}ms")

            # Phase 6: Response generation
            if await self._check_cancellation():
                await self._cleanup_on_cancel(flow_id)
                return

            generation_node_id = None
            if self._agent_tree:
                generation_node_id = self._agent_tree.start_node(
                    node_type=NodeType.GENERATION,
                    name="response_generation",
                    parent_id=root_node_id
                )

            state.start_phase("generation", generation_node_id)
            self.logger.info(f"[{flow_id}] PHASE:GENERATION | Streaming response...")
            
            complete_response = ""
            async for event in self._phase6_generate_response_stream(
                flow_id=flow_id,
                query=query,
                assembled_context=assembled_context,
                execution_data=execution_data,
                user_id=user_id,
                session_id=session_id,
                model_name=model_name,
                provider_type=provider_type,
                enable_thinking=enable_thinking,
                state=state,
                parent_node_id=generation_node_id
            ):
                # Check cancellation during generation
                if await self._check_cancellation():
                    # Still yield partial response before cancelling
                    await self._cleanup_on_cancel(flow_id)
                    return
                
                if isinstance(event, TextDeltaEvent):
                    complete_response += event.data.get('chunk', '')
                yield event
            
            generation_duration = state.end_phase()

            if self._agent_tree and generation_node_id:
                self._agent_tree.end_node(
                    generation_node_id,
                    success=True,
                    metadata={"response_length": len(complete_response)}
                )

            self.logger.info(f"[{flow_id}] PHASE:GENERATION | Complete | {generation_duration:.0f}ms")

            # Phase 7: Post-processing
            self.logger.debug(f"[{flow_id}] PHASE:COMPLETION | Post-processing...")
            
            # Save response
            if self.config.save_messages and complete_response:
                await self._save_assistant_response(
                    flow_id=flow_id,
                    session_id=session_id,
                    response=complete_response,
                    state=state
                )
            
            # Cleanup Working Memory
            working_memory_context = None
            if self._wm_integration:
                working_memory_context = self._wm_integration.get_context_for_memory_update()
                self.logger.debug(
                    f"[{flow_id}] Working Memory context for memory update: "
                    f"{working_memory_context[:200] if working_memory_context else 'None'}..."
                )
            
            if self._wm_integration:
                self._wm_integration.complete_request(clear_task_data=True)
            
            # Background memory updates
            asyncio.create_task(
                self._background_memory_updates(
                    flow_id=flow_id,
                    user_id=user_id,
                    session_id=session_id,
                    query=query,
                    response=complete_response,
                    organization_id=organization_id,
                    provider_type=provider_type,
                    model_name=model_name,
                    working_memory_context=working_memory_context
                )
            )
            
            # Final stats
            total_time = time.time() - flow_start
            stats = {
                "total_time_ms": int(total_time * 1000),
                "phases": {
                    "context": int(context_duration),
                    "planning": int(planning_duration),
                    "execution": int(execution_duration) if execution_duration else 0,
                    "assembly": int(assembly_duration),
                    "generation": int(generation_duration)
                },
                "tools_executed": len(execution_data.get('tools_executed', [])),
                "response_length": len(complete_response)
            }
            
            # Get agent tree summary
            agent_tree_summary = None
            if self._agent_tree:
                # End root node
                self._agent_tree.end_node(
                    self._agent_tree.root_id,
                    success=True,
                    metadata=stats
                )
                agent_tree_summary = self._agent_tree.get_summary()
                
                # Log tree visualization for debugging
                self.logger.debug(f"[{flow_id}] Agent Tree:\n{self._agent_tree.visualize()}")
            
            yield DoneEvent(
                session_id=session_id,
                flow_id=flow_id,
                total_duration_ms=int(total_time * 1000),
                stats=stats,
                agent_tree=agent_tree_summary,
                node_id=root_node_id
            )

            self.logger.info(
                f"[{flow_id}] COMPLETE | total={total_time:.2f}s | "
                f"tools={len(execution_data.get('tools_executed', []))} | "
                f"response={len(complete_response)} chars"
            )

        except asyncio.CancelledError:
            self.logger.info(f"[{flow_id}] Stream cancelled via asyncio")
            await self._cleanup_on_cancel(flow_id)
            yield ErrorEvent(
                error_message="Stream cancelled",
                error_type="Cancelled",
                phase=state.current_phase or "unknown",
                recoverable=True
            )

        except Exception as e:
            self.logger.error(f"[{flow_id}] STREAM ERROR: {e}", exc_info=True)

            if self._wm_integration:
                self._wm_integration.save_error(
                    error_type="stream_error",
                    message=str(e),
                    recoverable=True
                )
                self._wm_integration.complete_request(clear_task_data=False)

            if self._agent_tree:
                self._agent_tree.update_node(
                    self._agent_tree.root_id,
                    status=NodeStatus.FAILED,
                    error=str(e)
                )

            yield ErrorEvent(
                error_message=str(e),
                error_type="StreamError",
                phase=state.current_phase or "unknown",
                recoverable=True
            )
    
    def _build_context_summary(self, context_data: Dict[str, Any]) -> str:
        """Build context loading summary for UI display."""
        components = context_data.get('components', {})

        parts = []
        if components.get('core_memory', 0) > 0:
            parts.append("your profile")
        if components.get('summary', 0) > 0:
            parts.append("conversation summary")
        if components.get('chat_history', 0) > 0:
            parts.append("recent messages")

        if parts:
            return f"Loaded {', '.join(parts)} for context"
        return "Ready to respond"

    async def _save_user_message(
        self,
        flow_id: str,
        session_id: str,
        user_id: int,
        query: str
    ) -> Optional[str]:
        """Save user message to database."""
        if not self.chat_repo:
            return None

        try:
            question_id = self.chat_repo.save_user_question(
                session_id=session_id,
                created_at=datetime.now(),
                created_by=user_id,
                content=query
            )
            self._current_question_id = question_id
            self.logger.debug(f"[{flow_id}] User message saved | id={question_id}")
            return question_id
        except Exception as e:
            self.logger.error(f"[{flow_id}] Failed to save user message: {e}")
            return None

    async def _save_assistant_response(
        self,
        flow_id: str,
        session_id: str,
        response: str,
        state: StreamState
    ) -> Optional[str]:
        """Save assistant response to database."""
        if not self.chat_repo or not self._current_question_id:
            return None

        try:
            response_time = state.get_elapsed_ms() / 1000.0
            message_id = self.chat_repo.save_assistant_response(
                session_id=session_id,
                created_at=datetime.now(),
                question_id=self._current_question_id,
                content=response,
                response_time=response_time
            )
            self.logger.debug(
                f"[{flow_id}] Response saved | id={message_id} | "
                f"{len(response)} chars | {response_time:.2f}s"
            )
            return message_id
        except Exception as e:
            self.logger.error(f"[{flow_id}] Failed to save response: {e}")
            return None

    async def _phase1_load_context_stream(
        self,
        flow_id: str,
        user_id: int,
        session_id: str,
        state: StreamState,
        enable_events: bool = True,
        enable_compaction: bool = True,
        wm_context: str = "",
        parent_node_id: str = None
    ) -> Dict[str, Any]:
        """
        Load context with progress events and optional compaction.

        Components: Core Memory, Summary, Chat History, Compaction (if needed)
        """
        events = []
        core_memory_str = ""
        summary = ""
        recent_chat = []
        usage_percent = 0.0
        total_tokens = 0
        was_compacted = False

        # 1.1 Load Core Memory
        self.logger.debug(f"[{flow_id}] Loading core memory...")
        
        if enable_events:
            events.append(ContextLoadingEvent(
                component="core_memory",
                status="loading",
                display_name="Loading your profile...",
                node_id=parent_node_id
            ))
        
        core_memory = {}
        try:
            core_memory = await self.core_memory.load_core_memory(user_id)
            core_memory_str = self.core_memory.format_for_context(core_memory) if hasattr(self.core_memory, 'format_for_context') else str(core_memory)
            core_size = len(core_memory_str)
            self.logger.debug(f"[{flow_id}] Core memory loaded | {core_size} chars")
            
            if enable_events:
                events.append(ContextLoadingEvent(
                    component="core_memory",
                    status="loaded",
                    display_name="Your profile loaded" if core_size > 0 else "No profile data",
                    size_chars=core_size,
                    node_id=parent_node_id
                ))
        except Exception as e:
            self.logger.warning(f"[{flow_id}] Core memory error: {e}")
            if enable_events:
                events.append(ContextLoadingEvent(
                    component="core_memory",
                    status="error",
                    display_name="Profile unavailable",
                    node_id=parent_node_id
                ))

        # 1.2 Load Summary
        self.logger.debug(f"[{flow_id}] Loading summary...")
        
        if enable_events:
            events.append(ContextLoadingEvent(
                component="summary",
                status="loading",
                display_name="Loading conversation context...",
                node_id=parent_node_id
            ))
        
        try:
            summary = await self.summary_manager.get_active_summary(session_id) or ""
            if summary:
                self.logger.debug(f"[{flow_id}] Summary loaded | {len(summary)} chars")
            
            if enable_events:
                events.append(ContextLoadingEvent(
                    component="summary",
                    status="loaded",
                    display_name="Conversation context loaded" if summary else "New conversation",
                    size_chars=len(summary),
                    node_id=parent_node_id
                ))
        except Exception as e:
            self.logger.warning(f"[{flow_id}] Summary error: {e}")
            if enable_events:
                events.append(ContextLoadingEvent(
                    component="summary",
                    status="error",
                    display_name="Context unavailable",
                    node_id=parent_node_id
                ))

        # 1.3 Load Chat History
        self.logger.debug(f"[{flow_id}] Loading chat history...")
        
        if enable_events:
            events.append(ContextLoadingEvent(
                component="chat_history",
                status="loading",
                display_name="Loading recent messages...",
                node_id=parent_node_id
            ))
        
        try:
            recent_chat = await self.session_repo.get_session_messages(
                session_id=session_id,
                limit=10
            )
            
            msg_count = len(recent_chat)
            chat_size = sum(len(str(m.get('content', ''))) for m in recent_chat)
            self.logger.debug(f"[{flow_id}] Chat history loaded | {msg_count} messages")
            
            if enable_events:
                display = f"Loaded {msg_count} recent messages" if msg_count > 0 else "Starting fresh conversation"
                events.append(ContextLoadingEvent(
                    component="chat_history",
                    status="loaded",
                    display_name=display,
                    size_chars=chat_size,
                    node_id=parent_node_id
                ))
        except Exception as e:
            self.logger.warning(f"[{flow_id}] Chat history error: {e}")
            if enable_events:
                events.append(ContextLoadingEvent(
                    component="chat_history",
                    status="error",
                    display_name="History unavailable",
                    node_id=parent_node_id
                ))

        # 1.4 Check context usage and auto-compact if needed
        if enable_compaction and recent_chat:
            try:
                if not hasattr(self, 'context_manager') or self.context_manager is None:
                    self.logger.warning(f"[{flow_id}] Context manager not initialized, skipping compaction")
                else:
                    # Combine all context for token counting
                    additional_context = core_memory_str + summary + wm_context
                    
                    # Check threshold
                    needs_compaction, stats = self.context_manager.compressor.should_compact(
                        messages=recent_chat,
                        system_prompt="",
                        additional_context=additional_context
                    )
                    
                    usage_percent = stats.usage_percent
                    total_tokens = stats.total_tokens

                    self.logger.debug(
                        f"[{flow_id}] Context usage: {usage_percent:.1f}% ({total_tokens:,} tokens)"
                    )

                    if needs_compaction:
                        self.logger.info(f"[{flow_id}] Auto-compacting context (>= 90%)...")
                        
                        if enable_events:
                            events.append(ContextLoadingEvent(
                                component="compaction",
                                status="compacting",
                                display_name=f"Optimizing context ({usage_percent:.0f}%)...",
                                node_id=parent_node_id
                            ))
                        
                        # Extract symbols to preserve
                        symbols = self.context_manager.compressor.extract_symbols_from_messages(
                            recent_chat
                        )
                        
                        # Compact
                        result = await self.context_manager.compact_now(
                            messages=recent_chat,
                            preserve_keywords=symbols,
                            strategy="smart_summary"
                        )
                        
                        if result.success:
                            recent_chat = result.preserved_messages or recent_chat
                            was_compacted = True
                            self.logger.info(
                                f"[{flow_id}] Compaction complete | saved {result.tokens_saved:,} tokens"
                            )

                            if enable_events:
                                events.append(ContextLoadingEvent(
                                    component="compaction",
                                    status="completed",
                                    display_name=f"Saved {result.tokens_saved:,} tokens",
                                    size_chars=result.tokens_saved,
                                    node_id=parent_node_id
                                ))
                        else:
                            self.logger.warning(f"[{flow_id}] Compaction failed: {result.error}")

            except Exception as e:
                self.logger.error(f"[{flow_id}] Compaction error: {e}")

        return {
            'events': events,
            'core_memory': core_memory,
            'core_memory_str': core_memory_str,
            'summary': summary,
            'recent_chat': recent_chat,
            'wm_context': wm_context,
            
            # Token stats
            'total_tokens': total_tokens,
            'usage_percent': usage_percent,
            
            # Compaction info
            'was_compacted': was_compacted,
            
            # Components summary (for ContextLoadedEvent)
            'components': {
                'core_memory': len(core_memory_str) > 0,
                'summary': len(summary) > 0,
                'history': len(recent_chat) > 0,
                'working_memory': len(wm_context) > 0
            }
        }

    def _extract_symbols_from_context(
        self,
        recent_chat: List[Dict],
        core_memory: Dict
    ) -> List[str]:
        """Extract stock/crypto symbols from context for compaction preservation."""
        import re
        
        symbols = set()
        
        # Pattern for stock/crypto symbols (1-5 uppercase letters)
        symbol_pattern = r'\\b[A-Z]{1,5}\\b'
        # symbol_pattern = r'\b[A-Z][A-Z0-9]{0,14}\b'
        
        # Common words to exclude
        exclude = {
            'I', 'A', 'THE', 'AND', 'OR', 'NOT', 'IS', 'ARE', 'WAS', 'BE',
            'TO', 'OF', 'IN', 'FOR', 'ON', 'WITH', 'AS', 'AT', 'BY', 'AN',
            'IT', 'IF', 'SO', 'UP', 'DO', 'NO', 'HE', 'WE', 'MY', 'OK',
            'API', 'LLM', 'AI', 'ML', 'USD', 'EUR', 'JPY', 'ETF', 'IPO',
            'CEO', 'CFO', 'COO', 'CTO', 'FAQ', 'ROI', 'YTD', 'QTD', 'MTD'
        }
        
        # Search in recent chat
        for msg in recent_chat:
            content = msg.get("content", "")
            matches = re.findall(symbol_pattern, content)
            for match in matches:
                if match not in exclude and len(match) >= 2:
                    symbols.add(match)
        
        # Search in core memory (user preferences may contain watchlist)
        if core_memory:
            core_str = str(core_memory)
            matches = re.findall(symbol_pattern, core_str)
            for match in matches:
                if match not in exclude and len(match) >= 2:
                    symbols.add(match)
        
        return list(symbols)[:20]  # Limit to 20 symbols

    def _extract_symbols_heuristic(self, recent_chat: List[Dict]) -> List[str]:
        """Extract symbols using heuristics (no LLM)"""
        import re
        
        symbols = set()
        symbol_pattern = r'\b[A-Z][A-Z0-9]{0,14}\b'
        
        exclude = {
            'I', 'A', 'THE', 'AND', 'OR', 'NOT', 'IS', 'ARE', 'TO', 'OF',
            'IN', 'FOR', 'ON', 'AT', 'BY', 'WITH', 'FROM', 'UP', 'DOWN',
            'BE', 'IT', 'AS', 'AN', 'IF', 'SO', 'NO', 'YES', 'OK',
            'API', 'URL', 'HTTP', 'HTTPS', 'JSON', 'XML', 'HTML', 'CSS',
            'USD', 'EUR', 'VND', 'JPY', 'GBP',
        }
        
        for msg in recent_chat[-5:]:
            content = msg.get('content', '')
            matches = re.findall(symbol_pattern, content)
            for match in matches:
                if match not in exclude and len(match) >= 2:
                    symbols.add(match)
        
        return list(symbols)[:10]

    async def _phase2_planning_stream(
        self,
        flow_id: str,
        query: str,
        context_data: Dict[str, Any],
        state: StreamState,
        enable_thinking: bool = True,
        wm_context: str = "",
        model_name: str = "gpt-4.1-nano",
        provider_type: str = "openai",
        parent_node_id: str = None
    ) -> AsyncGenerator[Any, None]:
        """Stream planning process with progress events."""
        # Format recent chat
        formatted_recent = [
            {
                'role': msg.get('role', 'user'),
                'content': msg.get('content', ''),
                'created_at': msg.get('created_at', '')
            }
            for msg in context_data['recent_chat']
        ]
        
        # Emit thinking start
        if enable_thinking and self.config.enable_thinking_display:
            yield ThinkingStartEvent(
                phase="planning",
                message="Analyzing your request...",
                estimated_steps=3,
                node_id=parent_node_id
            )
        
        # Emit LLM thought about query understanding
        if self.config.enable_llm_decision_events:
            yield LLMThoughtEvent(
                thought=f"Analyzing query: \"{query[:100]}...\"",
                thought_type="analysis",
                context="query_understanding",
                parent_id=parent_node_id
            )
        
        # Stage progress events (static but informative)
        yield PlanningProgressEvent(
            stage=1,
            total_stages=3,
            stage_name="Understanding",
            details="Analyzing intent and identifying relevant data",
            parent_id=parent_node_id
        )
        
        yield PlanningProgressEvent(
            stage=2,
            total_stages=3,
            stage_name="Classification", 
            details="Determining query type and required tools",
            parent_id=parent_node_id
        )
        
        # Execute actual planning (ONLY 1 LLM call here)
        task_plan = await self.planning_agent.think_and_plan(
            query=query,
            recent_chat=formatted_recent,
            core_memory=context_data['core_memory'],
            summary=context_data['summary'],
            working_memory_context=wm_context
        )
        
        language = 'auto'
        classification_reasoning = ""
        
        if task_plan:
            language = task_plan.response_language
            classification_reasoning = task_plan.reasoning or ""
        
        yield PlanningProgressEvent(
            stage=3,
            total_stages=3,
            stage_name="Plan Creation",
            details="Building execution strategy",
            parent_id=parent_node_id
        )
        
        # Emit LLM decision about tool selection
        if self.config.enable_llm_decision_events and task_plan:
            tools_selected = []
            for task in task_plan.tasks:
                for tool in task.tools_needed:
                    tools_selected.append(tool.tool_name)
            
            yield LLMDecisionEvent(
                decision=f"Selected {len(tools_selected)} tools",
                decision_type="tool_selection",
                options_considered=list(set(tools_selected)),
                reasoning=classification_reasoning[:200] if classification_reasoning else None,
                confidence=0.9,
                parent_id=parent_node_id
            )
        
        # Save to working memory
        if task_plan and self._wm_integration:
            self._wm_integration.save_classification(
                query_type=task_plan.query_intent,
                categories=[],
                symbols=task_plan.symbols or [],
                language=language,
                reasoning=classification_reasoning
            )
            if task_plan.tasks:
                self._wm_integration.save_plan(task_plan)
        
        self._current_task_plan = task_plan
        
        # Emit planning complete
        yield PlanningCompleteEvent(
            task_count=len(task_plan.tasks) if task_plan else 0,
            strategy=task_plan.strategy if task_plan else 'direct_answer',
            symbols=task_plan.symbols if task_plan else [],
            query_intent=task_plan.query_intent if task_plan else 'conversational',
            node_id=parent_node_id
        )
        
        # Emit thinking end
        if enable_thinking:
            task_count = len(task_plan.tasks) if task_plan else 0
            intent = task_plan.query_intent if task_plan else 'N/A'
            
            if task_count > 0:
                summary = f"Identified {task_count} analysis tasks for {intent} query"
            else:
                summary = f"Conversational response - no data retrieval needed"
            
            yield ThinkingEndEvent(
                phase="planning",
                summary=summary,
                node_id=parent_node_id
            )
        
        # Return plan data
        yield {
            'task_plan': task_plan,
            'language': language,
            'symbols': task_plan.symbols if task_plan and hasattr(task_plan, 'symbols') else [],
            'query_intent': task_plan.query_intent if task_plan else '',
            'strategy': task_plan.strategy if task_plan else '',
            'reasoning': classification_reasoning
        }

    async def _phase4_tool_execution_stream_fixed(
        self,
        flow_id: str,
        query: str,
        plan_data: Dict[str, Any],
        context_data: Dict[str, Any],
        provider_type: str,
        model_name: str,
        user_id: int,
        session_id: str,
        state: StreamState,
        parent_node_id: str = None
    ) -> AsyncGenerator[Any, None]:
        """Execute tools via TaskExecutor with streaming events."""
        task_plan = plan_data.get('task_plan')
        
        if not task_plan or len(task_plan.tasks) == 0:
            yield {
                'all_tool_results': {},
                'tools_executed': [],
                'mode': None,
                'stats': {}
            }
            return

        self.logger.debug(f"[{flow_id}] Executing {len(task_plan.tasks)} tasks via TaskExecutor")

        chat_history = [
            [msg.get('content', ''), msg.get('role', 'user')]
            for msg in context_data['recent_chat'][-5:]
        ]

        execution_result = await self.task_executor.execute_task_plan(
            plan=task_plan,
            query=query,
            chat_history=chat_history,
            system_language=plan_data.get('language', 'vi'),
            provider_type=provider_type,
            model_name=model_name,
            flow_id=flow_id,
            user_id=user_id,
            session_id=session_id
        )
        
        # Extract results
        task_results = execution_result.get('task_results', [])
        stats = execution_result.get('stats', {})
        accumulated_context = execution_result.get('accumulated_context', {})

        self.logger.debug(
            f"[{flow_id}] TaskExecutor completed | "
            f"{stats.get('completed_tasks', 0)}/{stats.get('total_tasks', 0)} tasks | "
            f"retries={stats.get('total_retries', 0)}"
        )

        all_tool_results = {}
        tools_executed = []
        total_tools = len(task_results)
        
        for idx, task_result in enumerate(task_results):
            task_id = getattr(task_result, 'task_id', idx + 1)
            is_success = getattr(task_result, 'success', False)
            task_data = getattr(task_result, 'data', {})
            task_tools = getattr(task_result, 'tools_executed', [])
            execution_time_s = getattr(task_result, 'execution_time', 0)
            error_msg = getattr(task_result, 'error', None)
            
            # Get tool name from task_tools or task_data
            tool_name = task_tools[0] if task_tools else f"task_{task_id}"
            
            # Generate call_id for correlation
            call_id = generate_call_id(tool_name)
            
            # Start tool node in tree
            tool_node_id = None
            if self._agent_tree:
                tool_node_id = self._agent_tree.start_node(
                    node_type=NodeType.TOOL,
                    name=tool_name,
                    parent_id=parent_node_id,
                    metadata={"call_id": call_id, "task_id": task_id}
                )
            
            # Track in state
            state.start_tool(call_id, tool_name, tool_node_id)
            
            # Emit ToolStartEvent
            yield ToolStartEvent(
                task_id=task_id,
                tool_name=tool_name,
                params={},  # Params already executed
                call_id=call_id,
                node_id=tool_node_id,
                parent_id=parent_node_id
            )
            
            # Emit LLM action event
            if self.config.enable_llm_decision_events:
                yield LLMActionEvent(
                    action="call_tool",
                    action_target=tool_name,
                    parameters={},
                    reasoning=f"Task {task_id}: {tool_name}",
                    node_id=tool_node_id
                )
            
            # Thinking delta for progress
            if self.config.enable_thinking_display:
                yield ThinkingDeltaEvent(
                    phase="execution",
                    thought=f"Completed {tool_name}",
                    progress=(idx + 1) / total_tools,
                    step_number=idx + 1,
                    total_steps=total_tools,
                    parent_id=parent_node_id
                )
            
            # End tool tracking
            state.end_tool(call_id, is_success)
            
            # End tool node in tree
            if self._agent_tree and tool_node_id:
                self._agent_tree.end_node(
                    tool_node_id,
                    success=is_success,
                    error=error_msg,
                    metadata={"duration_ms": int(execution_time_s * 1000)}
                )
            
            # Get preview for event
            preview = self._get_result_preview(task_data) if task_data else None
            
            # Emit ToolCompleteEvent
            yield ToolCompleteEvent(
                task_id=task_id,
                tool_name=tool_name,
                success=is_success,
                call_id=call_id,
                preview=preview,
                duration_ms=int(execution_time_s * 1000),
                error=error_msg,
                node_id=tool_node_id,
                parent_id=parent_node_id
            )
            
            # Collect results
            if is_success and task_data:
                all_tool_results[f"task_{task_id}"] = task_data
                tools_executed.extend(task_tools)
                
                # Save to working memory
                if self._wm_integration:
                    for tool in task_tools:
                        tool_output = task_data.get(tool, task_data)
                        self._wm_integration.save_tool_result(
                            tool_name=tool,
                            result=tool_output,
                            task_id=str(task_id),
                            execution_time_ms=int(execution_time_s * 1000),
                            status="success" if is_success else "failed"
                        )
        
        # Store in instance for context assembly
        self._execution_results = all_tool_results
        
        # Return execution data
        yield {
            'all_tool_results': all_tool_results,
            'tools_executed': list(set(tools_executed)),
            'mode': 'task_based',
            'stats': stats,
            'accumulated_context': accumulated_context
        }
        
    def _get_result_preview(self, result: Dict[str, Any]) -> str:
        """Get a short preview of tool result for UI display."""
        if not isinstance(result, dict):
            return str(result)[:100]
        
        # Check for formatted_context
        if 'formatted_context' in result:
            return result['formatted_context'][:self.config.max_tool_preview_length]
        
        # Check for data
        if 'data' in result:
            import json
            try:
                return json.dumps(result['data'], ensure_ascii=False)[:self.config.max_tool_preview_length]
            except:
                return str(result['data'])[:self.config.max_tool_preview_length]
        
        return str(result)[:self.config.max_tool_preview_length]

    async def _phase5_context_assembly(
        self,
        flow_id: str,
        query: str,
        chart_displayed: bool,
        plan_data: Dict[str, Any],
        execution_data: Dict[str, Any],
        model_name: str,
        enable_thinking: bool,
        context_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Assemble context for LLM response generation."""
        all_tool_results = execution_data.get('all_tool_results', {})

        tool_outputs_dict = self._convert_to_tool_outputs(
            list(all_tool_results.values()) if isinstance(all_tool_results, dict) else all_tool_results
        )
        self.logger.debug(f"[{flow_id}] Tool outputs assembled | {len(tool_outputs_dict)} tools")
        
        language = plan_data.get('language', 'vi')
        plan_symbols = plan_data.get('symbols', [])
        query_intent = plan_data.get('query_intent', '')
        
        # Build system prompt
        from src.helpers.system_prompts import get_system_message_general_chat
        
        system_prompt = get_system_message_general_chat(
            enable_thinking=enable_thinking,
            model_name=model_name,
            detected_language=language,
            chart_displayed=chart_displayed
        )

        summary = context_data.get('summary', '') if context_data else ''
        if summary:
            system_prompt += f"""

<conversation_summary>
## Previous Conversation Context
The following is a summary of earlier conversation turns that may provide relevant context:

{summary}

Note: Use this context to maintain conversation continuity. Current query data below takes precedence.
</conversation_summary>
"""
        self.logger.debug(f"[{flow_id}] Summary included in context | {len(summary)} chars")

        # Format tool context
        tool_context = ""
        if all_tool_results:
            tool_context = self._format_tool_context_for_llm(tool_outputs_dict, language)
            if tool_context:
                system_prompt += f"""

<current_data>
## Real-Time Data for Current Query
The following data was retrieved specifically for the current user query:

{tool_context}
</current_data>
"""
        
        # Extract symbols from results
        symbols = self._extract_symbols_from_results(all_tool_results)
        active_symbols = plan_symbols if plan_symbols else symbols
        
        # Add symbol awareness
        if active_symbols:
            symbol_awareness = f"""
{'='*70}
⚠️ CURRENT QUERY SYMBOLS: {', '.join(active_symbols)}
Query Intent: {query_intent}
IMPORTANT: Analyze ONLY these symbol(s). Do NOT confuse with history!
{'='*70}
"""
            system_prompt += symbol_awareness
        
        return {
            'system_prompt': system_prompt,
            'tool_context': tool_context,
            'symbols': active_symbols
        }
    

    def _convert_to_tool_outputs(self, task_results: List[Any]) -> Dict[str, Any]:
        """Convert TaskExecutionResult list to tool outputs dict"""
        tool_outputs = {}
        
        if not task_results:
            return tool_outputs
        
        for idx, result in enumerate(task_results):
            source_data = {}
            
            if isinstance(result, dict) and 'data' in result:
                source_data = result['data']
            elif hasattr(result, 'data') and result.data:
                source_data = result.data
            elif isinstance(result, dict):
                is_tool_dict = any(
                    isinstance(v, dict) and (
                        'status' in v or 
                        'formatted_context' in v or 
                        'data' in v or
                        'symbol' in v or
                        'price' in v
                    )
                    for v in result.values()
                )
                if is_tool_dict:
                    source_data = result
            
            if not source_data:
                continue
            
            for tool_name, tool_data in source_data.items():
                if not isinstance(tool_data, dict):
                    continue
                
                if tool_name in tool_outputs:
                    existing = tool_outputs[tool_name]
                    new_context = tool_data.get('formatted_context', '')
                    if new_context:
                        existing['formatted_context'] = (
                            existing.get('formatted_context', '') + "\n\n" + new_context
                        )
                    new_symbols = tool_data.get('symbols', [])
                    if new_symbols:
                        existing_symbols = existing.get('symbols', [])
                        existing['symbols'] = list(set(existing_symbols + new_symbols))
                else:
                    tool_outputs[tool_name] = tool_data
        
        return tool_outputs

    def _format_tool_context_for_llm(self, tool_outputs: Dict[str, Any], language: str = 'en') -> str:
        sections = []
        
        for tool_name, tool_output in tool_outputs.items():
            if not isinstance(tool_output, dict):
                continue
            
            status = tool_output.get('status', 'success')
            if status in ['error', 'failed']:
                continue
            
            # Priority 1: formatted_context
            if 'formatted_context' in tool_output and tool_output['formatted_context']:
                sections.append(tool_output['formatted_context'])
            
            # Priority 2: nested data
            elif 'data' in tool_output and isinstance(tool_output['data'], dict):
                import json
                formatted = json.dumps(tool_output['data'], indent=2, ensure_ascii=False, default=str)
                sections.append(f"=== {tool_name} ===\n{formatted}")
            
            # Priority 3: raw data (CRITICAL FIX!)
            elif any(key in tool_output for key in ['symbol', 'price', 'name', 'value', 'items']):
                import json
                clean_data = {k: v for k, v in tool_output.items() 
                            if k not in ['status', 'formatted_context', 'error', 'execution_time_ms', 'metadata']}
                if clean_data:
                    formatted = json.dumps(clean_data, indent=2, ensure_ascii=False, default=str)
                    sections.append(f"=== {tool_name} ===\n{formatted}")
        
        return "\n\n".join(sections)

    def _extract_symbols_from_results(self, tool_outputs: Dict[str, Any]) -> List[str]:
        """Extract unique symbols from tool outputs."""
        all_symbols = set()
        
        for tool_output in tool_outputs.values():
            if not isinstance(tool_output, dict):
                continue
            
            symbols = tool_output.get('symbols', [])
            if isinstance(symbols, list):
                all_symbols.update(symbols)
            elif isinstance(symbols, str):
                all_symbols.add(symbols)
            
            data = tool_output.get('data', {})
            if isinstance(data, dict) and 'symbol' in data:
                all_symbols.add(data['symbol'])
        
        return sorted(list(all_symbols))

    async def _phase6_generate_response_stream(
        self,
        flow_id: str,
        query: str,
        assembled_context: Dict[str, Any],
        execution_data: Dict[str, Any],
        user_id: int,
        session_id: str,
        model_name: str,
        provider_type: str,
        enable_thinking: bool,
        state: StreamState,
        parent_node_id: str = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream LLM response generation with text delta events."""
        system_prompt = assembled_context['system_prompt']
        has_tool_results = bool(execution_data.get('all_tool_results'))
        
        max_history = 2 if has_tool_results else 10
        
        try:
            from src.helpers.context_assembler import ContextAssembler
            context_assembler = ContextAssembler()
            
            messages, llm_metadata = await context_assembler.prepare_messages_for_llm(
                user_id=user_id,
                session_id=session_id,
                current_query=query,
                system_prompt=system_prompt,
                enable_thinking=enable_thinking,
                model_name=model_name,
                max_history_messages=max_history
            )
        except Exception as e:
            self.logger.warning(f"[{flow_id}] Context assembler error: {e}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        
        from src.providers.provider_factory import ModelProviderFactory
        api_key = ModelProviderFactory._get_api_key(provider_type)

        complete_response = ""
        char_buffer = ""
        start_time = time.time()

        # Prepare streaming parameters with max_tokens
        stream_params = {
            "model_name": model_name,
            "messages": messages,
            "provider_type": provider_type,
            "api_key": api_key,
            "clean_thinking": True,
            "max_tokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
        }

        self.logger.debug(
            f"[{flow_id}] LLM stream params | model={model_name} | "
            f"max_tokens={self.config.max_output_tokens} | msgs={len(messages)}"
        )

        async for chunk in self.llm_provider.stream_response(**stream_params):
            if chunk:
                char_buffer += chunk
                
                if len(char_buffer) >= self.config.text_chunk_min_size:
                    complete_response += char_buffer
                    yield TextDeltaEvent(
                        chunk=char_buffer,
                        accumulated_length=len(complete_response),
                        node_id=parent_node_id
                    )
                    char_buffer = ""
        
        if char_buffer:
            complete_response += char_buffer
            yield TextDeltaEvent(
                chunk=char_buffer,
                accumulated_length=len(complete_response),
                node_id=parent_node_id
            )
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        yield TextCompleteEvent(
            total_length=len(complete_response),
            duration_ms=duration_ms,
            node_id=parent_node_id
        )

    async def _background_memory_updates(
        self,
        flow_id: str,
        user_id: int,
        session_id: str,
        query: str,
        response: str,
        organization_id: Optional[int],
        provider_type: str,
        model_name: str,
        working_memory_context: Optional[str] = None
    ):
        """Run background memory updates (non-blocking)."""
        try:
            from src.agents.memory.memory_update_agent import get_memory_update_agent
            memory_update = get_memory_update_agent()

            update_result = await memory_update.analyze_for_updates(
                user_id=user_id,
                user_message=query,
                assistant_message=response,
                working_memory_context=working_memory_context,
                model_name=model_name,
                provider_type=provider_type
            )

            if update_result.get('updated'):
                self.logger.debug(f"[{flow_id}] BG: Core memory updated")

            summary_result = await self.summary_manager.check_and_create_summary(
                session_id=session_id,
                user_id=user_id,
                organization_id=organization_id,
                model_name=model_name,
                provider_type=provider_type
            )

            if summary_result.get('created'):
                self.logger.debug(f"[{flow_id}] BG: Summary v{summary_result.get('version')} created")

        except Exception as e:
            self.logger.error(f"[{flow_id}] BG: Memory update error: {e}")


async def stream_chat_sse(
    handler: StreamingChatHandler,
    query: str,
    session_id: str,
    user_id: int,
    cancellation_token: CancellationToken = None,
    **kwargs
) -> AsyncGenerator[str, None]:
    """Convenience function to stream chat as SSE-formatted strings."""
    async for event in handler.handle_chat_stream(
        query=query,
        session_id=session_id,
        user_id=user_id,
        cancellation_token=cancellation_token,
        **kwargs
    ):
        yield event.to_sse()
    
    # Final done marker
    yield "data: [DONE]\n\n"