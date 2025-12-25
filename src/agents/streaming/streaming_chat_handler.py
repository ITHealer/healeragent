# import asyncio
# import time
# from datetime import datetime
# from typing import AsyncGenerator, Dict, List, Any, Optional
# from dataclasses import dataclass, field

# from src.utils.logger.custom_logging import LoggerMixin
# from src.database.repository.chat import ChatRepository
# from src.agents.streaming.stream_events import (
#     StreamEvent,
#     StreamEventType,
#     StreamState,
#     StartEvent,
#     ThinkingStartEvent,
#     ThinkingDeltaEvent,
#     ThinkingEndEvent,
#     PlanningProgressEvent,
#     PlanningCompleteEvent,
#     ToolStartEvent,
#     ToolCompleteEvent,
#     ContextLoadingEvent,
#     ContextLoadedEvent,
#     TextDeltaEvent,
#     TextCompleteEvent,
#     MemoryUpdateEvent,
#     DoneEvent,
#     ErrorEvent,
#     HeartbeatEvent,
#     # LLM Decision Events
#     LLMThoughtEvent,
#     LLMDecisionEvent,
#     LLMActionEvent,
#     # Agent Tree Events
#     AgentNodeEvent,
#     # Utilities
#     generate_call_id,
#     generate_node_id,
# )

# from src.agents.streaming.agent_tree import (
#     AgentTree,
#     TreeNodeContext,
#     NodeType,
#     NodeStatus,
#     create_tree_for_request,
# )

# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# @dataclass
# class StreamingConfig:
#     """Configuration for streaming behavior"""
    
#     # Enable/disable streaming features
#     enable_thinking_display: bool = True
#     enable_tool_progress: bool = True
#     enable_context_events: bool = True
#     enable_memory_events: bool = False

#     # Thinking cutoff
#     max_thinking_length: int = 2000

#     # LLM decision events
#     enable_llm_decision_events: bool = True

#     # Agent tree tracking
#     enable_agent_tree: bool = True

#     # Content limits
#     max_tool_preview_length: int = 500
    
#     # Buffer settings
#     text_chunk_min_size: int = 5  # Minimum chars before emitting text chunk
    
#     # Persistence
#     save_messages: bool = True
    
#     # Context compaction (matching ChatHandler)
#     enable_compaction: bool = True
#     compaction_threshold: int = 100000  # tokens

#     # SSE Cancellation
#     cancellation_check_interval: float = 0.1  # seconds
#     enable_heartbeat: bool = True
#     heartbeat_interval: int = 30  # seconds


# # ============================================================================
# # CANCELLATION TOKEN
# # ============================================================================

# class CancellationToken:
#     """
#     Token to track cancellation state
    
#     Allows checking if client has disconnected
#     """
    
#     def __init__(self):
#         self._cancelled = False
#         self._cancel_event = asyncio.Event()
    
#     def cancel(self):
#         """Mark as cancelled"""
#         self._cancelled = True
#         self._cancel_event.set()
    
#     @property
#     def is_cancelled(self) -> bool:
#         """Check if cancelled"""
#         return self._cancelled
    
#     async def wait_for_cancel(self, timeout: float = None) -> bool:
#         """Wait for cancellation with optional timeout"""
#         try:
#             await asyncio.wait_for(self._cancel_event.wait(), timeout=timeout)
#             return True
#         except asyncio.TimeoutError:
#             return False

# # ============================================================================
# # STREAMING CHAT HANDLER
# # ============================================================================

# class StreamingChatHandler(LoggerMixin):
#     """
#     Streaming Chat Handler with Full 7-Phase Pipeline
    
#     Provides streaming responses while maintaining full feature parity
#     with ChatHandler, including:
#     - Context compaction
#     - Working memory
#     - Think Tool
#     - All 31 atomic tools
#     """
    
#     def __init__(
#         self,
#         planning_agent,
#         task_executor,
#         core_memory,
#         summary_manager,
#         session_repo,
#         llm_provider,
#         tool_execution_service=None,
#         chat_repo=None,
#         config: StreamingConfig = None
#     ):
#         """
#         Initialize StreamingChatHandler
        
#         Args:
#             planning_agent: Planning agent for task planning
#             task_executor: Task executor for tool execution
#             core_memory: Core memory manager
#             summary_manager: Recursive summary manager
#             session_repo: Session repository
#             llm_provider: LLM provider for response generation
#             tool_execution_service: Tool execution service
#             chat_repo: Chat repository for message persistence
#             config: Streaming configuration
#         """
#         super().__init__()
        
#         self.planning_agent = planning_agent
#         self.task_executor = task_executor
#         self.core_memory = core_memory
#         self.summary_manager = summary_manager
#         self.session_repo = session_repo
#         self.llm_provider = llm_provider
#         self.tool_execution_service = tool_execution_service
#         self.config = config or StreamingConfig()
#         self.chat_repo = chat_repo or ChatRepository()
        
#         # Instance state (reset per request)
#         self._current_task_plan = None
#         self._execution_results = {}
#         self._accumulated_response = ""
#         self._current_question_id = None
        
#         # Cancellation token
#         self._cancellation_token: Optional[CancellationToken] = None

#         # Agent tree
#         self._agent_tree: Optional[AgentTree] = None

#         # Context management service (lazy init)
#         self._context_manager = None
        
#         # Think tool service (lazy init per request)
#         self._think_service = None
        
#         # Working memory integration (lazy init per request)
#         self._wm_integration = None
        
#         self.logger.info("[STREAMING] StreamingChatHandler initialized")
    
#     def _get_context_manager(self):
#         """Lazy initialization of context manager"""
#         if self._context_manager is None:
#             try:
#                 from src.services.context_management_service import ContextManagementService
#                 self._context_manager = ContextManagementService(
#                     enable_compaction=self.config.enable_compaction,
#                     compaction_threshold=self.config.compaction_threshold,
#                     compaction_strategy="smart_summary"
#                 )
#             except ImportError:
#                 self.logger.warning("[STREAMING] ContextManagementService not available")
#         return self._context_manager
    

#     # ========================================================================
#     # SSE CANCELLATION CHECK
#     # ========================================================================
    
#     async def _check_cancellation(self) -> bool:
#         """
#         Check if request has been cancelled
        
#         Returns True if cancelled, False otherwise
#         """
#         if self._cancellation_token and self._cancellation_token.is_cancelled:
#             self.logger.info("[STREAMING] Request cancelled by client")
#             return True
#         return False
    
#     async def _cleanup_on_cancel(self, flow_id: str):
#         """
#         Cleanup resources when cancelled
#         """
#         self.logger.info(f"[{flow_id}] Cleaning up after cancellation...")
        
#         # Save partial state to working memory
#         if self._wm_integration:
#             self._wm_integration.save_error(
#                 error_type="client_disconnected",
#                 message="Client disconnected before completion",
#                 recoverable=True
#             )
#             self._wm_integration.complete_request(clear_task_data=False)
        
#         # End any open agent tree nodes
#         if self._agent_tree:
#             for node_id, node in self._agent_tree.nodes.items():
#                 if node.status in [NodeStatus.STARTED, NodeStatus.RUNNING]:
#                     self._agent_tree.end_node(
#                         node_id=node_id,
#                         success=False,
#                         error="cancelled"
#                     )
        
#         self.logger.info(f"[{flow_id}] Cleanup complete")

#     # ========================================================================
#     # MAIN ENTRY POINT
#     # ========================================================================
    
#     async def handle_chat_stream(
#         self,
#         query: str,
#         session_id: str,
#         user_id: int,
#         model_name: str = "gpt-4.1-nano",
#         provider_type: str = "openai",
#         organization_id: Optional[int] = None,
#         enable_thinking: bool = True,
#         # Additional parameters (matching ChatHandler)
#         chart_displayed: bool = False,
#         enable_compaction: bool = True,
#         enable_think_tool: bool = False,
#         cancellation_token: CancellationToken = None,
#         **kwargs
#     ) -> AsyncGenerator[StreamEvent, None]:
#         """
#         Main streaming chat handler - yields StreamEvent objects
        
#         Args:
#             query: User query
#             session_id: Session ID
#             user_id: User ID
#             model_name: LLM model name
#             provider_type: LLM provider
#             organization_id: Organization ID
#             enable_thinking: Enable thinking display
#             chart_displayed: Whether chart is displayed (affects response)
#             enable_compaction: Enable context compaction
#             enable_think_tool: Enable Think Tool for complex reasoning
#             **kwargs: Additional parameters
            
#         Yields:
#             StreamEvent objects for SSE streaming
#         """
#         # Generate flow ID for tracking
#         flow_id = f"STREAM-{datetime.now().strftime('%H%M%S')}-{session_id[:8]}"
#         flow_start = time.time()
        
#         # Reset instance state
#         self._current_task_plan = None
#         self._execution_results = {}
#         self._accumulated_response = ""
#         self._current_question_id = None
        
#         # Store cancellation token
#         self._cancellation_token = cancellation_token

#         # Create agent tree
#         if self.config.enable_agent_tree:
#             self._agent_tree = create_tree_for_request(flow_id)
        
#         # Initialize stream state
#         state = StreamState(
#             session_id=session_id,
#             flow_id=flow_id
#         )
        
#         self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════")
#         self.logger.info(f"[{flow_id}] [STREAMING] Starting progressive stream")
#         self.logger.info(f"[{flow_id}]   Query: {query[:100]}...")
#         self.logger.info(f"[{flow_id}]   Session: {session_id[:16]}...")
#         self.logger.info(f"[{flow_id}]   Model: {model_name}")
#         self.logger.info(f"[{flow_id}]   Features: cancellation={cancellation_token is not None}, "
#                         f"tree={self.config.enable_agent_tree}, "
#                         f"llm_events={self.config.enable_llm_decision_events}")
#         self.logger.info(f"[{flow_id}]   Compaction: {'ON' if enable_compaction else 'OFF'}")
#         self.logger.info(f"[{flow_id}]   Think Tool: {'ON' if enable_think_tool else 'OFF'}")
#         self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════")
        
#         # Setup Working Memory for this request
#         wm_context = ""
#         try:
#             from src.agents.memory.working_memory_integration import (
#                 setup_working_memory_for_request
#             )
#             self._wm_integration = setup_working_memory_for_request(
#                 session_id=session_id,
#                 user_id=str(user_id),
#                 flow_id=flow_id
#             )
#             wm_context = self._wm_integration.get_context_for_planning(max_tokens=1000)
#         except ImportError:
#             self.logger.warning(f"[{flow_id}] Working Memory not available")
#             self._wm_integration = None
        
#         # Setup Think Tool if enabled
#         if enable_think_tool:
#             try:
#                 from src.services.think_tool_service import ThinkToolService
#                 self._think_service = ThinkToolService(
#                     enabled=True,
#                     model_name=model_name
#                 )
#             except ImportError:
#                 self.logger.warning(f"[{flow_id}] ThinkToolService not available")
#                 self._think_service = None
        
#         try:
#             # Check for early cancellation
#             if await self._check_cancellation():
#                 yield ErrorEvent(
#                     error_message="Request cancelled",
#                     error_type="Cancelled",
#                     phase="initialization",
#                     recoverable=True
#                 )
#                 return
            
#             # Save user message
#             if self.config.save_messages:
#                 await self._save_user_message(
#                     flow_id=flow_id,
#                     session_id=session_id,
#                     user_id=user_id,
#                     query=query
#                 )
            
#             # Get root node ID for tree tracking
#             root_node_id = self._agent_tree.root_id if self._agent_tree else None
            
#             # ================================================================
#             # PHASE 1: START
#             # ================================================================
#             yield StartEvent(
#                 session_id=session_id,
#                 flow_id=flow_id,
#                 model_name=model_name,
#                 node_id=root_node_id
#             )
            
#             # ================================================================
#             # PHASE 2: CONTEXT LOADING (with compaction support)
#             # ================================================================

#             # Check cancellation
#             if await self._check_cancellation():
#                 await self._cleanup_on_cancel(flow_id)
#                 return
            
#             # Start context phase in tree
#             context_node_id = None
#             if self._agent_tree:
#                 context_node_id = self._agent_tree.start_node(
#                     node_type=NodeType.PHASE,
#                     name="context_loading",
#                     parent_id=root_node_id
#                 )

#             state.start_phase("context_loading")
#             self.logger.info(f"[{flow_id}] [PHASE 1] Loading Context...")
            
#             if enable_thinking and self.config.enable_context_events:
#                 yield ThinkingStartEvent(
#                     phase="context",
#                     message="Loading conversation context...",
#                     estimated_steps=3,
#                     node_id=context_node_id,
#                     parent_id=root_node_id
#                 )
            
#             context_data = await self._phase1_load_context_stream(
#                 flow_id=flow_id,
#                 user_id=user_id,
#                 session_id=session_id,
#                 state=state,
#                 enable_events=self.config.enable_context_events,
#                 enable_compaction=enable_compaction,
#                 wm_context=wm_context,
#                 parent_node_id=context_node_id
#             )
            
#             # Emit context events
#             for event in context_data.get('events', []):
#                 yield event
            
#             if self.config.enable_context_events:
#                 yield ContextLoadedEvent(
#                     total_tokens=context_data.get('total_tokens'),
#                     context_usage_percent=context_data.get('usage_percent'),
#                     components=context_data.get('components'),
#                     node_id=context_node_id
#                 )
            
#             context_duration = state.end_phase()

#             # End context node in tree
#             if self._agent_tree and context_node_id:
#                 self._agent_tree.end_node(context_node_id, success=True)

#             if enable_thinking and self.config.enable_context_events:
#                 yield ThinkingEndEvent(
#                     phase="context",
#                     summary=self._build_context_summary(context_data),
#                     duration_ms=int(context_duration),
#                     node_id=context_node_id
#                 )

#             self.logger.info(f"[{flow_id}] [PHASE 1] Complete ({context_duration:.0f}ms)")
            
#             # ================================================================
#             # PHASE 3: PLANNING (with thinking events) real thinking streaming
#             # ================================================================

#             # Check cancellation
#             if await self._check_cancellation():
#                 await self._cleanup_on_cancel(flow_id)
#                 return
            
#             # Start planning phase in tree
#             planning_node_id = None
#             if self._agent_tree:
#                 planning_node_id = self._agent_tree.start_node(
#                     node_type=NodeType.PLANNING,
#                     name="planning",
#                     parent_id=root_node_id
#                 )
                        
#             state.start_phase("planning")
#             self.logger.info(f"[{flow_id}] [PHASE 2] Planning...")
            
#             plan_data = None
#             async for event in self._phase2_planning_stream(
#                 flow_id=flow_id,
#                 query=query,
#                 context_data=context_data,
#                 state=state,
#                 enable_thinking=enable_thinking,
#                 wm_context=wm_context,
#                 model_name=model_name,
#                 provider_type=provider_type,
#                 parent_node_id=planning_node_id
#             ):
#                 # Check cancellation during planning
#                 if await self._check_cancellation():
#                     await self._cleanup_on_cancel(flow_id)
#                     return
                
#                 if isinstance(event, dict) and 'task_plan' in event:
#                     plan_data = event
#                 else:
#                     yield event
            
#             planning_duration = state.end_phase()

#             # End planning node
#             if self._agent_tree and planning_node_id:
#                 self._agent_tree.end_node(
#                     planning_node_id,
#                     success=plan_data is not None,
#                     metadata={"task_count": len(plan_data['task_plan'].tasks) if plan_data and plan_data.get('task_plan') else 0}
#                 )

#             self.logger.info(f"[{flow_id}] [PHASE 2] Complete ({planning_duration:.0f}ms)")
            
#             # Check if no tools needed
#             if not plan_data or not plan_data.get('task_plan') or len(plan_data['task_plan'].tasks) == 0:
#                 self.logger.info(f"[{flow_id}] No tools needed - conversational response")
#                 execution_data = {'all_tool_results': {}, 'tools_executed': [], 'mode': None}
#                 execution_duration = 0
#             else:
#                 # ================================================================
#                 # PHASE 4: TOOL EXECUTION (with progress events)
#                 # ================================================================

#                 # Check cancellation
#                 if await self._check_cancellation():
#                     await self._cleanup_on_cancel(flow_id)
#                     return
                
#                 # Start execution phase in tree
#                 execution_node_id = None
#                 if self._agent_tree:
#                     execution_node_id = self._agent_tree.start_node(
#                         node_type=NodeType.PHASE,
#                         name="execution",
#                         parent_id=root_node_id
#                     )

#                 state.start_phase("execution", execution_node_id)
#                 task_count = len(plan_data['task_plan'].tasks)
#                 self.logger.info(f"[{flow_id}] [PHASE 4] Tool Execution ({task_count} tasks)...")
                
#                 if enable_thinking and self.config.enable_thinking_display:
#                     yield ThinkingStartEvent(
#                         phase="execution",
#                         message=f"Executing {task_count} analysis tasks...",
#                         estimated_steps=task_count,
#                         node_id=execution_node_id,
#                         parent_id=root_node_id
#                     )

#                 execution_data = {'all_tool_results': {}, 'tools_executed': [], 'mode': 'task_based'}
                
#                 async for event in self._phase4_tool_execution_stream(
#                     flow_id=flow_id,
#                     query=query,
#                     plan_data=plan_data,
#                     context_data=context_data,
#                     provider_type=provider_type,
#                     model_name=model_name,
#                     user_id=user_id,
#                     session_id=session_id,
#                     state=state,
#                     parent_node_id=execution_node_id
#                 ):
#                     # Check cancellation during execution
#                     if await self._check_cancellation():
#                         await self._cleanup_on_cancel(flow_id)
#                         return
                    
#                     if isinstance(event, dict) and 'all_tool_results' in event:
#                         execution_data = event
#                     else:
#                         yield event
                
#                 execution_duration = state.end_phase()

#                 # End execution node
#                 if self._agent_tree and execution_node_id:
#                     self._agent_tree.end_node(
#                         execution_node_id,
#                         success=True,
#                         metadata={"tools_executed": len(execution_data.get('tools_executed', []))}
#                     )

#                 if enable_thinking and self.config.enable_thinking_display:
#                     tools_count = len(execution_data.get('tools_executed', []))
#                     yield ThinkingEndEvent(
#                         phase="execution",
#                         summary=f"Completed {tools_count} tool executions",
#                         duration_ms=int(execution_duration),
#                         node_id=execution_node_id
#                     )

#                 self.logger.info(f"[{flow_id}] [PHASE 4] Complete ({execution_duration:.0f}ms)")
            
#             # ================================================================
#             # PHASE 5: CONTEXT ASSEMBLY
#             # ================================================================

#             # Check cancellation
#             if await self._check_cancellation():
#                 await self._cleanup_on_cancel(flow_id)
#                 return
            
#             state.start_phase("assembly")
#             self.logger.info(f"[{flow_id}] [PHASE 5] Context Assembly...")
            
#             assembled_context = await self._phase5_context_assembly(
#                 flow_id=flow_id,
#                 query=query,
#                 chart_displayed=chart_displayed,
#                 plan_data=plan_data or {},
#                 execution_data=execution_data,
#                 model_name=model_name,
#                 enable_thinking=enable_thinking,
#                 context_data=context_data
#             )
            
#             assembly_duration = state.end_phase()
#             self.logger.info(f"[{flow_id}] [PHASE 5] Complete ({assembly_duration:.0f}ms)")
            
#             # ================================================================
#             # PHASE 6: RESPONSE GENERATION (streaming)
#             # ================================================================

#             # Check cancellation
#             if await self._check_cancellation():
#                 await self._cleanup_on_cancel(flow_id)
#                 return
            
#             # Start generation phase in tree
#             generation_node_id = None
#             if self._agent_tree:
#                 generation_node_id = self._agent_tree.start_node(
#                     node_type=NodeType.GENERATION,
#                     name="response_generation",
#                     parent_id=root_node_id
#                 )

#             state.start_phase("generation", generation_node_id)
#             self.logger.info(f"[{flow_id}] [PHASE 6] Generating Response...")
            
#             complete_response = ""
#             async for event in self._phase6_generate_response_stream(
#                 flow_id=flow_id,
#                 query=query,
#                 assembled_context=assembled_context,
#                 execution_data=execution_data,
#                 user_id=user_id,
#                 session_id=session_id,
#                 model_name=model_name,
#                 provider_type=provider_type,
#                 enable_thinking=enable_thinking,
#                 state=state,
#                 parent_node_id=generation_node_id
#             ):
#                 # Check cancellation during generation
#                 if await self._check_cancellation():
#                     # Still yield partial response before cancelling
#                     await self._cleanup_on_cancel(flow_id)
#                     return
                
#                 if isinstance(event, TextDeltaEvent):
#                     complete_response += event.data.get('chunk', '')
#                 yield event
            
#             generation_duration = state.end_phase()

#             # End generation node
#             if self._agent_tree and generation_node_id:
#                 self._agent_tree.end_node(
#                     generation_node_id,
#                     success=True,
#                     metadata={"response_length": len(complete_response)}
#                 )
                
#             self.logger.info(f"[{flow_id}] [PHASE 6] Complete ({generation_duration:.0f}ms)")
            
#             # ================================================================
#             # PHASE 7: COMPLETION
#             # ================================================================
#             self.logger.info(f"[{flow_id}] [PHASE 7] Post-Processing...")
            
#             # Save response
#             if self.config.save_messages and complete_response:
#                 await self._save_assistant_response(
#                     flow_id=flow_id,
#                     session_id=session_id,
#                     response=complete_response,
#                     state=state
#                 )
            
#             # Cleanup Working Memory
#             working_memory_context = None
#             if self._wm_integration:
#                 working_memory_context = self._wm_integration.get_context_for_memory_update()
#                 self.logger.debug(
#                     f"[{flow_id}] Working Memory context for memory update: "
#                     f"{working_memory_context[:200] if working_memory_context else 'None'}..."
#                 )
            
#             if self._wm_integration:
#                 self._wm_integration.complete_request(clear_task_data=True)
            
#             # Background memory updates
#             asyncio.create_task(
#                 self._background_memory_updates(
#                     flow_id=flow_id,
#                     user_id=user_id,
#                     session_id=session_id,
#                     query=query,
#                     response=complete_response,
#                     organization_id=organization_id,
#                     provider_type=provider_type,
#                     model_name=model_name,
#                     working_memory_context=working_memory_context
#                 )
#             )
            
#             # Final stats
#             total_time = time.time() - flow_start
#             stats = {
#                 "total_time_ms": int(total_time * 1000),
#                 "phases": {
#                     "context": int(context_duration),
#                     "planning": int(planning_duration),
#                     "execution": int(execution_duration) if execution_duration else 0,
#                     "assembly": int(assembly_duration),
#                     "generation": int(generation_duration)
#                 },
#                 "tools_executed": len(execution_data.get('tools_executed', [])),
#                 "response_length": len(complete_response)
#             }
            
#             # Get agent tree summary
#             agent_tree_summary = None
#             if self._agent_tree:
#                 # End root node
#                 self._agent_tree.end_node(
#                     self._agent_tree.root_id,
#                     success=True,
#                     metadata=stats
#                 )
#                 agent_tree_summary = self._agent_tree.get_summary()
                
#                 # Log tree visualization for debugging
#                 self.logger.debug(f"[{flow_id}] Agent Tree:\n{self._agent_tree.visualize()}")
            
#             # DoneEvent signature
#             yield DoneEvent(
#                 session_id=session_id,
#                 flow_id=flow_id,
#                 total_duration_ms=int(total_time * 1000),
#                 stats=stats,
#                 agent_tree=agent_tree_summary,  
#                 node_id=root_node_id
#             )
            
#             self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════")
#             self.logger.info(f"[{flow_id}] [STREAMING] COMPLETE ({total_time:.2f}s)")
#             self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════")

#         except asyncio.CancelledError:
#             # Handle asyncio cancellation
#             self.logger.info(f"[{flow_id}] Stream cancelled via asyncio")
#             await self._cleanup_on_cancel(flow_id)
#             yield ErrorEvent(
#                 error_message="Stream cancelled",
#                 error_type="Cancelled",
#                 phase=state.current_phase or "unknown",
#                 recoverable=True
#             )

#         except Exception as e:
#             self.logger.error(f"[{flow_id}] ✗ STREAM ERROR: {e}", exc_info=True)
            
#             # Save error to working memory
#             if self._wm_integration:
#                 self._wm_integration.save_error(
#                     error_type="stream_error",
#                     message=str(e),
#                     recoverable=True
#                 )
#                 self._wm_integration.complete_request(clear_task_data=False)
            
#             # Mark tree node as failed
#             if self._agent_tree:
#                 self._agent_tree.update_node(
#                     self._agent_tree.root_id,
#                     status=NodeStatus.FAILED,
#                     error=str(e)
#                 )

#             # ErrorEvent signature - use error_message instead of error
#             yield ErrorEvent(
#                 error_message=str(e),
#                 error_type="StreamError",
#                 phase=state.current_phase or "unknown",
#                 recoverable=True
#             )
    
#     def _build_context_summary(self, context_data: Dict[str, Any]) -> str:
#         """Build professional context loading summary"""
#         components = context_data.get('components', {})
        
#         parts = []
        
#         # Check what was loaded
#         has_memory = components.get('core_memory', 0) > 0
#         has_summary = components.get('summary', 0) > 0
#         has_history = components.get('chat_history', 0) > 0
        
#         if has_memory:
#             parts.append("your profile")
#         if has_summary:
#             parts.append("conversation summary")
#         if has_history:
#             parts.append("recent messages")
        
#         if parts:
#             loaded_items = ", ".join(parts)
#             return f"Loaded {loaded_items} for context"
#         else:
#             return "Ready to respond"
        
#     # ========================================================================
#     # MESSAGE PERSISTENCE
#     # ========================================================================
    
#     async def _save_user_message(
#         self,
#         flow_id: str,
#         session_id: str,
#         user_id: int,
#         query: str
#     ) -> Optional[str]:
#         """Save user message to database"""
#         if not self.chat_repo:
#             return None
        
#         try:
#             question_id = self.chat_repo.save_user_question(
#                 session_id=session_id,
#                 created_at=datetime.now(),
#                 created_by=user_id,
#                 content=query
#             )
#             self._current_question_id = question_id
#             self.logger.info(f"[{flow_id}] User message saved (id={question_id})")
#             return question_id
#         except Exception as e:
#             self.logger.error(f"[{flow_id}] Failed to save user message: {e}")
#             return None
    
#     async def _save_assistant_response(
#         self,
#         flow_id: str,
#         session_id: str,
#         response: str,
#         state: StreamState
#     ) -> Optional[str]:
#         """Save assistant response to database"""
#         if not self.chat_repo or not self._current_question_id:
#             return None
        
#         try:
#             response_time = state.get_elapsed_ms() / 1000.0
            
#             message_id = self.chat_repo.save_assistant_response(
#                 session_id=session_id,
#                 created_at=datetime.now(),
#                 question_id=self._current_question_id,
#                 content=response,
#                 response_time=response_time
#             )
            
#             self.logger.info(
#                 f"[{flow_id}] Assistant response saved "
#                 f"(id={message_id}, {len(response)} chars, {response_time:.2f}s)"
#             )
#             return message_id
#         except Exception as e:
#             self.logger.error(f"[{flow_id}] Failed to save response: {e}")
#             return None
    
#     # ========================================================================
#     # PHASE 1: CONTEXT LOADING (with compaction)
#     # ========================================================================
    
#     async def _phase1_load_context_stream(
#         self,
#         flow_id: str,
#         user_id: int,
#         session_id: str,
#         state: StreamState,
#         enable_events: bool = True,
#         enable_compaction: bool = True,
#         wm_context: str = "",
#         parent_node_id: str = None
#     ) -> Dict[str, Any]:
#         """Load context with progress events and compaction support"""
#         events = []
        
#         # 1.1 Load Core Memory
#         self.logger.info(f"[{flow_id}] [PHASE 1.1] Loading Core Memory...")
#         if enable_events:
#             events.append(ContextLoadingEvent(
#                 component="core_memory",
#                 status="loading",
#                 display_name="Loading your profile...",
#                 node_id=parent_node_id
#             ))
        
#         core_memory = {}
#         try:
#             core_memory = await self.core_memory.load_core_memory(user_id)
#             core_size = len(str(core_memory))
#             self.logger.info(f"[{flow_id}] Core memory loaded: {core_size} chars")
            
#             if enable_events:
#                 events.append(ContextLoadingEvent(
#                     component="core_memory",
#                     status="loaded",
#                     display_name="Your profile loaded" if core_size > 0 else "No profile data",
#                     size_chars=core_size,
#                     node_id=parent_node_id
#                 ))
#         except Exception as e:
#             self.logger.warning(f"[{flow_id}] Core memory error: {e}")
        
#         # 1.2 Load Summary
#         self.logger.info(f"[{flow_id}] [PHASE 1.2] Loading Summary...")
#         if enable_events:
#             events.append(ContextLoadingEvent(
#                 component="summary",
#                 status="loading",
#                 display_name="Loading conversation context...",
#                 node_id=parent_node_id
#             ))
        
#         summary = ""
#         try:
#             summary = await self.summary_manager.get_active_summary(session_id) or ""
#             if summary:
#                 self.logger.info(f"[{flow_id}] Summary loaded: {len(summary)} chars")
            
#             if enable_events:
#                 events.append(ContextLoadingEvent(
#                     component="summary",
#                     status="loaded",
#                     display_name="Conversation context loaded" if summary else "New conversation",
#                     size_chars=len(summary),
#                     node_id=parent_node_id
#                 ))
#         except Exception as e:
#             self.logger.warning(f"[{flow_id}] Summary error: {e}")
        
#         # 1.3 Load Chat History
#         self.logger.info(f"[{flow_id}] [PHASE 1.3] Loading Chat History...")
#         if enable_events:
#             events.append(ContextLoadingEvent(
#                 component="chat_history",
#                 status="loading",
#                 display_name="Loading recent messages...",
#                 node_id=parent_node_id
#             ))
        
#         recent_chat = []
#         try:
#             recent_chat = await self.session_repo.get_session_messages(
#                 session_id=session_id,
#                 limit=10
#             )
            
#             chat_size = sum(len(str(m.get('content', ''))) for m in recent_chat)
#             msg_count = len(recent_chat)
#             self.logger.info(f"[{flow_id}] ✓ Chat history loaded: {len(recent_chat)} messages")
            
#             if enable_events:
#                 if msg_count > 0:
#                     display = f"Loaded {msg_count} recent messages"
#                 else:
#                     display = "Starting fresh conversation"
#                 events.append(ContextLoadingEvent(
#                     component="chat_history",
#                     status="loaded",
#                     display_name=display,
#                     size_chars=chat_size,
#                     node_id=parent_node_id
#                 ))
#         except Exception as e:
#             self.logger.warning(f"[{flow_id}] Chat history error: {e}")
        
#         # 1.4 Context Compaction
#         compaction_result = None
#         if enable_compaction and len(recent_chat) > 0:
#             context_manager = self._get_context_manager()
#             if context_manager:
#                 try:
#                     messages = [
#                         {"role": msg.get("role", "user"), "content": msg.get("content", "")}
#                         for msg in recent_chat
#                     ]
                    
#                     needs_compact, stats = context_manager.compressor.should_compact(messages)
                    
#                     if needs_compact:
#                         self.logger.info(
#                             f"[{flow_id}] Context at {stats.usage_percent:.1f}% - Compacting..."
#                         )
                        
#                         symbols_to_preserve = self._extract_symbols_heuristic(recent_chat)
                        
#                         compaction_result = await context_manager.compact_now(
#                             messages=messages,
#                             preserve_keywords=symbols_to_preserve
#                         )
                        
#                         if compaction_result.success:
#                             recent_chat = [
#                                 {"role": m["role"], "content": m["content"]}
#                                 for m in compaction_result.preserved_messages
#                             ]
#                             self.logger.info(
#                                 f"[{flow_id}] ✅ Compacted: {compaction_result.tokens_saved} tokens saved"
#                             )
#                 except Exception as e:
#                     self.logger.warning(f"[{flow_id}] Compaction error: {e}")
        
#         # Calculate totals
#         total_chars = (
#             len(str(core_memory)) +
#             len(summary) +
#             sum(len(str(m)) for m in recent_chat) +
#             len(wm_context)
#         )
#         total_tokens = total_chars // 4
        
#         return {
#             'core_memory': core_memory,
#             'summary': summary,
#             'recent_chat': recent_chat,
#             'working_memory_context': wm_context,
#             'compaction_result': compaction_result,
#             'events': events,
#             'total_tokens': total_tokens,
#             'usage_percent': round(total_tokens / 180000 * 100, 2),
#             'components': {
#                 'core_memory': len(str(core_memory)),
#                 'summary': len(summary),
#                 'chat_history': sum(len(str(m)) for m in recent_chat),
#                 'working_memory': len(wm_context)
#             }
#         }
    
#     def _extract_symbols_heuristic(self, recent_chat: List[Dict]) -> List[str]:
#         """Extract symbols using heuristics (no LLM)"""
#         import re
        
#         symbols = set()
#         symbol_pattern = r'\b[A-Z][A-Z0-9]{0,14}\b'
        
#         exclude = {
#             'I', 'A', 'THE', 'AND', 'OR', 'NOT', 'IS', 'ARE', 'TO', 'OF',
#             'IN', 'FOR', 'ON', 'AT', 'BY', 'WITH', 'FROM', 'UP', 'DOWN',
#             'BE', 'IT', 'AS', 'AN', 'IF', 'SO', 'NO', 'YES', 'OK',
#             'API', 'URL', 'HTTP', 'HTTPS', 'JSON', 'XML', 'HTML', 'CSS',
#             'USD', 'EUR', 'VND', 'JPY', 'GBP',
#         }
        
#         for msg in recent_chat[-5:]:
#             content = msg.get('content', '')
#             matches = re.findall(symbol_pattern, content)
#             for match in matches:
#                 if match not in exclude and len(match) >= 2:
#                     symbols.add(match)
        
#         return list(symbols)[:10]
    
#     # ========================================================================
#     # PHASE 2: PLANNING (with thinking events)
#     # ========================================================================
    
#     async def _phase2_planning_stream(
#         self,
#         flow_id: str,
#         query: str,
#         context_data: Dict[str, Any],
#         state: StreamState,
#         enable_thinking: bool = True,
#         wm_context: str = "",
#         model_name: str = "gpt-4.1-nano",
#         provider_type: str = "openai",
#         parent_node_id: str = None
#     ) -> AsyncGenerator[Any, None]:
#         """
#         Stream planning process with thinking events
        
#         SIMPLIFIED: No separate thinking LLM call
#         Just emit progress events during actual planning
#         """
#         # Format recent chat
#         formatted_recent = [
#             {
#                 'role': msg.get('role', 'user'),
#                 'content': msg.get('content', ''),
#                 'created_at': msg.get('created_at', '')
#             }
#             for msg in context_data['recent_chat']
#         ]
        
#         # Emit thinking start
#         if enable_thinking and self.config.enable_thinking_display:
#             yield ThinkingStartEvent(
#                 phase="planning",
#                 message="Analyzing your request...",
#                 estimated_steps=3,
#                 node_id=parent_node_id
#             )
        
#         # Emit LLM thought about query understanding
#         if self.config.enable_llm_decision_events:
#             yield LLMThoughtEvent(
#                 thought=f"Analyzing query: \"{query[:100]}...\"",
#                 thought_type="analysis",
#                 context="query_understanding",
#                 parent_id=parent_node_id
#             )
        
#         # Stage progress events (static but informative)
#         yield PlanningProgressEvent(
#             stage=1,
#             total_stages=3,
#             stage_name="Understanding",
#             details="Analyzing intent and identifying relevant data",
#             parent_id=parent_node_id
#         )
        
#         yield PlanningProgressEvent(
#             stage=2,
#             total_stages=3,
#             stage_name="Classification", 
#             details="Determining query type and required tools",
#             parent_id=parent_node_id
#         )
        
#         # Execute actual planning (ONLY 1 LLM call here)
#         task_plan = await self.planning_agent.think_and_plan(
#             query=query,
#             recent_chat=formatted_recent,
#             core_memory=context_data['core_memory'],
#             summary=context_data['summary'],
#             working_memory_context=wm_context
#         )
        
#         language = 'auto'
#         classification_reasoning = ""
        
#         if task_plan:
#             language = task_plan.response_language
#             classification_reasoning = task_plan.reasoning or ""
        
#         yield PlanningProgressEvent(
#             stage=3,
#             total_stages=3,
#             stage_name="Plan Creation",
#             details="Building execution strategy",
#             parent_id=parent_node_id
#         )
        
#         # Emit LLM decision about tool selection
#         if self.config.enable_llm_decision_events and task_plan:
#             tools_selected = []
#             for task in task_plan.tasks:
#                 for tool in task.tools_needed:
#                     tools_selected.append(tool.tool_name)
            
#             yield LLMDecisionEvent(
#                 decision=f"Selected {len(tools_selected)} tools",
#                 decision_type="tool_selection",
#                 options_considered=list(set(tools_selected)),
#                 reasoning=classification_reasoning[:200] if classification_reasoning else None,
#                 confidence=0.9,
#                 parent_id=parent_node_id
#             )
        
#         # Save to working memory
#         if task_plan and self._wm_integration:
#             self._wm_integration.save_classification(
#                 query_type=task_plan.query_intent,
#                 categories=[],
#                 symbols=task_plan.symbols or [],
#                 language=language,
#                 reasoning=classification_reasoning
#             )
#             if task_plan.tasks:
#                 self._wm_integration.save_plan(task_plan)
        
#         self._current_task_plan = task_plan
        
#         # Emit planning complete
#         yield PlanningCompleteEvent(
#             task_count=len(task_plan.tasks) if task_plan else 0,
#             strategy=task_plan.strategy if task_plan else 'direct_answer',
#             symbols=task_plan.symbols if task_plan else [],
#             query_intent=task_plan.query_intent if task_plan else 'conversational',
#             node_id=parent_node_id
#         )
        
#         # Emit thinking end
#         if enable_thinking:
#             task_count = len(task_plan.tasks) if task_plan else 0
#             intent = task_plan.query_intent if task_plan else 'N/A'
            
#             if task_count > 0:
#                 summary = f"Identified {task_count} analysis tasks for {intent} query"
#             else:
#                 summary = f"Conversational response - no data retrieval needed"
            
#             yield ThinkingEndEvent(
#                 phase="planning",
#                 summary=summary,
#                 node_id=parent_node_id
#             )
        
#         # Return plan data
#         yield {
#             'task_plan': task_plan,
#             'language': language,
#             'symbols': task_plan.symbols if task_plan and hasattr(task_plan, 'symbols') else [],
#             'query_intent': task_plan.query_intent if task_plan else '',
#             'strategy': task_plan.strategy if task_plan else '',
#             'reasoning': classification_reasoning
#         }
    
#     def _build_professional_thinking_summary(
#         self,
#         intent: str,
#         reasoning: str,
#         task_count: int,
#         strategy: str,
#         symbols: List[str]
#     ) -> str:
#         """
#         Build a professional, user-friendly thinking summary
        
#         Shows AI reasoning in a way that appears intelligent and helpful,
#         not like debug output.
#         """
#         parts = []
        
#         # 1. Intent/Goal - Always show
#         if intent:
#             # Clean up intent for display
#             intent_clean = intent.strip()
#             if not intent_clean.endswith('.'):
#                 intent_clean += '.'
#             parts.append(f"Goal: {intent_clean}")
        
#         # 2. Reasoning - Show if available (this is the key insight!)
#         if reasoning:
#             # Clean and truncate reasoning
#             reasoning_clean = reasoning.strip()
#             # Limit to ~200 chars for UI
#             if len(reasoning_clean) > 250:
#                 # Find last sentence break within limit
#                 truncated = reasoning_clean[:250]
#                 last_period = truncated.rfind('.')
#                 if last_period > 100:
#                     reasoning_clean = truncated[:last_period + 1]
#                 else:
#                     reasoning_clean = truncated + "..."
            
#             parts.append(f"Analysis: {reasoning_clean}")
        
#         # 3. Symbols - Show if relevant
#         if symbols:
#             symbols_str = ", ".join(symbols)
#             parts.append(f"Focus: {symbols_str}")
        
#         # 4. Action plan - User-friendly
#         if task_count > 0:
#             if task_count == 1:
#                 parts.append(f"Plan: Executing 1 analysis task")
#             else:
#                 execution_mode = "in parallel" if strategy == "parallel" else "sequentially"
#                 parts.append(f"Plan: Executing {task_count} analysis tasks {execution_mode}")
#         else:
#             parts.append(f"Plan: Direct response (no data retrieval needed)")
        
#         return " | ".join(parts)
    
#     # ========================================================================
#     # PHASE 4: TOOL EXECUTION (with progress events)
#     # ========================================================================
    
#     async def _phase4_tool_execution_stream(
#         self,
#         flow_id: str,
#         query: str,
#         plan_data: Dict[str, Any],
#         context_data: Dict[str, Any],
#         provider_type: str,
#         model_name: str,
#         user_id: int,
#         session_id: str,
#         state: StreamState,
#         parent_node_id: str = None
#     ) -> AsyncGenerator[Any, None]:
#         """
#         Task 3: Enhanced tool execution with call_id correlation
        
#         Each tool call gets a unique call_id that correlates
#         TOOL_START with TOOL_COMPLETE events.
#         """
#         task_plan = plan_data.get('task_plan')
        
#         if not task_plan or len(task_plan.tasks) == 0:
#             yield {
#                 'all_tool_results': {},
#                 'tools_executed': [],
#                 'mode': None,
#                 'stats': {}
#             }
#             return
        
#         self.logger.info(f"[{flow_id}] Executing {len(task_plan.tasks)} tasks...")
        
#         # Build chat history
#         chat_history = [
#             [msg.get('content', ''), msg.get('role', 'user')]
#             for msg in context_data['recent_chat'][-5:]
#         ]
        
#         all_tool_results = {}
#         tools_executed = []
#         total_tools = sum(len(task.tools_needed) for task in task_plan.tasks)
#         current_tool_idx = 0
        
#         # Execute each task
#         for task_idx, task in enumerate(task_plan.tasks):
#             for tool_call in task.tools_needed:
#                 tool_name = tool_call.tool_name
#                 current_tool_idx += 1
                
#                 # Task 3: Generate unique call_id
#                 call_id = generate_call_id(tool_name)
                
#                 # Task 6: Start tool node in tree
#                 tool_node_id = None
#                 if self._agent_tree:
#                     tool_node_id = self._agent_tree.start_node(
#                         node_type=NodeType.TOOL,
#                         name=tool_name,
#                         parent_id=parent_node_id,
#                         metadata={"call_id": call_id, "params": tool_call.params}
#                     )
                
#                 # Track in state (Task 3)
#                 state.start_tool(call_id, tool_name, tool_node_id)
                
#                 # Emit tool start with call_id
#                 yield ToolStartEvent(
#                     task_id=task.id,
#                     tool_name=tool_name,
#                     params=tool_call.params,
#                     call_id=call_id,  # Task 3
#                     node_id=tool_node_id,
#                     parent_id=parent_node_id
#                 )
                
#                 # Task 5: Emit LLM action event
#                 if self.config.enable_llm_decision_events:
#                     yield LLMActionEvent(
#                         action="call_tool",
#                         action_target=tool_name,
#                         parameters=tool_call.params,
#                         reasoning=f"Executing {tool_name} for task {task.id}",
#                         node_id=tool_node_id
#                     )
                
#                 # Thinking delta for tool execution
#                 if self.config.enable_thinking_display:
#                     yield ThinkingDeltaEvent(
#                         phase="execution",
#                         thought=f"Fetching data from {tool_name}...",
#                         progress=current_tool_idx / total_tools,
#                         step_number=current_tool_idx,
#                         total_steps=total_tools,
#                         parent_id=parent_node_id
#                     )
                
#                 # Execute tool
#                 tool_start = time.time()
#                 success = False
#                 error_msg = None
#                 result = None
                
#                 try:
#                     result = await self.tool_execution_service.execute_single_tool(
#                         tool_name=tool_name,
#                         tool_params=tool_call.params,
#                         query=query,
#                         chat_history=chat_history,
#                         system_language=plan_data.get('language', 'vi'),
#                         provider_type=provider_type,
#                         model_name=model_name,
#                         user_id=user_id,
#                         session_id=session_id
#                     )
                    
#                     execution_time = int((time.time() - tool_start) * 1000)
#                     status = result.get('status', 'unknown') if isinstance(result, dict) else 'unknown'
#                     success = status in ['200', 200, 'success']
                    
#                     # Store result
#                     all_tool_results[tool_name] = result
#                     tools_executed.append(tool_name)
                    
#                     # Save to working memory
#                     if self._wm_integration:
#                         self._wm_integration.save_tool_result(
#                             tool_name=tool_name,
#                             result=result,
#                             task_id=str(task.id),
#                             execution_time_ms=execution_time,
#                             status="success" if success else "failed"
#                         )
                    
#                 except Exception as e:
#                     execution_time = int((time.time() - tool_start) * 1000)
#                     error_msg = str(e)
#                     self.logger.error(f"[{flow_id}] ✗ {tool_name}: {e}")
                
#                 # Task 3: End tool tracking with call_id
#                 state.end_tool(call_id, success)
                
#                 # Task 6: End tool node in tree
#                 if self._agent_tree and tool_node_id:
#                     self._agent_tree.end_node(
#                         tool_node_id,
#                         success=success,
#                         error=error_msg,
#                         metadata={"duration_ms": execution_time}
#                     )
                
#                 # Emit tool complete with call_id (Task 3)
#                 preview = self._get_result_preview(result) if result else None
#                 yield ToolCompleteEvent(
#                     task_id=task.id,
#                     tool_name=tool_name,
#                     success=success,
#                     call_id=call_id,  # Task 3: Same call_id for correlation
#                     preview=preview,
#                     duration_ms=execution_time,
#                     error=error_msg,
#                     node_id=tool_node_id,
#                     parent_id=parent_node_id
#                 )
        
#         # Return execution data
#         yield {
#             'all_tool_results': all_tool_results,
#             'tools_executed': list(set(tools_executed)),
#             'mode': 'task_based',
#             'stats': {
#                 'total_tools': len(tools_executed),
#                 'unique_tools': len(set(tools_executed))
#             }
#         }
        
#     def _get_result_preview(self, result: Dict[str, Any]) -> str:
#         """Get a short preview of tool result"""
#         if not isinstance(result, dict):
#             return str(result)[:100]
        
#         # Check for formatted_context
#         if 'formatted_context' in result:
#             return result['formatted_context'][:self.config.max_tool_preview_length]
        
#         # Check for data
#         if 'data' in result:
#             import json
#             try:
#                 return json.dumps(result['data'], ensure_ascii=False)[:self.config.max_tool_preview_length]
#             except:
#                 return str(result['data'])[:self.config.max_tool_preview_length]
        
#         return str(result)[:self.config.max_tool_preview_length]
    
#     # ========================================================================
#     # PHASE 5: CONTEXT ASSEMBLY
#     # ========================================================================
    
#     async def _phase5_context_assembly(
#         self,
#         flow_id: str,
#         query: str,
#         chart_displayed: bool,
#         plan_data: Dict[str, Any],
#         execution_data: Dict[str, Any],
#         model_name: str,
#         enable_thinking: bool,
#         context_data: Dict[str, Any] = None
#     ) -> Dict[str, Any]:
#         """
#         Assemble context for LLM (NO LLM call)
#         """
#         all_tool_results = execution_data.get('all_tool_results', {})
#         language = plan_data.get('language', 'vi')
#         plan_symbols = plan_data.get('symbols', [])
#         query_intent = plan_data.get('query_intent', '')
        
#         # Build system prompt
#         from src.helpers.system_prompts import get_system_message_general_chat
        
#         system_prompt = get_system_message_general_chat(
#             enable_thinking=enable_thinking,
#             model_name=model_name,
#             detected_language=language,
#             chart_displayed=chart_displayed
#         )

#         summary = context_data.get('summary', '') if context_data else ''
#         if summary:
#             system_prompt += f"""

# <conversation_summary>
# ## Previous Conversation Context
# The following is a summary of earlier conversation turns that may provide relevant context:

# {summary}

# Note: Use this context to maintain conversation continuity. Current query data below takes precedence.
# </conversation_summary>
# """
#         self.logger.info(f"[{flow_id}] ✓ Summary included in context: {len(summary)} chars")

#         # Format tool context
#         tool_context = ""
#         if all_tool_results:
#             tool_context = self._format_tool_context_for_llm(all_tool_results, language)
#             if tool_context:
#                 system_prompt += f"""

# <current_data>
# ## Real-Time Data for Current Query
# The following data was retrieved specifically for the current user query:

# {tool_context}
# </current_data>
# """
        
#         # Extract symbols from results
#         symbols = self._extract_symbols_from_results(all_tool_results)
#         active_symbols = plan_symbols if plan_symbols else symbols
        
#         # Add symbol awareness
#         if active_symbols:
#             symbol_awareness = f"""
# {'='*70}
# ⚠️ CURRENT QUERY SYMBOLS: {', '.join(active_symbols)}
# Query Intent: {query_intent}
# IMPORTANT: Analyze ONLY these symbol(s). Do NOT confuse with history!
# {'='*70}
# """
#             system_prompt += symbol_awareness
        
#         return {
#             'system_prompt': system_prompt,
#             'tool_context': tool_context,
#             'symbols': active_symbols
#         }
    
#     def _format_tool_context_for_llm(
#         self,
#         tool_outputs: Dict[str, Any],
#         language: str = 'en'
#     ) -> str:
#         """Format tool context for LLM consumption"""
#         sections = []
        
#         for tool_name, tool_output in tool_outputs.items():
#             if not isinstance(tool_output, dict):
#                 continue
            
#             status = tool_output.get('status', 'success')
#             if status in ['error', 'failed']:
#                 continue
            
#             # Priority 1: formatted_context
#             if 'formatted_context' in tool_output and tool_output['formatted_context']:
#                 sections.append(tool_output['formatted_context'])
            
#             # Priority 2: nested data
#             elif 'data' in tool_output and isinstance(tool_output['data'], dict):
#                 import json
#                 formatted = json.dumps(tool_output['data'], indent=2, ensure_ascii=False, default=str)
#                 sections.append(f"=== {tool_name} ===\n{formatted}")
        
#         return "\n\n".join(sections)
    
#     def _extract_symbols_from_results(self, tool_outputs: Dict[str, Any]) -> List[str]:
#         """Extract unique symbols from tool outputs"""
#         all_symbols = set()
        
#         for tool_output in tool_outputs.values():
#             if not isinstance(tool_output, dict):
#                 continue
            
#             symbols = tool_output.get('symbols', [])
#             if isinstance(symbols, list):
#                 all_symbols.update(symbols)
#             elif isinstance(symbols, str):
#                 all_symbols.add(symbols)
            
#             data = tool_output.get('data', {})
#             if isinstance(data, dict) and 'symbol' in data:
#                 all_symbols.add(data['symbol'])
        
#         return sorted(list(all_symbols))
    
#     # ========================================================================
#     # PHASE 6: RESPONSE GENERATION (streaming)
#     # ========================================================================
    
#     async def _phase6_generate_response_stream(
#         self,
#         flow_id: str,
#         query: str,
#         assembled_context: Dict[str, Any],
#         execution_data: Dict[str, Any],
#         user_id: int,
#         session_id: str,
#         model_name: str,
#         provider_type: str,
#         enable_thinking: bool,
#         state: StreamState,
#         parent_node_id: str = None
#     ) -> AsyncGenerator[StreamEvent, None]:
#         """Stream LLM response generation"""
#         system_prompt = assembled_context['system_prompt']
#         has_tool_results = bool(execution_data.get('all_tool_results'))
        
#         max_history = 2 if has_tool_results else 10
        
#         try:
#             from src.helpers.context_assembler import ContextAssembler
#             context_assembler = ContextAssembler()
            
#             messages, llm_metadata = await context_assembler.prepare_messages_for_llm(
#                 user_id=user_id,
#                 session_id=session_id,
#                 current_query=query,
#                 system_prompt=system_prompt,
#                 enable_thinking=enable_thinking,
#                 model_name=model_name,
#                 max_history_messages=max_history
#             )
#         except Exception as e:
#             self.logger.warning(f"[{flow_id}] Context assembler error: {e}")
#             messages = [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": query}
#             ]
        
#         from src.providers.provider_factory import ModelProviderFactory
#         api_key = ModelProviderFactory._get_api_key(provider_type)
        
#         complete_response = ""
#         char_buffer = ""
#         start_time = time.time()
        
#         async for chunk in self.llm_provider.stream_response(
#             model_name=model_name,
#             messages=messages,
#             provider_type=provider_type,
#             api_key=api_key,
#             clean_thinking=True
#         ):
#             if chunk:
#                 char_buffer += chunk
                
#                 if len(char_buffer) >= self.config.text_chunk_min_size:
#                     complete_response += char_buffer
#                     yield TextDeltaEvent(
#                         chunk=char_buffer,
#                         accumulated_length=len(complete_response),
#                         node_id=parent_node_id
#                     )
#                     char_buffer = ""
        
#         if char_buffer:
#             complete_response += char_buffer
#             yield TextDeltaEvent(
#                 chunk=char_buffer,
#                 accumulated_length=len(complete_response),
#                 node_id=parent_node_id
#             )
        
#         duration_ms = int((time.time() - start_time) * 1000)
        
#         yield TextCompleteEvent(
#             total_length=len(complete_response),
#             duration_ms=duration_ms,
#             node_id=parent_node_id
#         )
    
#     # ========================================================================
#     # BACKGROUND TASKS
#     # ========================================================================
    
#     async def _background_memory_updates(
#         self,
#         flow_id: str,
#         user_id: int,
#         session_id: str,
#         query: str,
#         response: str,
#         organization_id: Optional[int],
#         provider_type: str,
#         model_name: str,
#         working_memory_context: Optional[str] = None
#     ):
#         """Background memory updates (non-blocking)"""
#         try:
#             # Import memory update agent
#             from src.agents.memory.memory_update_agent import MemoryUpdateAgent
#             memory_update = MemoryUpdateAgent()
            
#             # Update core memory
#             update_result = await memory_update.analyze_for_updates(
#                 user_id=user_id,
#                 user_message=query,
#                 assistant_message=response,
#                 working_memory_context=working_memory_context,
#                 model_name=model_name,
#                 provider_type=provider_type
#             )
            
#             if update_result.get('updated'):
#                 self.logger.info(f"[{flow_id}] [BG] ✓ Core memory updated")
            
#             # Check and create summary
#             summary_result = await self.summary_manager.check_and_create_summary(
#                 session_id=session_id,
#                 user_id=user_id,
#                 organization_id=organization_id,
#                 model_name=model_name,
#                 provider_type=provider_type
#             )
            
#             if summary_result.get('created'):
#                 self.logger.info(f"[{flow_id}] [BG] ✓ Summary v{summary_result.get('version')}")
            
#         except Exception as e:
#             self.logger.error(f"[{flow_id}] [BG] Memory update error: {e}")


# # ============================================================================
# # CONVENIENCE FUNCTION
# # ============================================================================

# async def stream_chat_sse(
#     handler: StreamingChatHandler,
#     query: str,
#     session_id: str,
#     user_id: int,
#     cancellation_token: CancellationToken = None,
#     **kwargs
# ) -> AsyncGenerator[str, None]:
#     """
#     Convenience function to stream chat as SSE strings
#     """
#     async for event in handler.handle_chat_stream(
#         query=query,
#         session_id=session_id,
#         user_id=user_id,
#         cancellation_token=cancellation_token,
#         **kwargs
#     ):
#         yield event.to_sse()
    
#     # Final done marker
#     yield "data: [DONE]\n\n"


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

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class StreamingConfig:
    """Configuration for streaming behavior"""
    
    # Enable/disable streaming features
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
    compaction_threshold: int = 100000  # tokens

    # SSE Cancellation
    cancellation_check_interval: float = 0.1  # seconds
    enable_heartbeat: bool = True
    heartbeat_interval: int = 30  # seconds


# ============================================================================
# CANCELLATION TOKEN
# ============================================================================

class CancellationToken:
    """
    Token to track cancellation state
    
    Allows checking if client has disconnected
    """
    
    def __init__(self):
        self._cancelled = False
        self._cancel_event = asyncio.Event()
    
    def cancel(self):
        """Mark as cancelled"""
        self._cancelled = True
        self._cancel_event.set()
    
    @property
    def is_cancelled(self) -> bool:
        """Check if cancelled"""
        return self._cancelled
    
    async def wait_for_cancel(self, timeout: float = None) -> bool:
        """Wait for cancellation with optional timeout"""
        try:
            await asyncio.wait_for(self._cancel_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

# ============================================================================
# STREAMING CHAT HANDLER
# ============================================================================

class StreamingChatHandler(LoggerMixin):
    """
    Streaming Chat Handler with Full 7-Phase Pipeline
    
    Provides streaming responses while maintaining full feature parity
    with ChatHandler, including:
    - Context compaction
    - Working memory
    - Think Tool
    - All 31 atomic tools
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
        """
        Initialize StreamingChatHandler
        
        Args:
            planning_agent: Planning agent for task planning
            task_executor: Task executor for tool execution
            core_memory: Core memory manager
            summary_manager: Recursive summary manager
            session_repo: Session repository
            llm_provider: LLM provider for response generation
            tool_execution_service: Tool execution service
            chat_repo: Chat repository for message persistence
            config: Streaming configuration
        """
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
        
        self.logger.info("[STREAMING] StreamingChatHandler initialized")
    
    def _get_context_manager(self):
        """Lazy initialization of context manager"""
        if self._context_manager is None:
            try:
                from src.services.context_management_service import ContextManagementService
                self._context_manager = ContextManagementService(
                    enable_compaction=self.config.enable_compaction,
                    compaction_threshold=self.config.compaction_threshold,
                    compaction_strategy="smart_summary"
                )
            except ImportError:
                self.logger.warning("[STREAMING] ContextManagementService not available")
        return self._context_manager
    

    # ========================================================================
    # SSE CANCELLATION CHECK
    # ========================================================================
    
    async def _check_cancellation(self) -> bool:
        """
        Check if request has been cancelled
        
        Returns True if cancelled, False otherwise
        """
        if self._cancellation_token and self._cancellation_token.is_cancelled:
            self.logger.info("[STREAMING] Request cancelled by client")
            return True
        return False
    
    async def _cleanup_on_cancel(self, flow_id: str):
        """
        Cleanup resources when cancelled
        """
        self.logger.info(f"[{flow_id}] Cleaning up after cancellation...")
        
        # Save partial state to working memory
        if self._wm_integration:
            self._wm_integration.save_error(
                error_type="client_disconnected",
                message="Client disconnected before completion",
                recoverable=True
            )
            self._wm_integration.complete_request(clear_task_data=False)
        
        # End any open agent tree nodes
        if self._agent_tree:
            for node_id, node in self._agent_tree.nodes.items():
                if node.status in [NodeStatus.STARTED, NodeStatus.RUNNING]:
                    self._agent_tree.end_node(
                        node_id=node_id,
                        success=False,
                        error="cancelled"
                    )
        
        self.logger.info(f"[{flow_id}] Cleanup complete")

    # ========================================================================
    # MAIN ENTRY POINT
    # ========================================================================
    
    async def handle_chat_stream(
        self,
        query: str,
        session_id: str,
        user_id: int,
        model_name: str = "gpt-4.1-nano",
        provider_type: str = "openai",
        organization_id: Optional[int] = None,
        enable_thinking: bool = True,
        # Additional parameters (matching ChatHandler)
        chart_displayed: bool = False,
        enable_compaction: bool = True,
        enable_think_tool: bool = False,
        cancellation_token: CancellationToken = None,
        **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Main streaming chat handler - yields StreamEvent objects
        
        Args:
            query: User query
            session_id: Session ID
            user_id: User ID
            model_name: LLM model name
            provider_type: LLM provider
            organization_id: Organization ID
            enable_thinking: Enable thinking display
            chart_displayed: Whether chart is displayed (affects response)
            enable_compaction: Enable context compaction
            enable_think_tool: Enable Think Tool for complex reasoning
            **kwargs: Additional parameters
            
        Yields:
            StreamEvent objects for SSE streaming
        """
        # Generate flow ID for tracking
        flow_id = f"STREAM-{datetime.now().strftime('%H%M%S')}-{session_id[:8]}"
        flow_start = time.time()
        
        # Reset instance state
        self._current_task_plan = None
        self._execution_results = {}
        self._accumulated_response = ""
        self._current_question_id = None
        
        # Store cancellation token
        self._cancellation_token = cancellation_token

        # Create agent tree
        if self.config.enable_agent_tree:
            self._agent_tree = create_tree_for_request(flow_id)
        
        # Initialize stream state
        state = StreamState(
            session_id=session_id,
            flow_id=flow_id
        )
        
        self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════")
        self.logger.info(f"[{flow_id}] [STREAMING] Starting progressive stream")
        self.logger.info(f"[{flow_id}]   Query: {query[:100]}...")
        self.logger.info(f"[{flow_id}]   Session: {session_id[:16]}...")
        self.logger.info(f"[{flow_id}]   Model: {model_name}")
        self.logger.info(f"[{flow_id}]   Features: cancellation={cancellation_token is not None}, "
                        f"tree={self.config.enable_agent_tree}, "
                        f"llm_events={self.config.enable_llm_decision_events}")
        self.logger.info(f"[{flow_id}]   Compaction: {'ON' if enable_compaction else 'OFF'}")
        self.logger.info(f"[{flow_id}]   Think Tool: {'ON' if enable_think_tool else 'OFF'}")
        self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════")
        
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
            
            # Get root node ID for tree tracking
            root_node_id = self._agent_tree.root_id if self._agent_tree else None
            
            # ================================================================
            # PHASE 1: START
            # ================================================================
            yield StartEvent(
                session_id=session_id,
                flow_id=flow_id,
                model_name=model_name,
                node_id=root_node_id
            )
            
            # ================================================================
            # PHASE 2: CONTEXT LOADING (with compaction support)
            # ================================================================

            # Check cancellation
            if await self._check_cancellation():
                await self._cleanup_on_cancel(flow_id)
                return
            
            # Start context phase in tree
            context_node_id = None
            if self._agent_tree:
                context_node_id = self._agent_tree.start_node(
                    node_type=NodeType.PHASE,
                    name="context_loading",
                    parent_id=root_node_id
                )

            state.start_phase("context_loading")
            self.logger.info(f"[{flow_id}] [PHASE 1] Loading Context...")
            
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

            self.logger.info(f"[{flow_id}] [PHASE 1] Complete ({context_duration:.0f}ms)")
            
            # ================================================================
            # PHASE 3: PLANNING (with thinking events) real thinking streaming
            # ================================================================

            # Check cancellation
            if await self._check_cancellation():
                await self._cleanup_on_cancel(flow_id)
                return
            
            # Start planning phase in tree
            planning_node_id = None
            if self._agent_tree:
                planning_node_id = self._agent_tree.start_node(
                    node_type=NodeType.PLANNING,
                    name="planning",
                    parent_id=root_node_id
                )
                        
            state.start_phase("planning")
            self.logger.info(f"[{flow_id}] [PHASE 2] Planning...")
            
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

            # End planning node
            if self._agent_tree and planning_node_id:
                self._agent_tree.end_node(
                    planning_node_id,
                    success=plan_data is not None,
                    metadata={"task_count": len(plan_data['task_plan'].tasks) if plan_data and plan_data.get('task_plan') else 0}
                )

            self.logger.info(f"[{flow_id}] [PHASE 2] Complete ({planning_duration:.0f}ms)")
            
            # Check if no tools needed
            if not plan_data or not plan_data.get('task_plan') or len(plan_data['task_plan'].tasks) == 0:
                self.logger.info(f"[{flow_id}] No tools needed - conversational response")
                execution_data = {'all_tool_results': {}, 'tools_executed': [], 'mode': None}
                execution_duration = 0
            else:
                # ================================================================
                # PHASE 4: TOOL EXECUTION (FIXED - Using TaskExecutor!)
                # ================================================================

                # Check cancellation
                if await self._check_cancellation():
                    await self._cleanup_on_cancel(flow_id)
                    return
                
                # Start execution phase in tree
                execution_node_id = None
                if self._agent_tree:
                    execution_node_id = self._agent_tree.start_node(
                        node_type=NodeType.PHASE,
                        name="execution",
                        parent_id=root_node_id
                    )

                state.start_phase("execution", execution_node_id)
                task_count = len(plan_data['task_plan'].tasks)
                self.logger.info(f"[{flow_id}] [PHASE 4] Tool Execution ({task_count} tasks)...")
                
                if enable_thinking and self.config.enable_thinking_display:
                    yield ThinkingStartEvent(
                        phase="execution",
                        message=f"Executing {task_count} analysis tasks...",
                        estimated_steps=task_count,
                        node_id=execution_node_id,
                        parent_id=root_node_id
                    )

                execution_data = {'all_tool_results': {}, 'tools_executed': [], 'mode': 'task_based'}
                
                # ============================================================
                # CRITICAL FIX: Use TaskExecutor instead of direct tool calls!
                # ============================================================
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

                self.logger.info(f"[{flow_id}] [PHASE 4] Complete ({execution_duration:.0f}ms)")
            
            # ================================================================
            # PHASE 5: CONTEXT ASSEMBLY
            # ================================================================

            # Check cancellation
            if await self._check_cancellation():
                await self._cleanup_on_cancel(flow_id)
                return
            
            state.start_phase("assembly")
            self.logger.info(f"[{flow_id}] [PHASE 5] Context Assembly...")
            
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
            self.logger.info(f"[{flow_id}] [PHASE 5] Complete ({assembly_duration:.0f}ms)")
            
            # ================================================================
            # PHASE 6: RESPONSE GENERATION (streaming)
            # ================================================================

            # Check cancellation
            if await self._check_cancellation():
                await self._cleanup_on_cancel(flow_id)
                return
            
            # Start generation phase in tree
            generation_node_id = None
            if self._agent_tree:
                generation_node_id = self._agent_tree.start_node(
                    node_type=NodeType.GENERATION,
                    name="response_generation",
                    parent_id=root_node_id
                )

            state.start_phase("generation", generation_node_id)
            self.logger.info(f"[{flow_id}] [PHASE 6] Generating Response...")
            
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

            # End generation node
            if self._agent_tree and generation_node_id:
                self._agent_tree.end_node(
                    generation_node_id,
                    success=True,
                    metadata={"response_length": len(complete_response)}
                )
                
            self.logger.info(f"[{flow_id}] [PHASE 6] Complete ({generation_duration:.0f}ms)")
            
            # ================================================================
            # PHASE 7: COMPLETION
            # ================================================================
            self.logger.info(f"[{flow_id}] [PHASE 7] Post-Processing...")
            
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
            
            # DoneEvent signature
            yield DoneEvent(
                session_id=session_id,
                flow_id=flow_id,
                total_duration_ms=int(total_time * 1000),
                stats=stats,
                agent_tree=agent_tree_summary,  
                node_id=root_node_id
            )
            
            self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════")
            self.logger.info(f"[{flow_id}] [STREAMING] COMPLETE ({total_time:.2f}s)")
            self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════")

        except asyncio.CancelledError:
            # Handle asyncio cancellation
            self.logger.info(f"[{flow_id}] Stream cancelled via asyncio")
            await self._cleanup_on_cancel(flow_id)
            yield ErrorEvent(
                error_message="Stream cancelled",
                error_type="Cancelled",
                phase=state.current_phase or "unknown",
                recoverable=True
            )

        except Exception as e:
            self.logger.error(f"[{flow_id}] ✗ STREAM ERROR: {e}", exc_info=True)
            
            # Save error to working memory
            if self._wm_integration:
                self._wm_integration.save_error(
                    error_type="stream_error",
                    message=str(e),
                    recoverable=True
                )
                self._wm_integration.complete_request(clear_task_data=False)
            
            # Mark tree node as failed
            if self._agent_tree:
                self._agent_tree.update_node(
                    self._agent_tree.root_id,
                    status=NodeStatus.FAILED,
                    error=str(e)
                )

            # ErrorEvent signature - use error_message instead of error
            yield ErrorEvent(
                error_message=str(e),
                error_type="StreamError",
                phase=state.current_phase or "unknown",
                recoverable=True
            )
    
    def _build_context_summary(self, context_data: Dict[str, Any]) -> str:
        """Build professional context loading summary"""
        components = context_data.get('components', {})
        
        parts = []
        
        # Check what was loaded
        has_memory = components.get('core_memory', 0) > 0
        has_summary = components.get('summary', 0) > 0
        has_history = components.get('chat_history', 0) > 0
        
        if has_memory:
            parts.append("your profile")
        if has_summary:
            parts.append("conversation summary")
        if has_history:
            parts.append("recent messages")
        
        if parts:
            loaded_items = ", ".join(parts)
            return f"Loaded {loaded_items} for context"
        else:
            return "Ready to respond"
        
    # ========================================================================
    # MESSAGE PERSISTENCE
    # ========================================================================
    
    async def _save_user_message(
        self,
        flow_id: str,
        session_id: str,
        user_id: int,
        query: str
    ) -> Optional[str]:
        """Save user message to database"""
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
            self.logger.info(f"[{flow_id}] User message saved (id={question_id})")
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
        """Save assistant response to database"""
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
            
            self.logger.info(
                f"[{flow_id}] Assistant response saved "
                f"(id={message_id}, {len(response)} chars, {response_time:.2f}s)"
            )
            return message_id
        except Exception as e:
            self.logger.error(f"[{flow_id}] Failed to save response: {e}")
            return None
    
    # ========================================================================
    # PHASE 1: CONTEXT LOADING (with compaction)
    # ========================================================================
    
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
        """Load context with progress events and compaction support"""
        events = []
        
        # 1.1 Load Core Memory
        self.logger.info(f"[{flow_id}] [PHASE 1.1] Loading Core Memory...")
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
            core_size = len(str(core_memory))
            self.logger.info(f"[{flow_id}] Core memory loaded: {core_size} chars")
            
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
        
        # 1.2 Load Summary
        self.logger.info(f"[{flow_id}] [PHASE 1.2] Loading Summary...")
        if enable_events:
            events.append(ContextLoadingEvent(
                component="summary",
                status="loading",
                display_name="Loading conversation context...",
                node_id=parent_node_id
            ))
        
        summary = ""
        try:
            summary = await self.summary_manager.get_active_summary(session_id) or ""
            if summary:
                self.logger.info(f"[{flow_id}] Summary loaded: {len(summary)} chars")
            
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
        
        # 1.3 Load Chat History
        self.logger.info(f"[{flow_id}] [PHASE 1.3] Loading Chat History...")
        if enable_events:
            events.append(ContextLoadingEvent(
                component="chat_history",
                status="loading",
                display_name="Loading recent messages...",
                node_id=parent_node_id
            ))
        
        recent_chat = []
        try:
            recent_chat = await self.session_repo.get_session_messages(
                session_id=session_id,
                limit=10
            )
            
            chat_size = sum(len(str(m.get('content', ''))) for m in recent_chat)
            msg_count = len(recent_chat)
            self.logger.info(f"[{flow_id}] ✓ Chat history loaded: {len(recent_chat)} messages")
            
            if enable_events:
                if msg_count > 0:
                    display = f"Loaded {msg_count} recent messages"
                else:
                    display = "Starting fresh conversation"
                events.append(ContextLoadingEvent(
                    component="chat_history",
                    status="loaded",
                    display_name=display,
                    size_chars=chat_size,
                    node_id=parent_node_id
                ))
        except Exception as e:
            self.logger.warning(f"[{flow_id}] Chat history error: {e}")
        
        # 1.4 Context Compaction
        compaction_result = None
        if enable_compaction and len(recent_chat) > 0:
            context_manager = self._get_context_manager()
            if context_manager:
                try:
                    messages = [
                        {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                        for msg in recent_chat
                    ]
                    
                    needs_compact, stats = context_manager.compressor.should_compact(messages)
                    
                    if needs_compact:
                        self.logger.info(
                            f"[{flow_id}] Context at {stats.usage_percent:.1f}% - Compacting..."
                        )
                        
                        symbols_to_preserve = self._extract_symbols_heuristic(recent_chat)
                        
                        compaction_result = await context_manager.compact_now(
                            messages=messages,
                            preserve_keywords=symbols_to_preserve
                        )
                        
                        if compaction_result.success:
                            recent_chat = [
                                {"role": m["role"], "content": m["content"]}
                                for m in compaction_result.preserved_messages
                            ]
                            self.logger.info(
                                f"[{flow_id}] ✅ Compacted: {compaction_result.tokens_saved} tokens saved"
                            )
                except Exception as e:
                    self.logger.warning(f"[{flow_id}] Compaction error: {e}")
        
        # Calculate totals
        total_chars = (
            len(str(core_memory)) +
            len(summary) +
            sum(len(str(m)) for m in recent_chat) +
            len(wm_context)
        )
        total_tokens = total_chars // 4
        
        return {
            'core_memory': core_memory,
            'summary': summary,
            'recent_chat': recent_chat,
            'working_memory_context': wm_context,
            'compaction_result': compaction_result,
            'events': events,
            'total_tokens': total_tokens,
            'usage_percent': round(total_tokens / 180000 * 100, 2),
            'components': {
                'core_memory': len(str(core_memory)),
                'summary': len(summary),
                'chat_history': sum(len(str(m)) for m in recent_chat),
                'working_memory': len(wm_context)
            }
        }
    
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
    
    # ========================================================================
    # PHASE 2: PLANNING (with thinking events)
    # ========================================================================
    
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
        """
        Stream planning process with thinking events
        
        SIMPLIFIED: No separate thinking LLM call
        Just emit progress events during actual planning
        """
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
    
    # ========================================================================
    # PHASE 4: TOOL EXECUTION - FIXED VERSION!
    # ========================================================================
    
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
        """
        FIXED: Tool execution using TaskExecutor with streaming events
        
        Key changes:
        1. Use self.task_executor.execute_task_plan() for proper dependency handling
        2. Emit streaming events based on execution results
        3. Auto-detect dependencies and expand symbols work correctly
        """
        task_plan = plan_data.get('task_plan')
        
        if not task_plan or len(task_plan.tasks) == 0:
            yield {
                'all_tool_results': {},
                'tools_executed': [],
                'mode': None,
                'stats': {}
            }
            return
        
        self.logger.info(f"[{flow_id}] Executing {len(task_plan.tasks)} tasks via TaskExecutor...")
        
        # Build chat history for TaskExecutor
        chat_history = [
            [msg.get('content', ''), msg.get('role', 'user')]
            for msg in context_data['recent_chat'][-5:]
        ]
        
        # ====================================================================
        # CRITICAL: Use TaskExecutor for proper dependency handling!
        # This enables auto-detect dependencies and symbol expansion
        # ====================================================================
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
        
        self.logger.info(
            f"[{flow_id}] TaskExecutor completed: "
            f"{stats.get('completed_tasks', 0)}/{stats.get('total_tasks', 0)} tasks, "
            f"retries={stats.get('total_retries', 0)}"
        )
        
        # ====================================================================
        # Emit streaming events based on TaskExecutor results
        # ====================================================================
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
        """Get a short preview of tool result"""
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
    
    # ========================================================================
    # PHASE 5: CONTEXT ASSEMBLY
    # ========================================================================
    
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
        """
        Assemble context for LLM (NO LLM call)
        """
        all_tool_results = execution_data.get('all_tool_results', {})

        tool_outputs_dict = self._convert_to_tool_outputs(
            list(all_tool_results.values()) if isinstance(all_tool_results, dict) else all_tool_results
        )
        self.logger.info(f"[{flow_id}] [PHASE 5] ✓ Tool outputs: {len(tool_outputs_dict)} tools")
        
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
        self.logger.info(f"[{flow_id}] ✓ Summary included in context: {len(summary)} chars")

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
        """Extract unique symbols from tool outputs"""
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
    
    # ========================================================================
    # PHASE 6: RESPONSE GENERATION (streaming)
    # ========================================================================
    
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
        """Stream LLM response generation"""
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
        
        async for chunk in self.llm_provider.stream_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key,
            clean_thinking=True
        ):
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
    
    # ========================================================================
    # BACKGROUND TASKS
    # ========================================================================
    
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
        """Background memory updates (non-blocking)"""
        try:
            # Import memory update agent
            from src.agents.memory.memory_update_agent import MemoryUpdateAgent
            memory_update = MemoryUpdateAgent()
            
            # Update core memory
            update_result = await memory_update.analyze_for_updates(
                user_id=user_id,
                user_message=query,
                assistant_message=response,
                working_memory_context=working_memory_context,
                model_name=model_name,
                provider_type=provider_type
            )
            
            if update_result.get('updated'):
                self.logger.info(f"[{flow_id}] [BG] ✓ Core memory updated")
            
            # Check and create summary
            summary_result = await self.summary_manager.check_and_create_summary(
                session_id=session_id,
                user_id=user_id,
                organization_id=organization_id,
                model_name=model_name,
                provider_type=provider_type
            )
            
            if summary_result.get('created'):
                self.logger.info(f"[{flow_id}] [BG] ✓ Summary v{summary_result.get('version')}")
            
        except Exception as e:
            self.logger.error(f"[{flow_id}] [BG] Memory update error: {e}")


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

async def stream_chat_sse(
    handler: StreamingChatHandler,
    query: str,
    session_id: str,
    user_id: int,
    cancellation_token: CancellationToken = None,
    **kwargs
) -> AsyncGenerator[str, None]:
    """
    Convenience function to stream chat as SSE strings
    """
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