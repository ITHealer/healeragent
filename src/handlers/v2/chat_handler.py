# File: src/handlers/v2/chat_handler.py
"""
Chat Handler V2 - 7 Phase Pipeline with Working Memory

Architecture based on Claude AI / ChatGPT patterns:
- Working Memory for task-specific state (scratchpad)
- Core Memory for long-term user info
- Context Compaction for long conversations
- 31 Atomic Tools with semantic planning

Phases:
1. Load Context (core memory, summary, recent chat, working memory)
2. Planning (3-stage semantic planning with working memory)
3. Memory Search (recall, archival if needed)
4. Tool Execution (save results to working memory)
5. Context Assembly (NO LLM - organize data for response)
6. Response Generation (1 LLM call)
7. Post-processing (save, memory update, clear working memory)

LLM Calls per Request: ~3-5 total
- Phase 2: 1-2 (classify + plan)
- Phase 6: 1 (response generation)
- Phase 7: 0-1 (background memory update)
"""

import time
import asyncio
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional, Any

from src.utils.logger.custom_logging import LoggerMixin

# Core components
from src.agents.planning.planning_agent import PlanningAgent
from src.agents.action.task_executor import TaskExecutor
from src.agents.validation.validation_agent import ValidationAgent
from src.services.v2.tool_execution_service import ToolExecutionService
from src.services.memory_search_service import MemorySearchService
from src.helpers.context_assembler import ContextAssembler

# Memory components
from src.agents.memory.core_memory import CoreMemory
from src.agents.memory.recursive_summary import RecursiveSummaryManager
from src.agents.memory.memory_update_agent import MemoryUpdateAgent

# Working Memory - NEW
from src.agents.memory.working_memory_integration import (
    WorkingMemoryIntegration,
    setup_working_memory_for_request,
)

# Think Tool & Context Compaction
from src.services.think_tool_service import ThinkToolService
from src.services.context_management_service import ContextManagementService

# Tool Registry
from src.agents.tools.tool_loader import get_registry

# LLM and chat components
from src.helpers.llm_helper import LLMGeneratorProvider
from src.helpers.chat_management_helper import ChatService
from src.providers.provider_factory import ModelProviderFactory, ProviderType

# Database
from src.database.repository.sessions import SessionRepository
from src.helpers.analysis_insights_extractor import AnalysisInsightsExtractor

from src.agents.planning.task_models import TaskExecutionResult, TaskPlan
from src.utils.config import settings


class ChatHandler(LoggerMixin):
    """
    Main Chat Handler with 7-Phase Pipeline and Working Memory
    
    Architecture (Claude AI/ChatGPT Pattern):
    ┌─────────────────────────────────────────────────────────────┐
    │  Phase 1: Context Loading (Memory + History + Compaction)   │
    │  Phase 2: Planning (3-stage semantic + working memory)      │
    │  Phase 3: Memory Search (if needed)                         │
    │  Phase 4: Tool Execution (with simple retry)                │
    │  Phase 5: Context Assembly (NO LLM - just organize)         │
    │  Phase 6: LLM Response Generation                           │
    │  Phase 7: Post-Processing (Background)                      │
    └─────────────────────────────────────────────────────────────┘
    
    Key Features:
    - Working Memory as scratchpad for current task state
    - NO Replanning Agent (uses simple retry with backoff)
    - LLM in Phase 6 handles all analysis
    - Multilingual support (no hardcoded keywords)
    """
    
    # Configuration flags
    ENABLE_TASK_BASED_EXECUTION = True
    
    # Limits
    MAX_TASKS_PER_QUERY = 15
    MAX_RETRIES_PER_TASK = 2
    
    # Validation
    VALIDATION_CONFIDENCE_THRESHOLD = 0.6
    ENABLE_VALIDATION = True
    
    def __init__(self):
        super().__init__()
        
        # ====================================================================
        # TOOL REGISTRY (31 atomic tools)
        # ====================================================================
        try:
            self.tool_registry = get_registry()
            tool_count = len(self.tool_registry.get_all_tools())
            categories = list(self.tool_registry.get_summary()['categories'].keys())
            self.logger.info(
                f"✅ Tool Registry: {tool_count} tools in {len(categories)} categories"
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to load registry: {e}")
            self.tool_registry = None
        
        # ====================================================================
        # CORE AGENTS
        # ====================================================================
        self.planning_agent = PlanningAgent(
            model_name=settings.MODEL_DEFAULT,
            provider_type=settings.PROVIDER_DEFAULT
        )
        self.validation_agent = ValidationAgent()
        self.tool_execution_service = ToolExecutionService()
        
        # TaskExecutor with simple retry (no replanning)
        self.task_executor = TaskExecutor(
            tool_execution_service=self.tool_execution_service,
            validation_agent=self.validation_agent,
            max_retries=self.MAX_RETRIES_PER_TASK,
        )
        
        # ====================================================================
        # MEMORY COMPONENTS
        # ====================================================================
        self.memory_search = MemorySearchService()
        self.context_assembler = ContextAssembler()
        self.core_memory = CoreMemory()
        self.summary_manager = RecursiveSummaryManager()
        self.memory_update = MemoryUpdateAgent()
        
        # ====================================================================
        # THINK TOOL SERVICE (initialized per request)
        # ====================================================================
        self.think_service: Optional[ThinkToolService] = None
        
        # ====================================================================
        # CONTEXT MANAGEMENT SERVICE
        # ====================================================================
        self.context_manager = ContextManagementService(
            enable_compaction=True,
            compaction_threshold=100000,
            compaction_strategy="smart_summary"
        )
        
        # ====================================================================
        # OTHER SERVICES
        # ====================================================================
        self.llm_provider = LLMGeneratorProvider()
        self.chat_service = ChatService()
        self.session_repo = SessionRepository()
        self.insights_extractor = AnalysisInsightsExtractor()
        
        self.logger.info("✅ ChatHandler initialized with Working Memory support")
    
    # ========================================================================
    # MAIN ENTRY POINT
    # ========================================================================
    
    async def handle_chat_with_reasoning(
        self,
        query: str,
        chart_displayed: Optional[bool],
        session_id: str,
        user_id: str,
        model_name: str,
        provider_type: str,
        organization_id: Optional[str] = None,
        enable_thinking: bool = True,
        stream: bool = True,
        enable_think_tool: bool = False,
        enable_compaction: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Main chat handler - orchestrates all 7 phases with Working Memory
        
        Args:
            query: User query
            chart_displayed: Whether chart is displayed
            session_id: Session ID
            user_id: User ID
            model_name: LLM model name
            provider_type: LLM provider
            organization_id: Organization ID (optional)
            enable_thinking: Enable extended thinking
            stream: Enable streaming
            enable_think_tool: Enable Think Tool for complex reasoning
            enable_compaction: Enable context compaction
            
        Yields:
            Response chunks
        """
        
        flow_id = f"FLOW-{datetime.now().strftime('%H%M%S')}-{session_id[:8]}"
        flow_start = time.time()
        
        # ====================================================================
        # SETUP WORKING MEMORY FOR THIS REQUEST
        # ====================================================================
        wm_integration = setup_working_memory_for_request(
            session_id=session_id,
            user_id=user_id,
            flow_id=flow_id,
        )
        
        try:
            self._log_flow_start(
                flow_id, query, user_id, session_id, 
                model_name, provider_type, enable_think_tool
            )
            
            # Initialize Think Tool (per request)
            self.think_service = ThinkToolService(
                enabled=enable_think_tool,
                model_name=model_name
            )
            
            # ================================================================
            # PHASE 1: Load Context (with Working Memory)
            # ================================================================
            context_data = await self._phase1_load_context(
                flow_id=flow_id,
                user_id=user_id,
                session_id=session_id,
                enable_compaction=enable_compaction,
                wm_integration=wm_integration
            )
            
            # ================================================================
            # PHASE 2: Planning (with Working Memory)
            # ================================================================
            plan_data = await self._phase2_planning(
                flow_id=flow_id,
                query=query,
                context_data=context_data,
                wm_integration=wm_integration
            )
            
            # Check if no tools needed (conversational query)
            if not plan_data.get('task_plan') or len(plan_data['task_plan'].tasks) == 0:
                self.logger.info(f"[{flow_id}] No tools needed - conversational response")
                execution_data = {'all_tool_results': {}, 'tools_executed': [], 'mode': None}
                memory_data = {'recall_results': [], 'archival_results': []}
            else:
                # ================================================================
                # PHASE 3: Memory Search (if needed)
                # ================================================================
                memory_data = await self._phase3_memory_search(
                    flow_id=flow_id,
                    query=query,
                    plan_data=plan_data,
                    session_id=session_id,
                    user_id=user_id
                )
                
                # ================================================================
                # PHASE 4: Tool Execution (save to Working Memory)
                # ================================================================
                execution_data = await self._phase4_tool_execution(
                    flow_id=flow_id,
                    query=query,
                    plan_data=plan_data,
                    context_data=context_data,
                    provider_type=provider_type,
                    model_name=model_name,
                    wm_integration=wm_integration
                )
            
            # ================================================================
            # PHASE 5: Context Assembly (NO LLM - just organize data)
            # ================================================================
            assembled_context = await self._phase5_context_assembly(
                flow_id=flow_id,
                query=query,
                chart_displayed=chart_displayed,
                plan_data=plan_data,
                execution_data=execution_data,
                memory_data=memory_data,
                model_name=model_name,
                enable_thinking=enable_thinking,
                wm_integration=wm_integration
            )
            
            # ================================================================
            # PHASE 6: LLM Response Generation
            # ================================================================
            complete_response = ""
            async for chunk in self._phase6_generate_response(
                flow_id=flow_id,
                query=query,
                assembled_context=assembled_context,
                execution_data=execution_data,
                user_id=user_id,
                session_id=session_id,
                model_name=model_name,
                provider_type=provider_type,
                enable_thinking=enable_thinking
            ):
                complete_response += chunk
                yield chunk
            
            # ================================================================
            # PHASE 7: Post-Processing (cleanup Working Memory)
            # ================================================================
            await self._phase7_post_processing(
                flow_id=flow_id,
                query=query,
                response=complete_response,
                user_id=user_id,
                session_id=session_id,
                organization_id=organization_id,
                provider_type=provider_type,
                model_name=model_name,
                flow_start=flow_start,
                wm_integration=wm_integration
            )
            
        except Exception as e:
            self.logger.error(f"[{flow_id}] ✗ FLOW ERROR: {e}", exc_info=True)
            
            # Save error to working memory for potential recovery
            wm_integration.save_error(
                error_type="flow_error",
                message=str(e),
                recoverable=True
            )
            wm_integration.complete_request(clear_task_data=False)
            
            yield f"I encountered an error: {str(e)}"
    
    # ========================================================================
    # PHASE 1: CONTEXT LOADING (with Working Memory)
    # ========================================================================
    
    async def _phase1_load_context(
        self,
        flow_id: str,
        user_id: str,
        session_id: str,
        enable_compaction: bool,
        wm_integration: WorkingMemoryIntegration
    ) -> Dict[str, Any]:
        """
        Load all existing context including Working Memory
        
        Components loaded:
        - Core Memory (long-term user info)
        - Recursive Summary (compressed history)
        - Recent Chat History
        - Working Memory (current task state)
        - Context Compaction (if needed)
        
        LLM Calls: 0-1 (if compaction triggers summarization)
        """
        phase_start = time.time()
        self.logger.info(f"[{flow_id}] [PHASE 1] Loading Context...")
        
        # 1.1 Load Core Memory (long-term user info)
        core_memory = await self.core_memory.load_core_memory(user_id)
        core_memory_size = len(core_memory.get('persona', '')) + len(core_memory.get('human', ''))
        self.logger.info(f"[{flow_id}] ✓ Core Memory: {core_memory_size} chars")
        
        # 1.2 Load Recursive Summary (compressed history)
        summary = await self.summary_manager.get_active_summary(session_id)
        if summary:
            self.logger.info(f"[{flow_id}] ✓ Summary: {len(summary)} chars")
        
        # 1.3 Load Recent Chat History
        recent_chat = await self.session_repo.get_session_messages(
            session_id=session_id,
            limit=10
        )
        self.logger.info(f"[{flow_id}] ✓ Recent chat: {len(recent_chat)} messages")
        
        # 1.4 Get Working Memory context (current task state)
        working_memory_context = wm_integration.get_context_for_planning(max_tokens=1000)
        current_symbols = wm_integration.get_current_symbols()
        
        if current_symbols:
            self.logger.info(f"[{flow_id}] ✓ Working Memory: symbols={current_symbols}")
        
        # 1.5 Check Context Compaction
        compaction_result = None
        if enable_compaction and len(recent_chat) > 0:
            messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in recent_chat
            ]
            
            needs_compact, stats = self.context_manager.compressor.should_compact(messages)
            
            if needs_compact:
                self.logger.info(f"[{flow_id}] ⚠️ Context at {stats.usage_percent:.1f}% - Compacting...")
                
                # Extract symbols to preserve
                symbols_to_preserve = current_symbols or self._extract_symbols_heuristic(recent_chat)
                
                compaction_result = await self.context_manager.compact_now(
                    messages=messages,
                    preserve_keywords=symbols_to_preserve
                )
                
                if compaction_result.success:
                    recent_chat = [
                        {"role": m["role"], "content": m["content"]}
                        for m in compaction_result.preserved_messages
                    ]
                    self.logger.info(f"[{flow_id}] ✅ Compacted: {compaction_result.tokens_saved} tokens saved")
        
        phase_time = time.time() - phase_start
        self.logger.info(f"[{flow_id}] [PHASE 1] Complete ({phase_time:.2f}s)")
        
        return {
            'core_memory': core_memory,
            'summary': summary,
            'recent_chat': recent_chat,
            'working_memory_context': working_memory_context,
            'current_symbols': current_symbols,
            'compaction_result': compaction_result,
            'timing': phase_time
        }
    
    # ========================================================================
    # PHASE 2: PLANNING (with Working Memory)
    # ========================================================================
    
    async def _phase2_planning(
        self,
        flow_id: str,
        query: str,
        context_data: Dict[str, Any],
        wm_integration: WorkingMemoryIntegration
    ) -> Dict[str, Any]:
        """
        Plan tasks using 3-stage semantic planning with Working Memory
        
        Working Memory is used for:
        - Task continuation (if existing task in progress)
        - Symbol context from previous turns
        - Classification results saved for later use
        
        LLM Calls: 1-2 (classify + plan)
        """
        phase_start = time.time()
        self.logger.info(f"[{flow_id}] [PHASE 2] Planning...")
        
        # Format recent chat
        formatted_recent = [
            {
                'role': msg.get('role', 'user'),
                'content': msg.get('content', ''),
                'created_at': msg.get('created_at', '')
            }
            for msg in context_data['recent_chat']
        ]
        
        # Think before planning (if enabled)
        if self.think_service and self.think_service.is_enabled():
            symbols_in_context = context_data.get('current_symbols', [])
            if not symbols_in_context:
                symbols_in_context = await self._extract_symbols_with_llm(
                    recent_chat=context_data['recent_chat'],
                    model_name=settings.MODEL_DEFAULT,
                    provider_type=settings.PROVIDER_DEFAULT
                )
            await self.think_service.think_before_planning(
                query=query,
                history_context=str(formatted_recent[-3:]),
                symbols_in_context=symbols_in_context
            )
        
        # Execute 3-stage planning with Working Memory context
        task_plan = None
        language = 'vi'
        
        try:
            # Pass working memory context to planning agent
            task_plan = await self.planning_agent.think_and_plan(
                query=query,
                recent_chat=formatted_recent,
                core_memory=context_data['core_memory'],
                summary=context_data['summary'],
                working_memory_context=context_data.get('working_memory_context', '')
            )
            
            # Limit tasks
            if task_plan and len(task_plan.tasks) > self.MAX_TASKS_PER_QUERY:
                self.logger.warning(f"[{flow_id}] Limiting tasks: {len(task_plan.tasks)} → {self.MAX_TASKS_PER_QUERY}")
                task_plan.tasks = task_plan.tasks[:self.MAX_TASKS_PER_QUERY]
            
            if task_plan:
                self._log_task_plan(flow_id, task_plan)
                language = task_plan.response_language
                
                # Save classification and plan to Working Memory
                wm_integration.save_classification(
                    query_type=task_plan.query_intent,
                    categories=self._get_categories_from_plan(task_plan),
                    symbols=task_plan.symbols or [],
                    language=language,
                    reasoning=task_plan.reasoning
                )
                
                if task_plan.tasks:
                    wm_integration.save_plan(task_plan)
                
        except Exception as e:
            self.logger.error(f"[{flow_id}] Planning failed: {e}", exc_info=True)
            
            # Save error to working memory
            wm_integration.save_error(
                error_type="planning_error",
                message=str(e),
                recoverable=True
            )
            
            task_plan = TaskPlan(
                tasks=[],
                query_intent="error",
                strategy="direct_answer",
                response_language="auto",
                reasoning=f"Planning error: {str(e)}"
            )
        
        phase_time = time.time() - phase_start
        self.logger.info(f"[{flow_id}] [PHASE 2] Complete ({phase_time:.2f}s)")
        
        return {
            'task_plan': task_plan,
            'language': language,
            'timing': phase_time,
            'symbols': task_plan.symbols if task_plan and hasattr(task_plan, 'symbols') else [],
            'query_intent': task_plan.query_intent if task_plan else '',
            'strategy': task_plan.strategy if task_plan else ''
        }
    
    # ========================================================================
    # PHASE 3: MEMORY SEARCH
    # ========================================================================
    
    async def _phase3_memory_search(
        self,
        flow_id: str,
        query: str,
        plan_data: Dict[str, Any],
        session_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Execute memory searches based on plan
        
        LLM Calls: 0 (vector search only)
        """
        phase_start = time.time()
        
        # Memory search is handled by tool executor if needed
        # This phase just prepares the context
        
        return {
            'recall_results': [],
            'archival_results': [],
            'timing': time.time() - phase_start
        }
    
    # ========================================================================
    # PHASE 4: TOOL EXECUTION (with Working Memory)
    # ========================================================================
    
    async def _phase4_tool_execution(
        self,
        flow_id: str,
        query: str,
        plan_data: Dict[str, Any],
        context_data: Dict[str, Any],
        provider_type: str,
        model_name: str,
        wm_integration: WorkingMemoryIntegration
    ) -> Dict[str, Any]:
        """
        Execute tools and save results to Working Memory
        
        Working Memory is used for:
        - Storing intermediate tool results
        - Tracking execution progress
        - Error recovery context
        
        LLM Calls: 0 (external API calls only)
        """
        phase_start = time.time()
        
        task_plan = plan_data.get('task_plan')
        
        if not task_plan or len(task_plan.tasks) == 0:
            return {
                'all_tool_results': {},
                'tools_executed': [],
                'mode': None,
                'stats': {},
                'timing': 0
            }
        
        self.logger.info(f"[{flow_id}] [PHASE 4] Executing {len(task_plan.tasks)} tasks...")
        
        # Build chat history
        chat_history = [
            [msg.get('content', ''), msg.get('role', 'user')]
            for msg in context_data['recent_chat'][-5:]
        ]
        
        # Execute task plan (TaskExecutor handles retry internally)
        execution_result = await self.task_executor.execute_task_plan(
            plan=task_plan,
            query=query,
            chat_history=chat_history,
            system_language=plan_data['language'],
            provider_type=provider_type,
            model_name=model_name,
            flow_id=flow_id
        )
        
        task_results = execution_result['task_results']
        stats = execution_result['stats']
        
        self._log_execution_stats(flow_id, stats)
        
        # Aggregate results and save to Working Memory
        all_tool_results = {}
        tools_executed = []
        
        if isinstance(task_results, list):
            for task_result in task_results:
                task_id = getattr(task_result, 'task_id', 'unknown')
                is_success = getattr(task_result, 'success', False)
                task_data = getattr(task_result, 'data', {})
                task_tools = getattr(task_result, 'tools_executed', [])
                execution_time = getattr(task_result, 'execution_time', 0)
                
                if is_success and task_data:
                    all_tool_results[f"task_{task_id}"] = task_data
                    tools_executed.extend(task_tools)
                    
                    # Save each tool result to Working Memory
                    for tool_name in task_tools:
                        tool_output = task_data.get(tool_name, task_data)
                        wm_integration.save_tool_result(
                            tool_name=tool_name,
                            result=tool_output,
                            task_id=task_id,
                            execution_time_ms=int(execution_time * 1000),
                            status="success" if is_success else "failed"
                        )
                
                # Think about tool output (if enabled)
                if self.think_service and self.think_service.is_enabled() and task_data:
                    await self.think_service.think_about_tool_output(
                        tool_name=task_tools[0] if task_tools else "unknown",
                        tool_output=task_data,
                        expected_data=[]
                    )
        
        phase_time = time.time() - phase_start
        self.logger.info(f"[{flow_id}] [PHASE 4] Complete ({phase_time:.2f}s)")
        
        return {
            'all_tool_results': all_tool_results,
            'tools_executed': list(set(tools_executed)),
            'mode': 'task_based',
            'stats': stats,
            'timing': phase_time
        }
    
    # ========================================================================
    # PHASE 5: CONTEXT ASSEMBLY (NO LLM - Just Organize Data)
    # ========================================================================
    
    async def _phase5_context_assembly(
        self,
        flow_id: str,
        query: str,
        chart_displayed: Optional[bool],
        plan_data: Dict[str, Any],
        execution_data: Dict[str, Any],
        memory_data: Dict[str, Any],
        model_name: str,
        enable_thinking: bool,
        wm_integration: WorkingMemoryIntegration
    ) -> Dict[str, Any]:
        """
        Context Assembly - Organize data for LLM (NO LLM call here)
        
        This phase:
        1. Builds system prompt
        2. Formats tool results
        3. Adds memory context
        4. Adds Working Memory context (reasoning, symbols)
        5. Formats everything for LLM consumption
        
        LLM Calls: 0 (just data organization)
        """
        phase_start = time.time()
        self.logger.info(f"[{flow_id}] [PHASE 5] Context Assembly...")
        
        all_tool_results = execution_data.get('all_tool_results', {})
        language = plan_data.get('language', 'vi')
        plan_symbols = plan_data.get('symbols', [])
        query_intent = plan_data.get('query_intent', '')
        
        recall_results = memory_data.get('recall_results', [])
        archival_results = memory_data.get('archival_results', [])
        
        # 5.1: Base System Prompt
        from src.helpers.system_prompts import get_system_message_general_chat
        
        system_prompt = get_system_message_general_chat(
            enable_thinking=enable_thinking,
            model_name=model_name,
            detected_language=language,
            chart_displayed=chart_displayed
        )
        
        # 5.2: Add Memory Context (if any)
        if recall_results or archival_results:
            try:
                search_context = await self.memory_search.format_search_results_for_context(
                    recall_results=recall_results,
                    archival_results=archival_results
                )
                if search_context:
                    system_prompt += f"\n\n{search_context}"
            except Exception as e:
                self.logger.warning(f"[{flow_id}] Memory context error: {e}")
        
        # 5.3: Format Tool Results (NO LLM - direct formatting)
        tool_outputs_dict = self._convert_to_tool_outputs(
            list(all_tool_results.values()) if isinstance(all_tool_results, dict) else all_tool_results
        )
        
        self.logger.info(f"[{flow_id}] [PHASE 5] ✓ Tool outputs: {len(tool_outputs_dict)} tools")
        
        # 5.4: Format Tool Context
        tool_context = ""
        if tool_outputs_dict:
            tool_context = self._format_tool_context_for_llm(tool_outputs_dict, language)
            if tool_context:
                system_prompt += f"\n\n{tool_context}"
                self.logger.info(f"[{flow_id}] [PHASE 5] ✓ Tool context: {len(tool_context)} chars")
        
        # 5.5: Symbol Awareness (prevent symbol confusion)
        symbols = self._extract_symbols_from_results(tool_outputs_dict)
        active_symbols = plan_symbols if plan_symbols else symbols
        
        if active_symbols:
            symbol_awareness = f"""
{'='*70}
⚠️ CURRENT QUERY SYMBOLS: {', '.join(active_symbols)}
Query Intent: {query_intent}
IMPORTANT: Analyze ONLY these symbol(s). Do NOT confuse with history!
{'='*70}
"""
            system_prompt += symbol_awareness
        
        phase_time = time.time() - phase_start
        self.logger.info(f"[{flow_id}] [PHASE 5] Complete ({phase_time:.2f}s)")
        
        return {
            'system_prompt': system_prompt,
            'tool_context': tool_context,
            'symbols': active_symbols
        }
    
    # ========================================================================
    # PHASE 6: LLM RESPONSE GENERATION
    # ========================================================================
    
    async def _phase6_generate_response(
        self,
        flow_id: str,
        query: str,
        assembled_context: Dict[str, Any],
        execution_data: Dict[str, Any],
        user_id: str,
        session_id: str,
        model_name: str,
        provider_type: str,
        enable_thinking: bool
    ) -> AsyncGenerator[str, None]:
        """
        Generate and stream LLM response
        
        LLM Calls: 1
        
        This is where the actual analysis happens.
        The system prompt + tool context guide the LLM.
        """
        phase_start = time.time()
        self.logger.info(f"[{flow_id}] [PHASE 6] Generating response...")
        
        system_prompt = assembled_context['system_prompt']
        has_tool_results = bool(execution_data.get('all_tool_results'))
        
        # Limit history if we have fresh tool data
        max_history = 2 if has_tool_results else 10
        
        # Prepare messages
        messages, llm_metadata = await self.context_assembler.prepare_messages_for_llm(
            user_id=user_id,
            session_id=session_id,
            current_query=query,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            model_name=model_name,
            max_history_messages=max_history
        )
        
        self.logger.info(f"[{flow_id}] ✓ Messages: {llm_metadata.get('llm_messages_tokens', 0)} tokens")
        
        # Get API key
        api_key = ModelProviderFactory._get_api_key(provider_type)
        
        # Stream response
        async for chunk in self.llm_provider.stream_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key,
            clean_thinking=True
        ):
            yield chunk
        
        phase_time = time.time() - phase_start
        self.logger.info(f"[{flow_id}] [PHASE 6] Complete ({phase_time:.2f}s)")
    
    # ========================================================================
    # PHASE 7: POST-PROCESSING (with Working Memory cleanup)
    # ========================================================================
    
    async def _phase7_post_processing(
        self,
        flow_id: str,
        query: str,
        response: str,
        user_id: str,
        session_id: str,
        organization_id: Optional[str],
        provider_type: str,
        model_name: str,
        flow_start: float,
        wm_integration: WorkingMemoryIntegration
    ) -> None:
        """
        Save conversation and trigger background tasks
        
        Working Memory cleanup:
        - Clear task-specific data (plan, tool outputs)
        - Keep symbols for potential follow-up queries
        
        LLM Calls: 0-1 (background - memory update only)
        """
        phase_start = time.time()
        self.logger.info(f"[{flow_id}] [PHASE 7] Post-Processing...")
        
        # Save conversation
        try:
            question_id = self.chat_service.save_user_question(
                session_id=session_id,
                created_at=datetime.now(),
                created_by=user_id,
                content=query
            )
            
            response_time = time.time() - flow_start
            
            self.chat_service.save_assistant_response(
                session_id=session_id,
                created_at=datetime.now(),
                question_id=question_id,
                content=response,
                response_time=response_time
            )
            
            self.logger.info(f"[{flow_id}] ✓ Conversation saved")
        except Exception as e:
            self.logger.error(f"[{flow_id}] ✗ Save failed: {e}")
        
        # Clear Working Memory task data (keep symbols for follow-up)
        wm_integration.complete_request(clear_task_data=True)
        
        # Log Working Memory stats
        wm_stats = wm_integration.get_stats()
        self.logger.info(f"[{flow_id}] ✓ Working Memory: {wm_stats}")
        
        # Background tasks
        asyncio.create_task(
            self._background_memory_updates(
                flow_id=flow_id,
                user_id=user_id,
                session_id=session_id,
                query=query,
                response=response,
                organization_id=organization_id,
                provider_type=provider_type,
                model_name=model_name
            )
        )
        
        # Log Think Tool stats
        if self.think_service:
            stats = self.think_service.get_think_stats()
            if stats['total_calls'] > 0:
                self.logger.info(f"[{flow_id}] 🧠 Think Tool: {stats['total_calls']} calls")
        
        total_time = time.time() - flow_start
        self.logger.info(f"[{flow_id}] ========== FLOW COMPLETE ({total_time:.2f}s) ==========")
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def get_available_tools(self) -> Dict[str, List[str]]:
        """Get all available atomic tools from registry"""
        if not self.tool_registry:
            return {}
        return self.tool_registry.list_tools()
    
    def _get_categories_from_plan(self, task_plan: TaskPlan) -> List[str]:
        """Extract unique tool categories from task plan"""
        categories = set()
        for task in task_plan.tasks:
            for tool_call in task.tools_needed:
                tool_name = tool_call.tool_name
                # Map tool name to category
                if self.tool_registry:
                    schema = self.tool_registry.get_schema(tool_name)
                    if schema and schema.category:
                        categories.add(schema.category)
        return list(categories)
    
    async def _extract_symbols_with_llm(
        self,
        recent_chat: List[Dict],
        model_name: str = None,
        provider_type: str = None
    ) -> List[str]:
        """
        Extract stock/crypto symbols from recent chat using LLM
        
        This is more accurate than regex because:
        - Handles variable length symbols (BTC, BTCUSDT, NVDA, etc.)
        - Understands context (not just pattern matching)
        - Works for any language
        """
        if not recent_chat:
            return []
        
        # Build conversation text from last 5 messages
        conversation_text = ""
        for msg in recent_chat[-5:]:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if content:
                conversation_text += f"{role}: {content}\n"
        
        if not conversation_text.strip():
            return []
        
        # Use default model if not specified
        model = model_name or settings.MODEL_DEFAULT
        provider = provider_type or settings.PROVIDER_DEFAULT
        
        prompt = f"""Extract all stock ticker symbols and cryptocurrency symbols from this conversation.

RULES:
- Stock symbols: Usually 1-5 uppercase letters (AAPL, NVDA, TSLA, VNM, FPT, VIC)
- Crypto symbols: Can be longer (BTC, ETH, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT)
- Only extract actual trading symbols, not regular words
- Return ONLY the symbols, comma-separated, no explanations
- If no symbols found, return "NONE"

CONVERSATION:
{conversation_text}

SYMBOLS (comma-separated):"""

        try:
            api_key = ModelProviderFactory._get_api_key(provider)
            
            response = ""
            async for chunk in self.llm_provider.stream_response(
                model_name=model,
                messages=[{"role": "user", "content": prompt}],
                provider_type=provider,
                api_key=api_key,
                max_tokens=100,
                temperature=0.0
            ):
                response += chunk
            
            response = response.strip().upper()
            
            if response == "NONE" or not response:
                return []
            
            symbols = []
            for symbol in response.split(','):
                symbol = symbol.strip()
                if symbol and symbol.isalnum() and 1 <= len(symbol) <= 15:
                    symbols.append(symbol)
            
            self.logger.debug(f"[SYMBOL_EXTRACT] Found symbols: {symbols}")
            return symbols[:10]
            
        except Exception as e:
            self.logger.warning(f"[SYMBOL_EXTRACT] LLM extraction failed: {e}, using fallback")
            return self._extract_symbols_heuristic(recent_chat)
    
    def _extract_symbols_heuristic(self, recent_chat: List[Dict]) -> List[str]:
        """Fallback: Extract symbols using heuristics (no LLM)"""
        import re
        
        symbols = set()
        symbol_pattern = r'\b[A-Z][A-Z0-9]{0,14}\b'
        
        exclude = {
            'I', 'A', 'THE', 'AND', 'OR', 'NOT', 'IS', 'ARE', 'TO', 'OF', 
            'IN', 'FOR', 'ON', 'AT', 'BY', 'WITH', 'FROM', 'UP', 'DOWN',
            'BE', 'IT', 'AS', 'AN', 'IF', 'SO', 'NO', 'YES', 'OK',
            'API', 'URL', 'HTTP', 'HTTPS', 'JSON', 'XML', 'HTML', 'CSS',
            'USD', 'EUR', 'VND', 'JPY', 'GBP',
            'VA', 'VE', 'CO', 'LA', 'MA', 'NE', 'SE', 'TA', 'TU',
        }
        
        for msg in recent_chat[-5:]:
            content = msg.get('content', '')
            matches = re.findall(symbol_pattern, content)
            for match in matches:
                if match not in exclude and len(match) >= 2:
                    symbols.add(match)
        
        return list(symbols)[:10]
    
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
    
    def _format_tool_context_for_llm(
        self,
        tool_outputs: Dict[str, Any],
        language: str = 'en'
    ) -> str:
        """Format tool context for LLM consumption"""
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
            
            # Priority 3: raw data
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
    # LOGGING HELPERS
    # ========================================================================
    
    def _log_flow_start(self, flow_id, query, user_id, session_id, model_name, provider_type, enable_think_tool=False):
        """Log flow start information"""
        self.logger.info(f"[{flow_id}] ========== STARTING FLOW ==========")
        self.logger.info(f"[{flow_id}] Query: '{query[:100]}...'")
        self.logger.info(f"[{flow_id}] User: {user_id}, Session: {session_id[:8]}")
        self.logger.info(f"[{flow_id}] Model: {model_name}, Provider: {provider_type}")
        self.logger.info(f"[{flow_id}] Think Tool: {'ON' if enable_think_tool else 'OFF'}")
        
        if self.tool_registry:
            summary = self.tool_registry.get_summary()
            self.logger.info(f"[{flow_id}] Tools: {summary['total_tools']} in {len(summary['categories'])} categories")
    
    def _log_task_plan(self, flow_id, task_plan):
        """Log task plan details"""
        self.logger.info(f"[{flow_id}] ── TASK PLAN ──")
        self.logger.info(f"[{flow_id}] Intent: {task_plan.query_intent}")
        self.logger.info(f"[{flow_id}] Strategy: {task_plan.strategy}")
        self.logger.info(f"[{flow_id}] Tasks: {len(task_plan.tasks)}")
        
        for idx, task in enumerate(task_plan.tasks, 1):
            tools = [t.tool_name for t in task.tools_needed]
            self.logger.info(f"[{flow_id}]   {idx}. {task.description[:40]}... → {tools}")
    
    def _log_execution_stats(self, flow_id, stats):
        """Log execution statistics"""
        self.logger.info(
            f"[{flow_id}] Execution: "
            f"completed={stats.get('completed_tasks', 0)}/{stats.get('total_tasks', 0)}, "
            f"failed={stats.get('failed_tasks', 0)}, "
            f"retries={stats.get('total_retries', 0)}"
        )
    
    # ========================================================================
    # BACKGROUND TASKS
    # ========================================================================
    
    async def _background_memory_updates(
        self,
        flow_id: str,
        user_id: str,
        session_id: str,
        query: str,
        response: str,
        organization_id: Optional[str],
        provider_type: str,
        model_name: str
    ):
        """
        Background memory updates
        
        LLM Calls: 0-1 (memory update only)
        """
        try:
            # Update core memory (1 LLM call for extraction)
            update_result = await self.memory_update.analyze_for_updates(
                user_id=user_id,
                user_message=query,
                assistant_message=response,
                model_name=model_name,
                provider_type=provider_type
            )
            
            if update_result.get('updated'):
                self.logger.info(f"[{flow_id}] [BG] ✓ Core memory updated")
            
            # Check and create summary (0-1 LLM call if threshold met)
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
            self.logger.error(f"[{flow_id}] [BG] Error: {e}")