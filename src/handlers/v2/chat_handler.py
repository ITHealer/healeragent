

# Task Decomposition System - Based on Claude AI and Dexter's task-based planning approach.
import time
import asyncio
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional, Any

from src.utils.logger.custom_logging import LoggerMixin

# Core components
from src.agents.planning.planning_agent import PlanningAgent
from src.agents.reasoning.inner_thoughts_agent import InnerThoughtsAgent
from src.agents.action.task_executor import TaskExecutor
from src.agents.validation.validation_agent import ValidationAgent
from src.services.v2.tool_execution_service import ToolExecutionService

from src.services.memory_search_service import MemorySearchService
from src.helpers.context_assembler import ContextAssembler

# Memory components
from src.agents.memory.core_memory import CoreMemory
from src.agents.memory.recursive_summary import RecursiveSummaryManager
from src.agents.memory.memory_update_agent import MemoryUpdateAgent

# LLM and chat components
from src.helpers.llm_helper import LLMGeneratorProvider
from src.helpers.chat_management_helper import ChatService
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.helpers.system_prompts import get_system_message_general_chat

# Database
from src.database.repository.sessions import SessionRepository
from src.helpers.analysis_insights_extractor import AnalysisInsightsExtractor

from src.helpers.data_formatter import FinancialDataFormatter
from src.agents.planning.task_models import TaskExecutionResult
from src.agents.tools.base import ToolOutput
from src.agents.planning.task_models import TaskPlan
from src.utils.config import settings

class ChatHandler(LoggerMixin):
    
    # Available tools
    AVAILABLE_TOOLS = [
        'showStockPrice',
        'showStockFinancials',
        'showStockChart',
        'showStockNews',
        'cryptoChart',
        'showMarketOverview',
        'showTrendingStocks',
        'showStockHeatmap'
    ]
    
    ENABLE_TASK_BASED_EXECUTION = True           # Master switch
    ENABLE_TASK_DECOMPOSITION = True             # Use complex planning
    ENABLE_REPLANNING = True                     # Enable 3-level replanning
    
    # Task execution limits
    MAX_TASKS_PER_QUERY = 7                      # Limit complexity
    MAX_RETRIES_PER_TASK = 2                     # Per-task retries
    TASK_VALIDATION_CONFIDENCE_THRESHOLD = 0.6   # Min confidence to accept
    
    # Fallback configuration
    FALLBACK_TO_SIMPLE_ON_ERROR = True          # Use simple plan if planning fails
    MAX_TOOLS_PER_QUERY = 5                     # For fallback mode
    
    # Validation configuration (for fallback mode)
    VALIDATION_CONFIDENCE_THRESHOLD = 0.6
    ENABLE_VALIDATION = True
    ENABLE_QUICK_CHECK = True
    
    def __init__(self):
        super().__init__()
        
        # Initialize Planning Agent
        self.planning_agent = PlanningAgent(
            model_name=settings.MODEL_DEFAULT,
            provider_type=settings.PROVIDER_DEFAULT
        )

        # Initialize Validation Agent
        self.validation_agent = ValidationAgent()

        # Initialize Tool Execution Service
        self.tool_execution_service = ToolExecutionService()
        
        # Initialize Task Executor with Replanning support
        self.task_executor = TaskExecutor(
            tool_execution_service=self.tool_execution_service,
            planning_agent=self.planning_agent,
            validation_agent=self.validation_agent,
            max_retries_per_task=self.MAX_RETRIES_PER_TASK,
            enable_replanning=self.ENABLE_REPLANNING
        )

        # Initialize Inner Thoughts to fallback
        self.inner_thoughts = InnerThoughtsAgent(
            model_name="gpt-oss:20b",
            provider_type=ProviderType.OLLAMA
        )
        
        # Initialize other services
        self.memory_search = MemorySearchService()
        self.context_assembler = ContextAssembler()
        self.core_memory = CoreMemory()
        self.summary_manager = RecursiveSummaryManager()
        self.memory_update = MemoryUpdateAgent()
        self.llm_provider = LLMGeneratorProvider()
        self.chat_service = ChatService()
        self.session_repo = SessionRepository()
        self.insights_extractor = AnalysisInsightsExtractor()
    
    def _extract_tool_data(self, tool_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract data from tool result - handles multiple data key names
        
        Args:
            tool_result: Tool result dict
            
        Returns:
            Extracted data dict or None
        """
        # Try multiple possible keys
        data = (
            tool_result.get('data') or
            tool_result.get('raw_data') or  
            tool_result.get('result') or
            tool_result.get('response')
        )
        
        if not data:
            self.logger.debug(
                f"[EXTRACT] No data found in keys: {list(tool_result.keys())}"
            )
            return None
        
        self.logger.debug(
            f"[EXTRACT] Found data: {len(str(data))} chars"
        )
        
        return data
    

    async def handle_chat_with_reasoning(
        self,
        query: str,
        session_id: str,
        user_id: str,
        model_name: str,
        provider_type: str,
        organization_id: Optional[str] = None,
        enable_thinking: bool = True,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Main chat handler - orchestrates all phases
        
        CLEAN ARCHITECTURE: Each phase is a separate method
        """
        
        flow_id = f"FLOW-{datetime.now().strftime('%H%M%S')}-{session_id[:8]}"
        flow_start = time.time()
        
        try:
            self._log_flow_start(flow_id, query, user_id, session_id, model_name, provider_type)
            
            # ════════════════════════════════════════════════════════════
            # PHASE 1: Load Context
            # ════════════════════════════════════════════════════════════
            context_data = await self._phase1_load_context(
                flow_id=flow_id,
                user_id=user_id,
                session_id=session_id
            )
            
            # ════════════════════════════════════════════════════════════
            # PHASE 2: Planning/Reasoning
            # ════════════════════════════════════════════════════════════
            plan_data = await self._phase2_planning(
                flow_id=flow_id,
                query=query,
                context_data=context_data
            )
            
            # ════════════════════════════════════════════════════════════
            # PHASE 3: Memory Search (if needed)
            # ════════════════════════════════════════════════════════════
            memory_data = await self._phase3_memory_search(
                flow_id=flow_id,
                query=query,
                plan_data=plan_data,
                session_id=session_id,
                user_id=user_id
            )
            
            # ════════════════════════════════════════════════════════════
            # PHASE 4: Tool Execution
            # ════════════════════════════════════════════════════════════
            execution_data = await self._phase4_tool_execution(
                flow_id=flow_id,
                query=query,
                plan_data=plan_data,
                context_data=context_data,
                provider_type=provider_type,
                model_name=model_name
            )
            
            # ════════════════════════════════════════════════════════════
            # PHASE 5: Synthesis & Context Assembly
            # ════════════════════════════════════════════════════════════
            assembled_context = await self._phase5_synthesis_and_assembly(
                flow_id=flow_id,
                query=query,
                plan_data=plan_data,
                execution_data=execution_data,
                memory_data=memory_data,
                model_name=model_name,
                enable_thinking=enable_thinking
            )
            
            # ════════════════════════════════════════════════════════════
            # PHASE 6: LLM Response Generation
            # ════════════════════════════════════════════════════════════
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
            
            # ════════════════════════════════════════════════════════════
            # PHASE 7: Post-Processing
            # ════════════════════════════════════════════════════════════
            await self._phase7_post_processing(
                flow_id=flow_id,
                query=query,
                response=complete_response,
                user_id=user_id,
                session_id=session_id,
                organization_id=organization_id,
                provider_type=provider_type,
                model_name=model_name,
                flow_start=flow_start
            )
            
        except Exception as e:
            self.logger.error(f"[{flow_id}] ✗ FLOW ERROR: {e}", exc_info=True)
            yield f"I encountered an error: {str(e)}"
    
    # ========================================================================
    # PHASE 1: CONTEXT LOADING
    # ========================================================================
    
    async def _phase1_load_context(
        self,
        flow_id: str,
        user_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Load all existing context: core memory, summary, chat history
        
        Returns:
            {
                'core_memory': dict,
                'summary': str,
                'recent_chat': list,
                'timing': float
            }
        """
        
        phase_start = time.time()
        self.logger.info(f"[{flow_id}] [PHASE 1] Loading Context...")
        
        # Load Core Memory
        self.logger.info(f"[{flow_id}] [PHASE 1.1] Loading Core Memory...")
        core_memory = await self.core_memory.load_core_memory(user_id)
        core_memory_size = len(core_memory.get('persona', '')) + len(core_memory.get('human', ''))
        self.logger.info(f"[{flow_id}] ✓ Core Memory loaded: {core_memory_size} chars")
        
        # Load Recursive Summary
        self.logger.info(f"[{flow_id}] [PHASE 1.2] Loading Recursive Summary...")
        summary = await self.summary_manager.get_active_summary(session_id)
        has_summary = bool(summary and len(summary.strip()) > 20)
        if has_summary:
            self.logger.info(f"[{flow_id}] ✓ Summary found: {len(summary)} chars")
        else:
            self.logger.info(f"[{flow_id}] ✓ No summary exists for this session")
        
        # Load Recent Chat History
        self.logger.info(f"[{flow_id}] [PHASE 1.3] Loading Recent Chat History...")
        recent_chat = await self.session_repo.get_session_messages(
            session_id=session_id,
            limit=10
        )
        self.logger.info(f"[{flow_id}] ✓ Recent chat loaded: {len(recent_chat)} messages")
        
        phase_time = time.time() - phase_start
        self.logger.info(f"[{flow_id}] [PHASE 1] Context Loading Complete ({phase_time:.2f}s)")
        
        return {
            'core_memory': core_memory,
            'summary': summary,
            'recent_chat': recent_chat,
            'timing': phase_time
        }
    
    # ========================================================================
    # PHASE 2: PLANNING/REASONING
    # ========================================================================
    
    async def _phase2_planning(
        self,
        flow_id: str,
        query: str,
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Plan tasks or make decisions
        
        Returns:
            {
                'task_plan': TaskPlan or None,
                'decision': Decision or None,
                'language': str,
                'timing': float
            }
        """
        
        phase_start = time.time()
        self.logger.info(f"[{flow_id}] [PHASE 2] Starting Reasoning...")
        
        # Format recent chat
        formatted_recent = []
        for msg in context_data['recent_chat']:
            formatted_recent.append({
                'role': msg.get('role', 'user'),
                'content': msg.get('content', ''),
                'created_at': msg.get('created_at', '')
            })
        
        task_plan = None
        decision = None
        language = 'vi'
        
        if self.ENABLE_TASK_BASED_EXECUTION:
            # Task Planning
            try:
                self.logger.info(f"[{flow_id}] Using Task-Based Planning...")
                
                task_plan = await self.planning_agent.think_and_plan(
                    query=query,
                    recent_chat=formatted_recent,
                    core_memory=context_data['core_memory'],
                    summary=context_data['summary']
                )
                
                # Limit tasks
                if len(task_plan.tasks) > self.MAX_TASKS_PER_QUERY:
                    self.logger.warning(
                        f"[{flow_id}] Limiting tasks from {len(task_plan.tasks)} "
                        f"to {self.MAX_TASKS_PER_QUERY}"
                    )
                    task_plan.tasks = task_plan.tasks[:self.MAX_TASKS_PER_QUERY]
                
                # Log plan
                self._log_task_plan(flow_id, task_plan)
                
                language = task_plan.response_language
                
            except Exception as e:
                self.logger.error(f"[{flow_id}] Task planning failed: {e}")
                if self.FALLBACK_TO_SIMPLE_ON_ERROR:
                    self.logger.info(f"[{flow_id}] Falling back to simple approach...")
                    task_plan = None
                else:
                    raise
        
        if task_plan is None:
            # Fallback: Inner Thoughts
            self.logger.info(f"[{flow_id}] Using Inner Thoughts...")
            
            decision = await self.inner_thoughts.think_and_decide(
                query=query,
                recent_chat=formatted_recent,
                core_memory=context_data['core_memory'],
                summary=context_data['summary'],
                available_tools=self.AVAILABLE_TOOLS
            )
            
            self._log_inner_thoughts(flow_id, decision)
            
            if hasattr(decision.response_strategy, 'language'):
                language = decision.response_strategy.language
        
        phase_time = time.time() - phase_start
        self.logger.info(f"[{flow_id}] [PHASE 2] Reasoning Complete ({phase_time:.2f}s)")
        
        return {
            'task_plan': task_plan,
            'decision': decision,
            'language': language,
            'timing': phase_time
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
        Execute memory searches if needed
        
        Returns:
            {
                'recall_results': list,
                'archival_results': list,
                'timing': float
            }
        """
        
        phase_start = time.time()
        
        decision = plan_data.get('decision')
        need_memory_search = False
        
        if decision:
            need_memory_search = (
                decision.memory_decision.need_recall_search or 
                decision.memory_decision.need_archival_search
            )
        
        recall_results = []
        archival_results = []
        
        if need_memory_search:
            self.logger.info(f"[{flow_id}] [PHASE 3] Starting Memory Search...")
            
            search_results = await self.memory_search.execute_memory_searches(
                query=query,
                recall_params=decision.memory_decision.recall_params,
                archival_query=decision.memory_decision.archival_query,
                need_recall=decision.memory_decision.need_recall_search,
                need_archival=decision.memory_decision.need_archival_search,
                session_id=session_id,
                user_id=user_id
            )
            
            recall_results = search_results.get('recall_results', [])
            archival_results = search_results.get('archival_results', [])
            
            self.logger.info(f"[{flow_id}] ✓ Recall: {len(recall_results)}, Archival: {len(archival_results)}")
        
        phase_time = time.time() - phase_start
        
        return {
            'recall_results': recall_results,
            'archival_results': archival_results,
            'timing': phase_time
        }
    
    # ========================================================================
    # PHASE 4: TOOL EXECUTION
    # ========================================================================
    
    async def _phase4_tool_execution(
        self,
        flow_id: str,
        query: str,
        plan_data: Dict[str, Any],
        context_data: Dict[str, Any],
        provider_type: str,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Execute tools based on plan
        
        Returns:
            {
                'all_tool_results': dict,
                'tools_executed': list,
                'mode': str ('task_based' or 'original'),
                'stats': dict,
                'timing': float
            }
        """
        
        phase_start = time.time()
        
        task_plan = plan_data.get('task_plan')
        decision = plan_data.get('decision')
        
        all_tool_results = {}
        tools_executed = []
        mode = None
        stats = {}
        
        if task_plan and len(task_plan.tasks) > 0:
            # Task-Based Execution
            self.logger.info(f"[{flow_id}] [PHASE 4] Starting Task-Based Execution...")
            
            execution_result = await self._execute_task_based(
                flow_id=flow_id,
                task_plan=task_plan,
                query=query,
                context_data=context_data,
                language=plan_data['language'],
                provider_type=provider_type,
                model_name=model_name
            )
            
            all_tool_results = execution_result['all_tool_results']
            tools_executed = execution_result['tools_executed']
            stats = execution_result['stats']
            mode = 'task_based'
            
        elif decision and decision.tool_decision.need_tool:
            # Original Tool Execution
            self.logger.info(f"[{flow_id}] [PHASE 4] Starting Original Tool Execution...")
            
            tool_results = await self._execute_tools_with_validation(
                flow_id=flow_id,
                decision=decision,
                query=query,
                recent_chat=context_data['recent_chat'],
                provider_type=provider_type,
                model_name=model_name
            )
            
            all_tool_results = tool_results['all_tool_results']
            tools_executed = tool_results['tools_executed']
            mode = 'original'
        
        phase_time = time.time() - phase_start
        
        if all_tool_results:
            self.logger.info(f"[{flow_id}] [PHASE 4] Tool Execution Complete ({phase_time:.2f}s)")
        
        return {
            'all_tool_results': all_tool_results,
            'tools_executed': tools_executed,
            'mode': mode,
            'stats': stats,
            'timing': phase_time
        }
    
    async def _execute_task_based(
        self,
        flow_id: str,
        task_plan: Any,
        query: str,
        context_data: Dict[str, Any],
        language: str,
        provider_type: str,
        model_name: str
    ) -> Dict[str, Any]:
        """Execute task plan and aggregate results"""
        
        # Build chat history
        chat_history = []
        for msg in context_data['recent_chat'][-5:]:
            chat_history.append([
                msg.get('content', ''),
                msg.get('role', 'user')
            ])
        
        # Execute task plan
        execution_result = await self.task_executor.execute_task_plan(
            plan=task_plan,
            query=query,
            chat_history=chat_history,
            system_language=language,
            provider_type=provider_type,
            model_name=model_name,
            flow_id=flow_id
        )
        
        # Extract results
        task_results = execution_result['task_results']
        stats = execution_result['stats']
        
        # Log stats
        self._log_execution_stats(flow_id, stats)
        
        # Aggregate results
        all_tool_results = {}
        tools_executed = []
        
        if isinstance(task_results, list):
            self.logger.debug(
                f"[{flow_id}] Processing {len(task_results)} task results"
            )
            
            for task_result in task_results:
                # Extract task data
                task_id = getattr(task_result, 'task_id', 'unknown')
                is_success = getattr(task_result, 'success', False)
                task_data = getattr(task_result, 'data', {})
                task_tools = getattr(task_result, 'tools_executed', [])
                
                # Aggregate successful tasks
                if is_success and task_data:
                    all_tool_results[f"task_{task_id}"] = task_data
                    tools_executed.extend(task_tools)
            
            self.logger.info(
                f"[{flow_id}] ✓ Aggregated {len(all_tool_results)} successful tasks, "
                f"{len(set(tools_executed))} unique tools"
            )
        
        return {
            'all_tool_results': all_tool_results,
            'tools_executed': tools_executed,
            'stats': stats
        }
    
    # ========================================================================
    # PHASE 5: SYNTHESIS & CONTEXT ASSEMBLY
    # ========================================================================
    async def _phase5_synthesis_and_assembly(
        self,
        flow_id: str,
        query: str,
        plan_data: Dict[str, Any],           # ✅ OLD - Keep for language, symbols
        execution_data: Dict[str, Any],      # ✅ OLD - Keep for all_tool_results
        memory_data: Dict[str, Any],         # ✅ OLD - Keep for recall/archival
        model_name: str,                     # ✅ OLD - Keep for system prompt
        enable_thinking: bool                # ✅ OLD - Keep for system prompt
    ) -> Dict[str, Any]:
        """
        Phase 5: Synthesis & Context Assembly
        
        MERGED VERSION: Combines old working signature with new formatted_context logic
        
        Args:
            flow_id: Flow ID for logging
            query: User query
            plan_data: Plan metadata (contains language, symbols, intent, strategy)
            execution_data: Execution results (contains all_tool_results)
            memory_data: Memory search results (recall_results, archival_results)
            model_name: Model name for system prompt
            enable_thinking: Enable thinking mode
            
        Returns:
            {
                'system_prompt': str,
                'insights_summary': str,
                'tool_context': str,
                'symbols': list
            }
        """
        
        phase_start = time.time()
        self.logger.info(f"[{flow_id}] [PHASE 5] Starting Synthesis & Context Assembly...")
        
        # ════════════════════════════════════════════════════════════
        # Extract Data from Parameters
        # ════════════════════════════════════════════════════════════
        
        # From execution_data
        all_tool_results = execution_data.get('all_tool_results', {})
        print(f"All tool results {all_tool_results}")
        # From plan_data
        language = plan_data.get('language', 'vi')
        plan_symbols = plan_data.get('symbols', [])
        query_intent = plan_data.get('query_intent', '')
        strategy = plan_data.get('strategy', 'sequential')
        
        # From memory_data
        recall_results = memory_data.get('recall_results', [])
        archival_results = memory_data.get('archival_results', [])
        
        self.logger.info(
            f"[{flow_id}] [PHASE 5] Context: "
            f"language={language}, "
            f"intent={query_intent}, "
            f"strategy={strategy}, "
            f"plan_symbols={plan_symbols}"
        )
        
        # ════════════════════════════════════════════════════════════
        # 5.1: Base System Prompt
        # ════════════════════════════════════════════════════════════
        from src.helpers.system_prompts import get_system_message_general_chat
        
        system_prompt = get_system_message_general_chat(
            enable_thinking=enable_thinking,
            model_name=model_name,
            detected_language=language
        )
        
        self.logger.info(f"[{flow_id}] ✓ Base system prompt: {len(system_prompt)} chars")
        
        # ════════════════════════════════════════════════════════════
        # 5.2: Add Memory Search Results
        # ════════════════════════════════════════════════════════════
        if recall_results or archival_results:
            try:
                search_context = await self.memory_search.format_search_results_for_context(
                    recall_results=recall_results,
                    archival_results=archival_results
                )
                if search_context:
                    system_prompt += f"\n\n{search_context}"
                    self.logger.info(
                        f"[{flow_id}] ✓ Added memory context: {len(search_context)} chars"
                    )
            except Exception as e:
                self.logger.warning(f"[{flow_id}] ⚠️ Memory context error: {e}")
        
        # ════════════════════════════════════════════════════════════
        # 5.3: Convert Tool Results to New Format
        # ════════════════════════════════════════════════════════════
        
        # Convert old Dict format to List[TaskExecutionResult] if needed
        if isinstance(all_tool_results, list):
            # Already List[TaskExecutionResult] - new format
            task_results_list = all_tool_results
        elif isinstance(all_tool_results, dict):
            # Old format: Dict[str, Any] - convert to list
            task_results_list = []
            for tool_name, tool_data in all_tool_results.items():
                # Create TaskExecutionResult-like object
                task_results_list.append({
                    'tool_name': tool_name,
                    'data': tool_data,
                    'status': tool_data.get('status', 'unknown')
                })
        else:
            task_results_list = []
        
        self.logger.info(
            f"[{flow_id}] Tool results format: "
            f"type={type(all_tool_results)}, "
            f"count={len(task_results_list)}"
        )
        
        # ════════════════════════════════════════════════════════════
        # 5.4: Convert to ToolOutput Dict
        # ════════════════════════════════════════════════════════════
        
        tool_outputs_dict = self._convert_to_tool_outputs(task_results_list)
        
        self.logger.info(
            f"[{flow_id}] ✓ Converted to tool outputs: {len(tool_outputs_dict)} tools, {tool_outputs_dict}"
        )
        
        # ════════════════════════════════════════════════════════════
        # 5.5: SYNTHESIS DECISION - Multi-Symbol vs Single
        # ════════════════════════════════════════════════════════════
        
        needs_synthesis = self._check_needs_synthesis(
            tool_outputs=tool_outputs_dict,
            plan_data=plan_data  # Pass plan_data for intent-aware decision
        )
        
        insights_summary = ""
        synthesized_context = ""
        tool_context = ""
        
        if tool_outputs_dict and needs_synthesis:
            # ════════════════════════════════════════════════════════════
            # 5.6: MULTI-SYMBOL SYNTHESIS
            # ════════════════════════════════════════════════════════════
            self.logger.info(f"[{flow_id}] [SYNTHESIS] Multi-symbol analysis detected...")
            
            try:
                from src.agents.synthesis.synthesis_agent import SynthesisAgent
                synthesis_agent = SynthesisAgent(self.llm_provider)
                
                # ✅ NEW: Pass tool_results with formatted_context
                synthesized_context = await synthesis_agent.synthesize_results(
                    query=query,
                    tool_results=tool_outputs_dict,  # Has formatted_context
                    language=language
                )
                
                if synthesized_context:
                    self.logger.info(
                        f"[{flow_id}] [SYNTHESIS] ✅ Generated synthesis: "
                        f"{len(synthesized_context)} chars"
                    )
                    
                    # Add to system prompt
                    system_prompt += f"\n\n{synthesized_context}"
                    tool_context = synthesized_context
                
            except Exception as e:
                self.logger.error(
                    f"[{flow_id}] [SYNTHESIS] Error: {e}", 
                    exc_info=True
                )
                # Fallback to detailed format
                needs_synthesis = False
        
        if tool_outputs_dict and not needs_synthesis:
            # ════════════════════════════════════════════════════════════
            # 5.7: SINGLE-SYMBOL DETAILED FORMAT
            # ════════════════════════════════════════════════════════════
            self.logger.info(f"[{flow_id}] [FORMAT] Single-symbol detailed format...")
            
            try:
                # ✅ NEW: Use formatted_context from tool outputs
                tool_context = self._format_tool_context_for_llm(
                    tool_outputs=tool_outputs_dict,
                    language=language
                )
                
                if tool_context:
                    system_prompt += f"\n\n{tool_context}"
                    self.logger.info(
                        f"[{flow_id}] [FORMAT] ✓ Detailed format: {len(tool_context)} chars"
                    )
            
            except Exception as e:
                self.logger.error(
                    f"[{flow_id}] [FORMAT] Error: {e}",
                    exc_info=True
                )
        
        # ════════════════════════════════════════════════════════════
        # 5.8: SYMBOL AWARENESS - Prevent Symbol Bleeding
        # ════════════════════════════════════════════════════════════
        
        symbols = self._extract_symbols_from_results(tool_outputs_dict)
        
        # Use plan_symbols if available, otherwise extracted symbols
        active_symbols = plan_symbols if plan_symbols else symbols
        
        if active_symbols:
            symbol_awareness = f"""

    {'='*70}
    ⚠️  CRITICAL - CURRENT QUERY SYMBOLS
    {'='*70}

    Query Intent: {query_intent}
    Strategy: {strategy}
    Symbols in scope: {', '.join(active_symbols)}

    IMPORTANT: Analyze ONLY these {len(active_symbols)} symbol(s) above.
    Do NOT confuse with symbols from chat history!

    {'='*70}
    """
            system_prompt += symbol_awareness
            self.logger.info(
                f"[{flow_id}] ✓ Symbol awareness added: {active_symbols}"
            )
        
        # ════════════════════════════════════════════════════════════
        # 5.9: RETURN ASSEMBLED CONTEXT
        # ════════════════════════════════════════════════════════════
        
        phase_time = time.time() - phase_start
        self.logger.info(
            f"[{flow_id}] [PHASE 5] ✅ Synthesis Complete "
            f"({phase_time:.2f}s, {len(system_prompt)} chars)"
        )
        
        self.logger.info(f"insights_summary {insights_summary}, tool_context {tool_context}, synthesized_context {synthesized_context}")
        return {
            'system_prompt': system_prompt,
            'insights_summary': insights_summary,
            'synthesized_context': synthesized_context,
            'tool_context': tool_context,  # Include for logging
            'symbols': active_symbols
        }


    def _convert_to_tool_outputs(
        self,
        task_results: List[Any]
    ) -> Dict[str, Any]:
        """
        Convert TaskExecutionResult list to tool outputs dict
        
        Handles multiple input formats:
        1. Dict with nested 'data' containing tool outputs
        2. Direct tool output dict
        3. TaskExecutionResult objects
        
        Returns:
            Dict[tool_name, ToolOutput] where each tool output has:
            - status: str
            - data: Dict
            - formatted_context: str
            - symbols: List[str]
        """
        
        tool_outputs = {}
        
        if not task_results:
            self.logger.warning("[CONVERT] Empty task_results")
            return tool_outputs
        
        self.logger.info(f"[CONVERT] Processing {len(task_results)} task results")
        
        for idx, result in enumerate(task_results):
            self.logger.debug(f"[CONVERT] Result {idx}: type={type(result)}")
            
            # ════════════════════════════════════════════════════════════
            # Format 1: Dict with nested 'data' (your current format)
            # ════════════════════════════════════════════════════════════
            if isinstance(result, dict):
                # Check for nested tool data
                if 'data' in result and isinstance(result['data'], dict):
                    self.logger.debug(f"[CONVERT] Found nested data structure")
                    
                    # Iterate through actual tool outputs
                    for potential_tool_name, potential_tool_data in result['data'].items():
                        # Verify it's a real tool output (has formatted_context)
                        if isinstance(potential_tool_data, dict):
                            if 'formatted_context' in potential_tool_data:
                                tool_outputs[potential_tool_name] = potential_tool_data
                                
                                context_len = len(potential_tool_data.get('formatted_context', ''))
                                status = potential_tool_data.get('status', 'unknown')
                                
                                self.logger.info(
                                    f"[CONVERT] ✓ {potential_tool_name}: "
                                    f"status={status}, context={context_len} chars"
                                )
                            else:
                                self.logger.debug(
                                    f"[CONVERT] Skipped {potential_tool_name}: "
                                    f"no formatted_context"
                                )
                
                # Fallback: Check if result itself is a tool output
                elif 'tool_name' in result and 'formatted_context' in result:
                    tool_name = result['tool_name']
                    tool_outputs[tool_name] = result
                    self.logger.info(f"[CONVERT] ✓ Added {tool_name} (direct format)")
            
            # ════════════════════════════════════════════════════════════
            # Format 2: TaskExecutionResult object
            # ════════════════════════════════════════════════════════════
            elif hasattr(result, 'data') and hasattr(result, 'tools_executed'):
                self.logger.debug(f"[CONVERT] TaskExecutionResult object")
                
                for tool_name in result.tools_executed:
                    tool_data = result.data.get(tool_name, {})
                    if tool_data:
                        tool_outputs[tool_name] = tool_data
                        self.logger.info(
                            f"[CONVERT] ✓ {tool_name} from TaskExecutionResult"
                        )
        
        # ════════════════════════════════════════════════════════════
        # Final verification
        # ════════════════════════════════════════════════════════════
        if not tool_outputs:
            self.logger.warning(
                "[CONVERT] ⚠️ No tool outputs extracted! "
                f"Input had {len(task_results)} results"
            )
        else:
            self.logger.info(
                f"[CONVERT] ✅ Success: {len(tool_outputs)} tools extracted → "
                f"{list(tool_outputs.keys())}"
            )
        
        return tool_outputs

    def _check_needs_synthesis(
        self,
        tool_outputs: Dict[str, Any],
        plan_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Determine if multi-symbol synthesis is needed
        
        ✅ ENHANCED: Uses plan_data for intent-aware decision
        
        Args:
            tool_outputs: Tool outputs dict
            plan_data: Plan metadata (intent, strategy, symbols)
            
        Returns:
            True if synthesis needed, False for detailed format
        """
        
        # Extract symbols from tool outputs
        all_symbols = set()
        for tool_output in tool_outputs.values():
            if isinstance(tool_output, dict) and 'symbols' in tool_output:
                symbols = tool_output.get('symbols', [])
                if isinstance(symbols, list):
                    all_symbols.update(symbols)
        
        symbol_count = len(all_symbols)
        
        # ✅ Use plan_data for context-aware decision
        if plan_data:
            query_intent = plan_data.get('query_intent', '')
            plan_symbols = plan_data.get('symbols', [])
            
            # Force synthesis for screening/comparison queries
            if query_intent in ['screening', 'comparison', 'discovery']:
                self.logger.info(
                    f"[SYNTHESIS CHECK] Intent={query_intent} → Force synthesis"
                )
                return True
            
            # Force synthesis if plan has multiple symbols
            if len(plan_symbols) > 1:
                self.logger.info(
                    f"[SYNTHESIS CHECK] Plan has {len(plan_symbols)} symbols → Synthesis"
                )
                return True
        
        # Fallback: Symbol count threshold
        needs_synthesis = symbol_count > 3
        
        self.logger.info(
            f"[SYNTHESIS CHECK] "
            f"Symbols={symbol_count}, "
            f"Intent={plan_data.get('query_intent') if plan_data else 'unknown'}, "
            f"Needs synthesis={needs_synthesis}"
        )
        
        return needs_synthesis
    
    def _format_tool_context_for_llm(
        self,
        tool_outputs: Dict[str, Any],
        language: str = 'en'
    ) -> str:
        """
        Format tool context for single-symbol queries
        
        ✅ NEW: Uses formatted_context from tool outputs
        
        Args:
            tool_outputs: Dict[tool_name, ToolOutput]
            language: Response language
            
        Returns:
            Formatted context string for LLM
        """
        
        sections = []
        
        for tool_name, tool_output in tool_outputs.items():
            if not isinstance(tool_output, dict):
                continue
            
            status = tool_output.get('status', 'unknown')
            if status not in ['success', '200']:
                continue
            
            # ✅ Use formatted_context if available
            if 'formatted_context' in tool_output and tool_output['formatted_context']:
                sections.append(tool_output['formatted_context'])
            
            # Fallback to raw data
            elif 'data' in tool_output:
                import json
                sections.append(
                    json.dumps(
                        tool_output['data'], 
                        indent=2, 
                        ensure_ascii=False,
                        default=str
                    )
                )
        
        return "\n\n".join(sections)
        
    def _extract_symbols_from_results(
        self,
        tool_outputs: Dict[str, Any]
    ) -> List[str]:
        """
        Extract unique symbols from tool outputs
        
        Args:
            tool_outputs: Dict[tool_name, ToolOutput]
            
        Returns:
            List of unique symbols
        """
        
        all_symbols = set()
        
        for tool_output in tool_outputs.values():
            if not isinstance(tool_output, dict):
                continue
            
            # Extract from 'symbols' field
            symbols = tool_output.get('symbols', [])
            if isinstance(symbols, list):
                all_symbols.update(symbols)
            elif isinstance(symbols, str):
                all_symbols.add(symbols)
            
            # Extract from 'data' field
            data = tool_output.get('data', {})
            if isinstance(data, dict):
                # Check for symbol in data
                if 'symbol' in data:
                    all_symbols.add(data['symbol'])
                
                # Check for symbols array in data
                if 'symbols' in data:
                    syms = data['symbols']
                    if isinstance(syms, list):
                        all_symbols.update(syms)
        
        return sorted(list(all_symbols))
    
    def _extract_tool_data(self, tool_result: Dict) -> Dict:
        """Extract data from tool result - supports multiple formats"""
        
        # Try different keys
        for key in ['data', 'result', 'raw_data', 'content']:
            data = tool_result.get(key)
            if data and isinstance(data, dict):
                return data
        
        # Fallback: return entire result minus metadata
        exclude_keys = {'status', 'error', 'tool_name', 'timestamp', 'symbols'}
        return {k: v for k, v in tool_result.items() if k not in exclude_keys}
    
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
        """Generate and stream LLM response"""
        
        phase_start = time.time()
        self.logger.info(f"[{flow_id}] [PHASE 6] Starting LLM Generation...")
        
        # print(f"Assemnled context {assembled_context}")
        system_prompt = assembled_context['system_prompt']
        has_tool_results = bool(execution_data.get('all_tool_results'))
        
        # Limit history if we have fresh tool data
        max_history = 2 if has_tool_results else 10
        
        if has_tool_results:
            self.logger.info(
                f"[{flow_id}] Tool results present - limiting history to {max_history}"
            )
        
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
        
        # self.logger.info(f"[{flow_id}] 📜 SYSTEM PROMPT:\n...{system_prompt}")
        # self.logger.info(f"[{flow_id}] 📜 Messages:\n...{messages}")
        self.logger.info(
            f"[{flow_id}] ✓ Messages prepared: {llm_metadata.get('llm_messages_tokens', 0)} tokens"
        )
        
        # Get API key
        from src.providers.provider_factory import ModelProviderFactory
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
        self.logger.info(f"[{flow_id}] [PHASE 6] LLM Generation Complete ({phase_time:.2f}s)")
    
    # ========================================================================
    # PHASE 7: POST-PROCESSING
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
        flow_start: float
    ) -> None:
        """Save conversation and trigger background tasks"""
        
        phase_start = time.time()
        self.logger.info(f"[{flow_id}] [PHASE 7] Starting Post-Processing...")
        
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
            self.logger.error(f"[{flow_id}] ✗ Failed to save conversation: {e}")
        
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
        
        phase_time = time.time() - phase_start
        total_time = time.time() - flow_start
        
        self.logger.info(f"[{flow_id}] [PHASE 7] Post-Processing Complete ({phase_time:.2f}s)")
        self.logger.info(f"[{flow_id}] ========== FLOW COMPLETE ({total_time:.2f}s) ==========")
    
    # ========================================================================
    # LOGGING HELPERS
    # ========================================================================
    
    def _log_flow_start(self, flow_id, query, user_id, session_id, model_name, provider_type):
        """Log flow start information"""
        self.logger.info(f"[{flow_id}] ========== STARTING FLOW ==========")
        self.logger.info(f"[{flow_id}] Query: '{query[:100]}...'")
        self.logger.info(f"[{flow_id}] User: {user_id}, Session: {session_id[:8]}")
        self.logger.info(f"[{flow_id}] Model: {model_name}, Provider: {provider_type}")
        self.logger.info(
            f"[{flow_id}] Task-based: "
            f"{'ENABLED' if self.ENABLE_TASK_BASED_EXECUTION else 'DISABLED'}"
        )
    
    def _log_task_plan(self, flow_id, task_plan):
        """Log task plan details"""
        self.logger.info(f"[{flow_id}] ======= TASK PLAN =======")
        self.logger.info(f"[{flow_id}] Query Intent: {task_plan.query_intent}")
        self.logger.info(f"[{flow_id}] Strategy: {task_plan.strategy}")
        self.logger.info(f"[{flow_id}] Complexity: {task_plan.estimated_complexity}")
        self.logger.info(f"[{flow_id}] Number of Tasks: {len(task_plan.tasks)}")
        
        for idx, task in enumerate(task_plan.tasks, 1):
            self.logger.info(f"[{flow_id}]   Task {idx}: {task.description}")
            self.logger.info(f"[{flow_id}]     Priority: {task.priority.value}")
            self.logger.info(f"[{flow_id}]     Tools: {[t.tool_name for t in task.tools_needed]}")
            if task.dependencies:
                self.logger.info(f"[{flow_id}]     Dependencies: {task.dependencies}")
    
    def _log_inner_thoughts(self, flow_id, decision):
        """Log inner thoughts decision"""
        self.logger.info(f"[{flow_id}] ======= INNER THOUGHTS =======")
        self.logger.info(f"[{flow_id}] Strategy: {decision.response_strategy.strategy}")
        self.logger.info(f"[{flow_id}] Need Tool: {decision.tool_decision.need_tool}")
        if decision.tool_decision.need_tool:
            num_tools = len(decision.tool_decision.tool_sequence)
            self.logger.info(f"[{flow_id}] Tool Sequence: {num_tools} tool(s)")
    
    def _log_execution_stats(self, flow_id, stats):
        """Log execution statistics"""
        self.logger.info(f"[{flow_id}] ═══════════════════════════════════")
        self.logger.info(f"[{flow_id}] Task Execution Summary:")
        self.logger.info(f"[{flow_id}]   Total tasks: {stats.get('total_tasks', 0)}")
        self.logger.info(f"[{flow_id}]   Completed: {stats.get('completed_tasks', 0)}")
        self.logger.info(f"[{flow_id}]   Failed: {stats.get('failed_tasks', 0)}")
        self.logger.info(f"[{flow_id}]   Skipped: {stats.get('skipped_tasks', 0)}")
        
    
    async def _execute_tools_with_validation(
        self,
        flow_id: str,
        decision,
        query: str,
        recent_chat: List,
        provider_type: str,
        model_name: str
    ) -> Dict:
        """
        Execute tools with validation
        """

        all_tool_results = {}
        tools_executed = []
        validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'retries_triggered': 0,
            'total_retries': 0
        }
        
        # Get tool sequence
        tool_sequence = decision.tool_decision.tool_sequence
        
        # Fallback: If tool_sequence is empty, create from legacy fields
        if not tool_sequence and decision.tool_decision.tool_name:
            params_dict = {}
            if decision.tool_decision.tool_params:
                params_dict = {
                    'symbols': decision.tool_decision.tool_params.symbols or [],
                    'timeframe': decision.tool_decision.tool_params.timeframe,
                    'additional_params': decision.tool_decision.tool_params.additional_params or {}
                }
            
            tool_sequence = [{
                'tool_name': decision.tool_decision.tool_name,
                'params': params_dict,
                'purpose': "Primary tool execution"
            }]
        
        if not tool_sequence:
            return {
                'all_tool_results': {},
                'tools_executed': [],
                'validation_stats': validation_stats
            }
        
        # Limit number of tools
        if len(tool_sequence) > self.MAX_TOOLS_PER_QUERY:
            self.logger.warning(
                f"[{flow_id}] Tool sequence has {len(tool_sequence)} tools, "
                f"limiting to {self.MAX_TOOLS_PER_QUERY}"
            )
            tool_sequence = tool_sequence[:self.MAX_TOOLS_PER_QUERY]
        
        # Build conversation context for tools
        tool_context = []
        for msg in recent_chat[-3:]:
            tool_context.append([
                msg.get('content', ''),
                msg.get('role', 'user')
            ])
        
        # Execute each tool with validation and retry
        for idx, tool_step in enumerate(tool_sequence, 1):
            tool_name = tool_step.get('tool_name')
            tool_params = tool_step.get('params', {})
            tool_purpose = tool_step.get('purpose', '')
            
            self.logger.info(f"[{flow_id}] ──────────────────────────────────")
            self.logger.info(f"[{flow_id}] ▶ Executing Tool {idx}/{len(tool_sequence)}")
            self.logger.info(f"[{flow_id}] Tool: {tool_name}")
            self.logger.info(f"[{flow_id}] Purpose: {tool_purpose}")
            
            # Retry loop with validation
            retry_count = 0
            max_retries = self.MAX_RETRIES_PER_TASK
            tool_result = None
            validation_result = None
            final_attempt = False
            
            while retry_count <= max_retries and not final_attempt:
                attempt_label = f"Attempt {retry_count + 1}/{max_retries + 1}"
                self.logger.info(f"[{flow_id}] {attempt_label}")
                
                tool_start = time.time()
                
                try:
                    # Execute tool
                    tool_result = await self.memory_search.execute_tool_with_context(
                        tool_name=tool_name,
                        tool_params=tool_params,
                        query=query,
                        conversation_history=tool_context,
                        provider_type=provider_type,
                        model_name=model_name
                    )
                    
                    tool_elapsed = time.time() - tool_start
                    status = tool_result.get('status', 'unknown')
                    
                    self.logger.info(f"[{flow_id}] Tool execution: {tool_elapsed:.2f}s, Status: {status}")
                    
                    # Quick check (skip LLM validation for obvious failures)
                    if self.ENABLE_QUICK_CHECK:
                        is_obviously_empty = self.validation_agent.quick_check_empty_results(tool_result)
                        if is_obviously_empty:
                            self.logger.warning(f"[{flow_id}] ⚠ Quick check: Obviously empty/failed result")
                            final_attempt = True
                            break
                    
                    # Validation with LLM
                    if self.ENABLE_VALIDATION:
                        validation_start = time.time()
                        
                        validation_result = await self.validation_agent.validate_tool_results(
                            original_query=query,
                            tool_name=tool_name,
                            tool_params=tool_params,
                            tool_results=tool_result,
                            query_intent=decision.query_analysis.intent,
                            symbols=decision.tool_decision.symbols
                        )
                        
                        validation_elapsed = time.time() - validation_start
                        validation_stats['total_validations'] += 1
                        
                        self.logger.info(
                            f"[{flow_id}] [VALIDATION] Sufficient: {validation_result.is_sufficient}, "
                            f"Confidence: {validation_result.confidence:.2f}, "
                            f"Time: {validation_elapsed:.2f}s"
                        )
                        
                        # Decide: Retry or complete?
                        if validation_result.is_sufficient:
                            validation_stats['successful_validations'] += 1
                            self.logger.info(f"[{flow_id}] ✓ Tool results validated successfully")
                            final_attempt = True
                            break
                        
                        elif retry_count < max_retries:
                            retry_count += 1
                            validation_stats['retries_triggered'] += 1
                            validation_stats['total_retries'] += 1
                            
                            self.logger.warning(
                                f"[{flow_id}] ⚠ Insufficient data, triggering retry {retry_count}/{max_retries}"
                            )
                            
                            # Adjust parameters based on validation feedback
                            adjusted_params = self._adjust_tool_params(tool_params, validation_result)
                            if adjusted_params != tool_params:
                                tool_params = adjusted_params
                            
                            continue
                        
                        else:
                            self.logger.warning(f"[{flow_id}] ⚠ Max retries reached")
                            final_attempt = True
                            break
                    
                    else:
                        # Validation disabled, accept result
                        final_attempt = True
                        break
                    
                except Exception as e:
                    self.logger.error(f"[{flow_id}] ✗ Tool execution error: {e}", exc_info=True)
                    tool_result = {
                        "status": "error",
                        "error": str(e),
                        "tool_executed": tool_name
                    }
                    final_attempt = True
                    break
            
            # Store final result
            if tool_result:
                all_tool_results[tool_name] = tool_result
                tools_executed.append(tool_name)
        
        return {
            'all_tool_results': all_tool_results,
            'tools_executed': tools_executed,
            'validation_stats': validation_stats
        }

    
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
        """Background memory updates with logging"""
        try:
            self.logger.info(f"[{flow_id}] [BACKGROUND] Starting memory updates...")
            
            # Update core memory
            update_result = await self.memory_update.analyze_for_updates(
                user_id=user_id,
                user_message=query,
                assistant_message=response,
                model_name=model_name,
                provider_type=provider_type
            )
            
            if update_result.get('updated'):
                self.logger.info(
                    f"[{flow_id}] [BACKGROUND] ✓ Core memory updated: "
                    f"categories={update_result.get('categories')}"
                )
            else:
                self.logger.info(
                    f"[{flow_id}] [BACKGROUND] Core memory not updated: "
                    f"{update_result.get('reason', 'No relevant info')}"
                )
            
            # Check and create summary
            summary_result = await self.summary_manager.check_and_create_summary(
                session_id=session_id,
                user_id=user_id,
                organization_id=organization_id,
                model_name=model_name,
                provider_type=provider_type
            )
            
            if summary_result.get('created'):
                self.logger.info(
                    f"[{flow_id}] [BACKGROUND] ✓ Summary created: "
                    f"version={summary_result.get('version')}"
                )
            else:
                self.logger.info(
                    f"[{flow_id}] [BACKGROUND] Summary not created: "
                    f"{summary_result.get('reason', 'Below threshold')}"
                )
            
            self.logger.info(f"[{flow_id}] [BACKGROUND] Memory updates complete")
            
        except Exception as e:
            self.logger.error(f"[{flow_id}] [BACKGROUND] Error in updates: {e}")

