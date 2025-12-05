# File: src/agents/action/task_executor.py
import time
import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from src.agents.planning.task_models import (
    Task,
    TaskPlan,
    TaskStatus,
    TaskPriority,
    TaskExecutionResult,
    ToolCall,
)
from src.agents.validation.validation_agent import (
    ValidationAgent,
    ValidationResult,
    is_retryable_error,
    calculate_backoff,
)
from src.agents.replanning.replanning_agent import (
    ReplanningAgent,
    ReplanAction,
    ReplanDecision,
    ErrorInfo,
    ExecutionHistory,
    ErrorCategory,
    create_error_info,
    create_execution_history,
)
from src.utils.logger.custom_logging import LoggerMixin


class TaskExecutor(LoggerMixin):
    """
    Task Executor với Sequential Dependencies & True Parallel Execution
    
    FEATURES:
    1. Symbol injection from Task N → Task N+1
    2. True parallel execution for independent tasks
    3. Dependency tracking
    4. Replanning support
    """
    
    ATOMIC_TOOLS = [
        'getStockPrice', 'getStockPerformance', 'getPriceTargets',
        'getTechnicalIndicators', 'detectChartPatterns',
        'getRelativeStrength', 'getSupportResistance',
        'assessRisk', 'getVolumeProfile', 'getSentiment', 'suggestStopLoss',
        'getIncomeStatement', 'getBalanceSheet', 'getCashFlow',
        'getFinancialRatios', 'getGrowthMetrics',
        "getStockNews", "getEarningsCalendar", "getCompanyEvents",
        "getMarketIndices", "getSectorPerformance", "getMarketMovers",
        "getMarketBreadth", "getStockHeatmap", "getMarketNews",
        "stockScreener",
        "getCryptoPrice", "getCryptoTechnicals"
    ]
    
    def __init__(
        self,
        tool_execution_service,
        planning_agent=None,
        validation_agent: Optional[ValidationAgent] = None,
        replanning_agent: Optional[ReplanningAgent] = None,
        max_retries_per_task: int = 2,
        enable_replanning: bool = True,
    ):
        super().__init__()
        
        self.tool_execution_service = tool_execution_service
        self.planning_agent = planning_agent
        self.validation_agent = validation_agent or ValidationAgent()
        self.enable_replanning = enable_replanning
        self.max_retries_per_task = max_retries_per_task
        
        # Initialize replanning agent
        if replanning_agent:
            self.replanning_agent = replanning_agent
        elif enable_replanning:
            self.replanning_agent = ReplanningAgent(
                planning_agent=planning_agent,
                max_replan_attempts=3
            )
        else:
            self.replanning_agent = None
        
        # Track outputs for dependency injection
        self.task_outputs: Dict[int, Dict[str, Any]] = {}
        self.accumulated_context: Dict[int, Dict[str, Any]] = {}
        
        # Execution history
        self.execution_history: ExecutionHistory = create_execution_history()
        self.user_messages: List[str] = []
    
    # ========================================================================
    # MAIN ENTRY POINT
    # ========================================================================
    
    async def execute_task_plan(
        self,
        plan: TaskPlan,
        query: str,
        chat_history: Any,
        system_language: str,
        provider_type: str,
        model_name: str,
        flow_id: str,
    ) -> Dict[str, Any]:
        """Execute complete task plan"""
        execution_start = time.time()
        
        try:
            self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════")
            self.logger.info(f"[{flow_id}] [TASK EXEC] Starting execution")
            self.logger.info(f"[{flow_id}]   Tasks: {len(plan.tasks)}")
            self.logger.info(f"[{flow_id}]   Strategy: {plan.strategy}")
            self.logger.info(f"[{flow_id}]   Symbols: {plan.symbols}")
            self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════")
            
            # Reset state
            self.task_outputs = {}
            self.accumulated_context = {}
            self.execution_history = create_execution_history()
            self.user_messages = []
            
            # Statistics
            stats: Dict[str, Any] = {
                "total_tasks": len(plan.tasks),
                "completed_tasks": 0,
                "failed_tasks": 0,
                "skipped_tasks": 0,
                "total_retries": 0,
                "replanning_attempts": 0,
                "task_timings": {},
            }
            
            # Execute based on strategy
            if plan.strategy == "parallel":
                results = await self._execute_parallel(
                    plan, query, chat_history, system_language,
                    provider_type, model_name, flow_id, stats,
                )
            else:
                results = await self._execute_sequential(
                    plan, query, chat_history, system_language,
                    provider_type, model_name, flow_id, stats,
                )
            
            # Update final stats
            stats["completed_tasks"] = sum(1 for t in plan.tasks if t.done)
            stats["failed_tasks"] = sum(1 for t in plan.tasks if t.status == TaskStatus.FAILED)
            stats["skipped_tasks"] = sum(1 for t in plan.tasks if t.status == TaskStatus.SKIPPED)
            stats["total_execution_time"] = time.time() - execution_start
            
            # Log summary
            self._log_execution_summary(flow_id, stats)
            
            return {
                "task_results": results,
                "stats": stats,
                "accumulated_context": self.accumulated_context,
                "plan": plan,
                "user_messages": self.user_messages,
                "replanning_stats": self.replanning_agent.get_stats() if self.replanning_agent else {},
            }
            
        except Exception as e:
            self.logger.error(f"[{flow_id}] [TASK EXEC] Fatal error: {e}", exc_info=True)
            return {
                "task_results": {},
                "stats": {"error": str(e)},
                "error": str(e),
                "user_messages": [f"An error occurred during execution: {str(e)[:100]}"],
            }
    
    # ========================================================================
    # SEQUENTIAL EXECUTION WITH SYMBOL INJECTION
    # ========================================================================
    
    async def _execute_sequential(
        self,
        plan: TaskPlan,
        query: str,
        chat_history: Any,
        system_language: str,
        provider_type: str,
        model_name: str,
        flow_id: str,
        stats: Dict[str, Any],
    ) -> List[TaskExecutionResult]:
        """Execute tasks sequentially with dependency management"""
        
        results: List[TaskExecutionResult] = []
        completed_task_ids: set[int] = set()
        current_plan = plan
        
        # Store task outputs for dependency injection
        task_outputs: Dict[int, Any] = {}  # ← FIX: Local variable to store outputs

        task_index = 0
        while task_index < len(current_plan.tasks):
            task = current_plan.tasks[task_index]
            
            self.logger.info(f"[{flow_id}] {'─' * 60}")
            self.logger.info(
                f"[{flow_id}] TASK {task_index + 1}/{len(current_plan.tasks)}: {task.description}"
            )
            
            # Log tools
            for tool_call in task.tools_needed:
                self.logger.info(f"[{flow_id}]   → Tool: {tool_call.tool_name}")
                self.logger.debug(f"[{flow_id}]     Params: {tool_call.params}")

            # Check dependencies
            if task.dependencies:
                missing_deps = [
                    dep for dep in task.dependencies if dep not in completed_task_ids
                ]
                if missing_deps:
                    self.logger.warning(
                        f"[{flow_id}] ⚠️ Skipping task {task.id}: Missing deps {missing_deps}"
                    )
                    task.status = TaskStatus.SKIPPED
                    results.append(self._create_skip_result(task, "Missing dependencies"))
                    task_index += 1
                    continue
                
                # ✅ FIX BUG 2: Pass completed_task_ids and use local task_outputs
                task = self._inject_dependency_outputs(
                    task=task,
                    task_outputs=task_outputs,  # ← Use local variable
                    completed_task_ids=completed_task_ids,  # ← Actually use this param
                    flow_id=flow_id
                )

            # Execute task
            result, replan_decision = await self._execute_single_task_with_replanning(
                task=task,
                query=query,
                chat_history=chat_history,
                system_language=system_language,
                provider_type=provider_type,
                model_name=model_name,
                flow_id=flow_id,
                stats=stats,
            )

            results.append(result)

            # ✅ FIX BUG 1: Check if replan_decision is None
            if result.success and result.data:
                task_outputs[task.id] = result.data  # ← Store in local dict
                completed_task_ids.add(task.id)
                
                # Extract and log symbols
                symbols = self._extract_symbols_from_output(result.data)
                self.logger.info(
                    f"[{flow_id}] ✅ Task {task.id} COMPLETED - Found {len(symbols)} symbols"
                    f"{': ' + str(symbols[:5]) if symbols else ''}"
                )
            
            elif result.status == TaskStatus.SKIPPED:
                self.logger.info(f"[{flow_id}] ⏭️ Task {task.id} SKIPPED")
            
            else:
                self.logger.warning(f"[{flow_id}] ❌ Task {task.id} FAILED")

            # ✅ FIX BUG 1: Handle None replan_decision
            if replan_decision is not None and replan_decision.should_replan:
                self.logger.info(
                    f"[{flow_id}] 🔄 REPLANNING: {replan_decision.reason}"
                )
                
                stats['replanning_attempts'] = stats.get('replanning_attempts', 0) + 1
                
                # Get new plan
                new_plan = await self._get_replanned_tasks(
                    original_plan=current_plan,
                    failed_task=task,
                    results_so_far=results,
                    query=query,
                    flow_id=flow_id
                )
                
                if new_plan:
                    current_plan = new_plan
                    task_index = 0  # Restart with new plan
                    continue
                else:
                    self.logger.warning(f"[{flow_id}] Replanning failed, continuing...")

            task_index += 1

        return results

    
    # ========================================================================
    # PARALLEL EXECUTION (TRUE PARALLEL)
    # ========================================================================
    
    async def _execute_parallel(
        self,
        plan: TaskPlan,
        query: str,
        chat_history: Any,
        system_language: str,
        provider_type: str,
        model_name: str,
        flow_id: str,
        stats: Dict[str, Any],
    ) -> List[TaskExecutionResult]:
        """
        Execute independent tasks in TRUE PARALLEL using asyncio.gather
        
        Key difference from sequential:
        - All independent tasks run simultaneously
        - Faster execution for stock-specific queries
        - No dependency tracking needed (all tasks are independent)
        """
        
        self.logger.info(f"[{flow_id}] [TASK EXEC] TRUE Parallel execution mode")
        self.logger.info(f"[{flow_id}] [TASK EXEC] Launching {len(plan.tasks)} tasks concurrently")
        
        # Group tasks by dependency
        independent_tasks = [t for t in plan.tasks if not t.dependencies]
        dependent_tasks = [t for t in plan.tasks if t.dependencies]
        
        if dependent_tasks:
            self.logger.warning(
                f"[{flow_id}] ⚠️ Parallel strategy has {len(dependent_tasks)} dependent tasks, "
                f"falling back to sequential"
            )
            return await self._execute_sequential(
                plan, query, chat_history, system_language,
                provider_type, model_name, flow_id, stats
            )
        
        # Create coroutines for all tasks
        task_coroutines = []
        for task in independent_tasks:
            self.logger.info(f"[{flow_id}] 🚀 Queueing Task {task.id}: {task.description}")
            
            coro = self._execute_single_task_with_replanning(
                task=task,
                query=query,
                chat_history=chat_history,
                system_language=system_language,
                provider_type=provider_type,
                model_name=model_name,
                flow_id=flow_id,
                stats=stats,
            )
            task_coroutines.append(coro)
        
        # Execute ALL tasks concurrently
        self.logger.info(f"[{flow_id}] ⚡ Executing {len(task_coroutines)} tasks in parallel...")
        start_time = time.time()
        
        # asyncio.gather runs all coroutines concurrently
        results_and_replans = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        elapsed = time.time() - start_time
        self.logger.info(f"[{flow_id}] ✅ All tasks completed in {elapsed:.2f}s")
        
        # Process results
        results: List[TaskExecutionResult] = []
        for i, result_tuple in enumerate(results_and_replans):
            if isinstance(result_tuple, Exception):
                self.logger.error(f"[{flow_id}] Task {i+1} raised exception: {result_tuple}")
                results.append(TaskExecutionResult(
                    task_id=i+1,
                    success=False,
                    data={},
                    error=str(result_tuple),
                    tools_executed=[],
                    execution_time_ms=0
                ))
                continue
            
            result, replan_decision = result_tuple
            results.append(result)
            
            # Track outputs
            if result.success:
                task_id = result.task_id
                self.task_outputs[task_id] = result.data
                self.accumulated_context[task_id] = result.data
                self.execution_history.successful_tools.extend(result.tools_executed)
                
                self.logger.info(f"[{flow_id}] ✅ Task {task_id} COMPLETED")
            else:
                self.logger.warning(f"[{flow_id}] ❌ Task {result.task_id} FAILED")
        
        return results
    
    # ========================================================================
    # SYMBOL INJECTION LOGIC
    # ========================================================================
    
    def _inject_dependency_outputs(
        self,
        task: Task,
        task_outputs: Dict[int, Any],
        completed_task_ids: set[int],  # ← NOW ACTUALLY USED
        flow_id: str
    ) -> Task:
        """
        Inject outputs from completed dependencies into task parameters
        
        CRITICAL: 
        - Only inject from COMPLETED tasks (use completed_task_ids)
        - Replace <FROM_TASK_X> with actual symbols
        """
        
        if not task.dependencies:
            return task
        
        self.logger.info(f"[{flow_id}] 🔗 [DEP] Task {task.id} depends on: {task.dependencies}")
        self.logger.info(f"[{flow_id}] 🔄 [INJECT] Injecting outputs from tasks: {task.dependencies}")
        
        # ✅ FIX BUG 2: Check which dependencies are actually completed
        available_deps = [dep for dep in task.dependencies if dep in completed_task_ids]
        missing_deps = [dep for dep in task.dependencies if dep not in completed_task_ids]
        
        if missing_deps:
            self.logger.warning(
                f"[{flow_id}] [INJECT] ⚠️ Missing dependencies: {missing_deps}"
            )
        
        if not available_deps:
            self.logger.error(
                f"[{flow_id}] [INJECT] ❌ No completed dependencies available!"
            )
            return task
        
        self.logger.debug(f"[{flow_id}] [INJECT] Available deps: {available_deps}")
        
        # Collect symbols from completed dependencies
        all_symbols = []
        
        for dep_id in available_deps:  # ← Only use completed deps
            if dep_id not in task_outputs:
                self.logger.warning(
                    f"[{flow_id}] [INJECT] ⚠️ Dependency {dep_id} not in task_outputs "
                    f"(available: {list(task_outputs.keys())})"
                )
                continue
            
            self.logger.debug(f"[{flow_id}] [INJECT] Processing Task {dep_id}")
            
            dep_output = task_outputs[dep_id]
            
            # Extract symbols
            symbols = self._extract_symbols_from_output(dep_output)
            
            if symbols:
                all_symbols.extend(symbols)
                self.logger.debug(
                    f"[{flow_id}] [INJECT] Extracted {len(symbols)} symbols from Task {dep_id}"
                )
        
        # Remove duplicates
        all_symbols = list(dict.fromkeys(all_symbols))
        
        if not all_symbols:
            self.logger.warning(
                f"[{flow_id}] [INJECT] ⚠️ No symbols extracted from dependencies"
            )
            return task
        
        self.logger.info(
            f"[{flow_id}] [INJECT] 📊 Total symbols: {len(all_symbols)} - {all_symbols[:5]}"
        )
        
        # Inject symbols into tool parameters
        for tool_call in task.tools_needed:
            if not tool_call.params:
                continue
            
            for param_name, param_value in list(tool_call.params.items()):
                if isinstance(param_value, str) and '<FROM_TASK_' in param_value:
                    # Inject first symbol
                    tool_call.params[param_name] = all_symbols[0]
                    
                    self.logger.info(
                        f"[{flow_id}] ✅ [INJECT] '{all_symbols[0]}' → "
                        f"{tool_call.tool_name}.{param_name}"
                    )
                    
                    if len(all_symbols) > 1:
                        self.logger.info(
                            f"[{flow_id}] ℹ️ [INJECT] {len(all_symbols) - 1} more available: "
                            f"{all_symbols[1:6]}"
                        )
        
        return task


    def _convert_to_batch_task(
        self,
        task: Task,
        symbols: List[str],
        flow_id: str
    ) -> Task:
        """
        Convert single-symbol task to BATCH multi-symbol task
        
        PATTERN FROM ANTHROPIC:
        - Parallel execution for independent operations
        - Each symbol gets own tool call
        - Results aggregated by Synthesis Agent
        
        Example:
        Input: getTechnicalIndicators(symbol=<FROM_TASK_1>)
        Output: [
            getTechnicalIndicators(symbol=TSM),
            getTechnicalIndicators(symbol=NVDA),
            getTechnicalIndicators(symbol=AAPL),
            ...
        ]
        """
        
        original_tools = task.tools_needed.copy()
        new_tools = []
        
        for tool_call in original_tools:
            # For EACH symbol, create a tool call
            for symbol in symbols:
                # Clone tool call
                new_tool = ToolCall(
                    tool_name=tool_call.tool_name,
                    params=tool_call.params.copy()
                )
                
                # Replace <FROM_TASK_X> with actual symbol
                for param_name, param_value in new_tool.params.items():
                    if isinstance(param_value, str) and '<FROM_TASK_' in param_value:
                        new_tool.params[param_name] = symbol
                    elif param_name in ['symbol', 'symbols']:
                        new_tool.params[param_name] = symbol
                
                new_tools.append(new_tool)
                
                self.logger.debug(
                    f"[{flow_id}] [BATCH] Created {tool_call.tool_name}(symbol={symbol})"
                )
        
        # Update task
        task.tools_needed = new_tools
        
        self.logger.info(
            f"[{flow_id}] ✅ [BATCH] Expanded to {len(new_tools)} tool calls "
            f"({len(original_tools)} tools × {len(symbols)} symbols)"
        )
        
        return task
    
        
    # ============================================================================
    # BUG 3 FIX: Fix symbol extraction to check ALL possible keys
    # ============================================================================

    def _extract_symbols_from_output(self, output: Any) -> List[str]:
        """
        Extract symbols from task output - ROBUST VERSION
        
        Handles deep nesting:
        1. Root level: {'symbols': ...}
        2. Tool wrapper: {'stockScreener': {'symbols': ...}}
        3. Data wrapper: {'data': {'stocks': ...}}
        4. Nested Data: {'stockScreener': {'data': {'stocks': ...}}} <--- Critical Fix
        """
        
        symbols = []
        if not isinstance(output, dict):
            return symbols

        # 🎯 STRATEGY: Flatten the structure into a list of dictionaries to inspect
        search_targets = [output] # Start with root
        
        # 1. Unwrap Tool Wrappers (e.g., output['stockScreener'])
        for key, value in output.items():
            if isinstance(value, dict):
                # Add the tool result wrapper itself
                search_targets.append(value)
                
                # 2. Unwrap 'data' inside Tool Wrapper (CRITICAL FIX)
                # This handles: stockScreener -> data -> stocks
                if 'data' in value and isinstance(value['data'], dict):
                    search_targets.append(value['data'])
        
        # 3. Check 'data' at root level
        if 'data' in output and isinstance(output['data'], dict):
            search_targets.append(output['data'])

        # 🕵️‍♀️ SCAN ALL TARGETS
        for target in search_targets:
            # Check 'symbols' list/string
            if 'symbols' in target:
                val = target['symbols']
                if isinstance(val, list):
                    symbols.extend([str(s) for s in val if s])
                elif isinstance(val, str) and val:
                    symbols.append(val)

            # Check 'top_symbols' list
            if 'top_symbols' in target:
                val = target['top_symbols']
                if isinstance(val, list):
                    symbols.extend([str(s) for s in val if s])

            # Check 'stocks' list of dicts (for Screeners)
            if 'stocks' in target and isinstance(target['stocks'], list):
                for item in target['stocks']:
                    if isinstance(item, dict) and 'symbol' in item:
                        symbols.append(str(item['symbol']))
            
            # Check 'results' list of dicts (Generic)
            if 'results' in target and isinstance(target['results'], list):
                for item in target['results']:
                    if isinstance(item, dict) and 'symbol' in item:
                        symbols.append(str(item['symbol']))

        # Clean up: Uppercase and Unique
        clean_symbols = []
        seen = set()
        
        for s in symbols:
            s_clean = s.strip().upper()
            # Basic validation: 1-10 chars, alphanumeric
            if s_clean and s_clean not in seen and len(s_clean) <= 10:
                seen.add(s_clean)
                clean_symbols.append(s_clean)

        # Log results
        if clean_symbols:
            self.logger.debug(
                f"[SYMBOL EXTRACT] ✅ Found {len(clean_symbols)} symbols: "
                f"{clean_symbols[:5]}{'...' if len(clean_symbols) > 5 else ''}"
            )
        else:
            self.logger.debug(
                f"[SYMBOL EXTRACT] ℹ️ No symbols found. Checked {len(search_targets)} layers."
            )
            
        return clean_symbols

    def _count_symbols_in_output(self, output: Any) -> int:
        """Count symbols in output for logging"""
        symbols = self._extract_symbols_from_output(output)
        return len(symbols)
    
    # ========================================================================
    # SINGLE TASK EXECUTION (UNCHANGED - KEEP EXISTING LOGIC)
    # ========================================================================
    async def _execute_task_tools(
        self,
        task: Task,
        query: str,
        chat_history: Any,
        system_language: str,
        provider_type: str,
        model_name: str,
        flow_id: str,
    ) -> Dict[str, Any]:
        """Execute all tools for a task"""
        
        tool_results: Dict[str, Any] = {}

        for tool_call in task.tools_needed:
            tool_start = time.time()
            self.logger.debug(f"[{flow_id}]   → Executing: {tool_call.tool_name}")
            self.logger.debug(f"[{flow_id}]     Params: {tool_call.params}")
            
            try:
                result = await self.tool_execution_service.execute_single_tool(
                    tool_name=tool_call.tool_name,
                    tool_params=tool_call.params,
                    query=query,
                    chat_history=chat_history,
                    system_language=system_language,
                    provider_type=provider_type,
                    model_name=model_name,
                )
                
                tool_results[tool_call.tool_name] = result
                
                tool_duration = (time.time() - tool_start) * 1000
                status = result.get('status', 'unknown') if isinstance(result, dict) else 'unknown'
                
                self.logger.info(
                    f"[{flow_id}]   ← {tool_call.tool_name}: {status} ({tool_duration:.0f}ms)"
                )
                    
            except Exception as e:
                self.logger.error(f"[{flow_id}]   ❌ {tool_call.tool_name} exception: {e}")
                tool_results[tool_call.tool_name] = {
                    "status": "error",
                    "error": str(e),
                    "tool_name": tool_call.tool_name
                }

        return tool_results

    def _create_success_result(
        self,
        task: Task,
        tool_results: Dict[str, Any],
        execution_time: float,
        retry_count: int,
        stats: Dict[str, Any],
        validation_result: Optional[ValidationResult] = None,
    ) -> TaskExecutionResult:
        """Create successful task execution result"""
        
        task.status = TaskStatus.COMPLETED
        task.done = True
        task.results = tool_results
        task.validation_confidence = 1.0
        task.validation_reasoning = "All tools executed successfully"
        
        stats["task_timings"][task.id] = execution_time
        
        return TaskExecutionResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            success=True,
            data=tool_results,
            tools_executed=list(tool_results.keys()),
            validation_result=validation_result.dict() if validation_result else {
                "is_sufficient": True,
                "confidence": 1.0,
                "reasoning": "All tools returned valid data"
            },
            execution_time=execution_time,
            retry_count=retry_count,
            suggested_next_action="continue",
        )

    def _create_failure_result(
        self,
        task: Task,
        error: str,
        tool_results: Dict[str, Any],
        execution_time: float,
        retry_count: int,
        validation_result: Optional[ValidationResult] = None,
    ) -> TaskExecutionResult:
        """Create failed task execution result"""
        
        task.status = TaskStatus.FAILED
        task.error = error
        
        return TaskExecutionResult(
            task_id=task.id,
            status=TaskStatus.FAILED,
            success=False,
            data=tool_results,  # Include partial data
            error=error,
            tools_executed=list(tool_results.keys()),
            validation_result=validation_result.dict() if validation_result else None,
            execution_time=execution_time,
            retry_count=retry_count,
            suggested_next_action="skip",
        )

    def _check_tool_results(
        self,
        task: Task,
        tool_results: Dict[str, Any],
        flow_id: str
    ) -> Tuple[bool, Optional[ErrorInfo]]:
        """
        Check tool results for success/failure
        
        Returns:
            Tuple of (is_success, ErrorInfo if failed)
        """
        for tool_name, result in tool_results.items():
            if not isinstance(result, dict):
                continue
            
            status = result.get('status', 'unknown')
            
            # Success cases
            if status in [200, '200', 'success']:
                self.logger.info(f"[{flow_id}]   ✅ {tool_name}: SUCCESS")
                continue
            
            # Error case
            if status == 'error':
                error_msg = result.get('error', 'Unknown error')
                params = {}
                
                # Extract params from task
                for tool_call in task.tools_needed:
                    if tool_call.tool_name == tool_name:
                        params = tool_call.params
                        break
                
                self.logger.warning(f"[{flow_id}]   ❌ {tool_name}: {error_msg[:80]}")
                
                return False, create_error_info(
                    tool_name=tool_name,
                    error_message=error_msg,
                    params=params
                )
        
        # All tools succeeded
        return True, None

    def _detect_language(self, query: str, system_language: str) -> str:
        """Detect language from query or use system language"""
        if system_language and system_language != "auto":
            return system_language
        
        # Simple Vietnamese detection
        vietnamese_chars = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
        has_vietnamese = any(c in query.lower() for c in vietnamese_chars)
        
        return "vi" if has_vietnamese else "en"

    async def _execute_fallback_tool(
        self,
        fallback_tool: str,
        params: Dict[str, Any],
        query: str,
        chat_history: Any,
        system_language: str,
        provider_type: str,
        model_name: str,
        flow_id: str,
    ) -> Dict[str, Any]:
        """Execute a fallback tool"""
        
        self.logger.info(f"[{flow_id}]   🔀 Executing fallback: {fallback_tool}")
        
        try:
            result = await self.tool_execution_service.execute_single_tool(
                tool_name=fallback_tool,
                tool_params=params,
                query=query,
                chat_history=chat_history,
                system_language=system_language,
                provider_type=provider_type,
                model_name=model_name,
            )
            return result
            
        except Exception as e:
            self.logger.error(f"[{flow_id}]   ❌ Fallback {fallback_tool} failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "tool_name": fallback_tool
            }

    async def _execute_single_task_with_replanning(
        self,
        task: Task,
        query: str,
        chat_history: Any,
        system_language: str,
        provider_type: str,
        model_name: str,
        flow_id: str,
        stats: Dict[str, Any],
    ) -> Tuple[TaskExecutionResult, Optional[ReplanDecision]]:
        """
        Execute single task with replanning support
        
        Returns:
            Tuple[TaskExecutionResult, Optional[ReplanDecision]]
            
        ✅ FIX: ALWAYS return a tuple with ReplanDecision (or None)
        """
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            self.logger.info(f"[{flow_id}]   Attempt {retry_count + 1}/{max_retries}")
            
            execution_start = time.time()
            
            # Execute task tools
            tool_results = await self._execute_task_tools(
                task=task,
                query=query,
                chat_history=chat_history,
                system_language=system_language,
                provider_type=provider_type,
                model_name=model_name,
                flow_id=flow_id,
            )
            
            execution_time = time.time() - execution_start
            
            # Check if all tools succeeded
            all_success = all(
                result.get('status') in ['success', 200, '200']
                for result in tool_results.values()
                if isinstance(result, dict)
            )
            
            if all_success:
                # Success - no replanning needed
                result = self._create_success_result(
                    task=task,
                    tool_results=tool_results,
                    execution_time=execution_time,
                    retry_count=retry_count,
                    stats=stats,
                    validation_result=None
                )
                
                # ✅ FIX: Return tuple with None for replan_decision
                return result, None
            
            else:
                # Failure - determine if should retry
                retry_count += 1
                
                if retry_count >= max_retries:
                    # Max retries reached - fail task
                    result = self._create_failure_result(
                        task=task,
                        error=f"All tools failed after {max_retries} attempts",
                        tool_results=tool_results,
                        execution_time=execution_time,
                        retry_count=retry_count,
                        validation_result=None
                    )
                    
                    # ✅ FIX: Return tuple with None (no replanning on max retries)
                    return result, None
                
                self.logger.warning(
                    f"[{flow_id}] Task {task.id} failed, retrying... ({retry_count}/{max_retries})"
                )
        
        # Should never reach here, but return safe defaults
        result = self._create_failure_result(
            task=task,
            error="Unexpected execution path",
            tool_results={},
            execution_time=0.0,
            retry_count=retry_count,
            validation_result=None
        )
        
        # ✅ FIX: Return tuple
        return result, None
    
    def _create_skip_result(self, task: Task, reason: str) -> TaskExecutionResult:
        """Create skip result"""
        return TaskExecutionResult(
            task_id=task.id,
            success=False,
            data={},
            error=reason,
            status=TaskStatus.SKIPPED,
            tools_executed=[]
        )
    
    def _log_execution_summary(self, flow_id: str, stats: Dict[str, Any]):
        """Log execution summary"""
        self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════")
        self.logger.info(f"[{flow_id}] [TASK EXEC] EXECUTION SUMMARY")
        self.logger.info(f"[{flow_id}]   Completed: {stats['completed_tasks']}/{stats['total_tasks']}")
        self.logger.info(f"[{flow_id}]   Failed: {stats['failed_tasks']}")
        self.logger.info(f"[{flow_id}]   Skipped: {stats['skipped_tasks']}")
        self.logger.info(f"[{flow_id}]   Total Retries: {stats['total_retries']}")
        self.logger.info(f"[{flow_id}]   Replanning Attempts: {stats['replanning_attempts']}")
        
        if 'total_execution_time' in stats:
            self.logger.info(f"[{flow_id}]   Total Time: {stats['total_execution_time']:.2f}s")
        
        self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════")