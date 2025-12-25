# import time
# import asyncio
# from typing import Dict, List, Optional, Any, Tuple
# from datetime import datetime

# from src.agents.planning.task_models import (
#     Task,
#     TaskPlan,
#     TaskStatus,
#     TaskExecutionResult,
#     ToolCall,
# )
# from src.agents.validation.validation_agent import (
#     ValidationAgent,
#     ValidationResult,
#     is_retryable_error,
#     calculate_backoff,
# )
# from src.utils.logger.custom_logging import LoggerMixin


# class TaskExecutor(LoggerMixin):
#     """
#     Executes task plans with sequential dependency handling and multi-symbol support.
    
#     Core capabilities:
#     - Symbol injection between dependent tasks (Task N → Task N+1)
#     - Multi-symbol batch processing
#     - Exponential backoff retry for transient errors
#     - Graceful degradation on partial failures
#     """
    
#     # Max symbols to prevent API overload
#     MAX_SYMBOLS_PER_TASK = 5
#     MAX_RETRIES = 2
#     RETRY_BASE_DELAY = 1.0  # seconds
#     RETRY_MAX_DELAY = 8.0   # seconds
    
#     def __init__(
#         self,
#         tool_execution_service,
#         validation_agent: Optional[ValidationAgent] = None,
#         max_retries: int = 2,
#         max_symbols_per_task: int = 5,
#     ):
#         """
#         Initialize task executor with tool execution and validation services.
        
#         Args:
#             tool_execution_service: Service for executing individual tools
#             validation_agent: Optional validator for tool outputs
#             max_retries: Maximum retry attempts for transient failures
#             max_symbols_per_task: Symbol processing limit per task
#         """
#         super().__init__()
        
#         self.tool_execution_service = tool_execution_service
#         self.validation_agent = validation_agent or ValidationAgent()
#         self.max_retries = max_retries
#         self.max_symbols_per_task = max_symbols_per_task
        
#         # Store task outputs for dependency injection
#         self.task_outputs: Dict[int, Dict[str, Any]] = {}
        
#         self.logger.info("[TASK_EXECUTOR] Initialized successfully")

    
#     # ========================================================================
#     # MAIN ENTRY POINT
#     # ========================================================================
    
#     async def execute_task_plan(
#         self,
#         plan: TaskPlan,
#         query: str,
#         chat_history: Any,
#         system_language: str,
#         provider_type: str,
#         model_name: str,
#         flow_id: str,
#         user_id: Optional[int] = None,
#         session_id: Optional[str] = None
#     ) -> Dict[str, Any]:
#         """
#         Execute complete task plan and return aggregated results
        
#         Args:
#             plan: Task plan from Planning Agent
#             query: Original user query
#             chat_history: Conversation history
#             system_language: Response language
#             provider_type: LLM provider
#             model_name: LLM model
#             flow_id: Flow identifier for logging
#             user_id: Optional user identifier for personalization
#             session_id: Optional session identifier for context
            
#         Returns:
#             Dict containing:
#                 - task_results: List of TaskExecutionResult
#                 - stats: Execution statistics
#                 - accumulated_context: Task outputs for dependencies
#                 - plan: Original task plan
#         """
#         execution_start = time.time()
        
#         try:
#             self.logger.info(f"[{flow_id}] {'=' * 60}")
#             self.logger.info(f"[{flow_id}] [TASK EXEC] Starting execution")
#             self.logger.info(f"[{flow_id}]   Tasks: {len(plan.tasks)}")
#             self.logger.info(f"[{flow_id}]   Strategy: {plan.strategy}")
#             self.logger.info(f"[{flow_id}]   Symbols: {plan.symbols}")
#             self.logger.info(f"[{flow_id}] {'=' * 60}")
            
#             # Reset execution state
#             self.task_outputs = {}
            
#             # Initialize statistics
#             stats = {
#                 "total_tasks": len(plan.tasks),
#                 "completed_tasks": 0,
#                 "failed_tasks": 0,
#                 "skipped_tasks": 0,
#                 "total_retries": 0,
#                 "task_timings": {},
#             }
            
#             # Execute based on strategy
#             if plan.strategy == "parallel":
#                 results = await self._execute_parallel(
#                     plan=plan,
#                     query=query,
#                     chat_history=chat_history,
#                     system_language=system_language,
#                     provider_type=provider_type,
#                     model_name=model_name,
#                     flow_id=flow_id,
#                     stats=stats,
#                     user_id=user_id,
#                     session_id=session_id
#                 )
#             else:
#                 results = await self._execute_sequential(
#                     plan=plan,
#                     query=query,
#                     chat_history=chat_history,
#                     system_language=system_language,
#                     provider_type=provider_type,
#                     model_name=model_name,
#                     flow_id=flow_id,
#                     stats=stats,
#                     user_id=user_id, 
#                     session_id=session_id
#                 )
            
#             # Calculate final statistics
#             stats["completed_tasks"] = sum(1 for r in results if r.success)
#             stats["failed_tasks"] = sum(1 for r in results if not r.success and r.status != TaskStatus.SKIPPED)
#             stats["skipped_tasks"] = sum(1 for r in results if r.status == TaskStatus.SKIPPED)
#             stats["total_execution_time"] = time.time() - execution_start
            
#             self._log_execution_summary(flow_id, stats)
            
#             return {
#                 "task_results": results,
#                 "stats": stats,
#                 "accumulated_context": self.task_outputs,
#                 "plan": plan,
#             }
            
#         except Exception as e:
#             self.logger.error(f"[{flow_id}] [TASK EXEC] Fatal error: {e}", exc_info=True)
#             return {
#                 "task_results": [],
#                 "stats": {"error": str(e)},
#                 "error": str(e),
#             }
    
#     # ========================================================================
#     # SEQUENTIAL EXECUTION
#     # ========================================================================
    
#     async def _execute_sequential(
#         self,
#         plan: TaskPlan,
#         query: str,
#         chat_history: Any,
#         system_language: str,
#         provider_type: str,
#         model_name: str,
#         flow_id: str,
#         stats: Dict[str, Any],
#         user_id: Optional[int] = None,    
#         session_id: Optional[str] = None     
#     ) -> List[TaskExecutionResult]:
#         """
#         Execute tasks sequentially with dependency resolution.
        
#         Each task waits for its dependencies to complete before execution.
#         Dependent tasks receive symbol injection from completed tasks.
#         """
        
#         results: List[TaskExecutionResult] = []
#         completed_task_ids: set[int] = set()
        
#         for task_index, task in enumerate(plan.tasks):
#             self.logger.info(f"[{flow_id}] {'─' * 60}")
#             self.logger.info(
#                 f"[{flow_id}] TASK {task_index + 1}/{len(plan.tasks)}: {task.description}"
#             )
            
#             # Check dependencies
#             if task.dependencies:
#                 missing_deps = [
#                     dep for dep in task.dependencies if dep not in completed_task_ids
#                 ]
#                 if missing_deps:
#                     self.logger.warning(
#                         f"[{flow_id}] Skipping task {task.id}: Missing deps {missing_deps}"
#                     )
#                     task.status = TaskStatus.SKIPPED
#                     results.append(self._create_skip_result(task, "Missing dependencies"))
#                     continue
                
#                 # Expand task for multi-symbols from dependencies
#                 task = self._expand_for_multi_symbols(
#                     task=task,
#                     completed_task_ids=completed_task_ids,
#                     flow_id=flow_id
#                 )
            
#             # Execute task with retry
#             result = await self._execute_single_task(
#                 task=task,
#                 query=query,
#                 chat_history=chat_history,
#                 system_language=system_language,
#                 provider_type=provider_type,
#                 model_name=model_name,
#                 flow_id=flow_id,
#                 stats=stats,
#                 user_id=user_id,
#                 session_id=session_id
#             )
            
#             results.append(result)
            
#             # Track outputs for dependency injection
#             if result.success and result.data:
#                 self.task_outputs[task.id] = result.data
#                 completed_task_ids.add(task.id)
                
#                 symbols = self._extract_symbols_from_output(result.data)
#                 self.logger.info(
#                     f"[{flow_id}] Task {task.id} COMPLETED - {len(symbols)} symbols"
#                 )
#             else:
#                 self.logger.warning(f"[{flow_id}] Task {task.id} FAILED")
        
#         return results
    
#     # ========================================================================
#     # PARALLEL EXECUTION
#     # ========================================================================
    
#     async def _execute_parallel(
#         self,
#         plan: TaskPlan,
#         query: str,
#         chat_history: Any,
#         system_language: str,
#         provider_type: str,
#         model_name: str,
#         flow_id: str,
#         stats: Dict[str, Any],
#         user_id: Optional[int] = None,
#         session_id: Optional[str] = None
#     ) -> List[TaskExecutionResult]:
#         """Execute independent tasks in parallel"""
        
#         # Check for dependencies - parallel only works for independent tasks
#         dependent_tasks = [t for t in plan.tasks if t.dependencies]
#         if dependent_tasks:
#             self.logger.warning(
#                 f"[{flow_id}] Parallel strategy has {len(dependent_tasks)} "
#                 f"dependent tasks, falling back to sequential"
#             )
#             return await self._execute_sequential(
#                 plan, query, chat_history, system_language,
#                 provider_type, model_name, flow_id, stats, user_id, session_id
#             )
        
#         self.logger.info(f"[{flow_id}] ⚡ Executing {len(plan.tasks)} tasks in parallel")
        
#         # Create coroutines for all tasks
#         coroutines = [
#             self._execute_single_task(
#                 task=task,
#                 query=query,
#                 chat_history=chat_history,
#                 system_language=system_language,
#                 provider_type=provider_type,
#                 model_name=model_name,
#                 flow_id=flow_id,
#                 stats=stats,
#                 user_id=user_id,
#                 session_id=session_id
#             )
#             for task in plan.tasks
#         ]
        
#         # Execute all tasks concurrently
#         start_time = time.time()
#         results_or_exceptions = await asyncio.gather(*coroutines, return_exceptions=True)
#         elapsed = time.time() - start_time
        
#         self.logger.info(f"[{flow_id}] All tasks completed in {elapsed:.2f}s")
        
#         # Process results and handle exceptions
#         results: List[TaskExecutionResult] = []
#         for i, result in enumerate(results_or_exceptions):
#             if isinstance(result, Exception):
#                 self.logger.error(f"[{flow_id}] Task {i+1} raised exception: {result}")
#                 results.append(TaskExecutionResult(
#                     task_id=i+1,
#                     success=False,
#                     status=TaskStatus.FAILED,
#                     data={},
#                     error=str(result),
#                     tools_executed=[],
#                     execution_time=0
#                 ))
#             else:
#                 results.append(result)
#                 if result.success:
#                     self.task_outputs[result.task_id] = result.data
        
#         return results
    
#     # ========================================================================
#     # SINGLE TASK EXECUTION WITH RETRY
#     # ========================================================================
    
#     async def _execute_single_task(
#         self,
#         task: Task,
#         query: str,
#         chat_history: Any,
#         system_language: str,
#         provider_type: str,
#         model_name: str,
#         flow_id: str,
#         stats: Dict[str, Any],
#         user_id: Optional[int] = None,
#         session_id: Optional[str] = None
#     ) -> TaskExecutionResult:
#         """
#         Execute a single task with exponential backoff retry.
        
#         Retry strategy:
#         - Transient errors (network, rate limit): Retry with backoff
#         - Permanent errors (invalid input, not found): Fail immediately
#         """
        
#         retry_count = 0
#         last_error = None
        
#         while retry_count <= self.max_retries:
#             execution_start = time.time()
            
#             if retry_count > 0:
#                 # Apply exponential backoff on retry
#                 delay = calculate_backoff(retry_count - 1, self.RETRY_BASE_DELAY, self.RETRY_MAX_DELAY)
#                 self.logger.info(f"[{flow_id}] Retry {retry_count}/{self.max_retries} after {delay:.1f}s")
#                 await asyncio.sleep(delay)
            
#             try:
#                 # Execute all tools for this task
#                 tool_results = await self._execute_task_tools(
#                     task=task,
#                     query=query,
#                     chat_history=chat_history,
#                     system_language=system_language,
#                     provider_type=provider_type,
#                     model_name=model_name,
#                     flow_id=flow_id,
#                     user_id=user_id,
#                     session_id=session_id
#                 )
                
#                 execution_time = time.time() - execution_start
                
#                 # Check if all tools succeeded
#                 all_success, error_info = self._check_tool_results(tool_results, flow_id)
                
#                 if all_success:
#                     stats["task_timings"][task.id] = execution_time
#                     return self._create_success_result(task, tool_results, execution_time, retry_count)
                
#                 # Check if error is retryable
#                 if error_info and is_retryable_error(error_info.get('message', '')):
#                     last_error = error_info.get('message', 'Unknown error')
#                     retry_count += 1
#                     stats["total_retries"] = stats.get("total_retries", 0) + 1
#                     continue
#                 else:
#                     # Permanent error - skip immediately
#                     return self._create_failure_result(
#                         task=task,
#                         error=error_info.get('message', 'Permanent error'),
#                         tool_results=tool_results,
#                         execution_time=execution_time,
#                         retry_count=retry_count
#                     )
                    
#             except Exception as e:
#                 execution_time = time.time() - execution_start
#                 last_error = str(e)
                
#                 if is_retryable_error(str(e)):
#                     retry_count += 1
#                     stats["total_retries"] = stats.get("total_retries", 0) + 1
#                     continue
#                 else:
#                     return self._create_failure_result(
#                         task=task,
#                         error=str(e),
#                         tool_results={},
#                         execution_time=execution_time,
#                         retry_count=retry_count
#                     )
        
#         # Max retries exhausted
#         return self._create_failure_result(
#             task=task,
#             error=f"Max retries exhausted: {last_error}",
#             tool_results={},
#             execution_time=0,
#             retry_count=retry_count
#         )
    
#     # ========================================================================
#     # TOOL EXECUTION WITH AGGREGATION
#     # ========================================================================
    
#     async def _execute_task_tools(
#         self,
#         task: Task,
#         query: str,
#         chat_history: Any,
#         system_language: str,
#         provider_type: str,
#         model_name: str,
#         flow_id: str,
#         user_id: Optional[int] = None,
#         session_id: Optional[str] = None
#     ) -> Dict[str, Any]:
#         """
#         Execute all tools for a task with aggregation
        
#         When multiple symbols use the same tool, results are aggregated instead of being overwritten.
#         """
        
#         # Group results by tool name (for aggregation)
#         raw_results: Dict[str, List[Dict[str, Any]]] = {}
        
#         for tool_call in task.tools_needed:
#             tool_start = time.time()
            
#             self.logger.debug(f"[{flow_id}]   → {tool_call.tool_name}: {tool_call.params}")
            
#             try:
#                 result = await self.tool_execution_service.execute_single_tool(
#                     tool_name=tool_call.tool_name,
#                     tool_params=tool_call.params,
#                     query=query,
#                     chat_history=chat_history,
#                     system_language=system_language,
#                     provider_type=provider_type,
#                     model_name=model_name,
#                     user_id=user_id,
#                     session_id=session_id
#                 )
                
#                 tool_duration = (time.time() - tool_start) * 1000
#                 status = result.get('status', 'unknown') if isinstance(result, dict) else 'unknown'
                
#                 self.logger.info(
#                     f"[{flow_id}]   ← {tool_call.tool_name}: {status} ({tool_duration:.0f}ms)"
#                 )
                
#                 # Aggregate by tool name
#                 if tool_call.tool_name not in raw_results:
#                     raw_results[tool_call.tool_name] = []
#                 raw_results[tool_call.tool_name].append(result)
                
#             except Exception as e:
#                 self.logger.error(f"[{flow_id}] {tool_call.tool_name}: {e}")
#                 error_result = {
#                     "status": "error",
#                     "error": str(e),
#                     "tool_name": tool_call.tool_name
#                 }
#                 if tool_call.tool_name not in raw_results:
#                     raw_results[tool_call.tool_name] = []
#                 raw_results[tool_call.tool_name].append(error_result)
        
#         # Aggregate results
#         final_results: Dict[str, Any] = {}
#         for tool_name, results_list in raw_results.items():
#             if len(results_list) == 1:
#                 final_results[tool_name] = results_list[0]
#             else:
#                 self.logger.info(
#                     f"[{flow_id}] Aggregating {len(results_list)} results for '{tool_name}'"
#                 )
#                 final_results[tool_name] = self._aggregate_results(results_list)
        
#         return final_results
    
#     def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """
#         Aggregate multiple tool results into single result
        
#         Used when same tool is called for multiple symbols.
#         """
#         if not results:
#             return {}
        
#         first = results[0]
#         tool_name = first.get('tool_name', 'unknown')
        
#         # Merge formatted contexts (for LLM)
#         contexts = [
#             str(r.get('formatted_context', '')) 
#             for r in results 
#             if r.get('formatted_context')
#         ]
#         combined_context = "\n\n".join(contexts)
        
#         # Merge symbols
#         combined_symbols = []
#         for r in results:
#             syms = r.get('symbols', [])
#             if isinstance(syms, list):
#                 combined_symbols.extend(syms)
#             elif isinstance(syms, str):
#                 combined_symbols.append(syms)
#         combined_symbols = list(set(combined_symbols))
        
#         # Merge data
#         combined_data = [r.get('data') for r in results if r.get('data')]
        
#         # Status: success if at least one succeeded
#         has_success = any(
#             r.get('status') in ['200', 200, 'success'] 
#             for r in results
#         )
        
#         return {
#             'tool_name': tool_name,
#             'status': '200' if has_success else 'error',
#             'symbols': combined_symbols,
#             'data': {
#                 'batch_results': combined_data,
#                 'count': len(combined_data),
#             },
#             'execution_time_ms': sum(r.get('execution_time_ms', 0) for r in results),
#             'formatted_context': combined_context,
#             'metadata': {'aggregated': True, 'count': len(results)}
#         }
    
#     # ========================================================================
#     # MULTI-SYMBOL EXPANSION
#     # ========================================================================
    
#     def _expand_for_multi_symbols(
#         self,
#         task: Task,
#         completed_task_ids: set[int],
#         flow_id: str
#     ) -> Task:
#         """
#         Expand task to process ALL symbols from dependencies
        
#         Example:
#         - Input: 1 ToolCall with symbol=<FROM_TASK_1>
#         - Output: N ToolCalls with actual symbols from Task 1
#         """
        
#         if not task.dependencies:
#             return task
        
#         self.logger.info(f"[{flow_id}] Task {task.id} depends on: {task.dependencies}")
        
#         # Collect symbols from dependencies
#         all_symbols = []
#         for dep_id in task.dependencies:
#             if dep_id not in self.task_outputs:
#                 continue
            
#             dep_output = self.task_outputs[dep_id]
#             symbols = self._extract_symbols_from_output(dep_output)
#             all_symbols.extend(symbols)
        
#         # Remove duplicates
#         all_symbols = list(dict.fromkeys(all_symbols))
        
#         if not all_symbols:
#             self.logger.warning(f"[{flow_id}] No symbols from dependencies")
#             return task
        
#         # Limit symbols
#         if len(all_symbols) > self.max_symbols_per_task:
#             self.logger.warning(
#                 f"[{flow_id}] Limiting {len(all_symbols)} symbols to {self.max_symbols_per_task}"
#             )
#             all_symbols = all_symbols[:self.max_symbols_per_task]
        
#         self.logger.info(f"[{flow_id}] Expanding for {len(all_symbols)} symbols: {all_symbols}")
        
#         # Expand tool calls
#         original_tools = task.tools_needed.copy()
#         new_tools = []
        
#         for tool_call in original_tools:
#             # Check for dependency placeholder
#             has_placeholder = any(
#                 isinstance(v, str) and '<FROM_TASK_' in v
#                 for v in tool_call.params.values()
#             )
            
#             if not has_placeholder:
#                 new_tools.append(tool_call)
#                 continue
            
#             # Create one tool call per symbol
#             for symbol in all_symbols:
#                 new_params = tool_call.params.copy()
#                 for param_name, param_value in new_params.items():
#                     if isinstance(param_value, str) and '<FROM_TASK_' in param_value:
#                         new_params[param_name] = symbol
                
#                 new_tools.append(ToolCall(
#                     tool_name=tool_call.tool_name,
#                     params=new_params
#                 ))
        
#         task.tools_needed = new_tools
        
#         self.logger.info(
#             f"[{flow_id}] Expanded: {len(original_tools)} → {len(new_tools)} tool calls"
#         )
        
#         return task
    
#     # ========================================================================
#     # SYMBOL EXTRACTION
#     # ========================================================================
    
#     def _extract_symbols_from_output(self, output: Any) -> List[str]:
#         """
#         Extract symbols from task output
        
#         Handles nested structures from various tools.
#         """
#         symbols = []
        
#         if not isinstance(output, dict):
#             return symbols
        
#         # Locations to search
#         search_targets = [output]
        
#         # Unwrap tool wrappers
#         for key, value in output.items():
#             if isinstance(value, dict):
#                 search_targets.append(value)
#                 if 'data' in value and isinstance(value['data'], dict):
#                     search_targets.append(value['data'])
#             elif isinstance(value, list):
#                 for item in value:
#                     if isinstance(item, dict):
#                         search_targets.append(item)
        
#         # Check 'data' at root
#         if 'data' in output and isinstance(output['data'], dict):
#             search_targets.append(output['data'])
        
#         # Scan all targets
#         for target in search_targets:
#             # Direct symbols field
#             if 'symbols' in target:
#                 val = target['symbols']
#                 if isinstance(val, list):
#                     symbols.extend([str(s) for s in val if s])
#                 elif isinstance(val, str) and val:
#                     symbols.append(val)
            
#             # top_symbols field
#             if 'top_symbols' in target:
#                 val = target['top_symbols']
#                 if isinstance(val, list):
#                     symbols.extend([str(s) for s in val if s])
            
#             # stocks list
#             if 'stocks' in target and isinstance(target['stocks'], list):
#                 for item in target['stocks']:
#                     if isinstance(item, dict) and 'symbol' in item:
#                         symbols.append(str(item['symbol']))
            
#             # results list
#             if 'results' in target and isinstance(target['results'], list):
#                 for item in target['results']:
#                     if isinstance(item, dict) and 'symbol' in item:
#                         symbols.append(str(item['symbol']))
        
#         # Clean and deduplicate
#         clean_symbols = []
#         seen = set()
#         for s in symbols:
#             s_clean = s.strip().upper()
#             if s_clean and s_clean not in seen and len(s_clean) <= 10:
#                 seen.add(s_clean)
#                 clean_symbols.append(s_clean)
        
#         return clean_symbols
    
#     # ========================================================================
#     # RESULT CHECKING
#     # ========================================================================
    
#     def _check_tool_results(
#         self,
#         tool_results: Dict[str, Any],
#         flow_id: str
#     ) -> Tuple[bool, Optional[Dict[str, str]]]:
#         """
#         Check if all tool results are successful
        
#         Returns:
#             (all_success, error_info)
#         """
#         for tool_name, result in tool_results.items():
#             if not isinstance(result, dict):
#                 continue
            
#             status = result.get('status', 'unknown')
            
#             if status in [200, '200', 'success']:
#                 continue
            
#             if status == 'error':
#                 error_msg = result.get('error', 'Unknown error')
#                 self.logger.warning(f"[{flow_id}] {tool_name}: {error_msg[:80]}")
#                 return False, {'tool': tool_name, 'message': error_msg}
        
#         return True, None
    
#     # ========================================================================
#     # RESULT BUILDERS
#     # ========================================================================
    
#     def _create_success_result(
#         self,
#         task: Task,
#         tool_results: Dict[str, Any],
#         execution_time: float,
#         retry_count: int,
#     ) -> TaskExecutionResult:
#         """Create successful task result"""
        
#         task.status = TaskStatus.COMPLETED
#         task.done = True
#         task.results = tool_results
        
#         return TaskExecutionResult(
#             task_id=task.id,
#             status=TaskStatus.COMPLETED,
#             success=True,
#             data=tool_results,
#             tools_executed=list(tool_results.keys()),
#             execution_time=execution_time,
#             retry_count=retry_count,
#         )
    
#     def _create_failure_result(
#         self,
#         task: Task,
#         error: str,
#         tool_results: Dict[str, Any],
#         execution_time: float,
#         retry_count: int,
#     ) -> TaskExecutionResult:
#         """Create failed task result"""
        
#         task.status = TaskStatus.FAILED
#         task.error = error
        
#         return TaskExecutionResult(
#             task_id=task.id,
#             status=TaskStatus.FAILED,
#             success=False,
#             data=tool_results,
#             error=error,
#             tools_executed=list(tool_results.keys()),
#             execution_time=execution_time,
#             retry_count=retry_count,
#         )
    
#     def _create_skip_result(self, task: Task, reason: str) -> TaskExecutionResult:
#         """Create skipped task result"""
        
#         task.status = TaskStatus.SKIPPED
        
#         return TaskExecutionResult(
#             task_id=task.id,
#             status=TaskStatus.SKIPPED,
#             success=False,
#             data={},
#             error=reason,
#             tools_executed=[],
#             execution_time=0,
#             retry_count=0,
#         )
    
#     # ========================================================================
#     # LOGGING
#     # ========================================================================
    
#     def _log_execution_summary(self, flow_id: str, stats: Dict[str, Any]):
#         """Log execution summary with statistics."""
        
#         self.logger.info(f"[{flow_id}] {'=' * 60}")
#         self.logger.info(f"[{flow_id}] [TASK EXEC] EXECUTION SUMMARY")
#         self.logger.info(f"[{flow_id}]   Completed: {stats['completed_tasks']}/{stats['total_tasks']}")
#         self.logger.info(f"[{flow_id}]   Failed: {stats['failed_tasks']}")
#         self.logger.info(f"[{flow_id}]   Skipped: {stats['skipped_tasks']}")
#         self.logger.info(f"[{flow_id}]   Retries: {stats['total_retries']}")
        
#         if 'total_execution_time' in stats:
#             self.logger.info(f"[{flow_id}]   Time: {stats['total_execution_time']:.2f}s")
        
#         self.logger.info(f"[{flow_id}] {'=' * 60}")

"""
Task Executor - Execute task plans with dependency resolution

Location: src/agents/action/task_executor.py

FIXED v2.2:
- SKIP dependent tasks when dependency returns 0 symbols
- Don't execute tools with raw <FROM_TASK_N> placeholder
- Better handling of empty dependency results
"""

import re
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from src.agents.planning.task_models import (
    Task,
    TaskPlan,
    TaskStatus,
    TaskExecutionResult,
    ToolCall,
)
from src.agents.validation.validation_agent import (
    ValidationAgent,
    ValidationResult,
    is_retryable_error,
    calculate_backoff,
)
from src.utils.logger.custom_logging import LoggerMixin


class TaskExecutor(LoggerMixin):
    """
    Executes task plans with sequential dependency handling and multi-symbol support.
    
    Core capabilities:
    - Auto-detect dependencies from <FROM_TASK_N> placeholders
    - Symbol injection between dependent tasks (Task N → Task N+1)
    - Multi-symbol batch processing
    - SKIP tasks when dependencies return no symbols (v2.2 FIX!)
    - Exponential backoff retry for transient errors
    - Graceful degradation on partial failures
    """
    
    # Max symbols to prevent API overload
    MAX_SYMBOLS_PER_TASK = 5
    MAX_RETRIES = 2
    RETRY_BASE_DELAY = 1.0  # seconds
    RETRY_MAX_DELAY = 8.0   # seconds
    
    # Regex pattern for placeholder detection
    PLACEHOLDER_PATTERN = re.compile(r'<FROM_TASK_(\d+)>')
    
    # Known foreign exchange suffixes (>1 char) to skip
    FOREIGN_SUFFIXES = {
        'BA', 'MX', 'NE', 'TO', 'DE', 'PA', 'MI', 'AS', 
        'SW', 'HK', 'SI', 'TW', 'KS', 'AX', 'SA', 'KQ',
        'F',   # Frankfurt
        'L',   # London  
        'T',   # Tokyo
    }

    # US share class
    VALID_US_SUFFIXES = {'A', 'B'}  # BRK.A, BRK.B

    def __init__(
        self,
        tool_execution_service,
        validation_agent: Optional[ValidationAgent] = None,
        max_retries: int = 2,
        max_symbols_per_task: int = 5,
    ):
        """Initialize task executor."""
        super().__init__()
        
        self.tool_execution_service = tool_execution_service
        self.validation_agent = validation_agent or ValidationAgent()
        self.max_retries = max_retries
        self.max_symbols_per_task = max_symbols_per_task
        
        # Track outputs for dependency injection
        self.task_outputs: Dict[int, Dict[str, Any]] = {}
        
        self.logger.info("[TASK_EXECUTOR] Initialized v2.2 (skip-on-empty-deps)")

    
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
        user_id: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute complete task plan."""
        execution_start = time.time()
        
        try:
            self.logger.info(f"[{flow_id}] {'=' * 60}")
            self.logger.info(f"[{flow_id}] [TASK EXEC] Starting execution")
            self.logger.info(f"[{flow_id}]   Tasks: {len(plan.tasks)}")
            self.logger.info(f"[{flow_id}]   Strategy: {plan.strategy}")
            self.logger.info(f"[{flow_id}]   Symbols: {plan.symbols}")
            self.logger.info(f"[{flow_id}] {'=' * 60}")
            
            # Reset execution state
            self.task_outputs = {}
            
            # Initialize statistics
            stats = {
                "total_tasks": len(plan.tasks),
                "completed_tasks": 0,
                "failed_tasks": 0,
                "skipped_tasks": 0,
                "total_retries": 0,
                "task_timings": {},
            }
            
            # Execute based on strategy
            if plan.strategy == "parallel":
                results = await self._execute_parallel(
                    plan=plan,
                    query=query,
                    chat_history=chat_history,
                    system_language=system_language,
                    provider_type=provider_type,
                    model_name=model_name,
                    flow_id=flow_id,
                    stats=stats,
                    user_id=user_id,
                    session_id=session_id
                )
            else:
                results = await self._execute_sequential(
                    plan=plan,
                    query=query,
                    chat_history=chat_history,
                    system_language=system_language,
                    provider_type=provider_type,
                    model_name=model_name,
                    flow_id=flow_id,
                    stats=stats,
                    user_id=user_id, 
                    session_id=session_id
                )
            
            # Calculate final statistics
            stats["completed_tasks"] = sum(1 for r in results if r.success)
            stats["failed_tasks"] = sum(1 for r in results if not r.success and r.status != TaskStatus.SKIPPED)
            stats["skipped_tasks"] = sum(1 for r in results if r.status == TaskStatus.SKIPPED)
            stats["total_execution_time"] = time.time() - execution_start
            
            self._log_execution_summary(flow_id, stats)
            
            return {
                "task_results": results,
                "stats": stats,
                "accumulated_context": self.task_outputs,
                "plan": plan,
            }
            
        except Exception as e:
            self.logger.error(f"[{flow_id}] [TASK EXEC] Fatal error: {e}", exc_info=True)
            return {
                "task_results": [],
                "stats": {"error": str(e)},
                "error": str(e),
            }
    
    # ========================================================================
    # SEQUENTIAL EXECUTION - FIXED v2.2
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
        user_id: Optional[int] = None,    
        session_id: Optional[str] = None     
    ) -> List[TaskExecutionResult]:
        """
        Execute tasks sequentially with dependency resolution.
        
        FIXED v2.2: Skip tasks when dependencies return no symbols.
        """
        
        results: List[TaskExecutionResult] = []
        completed_task_ids: set[int] = set()
        
        for task_index, task in enumerate(plan.tasks):
            self.logger.info(f"[{flow_id}] {'─' * 60}")
            self.logger.info(
                f"[{flow_id}] TASK {task_index + 1}/{len(plan.tasks)}: {task.description}"
            )
            
            # Auto-detect dependencies from <FROM_TASK_N> placeholders
            self._auto_detect_dependencies_from_placeholders(task, flow_id)
            
            # Check dependencies
            if task.dependencies:
                missing_deps = [
                    dep for dep in task.dependencies if dep not in completed_task_ids
                ]
                if missing_deps:
                    self.logger.warning(
                        f"[{flow_id}] ⏭️ Skipping task {task.id}: Missing deps {missing_deps}"
                    )
                    task.status = TaskStatus.SKIPPED
                    results.append(self._create_skip_result(task, "Missing dependencies"))
                    continue
                
                # ============================================================
                # FIXED v2.2: Expand and CHECK if we got any symbols
                # ============================================================
                task, has_symbols = self._expand_for_multi_symbols_safe(
                    task=task,
                    completed_task_ids=completed_task_ids,
                    flow_id=flow_id
                )
                
                # SKIP if no symbols from dependencies
                if not has_symbols:
                    self.logger.warning(
                        f"[{flow_id}] ⏭️ SKIPPING task {task.id}: Dependencies returned 0 symbols"
                    )
                    task.status = TaskStatus.SKIPPED
                    results.append(self._create_skip_result(
                        task, 
                        "Dependencies returned no symbols to process"
                    ))
                    continue
            
            # Execute task with retry
            result = await self._execute_single_task(
                task=task,
                query=query,
                chat_history=chat_history,
                system_language=system_language,
                provider_type=provider_type,
                model_name=model_name,
                flow_id=flow_id,
                stats=stats,
                user_id=user_id,
                session_id=session_id
            )
            
            results.append(result)
            
            # Track outputs for dependency injection
            if result.success and result.data:
                self.task_outputs[task.id] = result.data
                completed_task_ids.add(task.id)
                
                symbols = self._extract_symbols_from_output(result.data)
                self.logger.info(
                    f"[{flow_id}] ✅ Task {task.id} COMPLETED - {len(symbols)} symbols"
                )
            else:
                self.logger.warning(f"[{flow_id}] ❌ Task {task.id} FAILED")
        
        return results
    
    # ========================================================================
    # PARALLEL EXECUTION
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
        user_id: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> List[TaskExecutionResult]:
        """Execute independent tasks in parallel"""
        
        # Check for dependencies - parallel only works for independent tasks
        dependent_tasks = [t for t in plan.tasks if t.dependencies]
        has_placeholders = any(self._task_has_placeholder(t) for t in plan.tasks)
        
        if dependent_tasks or has_placeholders:
            self.logger.warning(
                f"[{flow_id}] Parallel strategy has dependencies/placeholders, "
                f"falling back to sequential"
            )
            return await self._execute_sequential(
                plan, query, chat_history, system_language,
                provider_type, model_name, flow_id, stats, user_id, session_id
            )
        
        self.logger.info(f"[{flow_id}] ⚡ Executing {len(plan.tasks)} tasks in parallel")
        
        coroutines = [
            self._execute_single_task(
                task=task,
                query=query,
                chat_history=chat_history,
                system_language=system_language,
                provider_type=provider_type,
                model_name=model_name,
                flow_id=flow_id,
                stats=stats,
                user_id=user_id,
                session_id=session_id
            )
            for task in plan.tasks
        ]
        
        start_time = time.time()
        results_or_exceptions = await asyncio.gather(*coroutines, return_exceptions=True)
        elapsed = time.time() - start_time
        
        self.logger.info(f"[{flow_id}] All tasks completed in {elapsed:.2f}s")
        
        results: List[TaskExecutionResult] = []
        for i, result in enumerate(results_or_exceptions):
            if isinstance(result, Exception):
                self.logger.error(f"[{flow_id}] Task {i+1} raised exception: {result}")
                results.append(TaskExecutionResult(
                    task_id=i+1,
                    success=False,
                    status=TaskStatus.FAILED,
                    data={},
                    error=str(result),
                    tools_executed=[],
                    execution_time=0
                ))
            else:
                results.append(result)
                if result.success:
                    self.task_outputs[result.task_id] = result.data
        
        return results
    
    # ========================================================================
    # AUTO-DETECT DEPENDENCIES
    # ========================================================================
    
    def _task_has_placeholder(self, task: Task) -> bool:
        """Check if task has any <FROM_TASK_N> placeholder in params."""
        for tool_call in task.tools_needed:
            for param_value in tool_call.params.values():
                if isinstance(param_value, str) and '<FROM_TASK_' in param_value:
                    return True
        return False
    
    def _auto_detect_dependencies_from_placeholders(self, task: Task, flow_id: str) -> None:
        """Auto-detect dependencies from <FROM_TASK_N> placeholders."""
        detected_deps = set()
        
        for tool_call in task.tools_needed:
            for param_name, param_value in tool_call.params.items():
                if isinstance(param_value, str):
                    matches = self.PLACEHOLDER_PATTERN.findall(param_value)
                    for match in matches:
                        detected_deps.add(int(match))
        
        if detected_deps:
            existing_deps = set(task.dependencies) if task.dependencies else set()
            new_deps = detected_deps - existing_deps
            
            if new_deps:
                task.dependencies = sorted(list(existing_deps | detected_deps))
                self.logger.info(
                    f"[{flow_id}] 🔧 AUTO-DETECTED dependencies: {list(new_deps)} "
                    f"→ task.dependencies = {task.dependencies}"
                )
    
    # ========================================================================
    # SINGLE TASK EXECUTION WITH RETRY
    # ========================================================================
    
    async def _execute_single_task(
        self,
        task: Task,
        query: str,
        chat_history: Any,
        system_language: str,
        provider_type: str,
        model_name: str,
        flow_id: str,
        stats: Dict[str, Any],
        user_id: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> TaskExecutionResult:
        """Execute a single task with exponential backoff retry."""
        
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            execution_start = time.time()
            
            if retry_count > 0:
                delay = calculate_backoff(retry_count - 1, self.RETRY_BASE_DELAY, self.RETRY_MAX_DELAY)
                self.logger.info(f"[{flow_id}] Retry {retry_count}/{self.max_retries} after {delay:.1f}s")
                await asyncio.sleep(delay)
            
            try:
                tool_results = await self._execute_task_tools(
                    task=task,
                    query=query,
                    chat_history=chat_history,
                    system_language=system_language,
                    provider_type=provider_type,
                    model_name=model_name,
                    flow_id=flow_id,
                    user_id=user_id,
                    session_id=session_id
                )
                
                execution_time = time.time() - execution_start
                all_success, error_info = self._check_tool_results(tool_results, flow_id)
                
                if all_success:
                    stats["task_timings"][task.id] = execution_time
                    return self._create_success_result(task, tool_results, execution_time, retry_count)
                
                if error_info and is_retryable_error(error_info.get('message', '')):
                    last_error = error_info.get('message', 'Unknown error')
                    retry_count += 1
                    stats["total_retries"] = stats.get("total_retries", 0) + 1
                    continue
                else:
                    return self._create_failure_result(
                        task=task,
                        error=error_info.get('message', 'Permanent error'),
                        tool_results=tool_results,
                        execution_time=execution_time,
                        retry_count=retry_count
                    )
                    
            except Exception as e:
                execution_time = time.time() - execution_start
                last_error = str(e)
                
                if is_retryable_error(str(e)):
                    retry_count += 1
                    stats["total_retries"] = stats.get("total_retries", 0) + 1
                    continue
                else:
                    return self._create_failure_result(
                        task=task,
                        error=str(e),
                        tool_results={},
                        execution_time=execution_time,
                        retry_count=retry_count
                    )
        
        return self._create_failure_result(
            task=task,
            error=f"Max retries exhausted: {last_error}",
            tool_results={},
            execution_time=0,
            retry_count=retry_count
        )
    
    # ========================================================================
    # TOOL EXECUTION WITH AGGREGATION
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
        user_id: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute all tools for a task with aggregation."""
        
        raw_results: Dict[str, List[Dict[str, Any]]] = {}
        
        for tool_call in task.tools_needed:
            tool_start = time.time()
            
            self.logger.debug(f"[{flow_id}]   → {tool_call.tool_name}: {tool_call.params}")
            
            try:
                result = await self.tool_execution_service.execute_single_tool(
                    tool_name=tool_call.tool_name,
                    tool_params=tool_call.params,
                    query=query,
                    chat_history=chat_history,
                    system_language=system_language,
                    provider_type=provider_type,
                    model_name=model_name,
                    user_id=user_id,
                    session_id=session_id
                )
                
                tool_duration = (time.time() - tool_start) * 1000
                status = result.get('status', 'unknown') if isinstance(result, dict) else 'unknown'
                
                self.logger.info(
                    f"[{flow_id}]   ← {tool_call.tool_name}: {status} ({tool_duration:.0f}ms)"
                )
                
                if tool_call.tool_name not in raw_results:
                    raw_results[tool_call.tool_name] = []
                raw_results[tool_call.tool_name].append(result)
                
            except Exception as e:
                self.logger.error(f"[{flow_id}] {tool_call.tool_name}: {e}")
                error_result = {
                    "status": "error",
                    "error": str(e),
                    "tool_name": tool_call.tool_name
                }
                if tool_call.tool_name not in raw_results:
                    raw_results[tool_call.tool_name] = []
                raw_results[tool_call.tool_name].append(error_result)
        
        final_results: Dict[str, Any] = {}
        for tool_name, results_list in raw_results.items():
            if len(results_list) == 1:
                final_results[tool_name] = results_list[0]
            else:
                self.logger.info(
                    f"[{flow_id}] Aggregating {len(results_list)} results for '{tool_name}'"
                )
                final_results[tool_name] = self._aggregate_results(results_list)
        
        return final_results
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple tool results into single result."""
        if not results:
            return {}
        
        first = results[0]
        tool_name = first.get('tool_name', 'unknown')
        
        contexts = [
            str(r.get('formatted_context', '')) 
            for r in results 
            if r.get('formatted_context')
        ]
        combined_context = "\n\n".join(contexts)
        
        combined_symbols = []
        for r in results:
            syms = r.get('symbols', [])
            if isinstance(syms, list):
                combined_symbols.extend(syms)
            elif isinstance(syms, str):
                combined_symbols.append(syms)
        combined_symbols = list(set(combined_symbols))
        
        combined_data = [r.get('data') for r in results if r.get('data')]
        
        has_success = any(
            r.get('status') in ['200', 200, 'success'] 
            for r in results
        )
        
        return {
            'tool_name': tool_name,
            'status': '200' if has_success else 'error',
            'symbols': combined_symbols,
            'data': {
                'batch_results': combined_data,
                'count': len(combined_data),
            },
            'execution_time_ms': sum(r.get('execution_time_ms', 0) for r in results),
            'formatted_context': combined_context,
            'metadata': {'aggregated': True, 'count': len(results)}
        }
    
    # ========================================================================
    # MULTI-SYMBOL EXPANSION - FIXED v2.2
    # ========================================================================
    
    def _expand_for_multi_symbols_safe(
        self,
        task: Task,
        completed_task_ids: set[int],
        flow_id: str
    ) -> Tuple[Task, bool]:
        """
        Expand task to process ALL symbols from dependencies.
        
        FIXED v2.2: Returns tuple (task, has_symbols) to indicate success.
        
        Returns:
            Tuple of (modified_task, has_symbols_to_process)
        """
        
        if not task.dependencies:
            return task, True  # No dependencies = proceed normally
        
        self.logger.info(f"[{flow_id}] 🔗 Task {task.id} depends on: {task.dependencies}")
        
        # Collect symbols from dependencies
        all_symbols = []
        for dep_id in task.dependencies:
            if dep_id not in self.task_outputs:
                self.logger.warning(f"[{flow_id}] ⚠️ Dependency {dep_id} not in task_outputs")
                continue
            
            dep_output = self.task_outputs[dep_id]
            symbols = self._extract_symbols_from_output(dep_output)
            self.logger.info(f"[{flow_id}] 📊 Task {dep_id} provided {len(symbols)} symbols: {symbols[:10]}")
            all_symbols.extend(symbols)
        
        # Remove duplicates
        all_symbols = list(dict.fromkeys(all_symbols))
        
        # ================================================================
        # FIXED v2.2: Return False if no symbols to process
        # ================================================================
        if not all_symbols:
            self.logger.warning(f"[{flow_id}] ⚠️ No symbols from dependencies - will SKIP task")
            return task, False  # Signal to SKIP this task
        
        # Limit symbols
        if len(all_symbols) > self.max_symbols_per_task:
            self.logger.warning(
                f"[{flow_id}] ⚠️ Limiting {len(all_symbols)} symbols to {self.max_symbols_per_task}"
            )
            all_symbols = all_symbols[:self.max_symbols_per_task]
        
        self.logger.info(f"[{flow_id}] 📊 Expanding for {len(all_symbols)} symbols: {all_symbols}")
        
        # Expand tool calls
        original_tools = task.tools_needed.copy()
        new_tools = []
        
        for tool_call in original_tools:
            has_placeholder = any(
                isinstance(v, str) and '<FROM_TASK_' in v
                for v in tool_call.params.values()
            )
            
            if not has_placeholder:
                new_tools.append(tool_call)
                continue
            
            # Create one tool call per symbol
            for symbol in all_symbols:
                new_params = tool_call.params.copy()
                for param_name, param_value in new_params.items():
                    if isinstance(param_value, str) and '<FROM_TASK_' in param_value:
                        new_params[param_name] = symbol
                
                new_tools.append(ToolCall(
                    tool_name=tool_call.tool_name,
                    params=new_params
                ))
        
        task.tools_needed = new_tools
        
        self.logger.info(
            f"[{flow_id}] ✅ EXPANDED: {len(original_tools)} templates → {len(new_tools)} tool calls"
        )
        
        return task, True  # Success - has symbols to process
    
    # ========================================================================
    # SYMBOL EXTRACTION
    # ========================================================================
    
    def _extract_symbols_from_output(self, output: Any) -> List[str]:
        """Extract symbols from task output with relaxed filtering."""
        symbols = []
        
        if not isinstance(output, dict):
            return symbols
        
        search_targets = [output]
        
        for key, value in output.items():
            if isinstance(value, dict):
                search_targets.append(value)
                if 'data' in value and isinstance(value['data'], dict):
                    search_targets.append(value['data'])
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        search_targets.append(item)
        
        if 'data' in output and isinstance(output['data'], dict):
            search_targets.append(output['data'])
        
        for target in search_targets:
            if 'top_symbols' in target:
                val = target['top_symbols']
                if isinstance(val, list):
                    symbols.extend([str(s) for s in val if s])
            
            if 'symbols' in target:
                val = target['symbols']
                if isinstance(val, list):
                    symbols.extend([str(s) for s in val if s])
                elif isinstance(val, str) and val:
                    symbols.append(val)
            
            if 'stocks' in target and isinstance(target['stocks'], list):
                for item in target['stocks']:
                    if isinstance(item, dict) and 'symbol' in item:
                        symbols.append(str(item['symbol']))
            
            if 'results' in target and isinstance(target['results'], list):
                for item in target['results']:
                    if isinstance(item, dict) and 'symbol' in item:
                        symbols.append(str(item['symbol']))
        
        clean_symbols = []
        seen = set()
        
        for s in symbols:
            s_clean = s.strip().upper()
            
            if not s_clean or s_clean in seen or len(s_clean) > 10:
                continue
            
            if '.' in s_clean:
                parts = s_clean.split('.')
                suffix = parts[-1]
                
                # Skip if foreign suffix (F, L, T, DE, BA, etc.)
                if suffix in self.FOREIGN_SUFFIXES:
                    self.logger.debug(f"[EXTRACT] Skipping foreign symbol: {s_clean}")
                    continue

                # Only allow .A, .B (US share class like BRK.A)
                if suffix not in self.VALID_US_SUFFIXES:
                    self.logger.debug(f"[EXTRACT] Skipping unknown suffix: {s_clean}")
                    continue
            
            seen.add(s_clean)
            clean_symbols.append(s_clean)
        
        if clean_symbols:
            self.logger.info(f"[EXTRACT] Extracted {len(clean_symbols)} symbols: {clean_symbols[:10]}")
        
        return clean_symbols
    
    # ========================================================================
    # RESULT CHECKING
    # ========================================================================
    
    def _check_tool_results(
        self,
        tool_results: Dict[str, Any],
        flow_id: str
    ) -> Tuple[bool, Optional[Dict[str, str]]]:
        """Check if all tool results are successful."""
        for tool_name, result in tool_results.items():
            if not isinstance(result, dict):
                continue
            
            status = result.get('status', 'unknown')
            
            if status in [200, '200', 'success']:
                continue
            
            if status == 'error':
                error_msg = result.get('error', 'Unknown error')
                self.logger.warning(f"[{flow_id}] {tool_name}: {error_msg[:80]}")
                return False, {'tool': tool_name, 'message': error_msg}
        
        return True, None
    
    # ========================================================================
    # RESULT BUILDERS
    # ========================================================================
    
    def _create_success_result(
        self,
        task: Task,
        tool_results: Dict[str, Any],
        execution_time: float,
        retry_count: int,
    ) -> TaskExecutionResult:
        """Create successful task result."""
        task.status = TaskStatus.COMPLETED
        task.done = True
        task.results = tool_results
        
        return TaskExecutionResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            success=True,
            data=tool_results,
            tools_executed=list(tool_results.keys()),
            execution_time=execution_time,
            retry_count=retry_count,
        )
    
    def _create_failure_result(
        self,
        task: Task,
        error: str,
        tool_results: Dict[str, Any],
        execution_time: float,
        retry_count: int,
    ) -> TaskExecutionResult:
        """Create failed task result."""
        task.status = TaskStatus.FAILED
        task.error = error
        
        return TaskExecutionResult(
            task_id=task.id,
            status=TaskStatus.FAILED,
            success=False,
            data=tool_results,
            error=error,
            tools_executed=list(tool_results.keys()),
            execution_time=execution_time,
            retry_count=retry_count,
        )
    
    def _create_skip_result(self, task: Task, reason: str) -> TaskExecutionResult:
        """Create skipped task result."""
        task.status = TaskStatus.SKIPPED
        
        return TaskExecutionResult(
            task_id=task.id,
            status=TaskStatus.SKIPPED,
            success=False,
            data={},
            error=reason,
            tools_executed=[],
            execution_time=0,
            retry_count=0,
        )
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    
    def _log_execution_summary(self, flow_id: str, stats: Dict[str, Any]):
        """Log execution summary."""
        self.logger.info(f"[{flow_id}] {'=' * 60}")
        self.logger.info(f"[{flow_id}] [TASK EXEC] EXECUTION SUMMARY")
        self.logger.info(f"[{flow_id}]   Completed: {stats['completed_tasks']}/{stats['total_tasks']}")
        self.logger.info(f"[{flow_id}]   Failed: {stats['failed_tasks']}")
        self.logger.info(f"[{flow_id}]   Skipped: {stats['skipped_tasks']}")
        self.logger.info(f"[{flow_id}]   Retries: {stats['total_retries']}")
        
        if 'total_execution_time' in stats:
            self.logger.info(f"[{flow_id}]   Time: {stats['total_execution_time']:.2f}s")
        
        self.logger.info(f"[{flow_id}] {'=' * 60}")