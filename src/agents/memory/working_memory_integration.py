# File: src/agents/memory/working_memory_integration.py
"""
Working Memory Integration Helpers

This module provides integration points between WorkingMemory and:
- Chat Handler (main flow)
- Planning Agent (save plans)
- Task Executor (save tool outputs)
- Synthesis Agent (retrieve context)

Usage in chat_handler.py:
    from src.agents.memory.working_memory_integration import (
        WorkingMemoryIntegration,
        setup_working_memory_for_request,
    )
    
    # At the start of request
    wm_integration = setup_working_memory_for_request(session_id, user_id)
    
    # After planning
    wm_integration.save_plan(task_plan)
    
    # After each tool execution
    wm_integration.save_tool_result(tool_name, result, task_id)
    
    # Before synthesis - get context
    context = wm_integration.get_context_for_synthesis()
    
    # After response - cleanup
    wm_integration.complete_request()

Architecture:
┌──────────────────────────────────────────────────────────────────────────┐
│                        CHAT REQUEST FLOW                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐          │
│   │  User   │───▶│ Planning │───▶│ Executor │───▶│ Synthesis │          │
│   │  Query  │    │  Agent   │    │          │    │   Agent   │          │
│   └─────────┘    └────┬─────┘    └────┬─────┘    └─────┬─────┘          │
│                       │               │                │                │
│                       ▼               ▼                ▼                │
│                 ┌─────────────────────────────────────────────┐         │
│                 │           WorkingMemoryIntegration          │         │
│                 │                                             │         │
│                 │  • save_classification()  ← from Planning   │         │
│                 │  • save_plan()            ← from Planning   │         │
│                 │  • save_tool_result()     ← from Executor   │         │
│                 │  • get_context_for_*()    → to Synthesis    │         │
│                 │  • complete_request()     ← cleanup         │         │
│                 │                                             │         │
│                 └─────────────────────────────────────────────┘         │
│                                     │                                   │
│                                     ▼                                   │
│                 ┌─────────────────────────────────────────────┐         │
│                 │               WorkingMemory                  │         │
│                 │              (Per Session)                   │         │
│                 └─────────────────────────────────────────────┘         │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.utils.logger.custom_logging import LoggerMixin
from src.agents.memory.working_memory import (
    WorkingMemory,
    WorkingMemoryManager,
    get_working_memory,
    EntryType,
    Priority,
)


# ============================================================================
# WORKING MEMORY INTEGRATION
# ============================================================================

class WorkingMemoryIntegration(LoggerMixin):
    """
    Integration layer between Working Memory and Chat Flow components
    
    This class provides convenient methods for:
    - Saving data from different stages of the flow
    - Retrieving context for different purposes
    - Managing lifecycle of working memory entries
    
    Designed to be instantiated per-request for clean state management.
    """
    
    def __init__(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        flow_id: Optional[str] = None,
    ):
        """
        Initialize integration for a request
        
        Args:
            session_id: Session identifier
            user_id: Optional user identifier
            flow_id: Optional flow identifier for logging
        """
        super().__init__()
        
        self.session_id = session_id
        self.user_id = user_id
        self.flow_id = flow_id or "UNKNOWN"
        
        # Get or create working memory
        self.memory = get_working_memory(session_id, user_id)
        
        # Track what we've saved in this request
        self._saved_in_request: List[str] = []
        
        self.logger.debug(
            f"[{self.flow_id}] [WORKING_MEMORY] Integration initialized "
            f"(session={session_id[:8]}...)"
        )
    
    # ========================================================================
    # SAVE METHODS (from different flow stages)
    # ========================================================================
    
    def save_classification(
        self,
        query_type: str,
        categories: List[str],
        symbols: List[str],
        language: str,
        reasoning: Optional[str] = None,
    ) -> None:
        """
        Save classification results from Planning Agent Stage 1
        
        Args:
            query_type: Classified query type
            categories: Tool categories
            symbols: Extracted symbols
            language: Response language
            reasoning: Optional reasoning
        """
        # Save query context
        entry_id = self.memory.add_query_context(
            intent=reasoning or f"Query type: {query_type}",
            language=language,
            categories=categories,
            query_type=query_type,
        )
        self._saved_in_request.append(entry_id)
        
        # Save symbols if present
        if symbols:
            entry_id = self.memory.add_symbols(symbols, source="classification")
            self._saved_in_request.append(entry_id)
        
        self.logger.info(
            f"[{self.flow_id}] [WORKING_MEMORY] Saved classification: "
            f"type={query_type}, categories={len(categories)}, symbols={symbols}"
        )
    
    def save_plan(
        self,
        task_plan: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save task plan from Planning Agent Stage 3
        
        Args:
            task_plan: TaskPlan object or dict
            metadata: Optional additional metadata
        """
        # Convert TaskPlan to dict if needed
        if hasattr(task_plan, 'to_dict'):
            plan_dict = task_plan.to_dict()
        elif hasattr(task_plan, '__dict__'):
            plan_dict = self._task_plan_to_dict(task_plan)
        else:
            plan_dict = task_plan
        
        entry_id = self.memory.add_plan(plan_dict, metadata)
        self._saved_in_request.append(entry_id)
        
        task_count = len(plan_dict.get('tasks', []))
        self.logger.info(
            f"[{self.flow_id}] [WORKING_MEMORY] Saved plan: "
            f"{task_count} tasks, strategy={plan_dict.get('strategy', 'unknown')}"
        )
    
    def save_tool_result(
        self,
        tool_name: str,
        result: Any,
        task_id: Optional[int] = None,
        execution_time_ms: Optional[int] = None,
        status: str = "success",
    ) -> None:
        """
        Save tool execution result from Task Executor
        
        Args:
            tool_name: Name of the executed tool
            result: Tool output
            task_id: Task ID this belongs to
            execution_time_ms: Execution time
            status: Execution status
        """
        # Wrap result with status if not already a dict
        if isinstance(result, dict):
            output = {**result, "_status": status}
        else:
            output = {"data": result, "_status": status}
        
        entry_id = self.memory.add_tool_output(
            tool_name=tool_name,
            output=output,
            task_id=task_id,
            execution_time_ms=execution_time_ms,
        )
        self._saved_in_request.append(entry_id)
        
        self.logger.debug(
            f"[{self.flow_id}] [WORKING_MEMORY] Saved tool result: "
            f"{tool_name} (task={task_id}, status={status})"
        )
    
    def save_reasoning(
        self,
        reasoning: str,
        category: str = "general",
    ) -> None:
        """
        Save agent's reasoning/analysis
        
        Args:
            reasoning: Reasoning text
            category: Category (intent, analysis, validation, etc.)
        """
        entry_id = self.memory.add_reasoning(reasoning, category)
        self._saved_in_request.append(entry_id)
        
        self.logger.debug(
            f"[{self.flow_id}] [WORKING_MEMORY] Saved reasoning: "
            f"category={category}, length={len(reasoning)}"
        )
    
    def save_intermediate(
        self,
        label: str,
        data: Any,
        step: Optional[int] = None,
    ) -> None:
        """
        Save intermediate result during multi-step task
        
        Args:
            label: Label for this result
            data: The data
            step: Step number
        """
        entry_id = self.memory.add_intermediate(label, data, step)
        self._saved_in_request.append(entry_id)
        
        self.logger.debug(
            f"[{self.flow_id}] [WORKING_MEMORY] Saved intermediate: "
            f"label={label}, step={step}"
        )
    
    def save_error(
        self,
        error_type: str,
        message: str,
        recoverable: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save error for potential recovery
        
        Args:
            error_type: Type of error
            message: Error message
            recoverable: Whether recoverable
            details: Additional details
        """
        entry_id = self.memory.add_error(
            error_type=error_type,
            message=message,
            recoverable=recoverable,
            details=details,
        )
        self._saved_in_request.append(entry_id)
        
        self.logger.warning(
            f"[{self.flow_id}] [WORKING_MEMORY] Saved error: "
            f"type={error_type}, recoverable={recoverable}"
        )
    
    # ========================================================================
    # RETRIEVE METHODS (for different stages)
    # ========================================================================
    
    def get_context_for_planning(
        self,
        max_tokens: int = 1000,
    ) -> str:
        """
        Get working memory context for Planning Agent
        
        Includes:
        - Current symbols (from previous queries)
        - User context
        
        Args:
            max_tokens: Max tokens to return
            
        Returns:
            Formatted context string
        """
        return self.memory.get_context_for_llm(
            max_tokens=max_tokens,
            include_types=[
                EntryType.SYMBOLS,
                EntryType.USER_CONTEXT,
            ],
        )
    
    def get_context_for_synthesis(
        self,
        max_tokens: int = 2000,
        include_tool_outputs: bool = True,
    ) -> str:
        """
        Get working memory context for Synthesis Agent
        
        Includes:
        - Query context (intent, language)
        - Symbols
        - Task plan
        - Tool outputs (if requested)
        - Reasoning
        
        Args:
            max_tokens: Max tokens to return
            include_tool_outputs: Whether to include tool outputs
            
        Returns:
            Formatted context string
        """
        include_types = [
            EntryType.QUERY_CONTEXT,
            EntryType.SYMBOLS,
            EntryType.TASK_PLAN,
            EntryType.REASONING,
        ]
        
        if include_tool_outputs:
            include_types.append(EntryType.TOOL_OUTPUT)
        
        return self.memory.get_context_for_llm(
            max_tokens=max_tokens,
            include_types=include_types,
        )
    
    def get_current_symbols(self) -> List[str]:
        """Get current symbols in context"""
        return self.memory.get_current_symbols()
    
    def get_current_plan(self) -> Optional[Dict[str, Any]]:
        """Get current task plan"""
        return self.memory.get_current_plan()
    
    def get_tool_outputs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent tool outputs"""
        return self.memory.get_tool_outputs(limit)
    
    def get_query_context(self) -> Optional[Dict[str, Any]]:
        """Get current query context"""
        return self.memory.get_query_context()
    
    # ========================================================================
    # LIFECYCLE METHODS
    # ========================================================================
    
    def complete_request(
        self,
        clear_task_data: bool = True,
    ) -> None:
        """
        Complete the current request
        
        Call this after the response has been sent to user.
        
        Args:
            clear_task_data: Whether to clear task-specific data
        """
        entries_saved = len(self._saved_in_request)
        
        if clear_task_data:
            cleared = self.memory.clear_task()
            self.logger.info(
                f"[{self.flow_id}] [WORKING_MEMORY] Request complete: "
                f"saved={entries_saved}, cleared={cleared}"
            )
        else:
            self.logger.info(
                f"[{self.flow_id}] [WORKING_MEMORY] Request complete: "
                f"saved={entries_saved} (not cleared)"
            )
        
        self._saved_in_request.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get working memory statistics"""
        stats = self.memory.get_stats()
        return {
            "session_id": self.session_id[:8] + "...",
            "total_entries": stats.total_entries,
            "total_tokens": stats.total_tokens,
            "entries_by_type": stats.entries_by_type,
            "saved_in_request": len(self._saved_in_request),
        }
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _task_plan_to_dict(self, task_plan) -> Dict[str, Any]:
        """Convert TaskPlan object to dictionary"""
        try:
            tasks = []
            for task in getattr(task_plan, 'tasks', []):
                tool_calls = []
                for tc in getattr(task, 'tools_needed', []):
                    tool_calls.append({
                        "tool_name": getattr(tc, 'tool_name', ''),
                        "params": getattr(tc, 'params', {}),
                    })
                
                tasks.append({
                    "id": getattr(task, 'id', 0),
                    "description": getattr(task, 'description', ''),
                    "tools_needed": tool_calls,
                    "dependencies": getattr(task, 'dependencies', []),
                    "priority": getattr(task, 'priority', 'medium'),
                })
            
            return {
                "query_intent": getattr(task_plan, 'query_intent', ''),
                "strategy": getattr(task_plan, 'strategy', 'sequential'),
                "estimated_complexity": getattr(task_plan, 'estimated_complexity', 'simple'),
                "symbols": getattr(task_plan, 'symbols', []),
                "response_language": getattr(task_plan, 'response_language', 'auto'),
                "reasoning": getattr(task_plan, 'reasoning', ''),
                "tasks": tasks,
            }
        except Exception as e:
            self.logger.error(f"Error converting TaskPlan: {e}")
            return {"error": str(e)}


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def setup_working_memory_for_request(
    session_id: str,
    user_id: Optional[str] = None,
    flow_id: Optional[str] = None,
) -> WorkingMemoryIntegration:
    """
    Setup working memory integration for a request
    
    Args:
        session_id: Session identifier
        user_id: Optional user identifier
        flow_id: Optional flow ID for logging
        
    Returns:
        WorkingMemoryIntegration instance
    """
    return WorkingMemoryIntegration(
        session_id=session_id,
        user_id=user_id,
        flow_id=flow_id,
    )


def get_working_memory_stats() -> Dict[str, Any]:
    """Get global working memory manager statistics"""
    manager = WorkingMemoryManager()
    return manager.get_stats()


def cleanup_expired_sessions() -> int:
    """Cleanup expired working memory sessions"""
    manager = WorkingMemoryManager()
    return manager.cleanup_expired()


# ============================================================================
# INTEGRATION EXAMPLES
# ============================================================================

"""
EXAMPLE: Integration with chat_handler.py

```python
# In handle_chat_with_reasoning() method:

async def handle_chat_with_reasoning(
    self,
    query: str,
    session_id: str,
    user_id: str,
    ...
):
    flow_id = generate_flow_id()
    
    # ================================================================
    # SETUP WORKING MEMORY
    # ================================================================
    wm_integration = setup_working_memory_for_request(
        session_id=session_id,
        user_id=user_id,
        flow_id=flow_id,
    )
    
    try:
        # ================================================================
        # PHASE 2: PLANNING
        # ================================================================
        task_plan = await planning_agent.think_and_plan(
            query=query,
            recent_chat=recent_chat,
            ...
        )
        
        # Save classification results
        wm_integration.save_classification(
            query_type=task_plan.query_intent,
            categories=task_plan.get_categories(),
            symbols=task_plan.symbols,
            language=task_plan.response_language,
            reasoning=task_plan.reasoning,
        )
        
        # Save plan
        wm_integration.save_plan(task_plan)
        
        # ================================================================
        # PHASE 4: EXECUTE TOOLS
        # ================================================================
        for result in task_results:
            wm_integration.save_tool_result(
                tool_name=result.tool_name,
                result=result.output,
                task_id=result.task_id,
                execution_time_ms=result.execution_time,
                status=result.status,
            )
        
        # ================================================================
        # PHASE 6: SYNTHESIS
        # ================================================================
        # Get context for synthesis
        wm_context = wm_integration.get_context_for_synthesis(max_tokens=2000)
        
        # Include in synthesis prompt
        synthesis_prompt = f'''
{system_prompt}

<working_memory>
{wm_context}
</working_memory>

User: {query}
'''
        
        response = await synthesis_agent.generate_response(...)
        
        # ================================================================
        # CLEANUP
        # ================================================================
        wm_integration.complete_request(clear_task_data=True)
        
        return response
        
    except Exception as e:
        wm_integration.save_error(
            error_type="flow_error",
            message=str(e),
            recoverable=True,
        )
        wm_integration.complete_request(clear_task_data=False)
        raise
```
"""


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Demo usage
    import asyncio
    
    async def demo():
        # Setup
        session_id = "demo_session_456"
        user_id = "user_1"
        flow_id = "FLOW-123456"
        
        # Create integration
        wm = setup_working_memory_for_request(
            session_id=session_id,
            user_id=user_id,
            flow_id=flow_id,
        )
        
        print("=== Demo: Working Memory Integration ===\n")
        
        # Simulate Planning Agent results
        print("1. Saving classification...")
        wm.save_classification(
            query_type="stock_specific",
            categories=["price", "technical"],
            symbols=["AAPL", "NVDA"],
            language="en",
            reasoning="User wants price and technical analysis for AAPL and NVDA",
        )
        
        # Simulate saving plan
        print("2. Saving plan...")
        wm.save_plan({
            "query_intent": "Analyze stock performance",
            "strategy": "parallel",
            "tasks": [
                {"id": 1, "tool": "getStockPrice", "symbol": "AAPL"},
                {"id": 2, "tool": "getStockPrice", "symbol": "NVDA"},
                {"id": 3, "tool": "getTechnicalIndicators", "symbol": "AAPL"},
            ],
        })
        
        # Simulate tool executions
        print("3. Saving tool results...")
        wm.save_tool_result(
            tool_name="getStockPrice",
            result={"symbol": "AAPL", "price": 195.50, "change": 2.3},
            task_id=1,
            execution_time_ms=150,
        )
        
        wm.save_tool_result(
            tool_name="getStockPrice",
            result={"symbol": "NVDA", "price": 875.20, "change": -1.5},
            task_id=2,
            execution_time_ms=120,
        )
        
        wm.save_tool_result(
            tool_name="getTechnicalIndicators",
            result={"symbol": "AAPL", "rsi": 62.5, "macd": 1.25},
            task_id=3,
            execution_time_ms=200,
        )
        
        # Get context for synthesis
        print("4. Getting context for synthesis...")
        context = wm.get_context_for_synthesis(max_tokens=2000)
        print("\n--- Context for Synthesis ---")
        print(context)
        print("--- End Context ---\n")
        
        # Get stats
        print("5. Getting stats...")
        stats = wm.get_stats()
        print(f"Stats: {stats}\n")
        
        # Complete request
        print("6. Completing request...")
        wm.complete_request(clear_task_data=True)
        
        # Check remaining
        print(f"7. Symbols after cleanup: {wm.get_current_symbols()}")
        
        # Global stats
        print("\n8. Global manager stats:")
        global_stats = get_working_memory_stats()
        print(f"   {global_stats}")
    
    asyncio.run(demo())