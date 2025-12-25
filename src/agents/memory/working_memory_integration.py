from typing import Dict, List, Optional, Any
from datetime import datetime

from src.utils.logger.custom_logging import LoggerMixin
from src.agents.memory.working_memory import (
    WorkingMemory,
    WorkingMemoryManager,
    WorkingMemoryStats,
    EntryType,
    Priority,
    get_working_memory,
    get_working_memory_manager,
)


class WorkingMemoryIntegration(LoggerMixin):
    """
    Integration layer for Working Memory in chat flow
    
    Features:
    - Symbol continuity: Symbols persist across turns
    - Turn tracking: Track which turn symbols were added
    - Stale symbol cleanup: Auto-clear symbols after N turns
    - Core Memory aware: Can include Core Memory context
    
    Provides methods matching the 7-phase pipeline stages:
    - Phase 1: Load Context (get_context_for_planning)
    - Phase 2: Planning (save_classification, save_plan)
    - Phase 4: Tool Execution (save_tool_result)
    - Phase 5: Context Assembly (get_context_for_synthesis)
    - Phase 7: Cleanup (complete_request)
    """
    
    # Configuration
    SYMBOL_MAX_TURNS = 5  # Keep symbols for this many turns
    AUTO_CLEAR_STALE_SYMBOLS = True
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        flow_id: str,
        max_tokens: int = 8000,
    ):
        """
        Initialize integration for a request
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            flow_id: Flow ID for logging
            max_tokens: Working memory token limit
        """
        super().__init__()
        
        self.session_id = session_id
        self.user_id = user_id
        self.flow_id = flow_id
        
        # Get or create working memory for this session
        self.memory = get_working_memory(
            session_id=session_id,
            user_id=user_id,
            max_tokens=max_tokens,
        )
        
        # Increment turn counter for this new request
        turn = self.memory.increment_turn()
        
        # Track entries saved in this request
        self._saved_in_request: List[str] = []
        
        # Auto-clear stale symbols at start of request
        if self.AUTO_CLEAR_STALE_SYMBOLS:
            self.memory.clear_stale_symbols(max_turns=self.SYMBOL_MAX_TURNS)
        
        self.logger.info(
            f"[{self.flow_id}] [WORKING_MEMORY] Integration initialized "
            f"(session={session_id[:8]}, turn={turn})"
        )
    
    # ========================================================================
    # PHASE 1 & 2: CONTEXT FOR PLANNING
    # ========================================================================
    
    def get_context_for_planning(
        self,
        max_tokens: int = 1000,
        include_symbols: bool = True,
    ) -> str:
        """
        Get working memory context for Planning Agent
        
        FIXED: Includes symbols from previous turns for continuity
        
        Args:
            max_tokens: Max tokens to return
            include_symbols: Whether to include symbols
            
        Returns:
            Formatted context string
        """
        include_types = [EntryType.USER_CONTEXT]
        
        if include_symbols:
            include_types.append(EntryType.SYMBOLS)
        
        context = self.memory.get_context_for_llm(
            max_tokens=max_tokens,
            include_types=include_types,
        )
        
        # Log symbol context
        if include_symbols:
            symbols = self.memory.get_current_symbols()
            if symbols:
                metadata = self.memory.get_symbols_metadata()
                turns_old = metadata.get('turns_old', 0) if metadata else 0
                self.logger.info(
                    f"[{self.flow_id}] [WORKING_MEMORY] Symbols in context: "
                    f"{symbols} (from {turns_old} turns ago)"
                )
        
        return context
    
    def get_current_symbols(self) -> List[str]:
        """
        Get current symbols in context
        
        Returns symbols from previous turns for follow-up queries
        """
        return self.memory.get_current_symbols()
    
    def get_symbols_with_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get current symbols with metadata (source, turn, staleness)
        """
        return self.memory.get_symbols_metadata()
    
    # ========================================================================
    # PHASE 2: SAVE CLASSIFICATION & PLAN
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
        
        FIXED: Symbols are merged with existing symbols (not replaced)
        
        Args:
            query_type: Type of query
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
        
        # Save symbols if present (merges with existing)
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
            f"[{self.flow_id}] [WORKING_MEMORY] Saved plan: {task_count} tasks"
        )
    
    # ========================================================================
    # PHASE 4: SAVE TOOL RESULTS
    # ========================================================================
    
    def save_tool_result(
        self,
        tool_name: str,
        result: Any,
        task_id: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
        status: str = "success",
    ) -> None:
        """
        Save tool execution result
        
        Args:
            tool_name: Name of the tool
            result: Tool result data
            task_id: Associated task ID
            execution_time_ms: Execution time
            status: Execution status
        """
        # Add status to result if it's a dict
        if isinstance(result, dict):
            result_with_status = {**result, "_status": status}
        else:
            result_with_status = {"data": result, "_status": status}
        
        entry_id = self.memory.add_tool_output(
            tool_name=tool_name,
            output=result_with_status,
            task_id=task_id,
            execution_time_ms=execution_time_ms,
        )
        self._saved_in_request.append(entry_id)
        
        self.logger.debug(
            f"[{self.flow_id}] [WORKING_MEMORY] Saved tool result: "
            f"{tool_name} ({status})"
        )
    
    # ========================================================================
    # PHASE 5: CONTEXT FOR SYNTHESIS
    # ========================================================================
    
    def get_context_for_synthesis(
        self,
        max_tokens: int = 2000,
        include_tool_outputs: bool = True,
    ) -> str:
        """
        Get working memory context for response synthesis
        
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
    # ERROR HANDLING
    # ========================================================================
    
    def save_error(
        self,
        error_type: str,
        message: str,
        recoverable: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save error for potential recovery
        
        Args:
            error_type: Type of error
            message: Error message
            recoverable: Whether error is recoverable
            context: Additional context
        """
        entry_id = self.memory.add_error(
            error_type=error_type,
            message=message,
            recoverable=recoverable,
            context=context,
        )
        self._saved_in_request.append(entry_id)
        
        self.logger.warning(
            f"[{self.flow_id}] [WORKING_MEMORY] Saved error: "
            f"type={error_type}, recoverable={recoverable}"
        )
    
    # ========================================================================
    # LIFECYCLE METHODS
    # ========================================================================
    
    def complete_request(
        self,
        clear_task_data: bool = True,
        clear_symbols: bool = False,
    ) -> None:
        """
        Complete the current request
        
        FIXED: Symbols are NOT cleared by default
        
        Args:
            clear_task_data: Whether to clear task-specific data
            clear_symbols: Whether to also clear symbols (default: False)
        """
        entries_saved = len(self._saved_in_request)
        
        if clear_task_data:
            cleared = self.memory.clear_task()
            
            # Optionally clear symbols too
            if clear_symbols:
                self.memory.clear_symbols()
                self.logger.info(
                    f"[{self.flow_id}] [WORKING_MEMORY] Request complete: "
                    f"saved={entries_saved}, cleared={cleared} (symbols cleared)"
                )
            else:
                # Log that symbols are preserved
                symbols = self.memory.get_current_symbols()
                self.logger.info(
                    f"[{self.flow_id}] [WORKING_MEMORY] Request complete: "
                    f"saved={entries_saved}, cleared={cleared}, "
                    f"symbols preserved: {symbols}"
                )
        else:
            self.logger.info(
                f"[{self.flow_id}] [WORKING_MEMORY] Request complete: "
                f"saved={entries_saved} (not cleared)"
            )
        
        self._saved_in_request.clear()
    
    def force_clear_symbols(self) -> None:
        """
        Force clear symbols (use when user clearly changes topic)
        """
        self.memory.clear_symbols()
        self.logger.info(
            f"[{self.flow_id}] [WORKING_MEMORY] Symbols force cleared"
        )
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get working memory statistics"""
        stats = self.memory.get_stats()
        return {
            "session_id": self.session_id[:8] + "...",
            "current_turn": self.memory.get_current_turn(),
            "total_entries": stats.total_entries,
            "total_tokens": stats.total_tokens,
            "entries_by_type": stats.entries_by_type,
            "symbols_turn_count": stats.symbols_turn_count,
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
                    "expected_data": getattr(task, 'expected_data', []),
                    "priority": getattr(task, 'priority', 'medium'),
                    "dependencies": getattr(task, 'dependencies', []),
                })
            
            return {
                "query_intent": getattr(task_plan, 'query_intent', ''),
                "strategy": getattr(task_plan, 'strategy', 'parallel'),
                "symbols": getattr(task_plan, 'symbols', []),
                "response_language": getattr(task_plan, 'response_language', 'auto'),
                "reasoning": getattr(task_plan, 'reasoning', ''),
                "tasks": tasks,
            }
        except Exception as e:
            self.logger.error(f"Error converting TaskPlan: {e}")
            return {}

    def get_context_for_memory_update(self) -> Optional[str]:
        """
        Get Working Memory context formatted for Memory Update Agent
        
        This context helps the extraction LLM understand:
        - What symbols are currently being discussed
        - What the user's intent was (from classification)
        - What tools were executed
        
        Returns:
            Formatted string with current context, or None if empty
        """
        context_parts = []
        
        # 1. Get current symbols (CRITICAL for reference resolution)
        symbols = self.memory.get_current_symbols()
        if symbols:
            context_parts.append(f"Current symbols in context: {', '.join(symbols)}")
        
        # 2. Get query context (user intent)
        query_context = self.memory.get_query_context()
        if query_context:
            intent = query_context.get('intent', '')
            if intent:
                context_parts.append(f"User intent: {intent[:200]}")
        
        # 3. Get tool outputs summary
        tool_outputs = self.memory.get_tool_outputs(limit=5)
        if tool_outputs:
            tool_names = []
            for output in tool_outputs:
                if isinstance(output, dict):
                    tool_name = output.get('tool_name', 'unknown')
                    tool_names.append(tool_name)
            if tool_names:
                context_parts.append(f"Tools executed: {', '.join(tool_names)}")
        
        if context_parts:
            return "\n".join(context_parts)
        
        return None

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def setup_working_memory_for_request(
    session_id: str,
    user_id: str,
    flow_id: str,
    max_tokens: int = 8000,
) -> WorkingMemoryIntegration:
    """
    Setup working memory integration for a new request
    
    This is the main entry point for chat handlers
    
    Args:
        session_id: Session identifier
        user_id: User identifier
        flow_id: Flow ID for logging
        max_tokens: Working memory token limit
        
    Returns:
        WorkingMemoryIntegration instance
    """
    return WorkingMemoryIntegration(
        session_id=session_id,
        user_id=user_id,
        flow_id=flow_id,
        max_tokens=max_tokens,
    )


def get_working_memory_stats(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get statistics for a session's working memory
    
    Args:
        session_id: Session identifier
        
    Returns:
        Stats dict or None if session not found
    """
    manager = get_working_memory_manager()
    wm = manager.get(session_id)
    
    if wm:
        return wm.get_stats().__dict__
    return None


def cleanup_expired_sessions() -> int:
    """
    Cleanup expired working memory sessions
    
    Call this periodically (e.g., via scheduler)
    
    Returns:
        Number of sessions cleaned up
    """
    manager = get_working_memory_manager()
    # This triggers internal cleanup
    manager._cleanup_old_sessions()
    return 0  # Cleanup happens internally