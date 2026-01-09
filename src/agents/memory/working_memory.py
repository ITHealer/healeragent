import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.token_counter import TokenCounter


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class EntryType(Enum):
    """Types of working memory entries"""
    QUERY_CONTEXT = "query_context"    # Current query intent, language, categories
    TASK_PLAN = "task_plan"            # Planned tasks from Planning Agent
    TOOL_OUTPUT = "tool_output"        # Results from tool execution
    INTERMEDIATE = "intermediate"      # Intermediate processing results
    REASONING = "reasoning"            # Agent's reasoning/analysis
    SYMBOLS = "symbols"                # Stock/crypto symbols in context
    USER_CONTEXT = "user_context"      # User-provided context for this request
    ERROR = "error"                    # Error information for recovery


class Priority(Enum):
    """Priority levels for entries (affects eviction order)"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4  # Never auto-evict


@dataclass
class WorkingMemoryEntry:
    """Single entry in working memory"""
    entry_id: str
    entry_type: EntryType
    content: Any
    priority: Priority
    created_at: datetime
    expires_at: Optional[datetime]
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0  # Track which turn this entry was created
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def is_stale_symbols(self, current_turn: int, max_turns: int = 5) -> bool:
        """Check if symbol entry is stale (too many turns old)"""
        if self.entry_type != EntryType.SYMBOLS:
            return False
        return (current_turn - self.turn_count) > max_turns


@dataclass
class WorkingMemoryStats:
    """Statistics about working memory state"""
    total_entries: int
    total_tokens: int
    entries_by_type: Dict[str, int]
    tokens_by_type: Dict[str, int]
    oldest_entry: Optional[datetime]
    newest_entry: Optional[datetime]
    symbols_turn_count: Optional[int] = None


# ============================================================================
# MAIN WORKING MEMORY CLASS
# ============================================================================

class WorkingMemory(LoggerMixin):
    """
    Working Memory - Session-scoped scratchpad for task execution
    
    Symbol continuity across turns
    - Symbols are NOT cleared in clear_task()
    - Symbols have TTL based on turn count
    - Explicit clear_symbols() for manual cleanup
    
    Features:
    - Token-based capacity management
    - Priority-based eviction
    - Type-based retrieval
    - Thread-safe operations
    """
    
    # Capacity limits
    DEFAULT_MAX_TOKENS = 8000
    DEFAULT_ENTRY_TTL_SECONDS = 3600  # 1 hour
    
    # Symbol configuration
    SYMBOL_MAX_TURNS = 5  # Keep symbols for N turns before considering stale
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize working memory for a session
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            max_tokens: Maximum token capacity
        """
        super().__init__()
        
        self.session_id = session_id
        self.user_id = user_id
        self.max_tokens = max_tokens
        
        # Storage
        self._entries: Dict[str, WorkingMemoryEntry] = {}
        self._by_type: Dict[EntryType, List[str]] = {t: [] for t in EntryType}
        
        # Token tracking
        self._total_tokens = 0
        self.token_counter = TokenCounter()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Turn tracking for symbol TTL
        self._current_turn = 0
        
        # Timestamps
        self.created_at = datetime.now()
        self.last_modified = datetime.now()
        
        self.logger.debug(
            f"[WORKING_MEMORY] Initialized for session {session_id[:8]}..., "
            f"max_tokens={max_tokens}"
        )
    
    # ========================================================================
    # TURN MANAGEMENT
    # ========================================================================
    
    def increment_turn(self) -> int:
        """Increment turn counter (call at start of each request)"""
        with self._lock:
            self._current_turn += 1
            return self._current_turn
    
    def get_current_turn(self) -> int:
        """Get current turn number"""
        return self._current_turn
    
    # ========================================================================
    # CORE ADD/REMOVE METHODS
    # ========================================================================
    
    def _add_entry(
        self,
        entry_type: EntryType,
        content: Any,
        priority: Priority = Priority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """
        Internal method to add entry with eviction handling
        
        Args:
            entry_type: Type of entry
            content: Entry content
            priority: Priority level
            metadata: Optional metadata
            ttl_seconds: Time-to-live in seconds
            
        Returns:
            Entry ID
        """
        with self._lock:
            # Calculate token count
            token_count = self._estimate_tokens(content)
            
            # Check if we need to evict
            while self._total_tokens + token_count > self.max_tokens:
                if not self._evict_lowest_priority():
                    self.logger.warning(
                        f"[WORKING_MEMORY] Cannot evict, skipping add"
                    )
                    return ""
            
            # Create entry
            entry_id = f"wm_{uuid.uuid4().hex[:8]}"
            expires_at = None
            if ttl_seconds:
                expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            
            entry = WorkingMemoryEntry(
                entry_id=entry_id,
                entry_type=entry_type,
                content=content,
                priority=priority,
                created_at=datetime.now(),
                expires_at=expires_at,
                token_count=token_count,
                metadata=metadata or {},
                turn_count=self._current_turn,
            )
            
            # Store
            self._entries[entry_id] = entry
            self._by_type[entry_type].append(entry_id)
            self._total_tokens += token_count
            self.last_modified = datetime.now()
            
            self.logger.debug(
                f"[WORKING_MEMORY] Added {entry_type.value}: "
                f"{token_count} tokens (total: {self._total_tokens})"
            )
            
            return entry_id
    
    def remove_entry(self, entry_id: str) -> bool:
        """Remove an entry by ID"""
        with self._lock:
            if entry_id not in self._entries:
                return False
            
            entry = self._entries[entry_id]
            
            # Remove from type index
            if entry_id in self._by_type[entry.entry_type]:
                self._by_type[entry.entry_type].remove(entry_id)
            
            # Update token count
            self._total_tokens -= entry.token_count
            
            # Remove from main storage
            del self._entries[entry_id]
            self.last_modified = datetime.now()
            
            return True
    
    def _evict_lowest_priority(self) -> bool:
        """
        Evict lowest priority non-critical entry
        
        FIXED: Never evict SYMBOLS entries during regular eviction
        
        Returns:
            True if eviction succeeded
        """
        with self._lock:
            candidates = []
            
            for entry_id, entry in self._entries.items():
                # Never evict critical entries
                if entry.priority == Priority.CRITICAL:
                    continue
                
                # FIXED: Don't evict symbols during regular eviction
                # Symbols should persist for cross-turn continuity
                if entry.entry_type == EntryType.SYMBOLS:
                    continue
                
                # Prefer expired entries
                if entry.is_expired():
                    candidates.insert(0, (entry_id, entry))
                else:
                    candidates.append((entry_id, entry))
            
            if not candidates:
                return False
            
            # Sort by priority (ascending) then by age (oldest first)
            candidates.sort(
                key=lambda x: (x[1].priority.value, x[1].created_at)
            )
            
            # Evict first candidate
            evict_id = candidates[0][0]
            evict_tokens = self._entries[evict_id].token_count
            
            self.remove_entry(evict_id)
            
            self.logger.debug(
                f"[WORKING_MEMORY] Evicted entry, freed {evict_tokens} tokens"
            )
            
            return True
    
    def _estimate_tokens(self, content: Any) -> int:
        """Estimate token count for content"""
        if isinstance(content, str):
            return self.token_counter.count_tokens(content)
        elif isinstance(content, (dict, list)):
            import json
            text = json.dumps(content, ensure_ascii=False, default=str)
            return self.token_counter.count_tokens(text)
        else:
            return self.token_counter.count_tokens(str(content))
    
    # ========================================================================
    # SPECIALIZED ADD METHODS
    # ========================================================================
    
    def add_plan(
        self,
        plan: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add task plan from Planning Agent
        
        Args:
            plan: Task plan dictionary
            metadata: Optional metadata
            
        Returns:
            Entry ID
        """
        # Clear previous plan first
        self.clear_type(EntryType.TASK_PLAN)
        
        return self._add_entry(
            entry_type=EntryType.TASK_PLAN,
            content=plan,
            priority=Priority.HIGH,
            metadata=metadata,
        )
    
    def add_tool_output(
        self,
        tool_name: str,
        output: Any,
        task_id: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
    ) -> str:
        """
        Add tool execution output
        
        Args:
            tool_name: Name of the tool
            output: Tool output data
            task_id: Associated task ID
            execution_time_ms: Execution time in milliseconds
            
        Returns:
            Entry ID
        """
        metadata = {
            "tool_name": tool_name,
            "task_id": task_id,
            "execution_time_ms": execution_time_ms,
            "executed_at": datetime.now().isoformat(),
        }
        
        return self._add_entry(
            entry_type=EntryType.TOOL_OUTPUT,
            content=output,
            priority=Priority.MEDIUM,
            metadata=metadata,
        )
    
    def add_reasoning(
        self,
        reasoning: str,
        category: Optional[str] = None,
    ) -> str:
        """
        Add agent's reasoning/analysis
        
        Args:
            reasoning: Reasoning text
            category: Optional category (e.g., "intent", "analysis")
            
        Returns:
            Entry ID
        """
        metadata = {"category": category} if category else {}
        
        return self._add_entry(
            entry_type=EntryType.REASONING,
            content=reasoning,
            priority=Priority.MEDIUM,
            metadata=metadata,
        )
    
    def add_symbols(
        self,
        symbols: List[str],
        source: str = "query",
    ) -> str:
        """
        Add extracted symbols for current context

        FIXED: Merges with existing symbols while PRESERVING chronological order.
        Most recent symbols are LAST in the list.

        Order: [oldest, ..., newest]

        Args:
            symbols: List of symbols (e.g., ["AAPL", "NVDA"])
            source: Where symbols came from (query, history, tool)

        Returns:
            Entry ID
        """
        with self._lock:
            # Get existing symbols
            existing_symbols = self.get_current_symbols()

            # =================================================================
            # FIXED: Preserve chronological order while deduplicating
            # Order: existing symbols (older) + new symbols (newer)
            # Duplicates are kept in their LATEST position (most recent mention)
            # =================================================================
            seen = set()
            merged_symbols = []

            # Process all symbols in order: existing first, then new
            # If a symbol appears again in new, it will be moved to the end
            all_symbols = existing_symbols + symbols

            # Reverse iterate to keep the LAST occurrence (most recent)
            for sym in reversed(all_symbols):
                if sym not in seen:
                    seen.add(sym)
                    merged_symbols.insert(0, sym)  # Insert at beginning to maintain order

            # Now merged_symbols is in chronological order: oldest first, newest last

            # Clear old symbols entry
            self.clear_type(EntryType.SYMBOLS)
            
            metadata = {
                "source": source,
                "count": len(merged_symbols),
                "added_symbols": symbols,
                "merged_from": existing_symbols,
            }
            
            return self._add_entry(
                entry_type=EntryType.SYMBOLS,
                content=merged_symbols,
                priority=Priority.HIGH,  # High priority - important for continuity
                metadata=metadata,
            )
    
    def add_intermediate(
        self,
        label: str,
        data: Any,
        step: Optional[int] = None,
    ) -> str:
        """
        Add intermediate result during multi-step task
        
        Args:
            label: Label for this intermediate result
            data: The data
            step: Optional step number
            
        Returns:
            Entry ID
        """
        metadata = {
            "label": label,
            "step": step,
        }
        
        return self._add_entry(
            entry_type=EntryType.INTERMEDIATE,
            content=data,
            priority=Priority.LOW,
            metadata=metadata,
        )
    
    def add_query_context(
        self,
        intent: str,
        language: str = "auto",
        categories: Optional[List[str]] = None,
        query_type: Optional[str] = None,
    ) -> str:
        """
        Add query-specific context from classification
        
        Args:
            intent: Query intent
            language: Response language
            categories: Tool categories
            query_type: Query type (stock_specific, screener, etc.)
            
        Returns:
            Entry ID
        """
        content = {
            "intent": intent,
            "language": language,
            "categories": categories or [],
            "query_type": query_type,
        }
        
        # Clear previous query context
        self.clear_type(EntryType.QUERY_CONTEXT)
        
        return self._add_entry(
            entry_type=EntryType.QUERY_CONTEXT,
            content=content,
            priority=Priority.HIGH,
        )
    
    def add_error(
        self,
        error_type: str,
        message: str,
        recoverable: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add error for potential recovery
        
        Args:
            error_type: Type of error
            message: Error message
            recoverable: Whether error is recoverable
            context: Additional context
            
        Returns:
            Entry ID
        """
        content = {
            "error_type": error_type,
            "message": message,
            "recoverable": recoverable,
            "context": context,
            "occurred_at": datetime.now().isoformat(),
        }
        
        return self._add_entry(
            entry_type=EntryType.ERROR,
            content=content,
            priority=Priority.HIGH if recoverable else Priority.CRITICAL,
        )
    
    # ========================================================================
    # GETTER METHODS
    # ========================================================================
    
    def get_entries_by_type(
        self,
        entry_type: EntryType,
        limit: int = 10,
    ) -> List[WorkingMemoryEntry]:
        """Get entries by type (most recent first)"""
        with self._lock:
            entry_ids = self._by_type.get(entry_type, [])
            entries = [
                self._entries[eid]
                for eid in entry_ids
                if eid in self._entries
            ]
            # Sort by created_at descending
            entries.sort(key=lambda e: e.created_at, reverse=True)
            return entries[:limit]
    
    def get_current_plan(self) -> Optional[Dict[str, Any]]:
        """Get current task plan"""
        entries = self.get_entries_by_type(EntryType.TASK_PLAN, limit=1)
        return entries[0].content if entries else None
    
    def get_current_symbols(self) -> List[str]:
        """Get current symbols in context"""
        entries = self.get_entries_by_type(EntryType.SYMBOLS, limit=1)
        return entries[0].content if entries else []
    
    def get_symbols_metadata(self) -> Optional[Dict[str, Any]]:
        """Get metadata about current symbols (source, turn, etc.)"""
        entries = self.get_entries_by_type(EntryType.SYMBOLS, limit=1)
        if entries:
            return {
                "symbols": entries[0].content,
                "source": entries[0].metadata.get("source"),
                "turn_count": entries[0].turn_count,
                "current_turn": self._current_turn,
                "turns_old": self._current_turn - entries[0].turn_count,
            }
        return None
    
    def get_tool_outputs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent tool outputs with metadata"""
        entries = self.get_entries_by_type(EntryType.TOOL_OUTPUT, limit=limit)
        return [
            {
                "tool_name": e.metadata.get("tool_name"),
                "task_id": e.metadata.get("task_id"),
                "output": e.content,
                "executed_at": e.metadata.get("executed_at"),
            }
            for e in entries
        ]
    
    def get_query_context(self) -> Optional[Dict[str, Any]]:
        """Get current query context"""
        entries = self.get_entries_by_type(EntryType.QUERY_CONTEXT, limit=1)
        return entries[0].content if entries else None
    
    # ========================================================================
    # CONTEXT FORMATTING
    # ========================================================================
    
    def get_context_for_llm(
        self,
        max_tokens: int = 2000,
        include_types: Optional[List[EntryType]] = None,
    ) -> str:
        """
        Format working memory for LLM context
        
        Args:
            max_tokens: Maximum tokens to include
            include_types: Types to include (default: all)
            
        Returns:
            Formatted string for LLM
        """
        with self._lock:
            if include_types is None:
                include_types = list(EntryType)
            
            sections = []
            current_tokens = 0
            
            # Order by priority for inclusion
            priority_order = [
                EntryType.SYMBOLS,      # Always include current symbols
                EntryType.QUERY_CONTEXT,
                EntryType.TASK_PLAN,
                EntryType.TOOL_OUTPUT,
                EntryType.REASONING,
                EntryType.INTERMEDIATE,
                EntryType.ERROR,
            ]
            
            for entry_type in priority_order:
                if entry_type not in include_types:
                    continue
                
                entries = self.get_entries_by_type(entry_type, limit=5)
                
                for entry in entries:
                    # Format entry
                    formatted = self._format_entry(entry)
                    entry_tokens = self.token_counter.count_tokens(formatted)
                    
                    if current_tokens + entry_tokens > max_tokens:
                        break
                    
                    sections.append(formatted)
                    current_tokens += entry_tokens
            
            return "\n\n".join(sections)
    
    def _format_entry(self, entry: WorkingMemoryEntry) -> str:
        """Format a single entry for LLM"""
        import json
        
        if entry.entry_type == EntryType.SYMBOLS:
            symbols = entry.content if isinstance(entry.content, list) else [entry.content]
            return f"Current Symbols: {', '.join(symbols)}"
        
        elif entry.entry_type == EntryType.QUERY_CONTEXT:
            ctx = entry.content
            return (
                f"Query Context:\n"
                f"- Intent: {ctx.get('intent', 'unknown')}\n"
                f"- Type: {ctx.get('query_type', 'unknown')}\n"
                f"- Language: {ctx.get('language', 'auto')}"
            )
        
        elif entry.entry_type == EntryType.TASK_PLAN:
            plan = entry.content
            task_count = len(plan.get('tasks', []))
            return (
                f"Task Plan:\n"
                f"- Strategy: {plan.get('strategy', 'unknown')}\n"
                f"- Tasks: {task_count}"
            )
        
        elif entry.entry_type == EntryType.TOOL_OUTPUT:
            tool_name = entry.metadata.get('tool_name', 'unknown')
            return f"Tool Output ({tool_name}): [data available]"
        
        else:
            content_str = str(entry.content)
            if len(content_str) > 200:
                content_str = content_str[:200] + "..."
            return f"{entry.entry_type.value}: {content_str}"
    
    # ========================================================================
    # CLEANUP METHODS
    # ========================================================================
    
    def clear_type(self, entry_type: EntryType) -> int:
        """
        Clear all entries of a specific type
        
        Args:
            entry_type: Type to clear
            
        Returns:
            Number of entries cleared
        """
        with self._lock:
            entry_ids = list(self._by_type.get(entry_type, []))
            count = 0
            
            for entry_id in entry_ids:
                if self.remove_entry(entry_id):
                    count += 1
            
            if count > 0:
                self.logger.debug(
                    f"[WORKING_MEMORY] Cleared {count} entries of type {entry_type.value}"
                )
            
            return count
    
    def clear_task(self) -> int:
        """
        Clear task-specific data (call after task completion)
        
        FIXED: Does NOT clear symbols for cross-turn continuity
        
        Clears:
        - Task plan
        - Tool outputs
        - Intermediate results
        - Query context
        - Errors
        
        Keeps:
        - Symbols (for follow-up queries)
        - User context
        - Reasoning (may be useful for context)
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            # FIXED: SYMBOLS is NOT in this list
            types_to_clear = [
                EntryType.TASK_PLAN,
                EntryType.TOOL_OUTPUT,
                EntryType.INTERMEDIATE,
                EntryType.QUERY_CONTEXT,
                EntryType.ERROR,
            ]
            
            total_cleared = 0
            for entry_type in types_to_clear:
                total_cleared += self.clear_type(entry_type)
            
            if total_cleared > 0:
                self.logger.info(
                    f"[WORKING_MEMORY] Cleared task data: {total_cleared} entries "
                    f"(symbols preserved)"
                )
            
            return total_cleared
    
    def clear_stale_symbols(self, max_turns: int = None) -> bool:
        """
        Clear symbols if they are stale (too many turns old)
        
        Call this periodically or when starting a clearly new topic
        
        Args:
            max_turns: Maximum turns before symbols are stale
            
        Returns:
            True if symbols were cleared
        """
        max_turns = max_turns or self.SYMBOL_MAX_TURNS
        
        with self._lock:
            entries = self.get_entries_by_type(EntryType.SYMBOLS, limit=1)
            
            if not entries:
                return False
            
            entry = entries[0]
            turns_old = self._current_turn - entry.turn_count
            
            if turns_old > max_turns:
                self.clear_type(EntryType.SYMBOLS)
                self.logger.info(
                    f"[WORKING_MEMORY] Cleared stale symbols "
                    f"({turns_old} turns old > {max_turns} max)"
                )
                return True
            
            return False
    
    def clear_symbols(self) -> int:
        """
        Explicitly clear symbols
        
        Use this when user clearly changes topic or explicitly requests
        """
        return self.clear_type(EntryType.SYMBOLS)
    
    def clear_all(self) -> int:
        """
        Clear all entries including symbols
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._entries)
            
            self._entries.clear()
            self._by_type = {t: [] for t in EntryType}
            self._total_tokens = 0
            
            self.last_modified = datetime.now()
            
            self.logger.info(f"[WORKING_MEMORY] Cleared all: {count} entries")
            
            return count
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_ids = [
                entry_id
                for entry_id, entry in self._entries.items()
                if entry.is_expired()
            ]
            
            for entry_id in expired_ids:
                self.remove_entry(entry_id)
            
            if expired_ids:
                self.logger.debug(
                    f"[WORKING_MEMORY] Cleaned up {len(expired_ids)} expired entries"
                )
            
            return len(expired_ids)
    
    # ========================================================================
    # STATISTICS & DEBUG
    # ========================================================================
    
    def get_stats(self) -> WorkingMemoryStats:
        """Get current statistics"""
        with self._lock:
            entries_by_type = {}
            tokens_by_type = {}
            
            for entry_type in EntryType:
                entry_ids = self._by_type.get(entry_type, [])
                entries_by_type[entry_type.value] = len(entry_ids)
                
                type_tokens = sum(
                    self._entries[eid].token_count
                    for eid in entry_ids
                    if eid in self._entries
                )
                tokens_by_type[entry_type.value] = type_tokens
            
            oldest = None
            newest = None
            symbols_turn = None
            
            if self._entries:
                entries_list = list(self._entries.values())
                oldest = min(e.created_at for e in entries_list)
                newest = max(e.created_at for e in entries_list)
                
                # Get symbol turn count
                symbol_entries = self.get_entries_by_type(EntryType.SYMBOLS, limit=1)
                if symbol_entries:
                    symbols_turn = symbol_entries[0].turn_count
            
            return WorkingMemoryStats(
                total_entries=len(self._entries),
                total_tokens=self._total_tokens,
                entries_by_type=entries_by_type,
                tokens_by_type=tokens_by_type,
                oldest_entry=oldest,
                newest_entry=newest,
                symbols_turn_count=symbols_turn,
            )


# ============================================================================
# WORKING MEMORY MANAGER (Session Management)
# ============================================================================

class WorkingMemoryManager(LoggerMixin):
    """
    Manages working memory instances across sessions
    
    Thread-safe singleton pattern for global access
    """
    
    _instance = None
    _lock = threading.Lock()
    
    # Session cleanup
    SESSION_TTL_SECONDS = 3600 * 4  # 4 hours
    MAX_SESSIONS = 1000
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        super().__init__()
        self._sessions: Dict[str, WorkingMemory] = {}
        self._session_lock = threading.RLock()
        self._initialized = True
        
        self.logger.info("[WORKING_MEMORY_MANAGER] Initialized")
    
    def get_or_create(
        self,
        session_id: str,
        user_id: str,
        max_tokens: int = WorkingMemory.DEFAULT_MAX_TOKENS,
    ) -> WorkingMemory:
        """Get existing or create new working memory for session"""
        with self._session_lock:
            if session_id not in self._sessions:
                # Cleanup if too many sessions
                if len(self._sessions) >= self.MAX_SESSIONS:
                    self._cleanup_old_sessions()
                
                self._sessions[session_id] = WorkingMemory(
                    session_id=session_id,
                    user_id=user_id,
                    max_tokens=max_tokens,
                )
                
                self.logger.debug(
                    f"[WORKING_MEMORY_MANAGER] Created new session {session_id[:8]}..."
                )
            
            return self._sessions[session_id]
    
    def get(self, session_id: str) -> Optional[WorkingMemory]:
        """Get existing working memory (or None)"""
        with self._session_lock:
            return self._sessions.get(session_id)
    
    def remove(self, session_id: str) -> bool:
        """Remove a session's working memory"""
        with self._session_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False
    
    def _cleanup_old_sessions(self):
        """Remove sessions older than TTL"""
        with self._session_lock:
            cutoff = datetime.now() - timedelta(seconds=self.SESSION_TTL_SECONDS)
            
            old_sessions = [
                sid for sid, wm in self._sessions.items()
                if wm.last_modified < cutoff
            ]
            
            for sid in old_sessions:
                del self._sessions[sid]
            
            if old_sessions:
                self.logger.info(
                    f"[WORKING_MEMORY_MANAGER] Cleaned up {len(old_sessions)} old sessions"
                )
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get stats for all sessions"""
        with self._session_lock:
            return {
                "total_sessions": len(self._sessions),
                "sessions": {
                    sid[:8]: wm.get_stats().__dict__
                    for sid, wm in self._sessions.items()
                }
            }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_working_memory_manager() -> WorkingMemoryManager:
    """Get the global working memory manager instance"""
    return WorkingMemoryManager()


def get_working_memory(
    session_id: str,
    user_id: str,
    max_tokens: int = WorkingMemory.DEFAULT_MAX_TOKENS,
) -> WorkingMemory:
    """Convenience function to get/create working memory"""
    manager = get_working_memory_manager()
    return manager.get_or_create(session_id, user_id, max_tokens)