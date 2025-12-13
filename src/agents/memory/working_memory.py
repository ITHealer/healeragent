# File: src/agents/memory/working_memory.py
"""
Working Memory (Scratchpad) - Task-specific Temporary Memory

Architecture inspired by:
- Anthropic's multi-agent researcher (save plan to Memory)
- RAISE architecture (Scratchpad + Examples)
- LangGraph thread-scoped memory
- Weaviate Context Engineering (Task Context Storage)

╔════════════════════════════════════════════════════════════════════════════╗
║                         WORKING MEMORY ARCHITECTURE                         ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  ┌─────────────────────────────────────────────────────────────────────┐   ║
║  │                        WorkingMemoryManager                          │   ║
║  │                     (Singleton - All Sessions)                       │   ║
║  └───────────────────────────────┬─────────────────────────────────────┘   ║
║                                  │                                         ║
║            ┌─────────────────────┼─────────────────────┐                   ║
║            │                     │                     │                   ║
║            ▼                     ▼                     ▼                   ║
║  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         ║
║  │ WorkingMemory    │  │ WorkingMemory    │  │ WorkingMemory    │         ║
║  │ (Session A)      │  │ (Session B)      │  │ (Session C)      │         ║
║  │                  │  │                  │  │                  │         ║
║  │ ┌──────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────┐ │         ║
║  │ │ Task Plan    │ │  │ │ Task Plan    │ │  │ │ Task Plan    │ │         ║
║  │ ├──────────────┤ │  │ ├──────────────┤ │  │ ├──────────────┤ │         ║
║  │ │ Tool Outputs │ │  │ │ Tool Outputs │ │  │ │ Tool Outputs │ │         ║
║  │ ├──────────────┤ │  │ ├──────────────┤ │  │ ├──────────────┤ │         ║
║  │ │ Reasoning    │ │  │ │ Reasoning    │ │  │ │ Reasoning    │ │         ║
║  │ ├──────────────┤ │  │ ├──────────────┤ │  │ ├──────────────┤ │         ║
║  │ │ Symbols      │ │  │ │ Symbols      │ │  │ │ Symbols      │ │         ║
║  │ └──────────────┘ │  │ └──────────────┘ │  │ └──────────────┘ │         ║
║  └──────────────────┘  └──────────────────┘  └──────────────────┘         ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Entry Types:
- TASK_PLAN: Current execution plan from Planning Agent
- TOOL_OUTPUT: Results from tool executions
- REASONING: Agent's thoughts, analysis, conclusions
- SYMBOLS: Extracted symbols for current query context
- INTERMEDIATE: Partial results during multi-step tasks
- USER_CONTEXT: Relevant user context for current task

Features:
- Session-scoped (isolated per conversation)
- Auto-expiry (configurable TTL)
- Priority-based retention
- Token-aware storage
- Redis caching support (optional)

Usage:
    from src.agents.memory.working_memory import (
        WorkingMemoryManager,
        EntryType
    )
    
    # Get manager (singleton)
    manager = WorkingMemoryManager()
    
    # Get working memory for session
    memory = manager.get_memory(session_id)
    
    # Add entries
    memory.add_plan(task_plan_dict)
    memory.add_tool_output("getStockPrice", {"AAPL": 195.50})
    memory.add_reasoning("User wants real-time price with technical analysis")
    memory.add_symbols(["AAPL", "NVDA"])
    
    # Get context for LLM
    context = memory.get_context_for_llm(max_tokens=2000)
    
    # Clear after task completion
    memory.clear_task()
"""

import json
import time
import threading
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import OrderedDict

from src.utils.logger.custom_logging import LoggerMixin

# Token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class EntryType(Enum):
    """Types of working memory entries"""
    TASK_PLAN = "task_plan"           # Execution plan from Planning Agent
    TOOL_OUTPUT = "tool_output"       # Results from tool executions
    REASONING = "reasoning"           # Agent's thoughts and analysis
    SYMBOLS = "symbols"               # Extracted symbols for context
    INTERMEDIATE = "intermediate"     # Partial results during multi-step
    USER_CONTEXT = "user_context"     # Relevant user context
    QUERY_CONTEXT = "query_context"   # Query-specific context (intent, language)
    ERROR = "error"                   # Error information for recovery


class Priority(Enum):
    """Entry priority for retention decisions"""
    LOW = 1       # Can be dropped first
    MEDIUM = 2    # Standard importance
    HIGH = 3      # Keep unless absolutely necessary
    CRITICAL = 4  # Never drop automatically


# Default TTL by entry type (seconds)
DEFAULT_TTL = {
    EntryType.TASK_PLAN: 3600,        # 1 hour
    EntryType.TOOL_OUTPUT: 1800,      # 30 minutes  
    EntryType.REASONING: 3600,        # 1 hour
    EntryType.SYMBOLS: 1800,          # 30 minutes
    EntryType.INTERMEDIATE: 900,      # 15 minutes
    EntryType.USER_CONTEXT: 7200,     # 2 hours
    EntryType.QUERY_CONTEXT: 1800,    # 30 minutes
    EntryType.ERROR: 600,             # 10 minutes
}

# Default priority by entry type
DEFAULT_PRIORITY = {
    EntryType.TASK_PLAN: Priority.HIGH,
    EntryType.TOOL_OUTPUT: Priority.MEDIUM,
    EntryType.REASONING: Priority.MEDIUM,
    EntryType.SYMBOLS: Priority.HIGH,
    EntryType.INTERMEDIATE: Priority.LOW,
    EntryType.USER_CONTEXT: Priority.HIGH,
    EntryType.QUERY_CONTEXT: Priority.HIGH,
    EntryType.ERROR: Priority.LOW,
}


# ============================================================================
# TOKEN COUNTER (Reused from context_compressor)
# ============================================================================

def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """Count tokens in text"""
    if not text:
        return 0
    
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    
    # Fallback: approximate 4 chars per token
    return len(text) // 4


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class WorkingMemoryEntry:
    """
    Single entry in working memory (scratchpad)
    
    Attributes:
        id: Unique identifier
        entry_type: Type of entry (plan, tool_output, reasoning, etc.)
        content: The actual data (can be dict, list, or string)
        priority: Importance level for retention
        created_at: Creation timestamp
        expires_at: Expiration timestamp (for auto-cleanup)
        metadata: Additional metadata (tool_name, symbol, etc.)
        token_count: Cached token count for the entry
    """
    id: str
    entry_type: EntryType
    content: Any
    priority: Priority = Priority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    
    def __post_init__(self):
        """Calculate token count after initialization"""
        if self.token_count == 0:
            content_str = self._content_to_string()
            self.token_count = count_tokens(content_str)
    
    def _content_to_string(self) -> str:
        """Convert content to string for token counting"""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, (dict, list)):
            return json.dumps(self.content, ensure_ascii=False)
        else:
            return str(self.content)
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "entry_type": self.entry_type.value,
            "content": self.content,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
            "token_count": self.token_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkingMemoryEntry":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            entry_type=EntryType(data["entry_type"]),
            content=data["content"],
            priority=Priority(data["priority"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count", 0),
        )


@dataclass
class WorkingMemoryStats:
    """Statistics for working memory usage"""
    total_entries: int = 0
    total_tokens: int = 0
    entries_by_type: Dict[str, int] = field(default_factory=dict)
    tokens_by_type: Dict[str, int] = field(default_factory=dict)
    oldest_entry: Optional[datetime] = None
    newest_entry: Optional[datetime] = None


# ============================================================================
# WORKING MEMORY (Per Session)
# ============================================================================

class WorkingMemory(LoggerMixin):
    """
    Working Memory for a single session - acts as a scratchpad
    
    This is the temporary memory that holds:
    - Current task plan
    - Tool execution results
    - Reasoning and analysis
    - Extracted symbols
    - Intermediate results
    
    All data is session-scoped and automatically cleaned up.
    """
    
    # Token budget for working memory
    MAX_TOKENS = 8000  # Max tokens to store
    CONTEXT_BUDGET = 2000  # Max tokens to return for LLM context
    
    def __init__(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        max_tokens: int = MAX_TOKENS,
        context_budget: int = CONTEXT_BUDGET,
    ):
        """
        Initialize working memory for a session
        
        Args:
            session_id: Unique session identifier
            user_id: Optional user identifier
            max_tokens: Maximum tokens to store
            context_budget: Maximum tokens for LLM context
        """
        super().__init__()
        
        self.session_id = session_id
        self.user_id = user_id
        self.max_tokens = max_tokens
        self.context_budget = context_budget
        
        # Storage: OrderedDict to maintain insertion order
        self._entries: OrderedDict[str, WorkingMemoryEntry] = OrderedDict()
        
        # Quick access indexes
        self._by_type: Dict[EntryType, List[str]] = {t: [] for t in EntryType}
        
        # Counters
        self._total_tokens = 0
        self._entry_counter = 0
        
        # Timestamps
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.last_modified = datetime.now()
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        self.logger.debug(
            f"[WORKING_MEMORY] Initialized for session {session_id[:8]}... "
            f"(max_tokens={max_tokens})"
        )
    
    # ========================================================================
    # CORE CRUD OPERATIONS
    # ========================================================================
    
    def _generate_entry_id(self) -> str:
        """Generate unique entry ID"""
        self._entry_counter += 1
        timestamp = int(time.time() * 1000)
        return f"wm_{timestamp}_{self._entry_counter}"
    
    def _add_entry(
        self,
        entry_type: EntryType,
        content: Any,
        priority: Optional[Priority] = None,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Internal method to add entry to working memory
        
        Args:
            entry_type: Type of entry
            content: Content to store
            priority: Optional priority override
            ttl_seconds: Optional TTL override
            metadata: Optional metadata
            
        Returns:
            Entry ID
        """
        with self._lock:
            # Generate ID
            entry_id = self._generate_entry_id()
            
            # Determine TTL
            if ttl_seconds is None:
                ttl_seconds = DEFAULT_TTL.get(entry_type, 1800)
            
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            
            # Determine priority
            if priority is None:
                priority = DEFAULT_PRIORITY.get(entry_type, Priority.MEDIUM)
            
            # Create entry
            entry = WorkingMemoryEntry(
                id=entry_id,
                entry_type=entry_type,
                content=content,
                priority=priority,
                expires_at=expires_at,
                metadata=metadata or {},
            )
            
            # Check if we need to make room
            if self._total_tokens + entry.token_count > self.max_tokens:
                self._evict_entries(entry.token_count)
            
            # Store
            self._entries[entry_id] = entry
            self._by_type[entry_type].append(entry_id)
            self._total_tokens += entry.token_count
            
            # Update timestamps
            self.last_modified = datetime.now()
            
            self.logger.debug(
                f"[WORKING_MEMORY] Added {entry_type.value}: "
                f"{entry.token_count} tokens (total: {self._total_tokens})"
            )
            
            return entry_id
    
    def get_entry(self, entry_id: str) -> Optional[WorkingMemoryEntry]:
        """Get entry by ID"""
        with self._lock:
            self.last_accessed = datetime.now()
            return self._entries.get(entry_id)
    
    def get_entries_by_type(
        self,
        entry_type: EntryType,
        limit: Optional[int] = None,
        include_expired: bool = False,
    ) -> List[WorkingMemoryEntry]:
        """
        Get all entries of a specific type
        
        Args:
            entry_type: Type to filter by
            limit: Optional limit on number of entries
            include_expired: Whether to include expired entries
            
        Returns:
            List of entries (newest first)
        """
        with self._lock:
            self.last_accessed = datetime.now()
            
            entry_ids = self._by_type.get(entry_type, [])
            entries = []
            
            # Reverse to get newest first
            for entry_id in reversed(entry_ids):
                entry = self._entries.get(entry_id)
                if entry:
                    if include_expired or not entry.is_expired():
                        entries.append(entry)
                        if limit and len(entries) >= limit:
                            break
            
            return entries
    
    def remove_entry(self, entry_id: str) -> bool:
        """Remove entry by ID"""
        with self._lock:
            if entry_id not in self._entries:
                return False
            
            entry = self._entries.pop(entry_id)
            
            # Update indexes
            if entry_id in self._by_type[entry.entry_type]:
                self._by_type[entry.entry_type].remove(entry_id)
            
            # Update counters
            self._total_tokens -= entry.token_count
            
            self.last_modified = datetime.now()
            
            self.logger.debug(
                f"[WORKING_MEMORY] Removed {entry.entry_type.value}: "
                f"-{entry.token_count} tokens (total: {self._total_tokens})"
            )
            
            return True
    
    def _evict_entries(self, tokens_needed: int) -> int:
        """
        Evict entries to make room for new content
        
        Strategy:
        1. Remove expired entries first
        2. Remove LOW priority entries
        3. Remove MEDIUM priority entries (oldest first)
        4. Remove HIGH priority entries (oldest first)
        5. Never remove CRITICAL entries
        
        Args:
            tokens_needed: Tokens we need to free up
            
        Returns:
            Tokens actually freed
        """
        tokens_freed = 0
        entries_to_remove = []
        
        # Step 1: Collect expired entries
        for entry_id, entry in self._entries.items():
            if entry.is_expired():
                entries_to_remove.append(entry_id)
                tokens_freed += entry.token_count
        
        # Step 2: Collect by priority (LOW → MEDIUM → HIGH)
        if tokens_freed < tokens_needed:
            for priority in [Priority.LOW, Priority.MEDIUM, Priority.HIGH]:
                for entry_id, entry in self._entries.items():
                    if entry_id in entries_to_remove:
                        continue
                    if entry.priority == priority:
                        entries_to_remove.append(entry_id)
                        tokens_freed += entry.token_count
                        if tokens_freed >= tokens_needed:
                            break
                if tokens_freed >= tokens_needed:
                    break
        
        # Remove collected entries
        for entry_id in entries_to_remove:
            if entry_id in self._entries:
                entry = self._entries.pop(entry_id)
                if entry_id in self._by_type[entry.entry_type]:
                    self._by_type[entry.entry_type].remove(entry_id)
                self._total_tokens -= entry.token_count
        
        if entries_to_remove:
            self.logger.info(
                f"[WORKING_MEMORY] Evicted {len(entries_to_remove)} entries, "
                f"freed {tokens_freed} tokens"
            )
        
        return tokens_freed
    
    # ========================================================================
    # CONVENIENCE METHODS (Type-specific)
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
        # Clear previous plan (only keep latest)
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
        task_id: Optional[int] = None,
        execution_time_ms: Optional[int] = None,
    ) -> str:
        """
        Add tool execution output
        
        Args:
            tool_name: Name of the tool
            output: Tool output data
            task_id: Optional task ID this belongs to
            execution_time_ms: Optional execution time
            
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
        
        Args:
            symbols: List of symbols (e.g., ["AAPL", "NVDA"])
            source: Where symbols came from (query, history, tool)
            
        Returns:
            Entry ID
        """
        # Clear previous symbols (keep only latest)
        self.clear_type(EntryType.SYMBOLS)
        
        metadata = {
            "source": source,
            "count": len(symbols),
        }
        
        return self._add_entry(
            entry_type=EntryType.SYMBOLS,
            content=symbols,
            priority=Priority.HIGH,
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
        # Clear previous query context
        self.clear_type(EntryType.QUERY_CONTEXT)
        
        content = {
            "intent": intent,
            "language": language,
            "categories": categories or [],
            "query_type": query_type,
        }
        
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
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add error information for recovery
        
        Args:
            error_type: Type of error
            message: Error message
            recoverable: Whether error is recoverable
            details: Additional error details
            
        Returns:
            Entry ID
        """
        content = {
            "error_type": error_type,
            "message": message,
            "recoverable": recoverable,
            "details": details or {},
        }
        
        return self._add_entry(
            entry_type=EntryType.ERROR,
            content=content,
            priority=Priority.LOW,
            ttl_seconds=600,  # 10 minutes
        )
    
    # ========================================================================
    # CONTEXT RETRIEVAL
    # ========================================================================
    
    def get_context_for_llm(
        self,
        max_tokens: Optional[int] = None,
        include_types: Optional[List[EntryType]] = None,
        exclude_types: Optional[List[EntryType]] = None,
    ) -> str:
        """
        Get formatted context for LLM prompt injection
        
        Priority order for inclusion:
        1. Query Context (intent, language)
        2. Symbols (current context)
        3. Task Plan (current plan)
        4. Tool Outputs (recent results)
        5. Reasoning (analysis)
        6. Intermediate Results
        
        Args:
            max_tokens: Token budget (default: context_budget)
            include_types: Only include these types
            exclude_types: Exclude these types
            
        Returns:
            Formatted context string
        """
        with self._lock:
            self.last_accessed = datetime.now()
            
            max_tokens = max_tokens or self.context_budget
            
            # Determine which types to include
            if include_types:
                types_to_include = include_types
            else:
                types_to_include = list(EntryType)
                if exclude_types:
                    types_to_include = [t for t in types_to_include if t not in exclude_types]
            
            # Priority order for inclusion
            priority_order = [
                EntryType.QUERY_CONTEXT,
                EntryType.SYMBOLS,
                EntryType.TASK_PLAN,
                EntryType.TOOL_OUTPUT,
                EntryType.REASONING,
                EntryType.INTERMEDIATE,
                EntryType.USER_CONTEXT,
                EntryType.ERROR,
            ]
            
            sections = []
            tokens_used = 0
            
            for entry_type in priority_order:
                if entry_type not in types_to_include:
                    continue
                
                entries = self.get_entries_by_type(entry_type, include_expired=False)
                
                if not entries:
                    continue
                
                section = self._format_entries_section(entry_type, entries)
                section_tokens = count_tokens(section)
                
                if tokens_used + section_tokens <= max_tokens:
                    sections.append(section)
                    tokens_used += section_tokens
                else:
                    # Try to fit partial content
                    remaining = max_tokens - tokens_used
                    if remaining > 100:  # Only if meaningful space left
                        truncated = self._truncate_section(section, remaining)
                        if truncated:
                            sections.append(truncated)
                    break
            
            if not sections:
                return ""
            
            # Build final context
            context = "\n\n".join(sections)
            
            self.logger.debug(
                f"[WORKING_MEMORY] Generated context: {tokens_used} tokens "
                f"from {len(sections)} sections"
            )
            
            return context
    
    def _format_entries_section(
        self,
        entry_type: EntryType,
        entries: List[WorkingMemoryEntry],
    ) -> str:
        """Format entries of a type into a section"""
        
        type_headers = {
            EntryType.QUERY_CONTEXT: "📋 QUERY CONTEXT",
            EntryType.SYMBOLS: "🎯 CURRENT SYMBOLS",
            EntryType.TASK_PLAN: "📝 TASK PLAN",
            EntryType.TOOL_OUTPUT: "🔧 TOOL RESULTS",
            EntryType.REASONING: "💭 REASONING",
            EntryType.INTERMEDIATE: "📊 INTERMEDIATE RESULTS",
            EntryType.USER_CONTEXT: "👤 USER CONTEXT",
            EntryType.ERROR: "⚠️ ERRORS",
        }
        
        header = type_headers.get(entry_type, entry_type.value.upper())
        lines = [f"<{header}>"]
        
        for entry in entries:
            if isinstance(entry.content, dict):
                content_str = json.dumps(entry.content, ensure_ascii=False, indent=2)
            elif isinstance(entry.content, list):
                if entry_type == EntryType.SYMBOLS:
                    content_str = ", ".join(entry.content)
                else:
                    content_str = json.dumps(entry.content, ensure_ascii=False)
            else:
                content_str = str(entry.content)
            
            # Add metadata if relevant
            if entry_type == EntryType.TOOL_OUTPUT and entry.metadata.get("tool_name"):
                lines.append(f"[{entry.metadata['tool_name']}]:")
            
            lines.append(content_str)
        
        lines.append(f"</{header}>")
        
        return "\n".join(lines)
    
    def _truncate_section(self, section: str, max_tokens: int) -> str:
        """Truncate section to fit token budget"""
        current_tokens = count_tokens(section)
        
        if current_tokens <= max_tokens:
            return section
        
        # Simple truncation: keep first N characters
        ratio = max_tokens / current_tokens
        target_chars = int(len(section) * ratio * 0.9)  # 10% safety margin
        
        if target_chars < 50:
            return ""
        
        return section[:target_chars] + "\n... (truncated)"
    
    # ========================================================================
    # GETTER METHODS
    # ========================================================================
    
    def get_current_plan(self) -> Optional[Dict[str, Any]]:
        """Get current task plan"""
        entries = self.get_entries_by_type(EntryType.TASK_PLAN, limit=1)
        return entries[0].content if entries else None
    
    def get_current_symbols(self) -> List[str]:
        """Get current symbols in context"""
        entries = self.get_entries_by_type(EntryType.SYMBOLS, limit=1)
        return entries[0].content if entries else []
    
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
        
        Clears:
        - Task plan
        - Tool outputs
        - Intermediate results
        - Query context
        
        Keeps:
        - Symbols (may be useful for follow-up)
        - User context
        - Reasoning (may be useful for context)
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
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
                    f"[WORKING_MEMORY] Cleared task data: {total_cleared} entries"
                )
            
            return total_cleared
    
    def clear_all(self) -> int:
        """
        Clear all entries
        
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
                entries = self._by_type.get(entry_type, [])
                entries_by_type[entry_type.value] = len(entries)
                
                tokens = sum(
                    self._entries[eid].token_count
                    for eid in entries
                    if eid in self._entries
                )
                tokens_by_type[entry_type.value] = tokens
            
            oldest = None
            newest = None
            
            if self._entries:
                entries_list = list(self._entries.values())
                oldest = min(e.created_at for e in entries_list)
                newest = max(e.created_at for e in entries_list)
            
            return WorkingMemoryStats(
                total_entries=len(self._entries),
                total_tokens=self._total_tokens,
                entries_by_type=entries_by_type,
                tokens_by_type=tokens_by_type,
                oldest_entry=oldest,
                newest_entry=newest,
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for Redis caching)"""
        with self._lock:
            return {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "created_at": self.created_at.isoformat(),
                "last_accessed": self.last_accessed.isoformat(),
                "last_modified": self.last_modified.isoformat(),
                "entries": [e.to_dict() for e in self._entries.values()],
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkingMemory":
        """Deserialize from dictionary (for Redis caching)"""
        memory = cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
        )
        
        memory.created_at = datetime.fromisoformat(data["created_at"])
        memory.last_accessed = datetime.fromisoformat(data["last_accessed"])
        memory.last_modified = datetime.fromisoformat(data["last_modified"])
        
        for entry_data in data.get("entries", []):
            entry = WorkingMemoryEntry.from_dict(entry_data)
            memory._entries[entry.id] = entry
            memory._by_type[entry.entry_type].append(entry.id)
            memory._total_tokens += entry.token_count
        
        return memory
    
    def __repr__(self) -> str:
        return (
            f"WorkingMemory(session={self.session_id[:8]}..., "
            f"entries={len(self._entries)}, tokens={self._total_tokens})"
        )


# ============================================================================
# WORKING MEMORY MANAGER (Singleton)
# ============================================================================

class WorkingMemoryManager(LoggerMixin):
    """
    Singleton manager for all working memories across sessions
    
    Responsibilities:
    - Create/retrieve WorkingMemory instances per session
    - Cleanup expired sessions
    - Optional Redis persistence
    - Thread-safe access
    """
    
    _instance = None
    _lock = threading.Lock()
    
    # Session TTL (default: 2 hours)
    SESSION_TTL_SECONDS = 7200
    
    # Max sessions to keep in memory
    MAX_SESSIONS = 1000
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize manager (only once)"""
        if self._initialized:
            return
        
        super().__init__()
        
        # Storage: session_id -> WorkingMemory
        self._memories: Dict[str, WorkingMemory] = {}
        self._access_times: Dict[str, datetime] = {}
        
        # Lock for thread safety
        self._manager_lock = threading.RLock()
        
        # Optional Redis client
        self._redis_client = None
        
        self._initialized = True
        
        self.logger.info(
            f"[WORKING_MEMORY_MANAGER] Initialized "
            f"(max_sessions={self.MAX_SESSIONS}, ttl={self.SESSION_TTL_SECONDS}s)"
        )
    
    def configure_redis(self, redis_client) -> None:
        """
        Configure Redis for persistence
        
        Args:
            redis_client: Redis client instance
        """
        self._redis_client = redis_client
        self.logger.info("[WORKING_MEMORY_MANAGER] Redis configured for persistence")
    
    def get_memory(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        create_if_missing: bool = True,
    ) -> Optional[WorkingMemory]:
        """
        Get or create working memory for a session
        
        Args:
            session_id: Session identifier
            user_id: Optional user identifier
            create_if_missing: Whether to create if not exists
            
        Returns:
            WorkingMemory instance or None
        """
        with self._manager_lock:
            # Check in-memory cache first
            if session_id in self._memories:
                memory = self._memories[session_id]
                self._access_times[session_id] = datetime.now()
                return memory
            
            # Try loading from Redis
            if self._redis_client:
                memory = self._load_from_redis(session_id)
                if memory:
                    self._memories[session_id] = memory
                    self._access_times[session_id] = datetime.now()
                    return memory
            
            # Create new if requested
            if create_if_missing:
                # Check if we need to evict old sessions
                if len(self._memories) >= self.MAX_SESSIONS:
                    self._evict_oldest_sessions(count=100)
                
                memory = WorkingMemory(
                    session_id=session_id,
                    user_id=user_id,
                )
                
                self._memories[session_id] = memory
                self._access_times[session_id] = datetime.now()
                
                self.logger.debug(
                    f"[WORKING_MEMORY_MANAGER] Created memory for session {session_id[:8]}..."
                )
                
                return memory
            
            return None
    
    def save_memory(self, session_id: str) -> bool:
        """
        Save working memory to Redis
        
        Args:
            session_id: Session to save
            
        Returns:
            Success status
        """
        if not self._redis_client:
            return False
        
        with self._manager_lock:
            if session_id not in self._memories:
                return False
            
            memory = self._memories[session_id]
            return self._save_to_redis(session_id, memory)
    
    def _load_from_redis(self, session_id: str) -> Optional[WorkingMemory]:
        """Load working memory from Redis"""
        try:
            key = f"working_memory:{session_id}"
            data = self._redis_client.get(key)
            
            if data:
                return WorkingMemory.from_dict(json.loads(data))
            
            return None
            
        except Exception as e:
            self.logger.error(f"[WORKING_MEMORY_MANAGER] Redis load error: {e}")
            return None
    
    def _save_to_redis(self, session_id: str, memory: WorkingMemory) -> bool:
        """Save working memory to Redis"""
        try:
            key = f"working_memory:{session_id}"
            data = json.dumps(memory.to_dict(), ensure_ascii=False)
            
            self._redis_client.setex(
                key,
                self.SESSION_TTL_SECONDS,
                data,
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"[WORKING_MEMORY_MANAGER] Redis save error: {e}")
            return False
    
    def _evict_oldest_sessions(self, count: int = 100) -> int:
        """
        Evict oldest sessions to make room
        
        Args:
            count: Number of sessions to evict
            
        Returns:
            Number evicted
        """
        # Sort by last access time
        sorted_sessions = sorted(
            self._access_times.items(),
            key=lambda x: x[1],
        )
        
        evicted = 0
        for session_id, _ in sorted_sessions[:count]:
            if session_id in self._memories:
                # Save to Redis before evicting
                if self._redis_client:
                    self._save_to_redis(session_id, self._memories[session_id])
                
                del self._memories[session_id]
                del self._access_times[session_id]
                evicted += 1
        
        self.logger.info(f"[WORKING_MEMORY_MANAGER] Evicted {evicted} old sessions")
        
        return evicted
    
    def cleanup_expired(self) -> int:
        """
        Cleanup expired sessions
        
        Returns:
            Number of sessions cleaned up
        """
        with self._manager_lock:
            now = datetime.now()
            expired_threshold = now - timedelta(seconds=self.SESSION_TTL_SECONDS)
            
            expired_sessions = [
                session_id
                for session_id, access_time in self._access_times.items()
                if access_time < expired_threshold
            ]
            
            for session_id in expired_sessions:
                if session_id in self._memories:
                    del self._memories[session_id]
                if session_id in self._access_times:
                    del self._access_times[session_id]
            
            if expired_sessions:
                self.logger.info(
                    f"[WORKING_MEMORY_MANAGER] Cleaned up {len(expired_sessions)} expired sessions"
                )
            
            return len(expired_sessions)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        with self._manager_lock:
            total_entries = sum(
                len(m._entries) for m in self._memories.values()
            )
            total_tokens = sum(
                m._total_tokens for m in self._memories.values()
            )
            
            return {
                "active_sessions": len(self._memories),
                "total_entries": total_entries,
                "total_tokens": total_tokens,
                "max_sessions": self.MAX_SESSIONS,
                "session_ttl_seconds": self.SESSION_TTL_SECONDS,
                "redis_enabled": self._redis_client is not None,
            }
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear and remove a session's working memory
        
        Args:
            session_id: Session to clear
            
        Returns:
            Success status
        """
        with self._manager_lock:
            if session_id in self._memories:
                self._memories[session_id].clear_all()
                del self._memories[session_id]
            
            if session_id in self._access_times:
                del self._access_times[session_id]
            
            # Also remove from Redis
            if self._redis_client:
                try:
                    key = f"working_memory:{session_id}"
                    self._redis_client.delete(key)
                except Exception:
                    pass
            
            return True


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_working_memory_manager() -> WorkingMemoryManager:
    """Get the singleton WorkingMemoryManager instance"""
    return WorkingMemoryManager()


def get_working_memory(
    session_id: str,
    user_id: Optional[str] = None,
) -> WorkingMemory:
    """
    Convenience function to get working memory for a session
    
    Args:
        session_id: Session identifier
        user_id: Optional user identifier
        
    Returns:
        WorkingMemory instance
    """
    manager = get_working_memory_manager()
    return manager.get_memory(session_id, user_id)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def demo():
        # Get manager
        manager = WorkingMemoryManager()
        
        # Get memory for session
        session_id = "demo_session_123"
        memory = manager.get_memory(session_id, user_id="user_1")
        
        # Add various entries
        memory.add_symbols(["AAPL", "NVDA", "MSFT"])
        
        memory.add_query_context(
            intent="Analyze stock performance",
            language="en",
            categories=["price", "technical"],
            query_type="stock_specific",
        )
        
        memory.add_plan({
            "tasks": [
                {"id": 1, "tool": "getStockPrice", "symbol": "AAPL"},
                {"id": 2, "tool": "getTechnicalIndicators", "symbol": "AAPL"},
            ],
            "strategy": "parallel",
        })
        
        memory.add_tool_output(
            tool_name="getStockPrice",
            output={"symbol": "AAPL", "price": 195.50, "change": 2.3},
            task_id=1,
        )
        
        memory.add_reasoning(
            "User wants comprehensive analysis with both price and technical data",
            category="intent_analysis",
        )
        
        # Get context for LLM
        context = memory.get_context_for_llm()
        print("=== Context for LLM ===")
        print(context)
        print()
        
        # Get stats
        stats = memory.get_stats()
        print("=== Stats ===")
        print(f"Total entries: {stats.total_entries}")
        print(f"Total tokens: {stats.total_tokens}")
        print(f"By type: {stats.entries_by_type}")
        print()
        
        # Manager stats
        manager_stats = manager.get_stats()
        print("=== Manager Stats ===")
        print(manager_stats)
        print()
        
        # Clear task data
        memory.clear_task()
        
        print("=== After clear_task ===")
        print(f"Remaining entries: {len(memory._entries)}")
        print(f"Current symbols: {memory.get_current_symbols()}")
    
    asyncio.run(demo())