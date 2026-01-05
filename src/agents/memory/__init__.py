"""
Memory System Components

Includes:
- Core Memory: Long-term user/persona information
- Recursive Summary: Conversation summarization
- Working Memory: Task-specific scratchpad
- Persistent Working Memory: Redis-backed persistence
- Memory Update Agent: Background memory updates
"""

from src.agents.memory.core_memory import CoreMemory
from src.agents.memory.recursive_summary import RecursiveSummaryManager
from src.agents.memory.working_memory import (
    WorkingMemory,
    WorkingMemoryManager,
    WorkingMemoryEntry,
    EntryType,
    Priority,
    get_working_memory,
    get_working_memory_manager,
    get_working_memory_with_recovery,
)
from src.agents.memory.persistent_working_memory import (
    PersistentWorkingMemoryService,
    WorkingMemorySnapshot,
    TaskState,
    TaskStatus,
    get_persistent_wm_service,
    create_task_state,
    save_working_memory_snapshot,
    recover_working_memory,
)
from src.agents.memory.working_memory_integration import (
    WorkingMemoryIntegration,
    setup_working_memory_for_request,
)
from src.agents.memory.memory_update_agent import MemoryUpdateAgent

__all__ = [
    # Core Memory
    "CoreMemory",

    # Recursive Summary
    "RecursiveSummaryManager",

    # Working Memory
    "WorkingMemory",
    "WorkingMemoryManager",
    "WorkingMemoryEntry",
    "EntryType",
    "Priority",
    "get_working_memory",
    "get_working_memory_manager",
    "get_working_memory_with_recovery",

    # Persistent Working Memory (Redis-backed)
    "PersistentWorkingMemoryService",
    "WorkingMemorySnapshot",
    "TaskState",
    "TaskStatus",
    "get_persistent_wm_service",
    "create_task_state",
    "save_working_memory_snapshot",
    "recover_working_memory",

    # Working Memory Integration
    "WorkingMemoryIntegration",
    "setup_working_memory_for_request",

    # Memory Update Agent
    "MemoryUpdateAgent",
]
