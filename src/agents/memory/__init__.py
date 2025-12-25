"""
Memory System Components

Includes:
- Core Memory: Long-term user/persona information
- Recursive Summary: Conversation summarization
- Working Memory: Task-specific scratchpad
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
    
    # Working Memory Integration
    "WorkingMemoryIntegration",
    "setup_working_memory_for_request",
    
    # Memory Update Agent
    "MemoryUpdateAgent",
]
