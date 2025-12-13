# # File: src/agents/memory/__init__.py
# """
# Memory System Components - HealerAgent

# Production-ready memory system inspired by MemGPT/Letta, Claude AI, and ChatGPT.

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         MEMORY SYSTEM ARCHITECTURE                          ║
# ╠════════════════════════════════════════════════════════════════════════════╣
# ║                                                                            ║
# ║  ┌─────────────────────────────────────────────────────────────────────┐   ║
# ║  │                        Memory Types (CoALA)                          │   ║
# ║  ├─────────────────────────────────────────────────────────────────────┤   ║
# ║  │                                                                     │   ║
# ║  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │   ║
# ║  │  │ Working Memory  │  │ Semantic Memory │  │ Episodic Memory │     │   ║
# ║  │  │ (Scratchpad)    │  │ (Core Memory)   │  │ (Chat History)  │     │   ║
# ║  │  │                 │  │                 │  │                 │     │   ║
# ║  │  │ • Task Plan     │  │ • User Profile  │  │ • Conversations │     │   ║
# ║  │  │ • Tool Outputs  │  │ • Preferences   │  │ • Sessions      │     │   ║
# ║  │  │ • Reasoning     │  │ • Watchlist     │  │ • Summaries     │     │   ║
# ║  │  │ • Symbols       │  │ • Portfolio     │  │                 │     │   ║
# ║  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │   ║
# ║  │           │                    │                    │               │   ║
# ║  │           └────────────────────┼────────────────────┘               │   ║
# ║  │                               │                                     │   ║
# ║  │                               ▼                                     │   ║
# ║  │                    ┌─────────────────────┐                          │   ║
# ║  │                    │  Context Assembler  │                          │   ║
# ║  │                    │  (Priority-based)   │                          │   ║
# ║  │                    └─────────────────────┘                          │   ║
# ║  │                               │                                     │   ║
# ║  │                               ▼                                     │   ║
# ║  │                    ┌─────────────────────┐                          │   ║
# ║  │                    │ Context Compressor  │                          │   ║
# ║  │                    │ (When needed)       │                          │   ║
# ║  │                    └─────────────────────┘                          │   ║
# ║  │                                                                     │   ║
# ║  └─────────────────────────────────────────────────────────────────────┘   ║
# ║                                                                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# Components:
# -----------

# 1. WorkingMemory (NEW)
#    - Task-specific temporary memory (scratchpad)
#    - Session-scoped, auto-cleanup
#    - Stores: plans, tool outputs, reasoning, symbols
   
# 2. CoreMemory  
#    - Always-loaded memory blocks (PERSONA, HUMAN)
#    - User profile, preferences, watchlist
   
# 3. MemoryUpdateAgent
#    - Extract & update memory from conversations
#    - Single LLM call approach
   
# 4. RecursiveSummaryManager
#    - Create summaries when history gets long
#    - Progressive summarization
   
# 5. ContextCompressor
#    - Compress context when approaching limits
#    - Multiple strategies (keep_last_n, smart_summary, etc.)

# Usage:
# ------

# # Working Memory (Scratchpad)
# from src.agents.memory import (
#     WorkingMemory,
#     WorkingMemoryManager,
#     WorkingMemoryIntegration,
#     get_working_memory,
#     setup_working_memory_for_request,
# )

# # Setup for request
# wm = setup_working_memory_for_request(session_id, user_id, flow_id)
# wm.save_classification(query_type, categories, symbols, language)
# wm.save_plan(task_plan)
# wm.save_tool_result(tool_name, result, task_id)
# context = wm.get_context_for_synthesis()
# wm.complete_request()

# # Core Memory
# from src.agents.memory import CoreMemory, MemoryUpdateAgent

# core_memory = CoreMemory()
# memory = await core_memory.load_core_memory(user_id)
# await core_memory.save_core_memory(user_id, memory)

# # Context Management
# from src.agents.memory import ContextCompressor

# compressor = ContextCompressor()
# result = await compressor.check_and_compress(messages, config)
# """

# # ============================================================================
# # WORKING MEMORY (Scratchpad) - NEW
# # ============================================================================

# from .working_memory import (
#     # Core classes
#     WorkingMemory,
#     WorkingMemoryManager,
#     WorkingMemoryEntry,
#     WorkingMemoryStats,
    
#     # Enums
#     EntryType,
#     Priority,
    
#     # Convenience functions
#     get_working_memory,
#     get_working_memory_manager,
# )

# from .working_memory_integration import (
#     # Integration class
#     WorkingMemoryIntegration,
    
#     # Convenience functions
#     setup_working_memory_for_request,
#     get_working_memory_stats,
#     cleanup_expired_sessions,
# )

# # ============================================================================
# # CORE MEMORY (Semantic Memory)
# # ============================================================================

# from .core_memory import CoreMemory

# # ============================================================================
# # MEMORY UPDATE AGENT
# # ============================================================================

# from .memory_update_agent import MemoryUpdateAgent

# # ============================================================================
# # RECURSIVE SUMMARY (Episodic Memory Compression)
# # ============================================================================

# from .recursive_summary import RecursiveSummaryManager

# # ============================================================================
# # CONTEXT COMPRESSOR
# # ============================================================================

# from .context_compressor import (
#     ContextCompressor,
#     CompactionStrategy,
#     CompactionConfig,
# )

# # ============================================================================
# # EXPORTS
# # ============================================================================

# __all__ = [
#     # Working Memory (NEW)
#     'WorkingMemory',
#     'WorkingMemoryManager',
#     'WorkingMemoryEntry',
#     'WorkingMemoryStats',
#     'WorkingMemoryIntegration',
#     'EntryType',
#     'Priority',
#     'get_working_memory',
#     'get_working_memory_manager',
#     'setup_working_memory_for_request',
#     'get_working_memory_stats',
#     'cleanup_expired_sessions',
    
#     # Core Memory
#     'CoreMemory',
    
#     # Memory Update
#     'MemoryUpdateAgent',
    
#     # Recursive Summary
#     'RecursiveSummaryManager',
    
#     # Context Compressor
#     'ContextCompressor',
#     'CompactionStrategy',
#     'CompactionConfig',
# ]

# # ============================================================================
# # VERSION INFO
# # ============================================================================

# __version__ = "2.0.0"
# __author__ = "HealerAgent Team"
# __description__ = "Production Memory System with Working Memory"

# # ============================================================================
# # MEMORY TYPE REFERENCE
# # ============================================================================
# """
# Memory Types (based on CoALA architecture):

# ┌─────────────────┬────────────────────────────────────────────────────────┐
# │ Memory Type     │ Description                                            │
# ├─────────────────┼────────────────────────────────────────────────────────┤
# │ Working Memory  │ Temporary task-specific memory (scratchpad)            │
# │                 │ • Current task plan                                    │
# │                 │ • Tool execution outputs                               │
# │                 │ • Agent reasoning/analysis                             │
# │                 │ • Extracted symbols                                    │
# │                 │ • Query context (intent, language)                     │
# │                 │ Lifecycle: Session-scoped, cleared after task          │
# ├─────────────────┼────────────────────────────────────────────────────────┤
# │ Semantic Memory │ Facts about user (Core Memory)                         │
# │                 │ • User profile                                         │
# │                 │ • Preferences                                          │
# │                 │ • Watchlist/Portfolio                                  │
# │                 │ Lifecycle: Persistent, user-scoped                     │
# ├─────────────────┼────────────────────────────────────────────────────────┤
# │ Episodic Memory │ Past interactions (Chat History)                       │
# │                 │ • Conversation history                                 │
# │                 │ • Session summaries                                    │
# │                 │ Lifecycle: Persistent, session-scoped                  │
# ├─────────────────┼────────────────────────────────────────────────────────┤
# │ Procedural      │ How to do things (System Prompt + Skills)              │
# │ Memory          │ • Agent instructions                                   │
# │                 │ • Domain knowledge                                     │
# │                 │ Lifecycle: Static, embedded in prompts                 │
# └─────────────────┴────────────────────────────────────────────────────────┘

# Reference:
# - CoALA: Cognitive Architectures for Language Agents
# - MemGPT/Letta: Memory-augmented LLM agents
# - Weaviate Context Engineering
# - Anthropic Multi-agent Research System
# """



"""
Memory System Components

Includes:
- Core Memory: Long-term user/persona information
- Recursive Summary: Conversation summarization
- Working Memory: Task-specific scratchpad (NEW)
- Memory Update Agent: Background memory updates
"""

# Core Memory
from src.agents.memory.core_memory import CoreMemory

# Recursive Summary
from src.agents.memory.recursive_summary import RecursiveSummaryManager

# Working Memory (NEW)
from src.agents.memory.working_memory import (
    WorkingMemory,
    WorkingMemoryManager,
    WorkingMemoryEntry,
    EntryType,
    Priority,
    get_working_memory,
    get_working_memory_manager,
)

# Working Memory Integration (NEW)
from src.agents.memory.working_memory_integration import (
    WorkingMemoryIntegration,
    setup_working_memory_for_request,
)

# Memory Update Agent
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
