# File: src/agents/tools/memory/__init__.py
"""
Memory Tools Package

Provides memory search capabilities as callable tools.
These tools allow the agent to dynamically search memory
during task execution (memory-as-a-tool pattern).

Tools:
- searchRecallMemory: Search conversation history
- searchArchivalMemory: Search knowledge base
- searchProceduralMemory: Search learned patterns

Based on:
- Google ADK: "memory-as-a-tool" pattern
- momo-research: Dynamic memory retrieval
"""

from src.agents.tools.memory.search_recall_memory import SearchRecallMemoryTool
from src.agents.tools.memory.search_archival_memory import SearchArchivalMemoryTool
from src.agents.tools.memory.search_procedural_memory import SearchProceduralMemoryTool

__all__ = [
    "SearchRecallMemoryTool",
    "SearchArchivalMemoryTool", 
    "SearchProceduralMemoryTool"
]