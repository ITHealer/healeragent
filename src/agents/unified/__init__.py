"""
Unified Agent package.

Provides a single agent that adapts execution strategy based on query complexity.
Merges concepts from Normal Mode Agent and Deep Research (Streaming Chat Handler).
"""

from src.agents.unified.unified_agent import (
    UnifiedAgent,
    UnifiedAgentResult,
    get_unified_agent,
    reset_unified_agent,
)

__all__ = [
    "UnifiedAgent",
    "UnifiedAgentResult",
    "get_unified_agent",
    "reset_unified_agent",
]
