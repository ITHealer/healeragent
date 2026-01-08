"""
Router package for LLM-based tool selection.

Provides ChatGPT-style 2-phase tool selection:
1. Router LLM sees ALL tool summaries → selects relevant tools
2. Agent LLM gets full schemas for selected tools only
"""

from src.agents.router.llm_tool_router import (
    LLMToolRouter,
    RouterDecision,
    Complexity,
    ExecutionStrategy,
    get_tool_router,
    reset_router,
)

__all__ = [
    "LLMToolRouter",
    "RouterDecision",
    "Complexity",
    "ExecutionStrategy",
    "get_tool_router",
    "reset_router",
]
