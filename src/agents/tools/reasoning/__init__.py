# File: src/agents/tools/reasoning/__init__.py
"""
Reasoning Tools Package

Provides reasoning and thinking capabilities for the agent.
These tools allow the agent to pause and reason during task execution.

Tools:
- think: Structured reasoning tool for complex scenarios

Based on:
- Anthropic τ-Bench: Think tool research
- Claude 3.7 Sonnet SWE-Bench implementation
"""

from src.agents.tools.reasoning.think_tool import (
    ThinkTool,
    ThinkToolPrompts,
    create_think_tool
)

__all__ = [
    "ThinkTool",
    "ThinkToolPrompts",
    "create_think_tool"
]