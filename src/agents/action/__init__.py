# File: src/agents/action/__init__.py
"""
Action Module - Task Execution

Components:
- TaskExecutor: Execute tasks with retry and multi-symbol support

REMOVED:
- ReplanningAgent: Removed for simplicity (Claude/ChatGPT don't use)
  Retry is now implicit via exponential backoff in TaskExecutor
"""

from .task_executor import TaskExecutor

__all__ = [
    'TaskExecutor',
]

# ============================================================================
# MIGRATION NOTE:
# ============================================================================
# 
# ReplanningAgent has been removed. The TaskExecutor now handles:
# - Retry with exponential backoff for transient errors
# - Graceful skip for permanent errors
# - Multi-symbol expansion
#
# This is the same pattern used by Claude and ChatGPT.
#
# If you need the old ReplanningAgent behavior, you can:
# 1. Use Think Tool for reasoning about failures (recommended)
# 2. Implement custom error handling in your chat handler
#
# ============================================================================