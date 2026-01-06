"""
Hooks Module - Guard Rails and Post-Processing

This module implements the HOOKS pattern from Claude AI Architecture:
- Pre-execution validation
- Post-execution learning (memory updates)
- Error handling and retry logic
- Future: Human-in-the-loop confirmation

Available Hooks:
    - LearnHook: Post-execution memory updates (LEARN phase)
    - ValidationHook: Pre-execution validation (existing)

Usage:
    from src.agents.hooks import LearnHook, get_learn_hook

    # Option 1: Create new instance
    learn_hook = LearnHook()
    await learn_hook.on_execution_complete(...)

    # Option 2: Use singleton
    learn_hook = get_learn_hook()
    await learn_hook.on_execution_complete(...)
"""

from src.agents.hooks.learn_hook import LearnHook, get_learn_hook

__all__ = [
    "LearnHook",
    "get_learn_hook",
]
