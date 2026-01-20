"""
Multi-Model Prompt System for HealerAgent

This module provides model-specific system prompts optimized for different LLM providers.
Based on analysis of Google, OpenAI, and Claude system prompt best practices.

Architecture:
- BasePromptTemplate: Golden principles shared across all models
- OpenAIPromptTemplate: GPT-5, GPT-5-mini, GPT-5.1-mini optimizations
- GeminiPromptTemplate: Gemini-2.5-pro, Gemini-2.5-flash, Gemini-3-flash optimizations
- PromptManager: Selects and composes prompts based on provider/model

Usage:
    from src.agents.prompts import PromptManager

    manager = PromptManager()
    system_prompt = manager.get_system_prompt(
        provider="openai",
        model="gpt-5",
        context={...}
    )
"""

from .base_prompt import BasePromptTemplate, PromptContext
from .openai_prompt import OpenAIPromptTemplate
from .gemini_prompt import GeminiPromptTemplate
from .prompt_manager import PromptManager

__all__ = [
    "BasePromptTemplate",
    "PromptContext",
    "OpenAIPromptTemplate",
    "GeminiPromptTemplate",
    "PromptManager",
]
