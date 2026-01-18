"""
Prompt Templates for Cross-Model Compatibility

This module provides standardized prompt templates that work consistently
across different LLM providers (OpenAI, Gemini, Anthropic).
"""

from src.agents.prompts.tool_calling_prompts import (
    ProviderHint,
    get_tool_calling_system_prompt,
    get_analysis_prompt_with_tools,
    get_router_system_prompt,
    get_router_output_format,
    get_synthesis_prompt,
    format_tools_for_routing,
    format_tools_for_execution,
    UNIVERSAL_TOOL_CALLING_PROTOCOL,
    DATA_INTEGRITY_RULES,
    THINK_TOOL_INSTRUCTION,
    WEB_SEARCH_INSTRUCTION,
)

__all__ = [
    "ProviderHint",
    "get_tool_calling_system_prompt",
    "get_analysis_prompt_with_tools",
    "get_router_system_prompt",
    "get_router_output_format",
    "get_synthesis_prompt",
    "format_tools_for_routing",
    "format_tools_for_execution",
    "UNIVERSAL_TOOL_CALLING_PROTOCOL",
    "DATA_INTEGRITY_RULES",
    "THINK_TOOL_INSTRUCTION",
    "WEB_SEARCH_INSTRUCTION",
]
