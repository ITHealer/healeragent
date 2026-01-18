"""
Tool Calling Prompts - Cross-Model Compatible System Prompts

This module provides standardized prompt templates for tool calling that work
consistently across OpenAI, Gemini, and Anthropic models.

Key Design Principles (from research):
1. Explicit tool calling workflow in system prompt
2. Clear parameter constraints in plain language
3. Examples in description, not just schema
4. Model-agnostic language (no provider-specific terms)

References:
- Mastra: "Passing schema constraints into tool instructions worked well"
- Google: "In prompts, mention that the model should use tools when it needs external data"
- Agenta: "Use proper JSON schema types and enums to validate input values strongly"

Usage:
    from src.agents.prompts.tool_calling_prompts import (
        get_tool_calling_system_prompt,
        get_analysis_prompt_with_tools,
        ProviderHint,
    )

    prompt = get_tool_calling_system_prompt(
        provider_hint=ProviderHint.GEMINI,
        enable_think_tool=True,
    )
"""

from enum import Enum
from typing import List, Optional, Dict, Any


class ProviderHint(str, Enum):
    """Provider hint for prompt optimization."""
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    UNIVERSAL = "universal"


# ============================================================================
# CORE TOOL CALLING PROMPTS
# ============================================================================

UNIVERSAL_TOOL_CALLING_PROTOCOL = """## Tool Calling Protocol

You have access to specialized data tools. Follow this workflow:

### Step 1: Analyze Query
- Identify what data is needed to answer the user's question
- Determine which tools can provide that data
- Plan which tools to call (can call multiple in parallel)

### Step 2: Call Tools
- Execute tools to fetch real data
- Provide all REQUIRED parameters for each tool
- Use valid values from enum/allowed options

### Step 3: Synthesize Response
- Analyze all tool results together
- Cite specific numbers from the data
- Acknowledge any missing or failed data
- Provide actionable insights based on the data

### Critical Rules
- NEVER fabricate data - only use actual tool results
- ALWAYS call tools before making claims about current data
- If a tool fails, note the limitation and proceed with available data
"""

GEMINI_TOOL_CALLING_ADDITION = """
### Function Calling Requirements (IMPORTANT)
- You MUST call available functions to get real-time data
- Provide exact values for required parameters
- Multiple function calls in one turn are encouraged
- Wait for function results before generating final response
"""

OPENAI_TOOL_CALLING_ADDITION = """
### Tool Usage Notes
- Use provided tools to gather data before responding
- Call multiple tools simultaneously when possible
- Process all tool results before final synthesis
"""

ANTHROPIC_TOOL_CALLING_ADDITION = """
### Tool Use Guidelines
- Use tools to retrieve current information
- Provide accurate parameter values
- Combine insights from multiple tool results
"""


# ============================================================================
# THINK TOOL PROMPT
# ============================================================================

THINK_TOOL_INSTRUCTION = """
### Think Tool (Recommended)
Use the `think` tool to organize your reasoning:
- **Before tools**: Plan what data you need
- **After tools**: Analyze results and identify insights
- **Before response**: Structure your final answer

Pattern: think(planning) -> call data tools -> think(analyzing) -> respond
"""


# ============================================================================
# WEB SEARCH PROMPT
# ============================================================================

WEB_SEARCH_INSTRUCTION = """
### Web Search (Available)
Use `webSearch` for:
- Latest news and market developments
- Information not in financial data APIs
- Strategy concepts and market context

Include source citations in your response: [Title](URL)
"""


# ============================================================================
# DATA INTEGRITY PROMPT
# ============================================================================

DATA_INTEGRITY_RULES = """
## Data Integrity (MANDATORY)

1. **Use Tool Data**: Every claim must be backed by tool results
2. **Cite Exact Numbers**: Quote specific values (e.g., "RSI at 67.3")
3. **No Fabrication**: Never invent numbers or fake statistics
4. **Source Attribution**: State which tool provided each data point
5. **Acknowledge Gaps**: Clearly note when data is missing or unavailable
"""


# ============================================================================
# BUILDER FUNCTIONS
# ============================================================================

def get_tool_calling_system_prompt(
    provider_hint: ProviderHint = ProviderHint.UNIVERSAL,
    enable_think_tool: bool = False,
    enable_web_search: bool = False,
    include_data_integrity: bool = True,
) -> str:
    """
    Build cross-model compatible tool calling system prompt.

    Args:
        provider_hint: Target LLM provider for optimization
        enable_think_tool: Include think tool instructions
        enable_web_search: Include web search instructions
        include_data_integrity: Include data integrity rules

    Returns:
        Complete system prompt section for tool calling
    """
    parts = [UNIVERSAL_TOOL_CALLING_PROTOCOL]

    # Add provider-specific additions
    if provider_hint == ProviderHint.GEMINI:
        parts.append(GEMINI_TOOL_CALLING_ADDITION)
    elif provider_hint == ProviderHint.OPENAI:
        parts.append(OPENAI_TOOL_CALLING_ADDITION)
    elif provider_hint == ProviderHint.ANTHROPIC:
        parts.append(ANTHROPIC_TOOL_CALLING_ADDITION)

    # Add optional sections
    if enable_think_tool:
        parts.append(THINK_TOOL_INSTRUCTION)

    if enable_web_search:
        parts.append(WEB_SEARCH_INSTRUCTION)

    if include_data_integrity:
        parts.append(DATA_INTEGRITY_RULES)

    return "\n".join(parts)


def get_analysis_prompt_with_tools(
    base_prompt: str,
    context_parts: List[str],
    user_context: str = "",
    provider_hint: ProviderHint = ProviderHint.UNIVERSAL,
    enable_think_tool: bool = False,
    enable_web_search: bool = False,
) -> str:
    """
    Build complete analysis prompt with tool calling instructions.

    Args:
        base_prompt: Domain skill prompt (e.g., StockSkill prompt)
        context_parts: List of context items (date, symbols, etc.)
        user_context: User profile and conversation summary
        provider_hint: Target LLM provider
        enable_think_tool: Include think tool instructions
        enable_web_search: Include web search instructions

    Returns:
        Complete system prompt for analysis
    """
    tool_prompt = get_tool_calling_system_prompt(
        provider_hint=provider_hint,
        enable_think_tool=enable_think_tool,
        enable_web_search=enable_web_search,
    )

    # Format context
    context_section = ""
    if context_parts:
        context_section = "\n## Current Context\n" + "\n".join(f"- {p}" for p in context_parts)

    # Assemble prompt
    return f"""{base_prompt}

---
{context_section}
{user_context}

{tool_prompt}
"""


# ============================================================================
# ROUTER PROMPTS (Simplified for cross-model)
# ============================================================================

def get_router_system_prompt() -> str:
    """Get cross-model compatible router system prompt."""
    return """You are a tool routing system. Given a user query and available tools, select the most appropriate tools.

Your task:
1. Analyze the query to understand what data is needed
2. Select tools that can provide that data
3. Determine complexity (simple/medium/complex)
4. Choose execution strategy

Output your decision as valid JSON (no markdown, no code blocks).
"""


def get_router_output_format() -> str:
    """Get cross-model compatible output format instruction."""
    return """
OUTPUT FORMAT (respond with valid JSON only):

{
  "selected_tools": ["tool1", "tool2"],
  "complexity": "simple",
  "execution_strategy": "direct",
  "reasoning": "Brief explanation",
  "confidence": 0.9
}

Values for complexity: "simple" (1-2 tools), "medium" (3-5 tools), "complex" (6+ tools)
Values for execution_strategy: "direct", "iterative", "parallel"
"""


# ============================================================================
# TOOL CATALOG FORMATTING
# ============================================================================

def format_tools_for_routing(
    tools: List[Dict[str, Any]],
    max_tools: int = 50,
) -> str:
    """
    Format tools for routing prompt (concise format).

    Args:
        tools: List of tool schemas
        max_tools: Maximum tools to include

    Returns:
        Formatted tool catalog string
    """
    lines = ["## Available Tools\n"]

    for tool in tools[:max_tools]:
        name = tool.get("name", "unknown")
        category = tool.get("metadata", {}).get("category", tool.get("category", "other"))
        description = tool.get("description", "")

        # Truncate description
        if len(description) > 100:
            description = description[:97] + "..."

        lines.append(f"- **{name}** [{category}]: {description}")

    return "\n".join(lines)


def format_tools_for_execution(
    tools: List[Dict[str, Any]],
    provider_hint: ProviderHint = ProviderHint.UNIVERSAL,
) -> str:
    """
    Format tool descriptions for execution context.

    Args:
        tools: List of tool schemas (full)
        provider_hint: Target provider for optimization

    Returns:
        Formatted tool descriptions
    """
    lines = ["## Available Tools\n"]

    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        description = func.get("description", "")

        lines.append(f"### {name}")
        lines.append(description)
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# SYNTHESIS PROMPTS
# ============================================================================

SYNTHESIS_PROMPT_TEMPLATE = """You have gathered data from multiple tools. Now synthesize a comprehensive response.

## Tool Results
{tool_results}

## Your Task
Analyze all the data above and provide a response that:
1. **Uses ALL data** - incorporate every relevant piece of information
2. **Cites specific numbers** - quote exact values from tool results
3. **Explains significance** - what does each metric mean for the user
4. **Provides actionable insights** - clear recommendations based on data
5. **Acknowledges limitations** - note any missing or failed data

Match the user's language and be conversational but data-driven.
"""


def get_synthesis_prompt(
    tool_results: List[Dict[str, Any]],
    analysis_framework: str = "",
) -> str:
    """
    Build synthesis prompt after tool execution.

    Args:
        tool_results: List of tool results
        analysis_framework: Optional analysis framework from skill

    Returns:
        Synthesis system prompt
    """
    # Format tool results
    results_text = ""
    for result in tool_results:
        tool_name = result.get("tool_name", "Unknown")
        status = result.get("status", "unknown")

        if status in ("success", "200"):
            data = result.get("formatted_context") or result.get("data", {})
            if isinstance(data, dict):
                import json
                data = json.dumps(data, indent=2)
            results_text += f"\n### {tool_name}\n{data}\n"
        else:
            error = result.get("error", "Unknown error")
            results_text += f"\n### {tool_name} (FAILED)\nError: {error}\n"

    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(tool_results=results_text)

    if analysis_framework:
        prompt = analysis_framework + "\n\n" + prompt

    return prompt
