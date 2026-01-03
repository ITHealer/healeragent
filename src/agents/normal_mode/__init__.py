"""
Normal Mode Agent Module

Provides the simplified agent loop for 90% of queries.
Uses OpenAI Runner.run() style with inline tool decisions.

Components:
- NormalModeAgent: Main agent loop with OpenAI function calling
- ToolSelectionService: 2-level tool loading (summaries vs full schemas)
- SmartToolLoader: Adaptive loading with semantic search for large registries

Usage:
    from src.agents.normal_mode import NormalModeAgent, get_normal_mode_agent

    # Using factory (singleton)
    agent = get_normal_mode_agent()
    result = await agent.run(query="What is AAPL's price?")

    # Or create instance directly
    agent = NormalModeAgent(model_name="gpt-4.1-nano")
    result = await agent.run(query="...", classification=classification)

    # 2-level tool loading
    from src.agents.normal_mode import get_tool_selection_service

    service = get_tool_selection_service()
    summaries = service.get_tool_summaries(categories=["price"])  # Level 1
    tools = service.get_tools_for_execution(tool_names=["getStockPrice"])  # Level 2

    # Smart loading with semantic search
    from src.agents.normal_mode import get_smart_tool_loader

    loader = get_smart_tool_loader()
    result = await loader.load_tools(classification, query="Analyze NVDA")
"""

from .normal_mode_agent import (
    NormalModeAgent,
    AgentResult,
    AgentTurn,
    ToolCall,
    get_normal_mode_agent,
    reset_agent,
)

from .tool_selection_service import (
    ToolSelectionService,
    ToolSummary,
    DetailLevel,
    get_tool_selection_service,
    reset_tool_selection_service,
)

from .smart_tool_loader import (
    SmartToolLoader,
    ToolLoadingResult,
    LoadingMethod,
    get_smart_tool_loader,
    reset_smart_tool_loader,
    LOAD_ALL_THRESHOLD,
    MAX_TOOLS_AFTER_FILTER,
)

__all__ = [
    # Agent
    "NormalModeAgent",
    "AgentResult",
    "AgentTurn",
    "ToolCall",
    "get_normal_mode_agent",
    "reset_agent",
    # Tool Selection (2-level)
    "ToolSelectionService",
    "ToolSummary",
    "DetailLevel",
    "get_tool_selection_service",
    "reset_tool_selection_service",
    # Smart Loader (semantic search)
    "SmartToolLoader",
    "ToolLoadingResult",
    "LoadingMethod",
    "get_smart_tool_loader",
    "reset_smart_tool_loader",
    "LOAD_ALL_THRESHOLD",
    "MAX_TOOLS_AFTER_FILTER",
]