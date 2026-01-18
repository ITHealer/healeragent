"""
Tool Search Handler - Integration for Agent Tool Execution

Handles `tool_search` function calls from GPT and manages dynamic tool injection.

When GPT calls `tool_search`:
1. Execute semantic search using ToolSearchService
2. Return search results as text to GPT
3. Expand the agent's available tools with found tools
4. GPT can then use those tools in subsequent turns

Usage in UnifiedAgent:
    handler = ToolSearchHandler()

    # In agent loop, before executing tool calls:
    tool_calls, new_tools = handler.process_tool_calls(
        tool_calls=parsed_tool_calls,
        current_tools=current_tools,
    )

    # Execute non-tool_search calls normally
    # Inject new_tools into the tools list for next LLM call
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from src.utils.logger.custom_logging import LoggerMixin
from src.services.tool_search_service import (
    get_tool_search_service,
    ToolSearchService,
    TOOL_SEARCH_DEFINITION,
)


@dataclass
class ToolSearchHandlerResult:
    """Result from processing tool_search call."""

    # Tool calls that are NOT tool_search (to execute normally)
    regular_tool_calls: List[Any]

    # Results from tool_search calls (tool_call_id -> result)
    search_results: Dict[str, Dict[str, Any]]

    # New tool definitions to inject into agent's tools
    discovered_tools: List[Dict[str, Any]]

    # Names of discovered tools
    discovered_tool_names: List[str]

    def has_search_results(self) -> bool:
        return len(self.search_results) > 0


class ToolSearchHandler(LoggerMixin):
    """
    Handler for tool_search function calls.

    Integrates with UnifiedAgent to enable dynamic tool discovery
    during the agent loop.
    """

    def __init__(self, service: Optional[ToolSearchService] = None):
        super().__init__()
        self._service = service or get_tool_search_service()
        self._discovered_tools: Dict[str, Dict[str, Any]] = {}

    def get_tool_search_definition(self) -> Dict[str, Any]:
        """Get the tool_search function definition."""
        return TOOL_SEARCH_DEFINITION.copy()

    async def process_tool_calls(
        self,
        tool_calls: List[Any],  # List of ToolCall objects
        current_tool_names: Optional[List[str]] = None,
    ) -> ToolSearchHandlerResult:
        """
        Process tool calls, handling any tool_search calls specially.

        Args:
            tool_calls: List of ToolCall objects from LLM
            current_tool_names: Currently available tool names

        Returns:
            ToolSearchHandlerResult with:
            - regular_tool_calls: Calls to execute normally
            - search_results: Results from tool_search calls
            - discovered_tools: New tools to add to agent
        """
        regular_calls = []
        search_results = {}
        discovered_tools = []
        discovered_names = []

        current_names = set(current_tool_names or [])

        for tool_call in tool_calls:
            tool_name = getattr(tool_call, 'name', None)
            if tool_name is None and isinstance(tool_call, dict):
                tool_name = tool_call.get('name')

            if tool_name == "tool_search":
                # Handle tool_search call
                call_id = getattr(tool_call, 'id', None)
                if call_id is None and isinstance(tool_call, dict):
                    call_id = tool_call.get('id', 'unknown')

                arguments = getattr(tool_call, 'arguments', {})
                if not arguments and isinstance(tool_call, dict):
                    arguments = tool_call.get('arguments', {})

                # Execute search
                query = arguments.get('query', '')
                top_k = arguments.get('top_k', 5)

                self.logger.info(
                    f"[TOOL_SEARCH_HANDLER] Processing search: '{query}'"
                )

                try:
                    response = await self._service.search(query, top_k)

                    # Get tool definitions for found tools
                    for tool_name_found in response.tool_names:
                        if tool_name_found not in current_names:
                            tool_def = self._service.get_tool_definition(tool_name_found)
                            if tool_def:
                                discovered_tools.append(tool_def)
                                discovered_names.append(tool_name_found)
                                current_names.add(tool_name_found)

                    # Store result
                    search_results[call_id] = {
                        "status": "success",
                        "result": response.format_for_llm(),
                        "tool_names": response.tool_names,
                        "search_time_ms": response.search_time_ms,
                    }

                    self.logger.info(
                        f"[TOOL_SEARCH_HANDLER] Found {len(response.tool_names)} tools, "
                        f"discovered {len(discovered_tools)} new"
                    )

                except Exception as e:
                    self.logger.error(
                        f"[TOOL_SEARCH_HANDLER] Search failed: {e}"
                    )
                    search_results[call_id] = {
                        "status": "error",
                        "error": str(e),
                    }
            else:
                # Regular tool call - pass through
                regular_calls.append(tool_call)

        return ToolSearchHandlerResult(
            regular_tool_calls=regular_calls,
            search_results=search_results,
            discovered_tools=discovered_tools,
            discovered_tool_names=discovered_names,
        )

    def format_search_result_for_message(
        self,
        call_id: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Format a tool_search result as a tool message for the LLM.

        Args:
            call_id: Tool call ID
            result: Search result dict

        Returns:
            Message dict for LLM
        """
        if result.get("status") == "error":
            content = json.dumps({
                "error": result.get("error", "Search failed"),
                "message": "Tool search failed. Please try a different query."
            })
        else:
            content = result.get("result", "No results found")

        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        }

    def get_expanded_tools(
        self,
        base_tools: List[Dict[str, Any]],
        include_tool_search: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get expanded tools list with tool_search and discovered tools.

        Args:
            base_tools: Base tool definitions
            include_tool_search: Whether to include tool_search meta-tool

        Returns:
            Expanded tools list
        """
        tools = list(base_tools)

        # Add tool_search if requested
        if include_tool_search:
            tool_search_def = self.get_tool_search_definition()
            # Check if already present
            if not any(
                t.get('function', {}).get('name') == 'tool_search'
                for t in tools
            ):
                tools.append(tool_search_def)

        return tools

    def clear_discovered(self) -> None:
        """Clear discovered tools cache."""
        self._discovered_tools.clear()


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_handler_instance: Optional[ToolSearchHandler] = None


def get_tool_search_handler() -> ToolSearchHandler:
    """Get singleton ToolSearchHandler instance."""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = ToolSearchHandler()
    return _handler_instance


def reset_tool_search_handler() -> None:
    """Reset singleton instance."""
    global _handler_instance
    _handler_instance = None


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

async def handle_tool_search_in_calls(
    tool_calls: List[Any],
    current_tool_names: List[str],
) -> Tuple[List[Any], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convenience function to handle tool_search in a list of tool calls.

    Args:
        tool_calls: List of tool calls from LLM
        current_tool_names: Currently available tool names

    Returns:
        Tuple of (regular_calls, search_results, new_tool_definitions)
    """
    handler = get_tool_search_handler()
    result = await handler.process_tool_calls(tool_calls, current_tool_names)
    return (
        result.regular_tool_calls,
        result.search_results,
        result.discovered_tools,
    )


def get_initial_tools_with_search(
    selected_tools: Optional[List[str]] = None,
    include_all: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Get initial tool list with tool_search meta-tool.

    For token-efficient setup, returns just tool_search initially.
    GPT can discover more tools as needed.

    Args:
        selected_tools: Optional list of pre-selected tool names
        include_all: If True, include all tools (no optimization)

    Returns:
        Tuple of (tool_definitions, tool_names)
    """
    service = get_tool_search_service()

    tools = []
    tool_names = []

    # Always add tool_search
    tools.append(TOOL_SEARCH_DEFINITION)
    tool_names.append("tool_search")

    # Add pre-selected tools if any
    if selected_tools:
        for name in selected_tools:
            tool_def = service.get_tool_definition(name)
            if tool_def:
                tools.append(tool_def)
                tool_names.append(name)

    # Or add all tools (no optimization)
    if include_all:
        all_names = service.get_all_tool_names()
        all_defs = service.get_tool_definitions(all_names)
        for tool_def in all_defs:
            func_name = tool_def.get('function', {}).get('name')
            if func_name and func_name not in tool_names:
                tools.append(tool_def)
                tool_names.append(func_name)

    return tools, tool_names
