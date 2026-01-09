"""
Log Formatter - Structured logging utilities

Provides consistent formatting for log messages across the application.
"""

from typing import Any, Dict, Optional


class LogFormatter:
    """
    Utility class for formatting structured log messages.

    Used by various components to create consistent, readable log output.
    """

    @staticmethod
    def format_classification(
        query_type: str,
        symbols: list,
        categories: list,
        requires_tools: bool,
        confidence: float,
        elapsed_ms: int,
    ) -> str:
        """Format classification result for logging."""
        return (
            f"Type={query_type} | "
            f"Symbols={symbols} | "
            f"Categories={categories} | "
            f"Tools={requires_tools} | "
            f"Confidence={confidence:.2f} | "
            f"Time={elapsed_ms}ms"
        )

    @staticmethod
    def format_tool_execution(
        tool_name: str,
        params: Dict[str, Any],
        status: str,
        elapsed_ms: int,
    ) -> str:
        """Format tool execution result for logging."""
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"[{tool_name}] {params_str} → {status} ({elapsed_ms}ms)"

    @staticmethod
    def format_agent_turn(
        turn_number: int,
        tool_calls: int,
        has_response: bool,
        elapsed_ms: int,
    ) -> str:
        """Format agent turn summary for logging."""
        response_status = "✓ Response" if has_response else "→ Continue"
        return (
            f"Turn {turn_number}: "
            f"{tool_calls} tool calls | "
            f"{response_status} | "
            f"{elapsed_ms}ms"
        )

    @staticmethod
    def section_header(title: str, width: int = 50) -> str:
        """Create a section header for log output."""
        return "─" * width + f"\n{title}\n" + "─" * width

    @staticmethod
    def key_value(key: str, value: Any, indent: int = 2) -> str:
        """Format a key-value pair for logging."""
        prefix = " " * indent + "├─ "
        return f"{prefix}{key}: {value}"
