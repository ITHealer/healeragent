"""
Enhanced Logging Utilities

Provides visual structure for logs to make debugging easier.
Uses emojis and box-drawing characters for clear phase separation.

Usage:
    from src.utils.logger.log_formatter import LogFormatter

    logger = LogFormatter(logger_instance)
    logger.phase_start("CLASSIFICATION", trace_id="abc123")
    logger.step("Extracting symbols...")
    logger.result({"type": "crypto_specific", "symbols": ["BTC"]})
    logger.phase_end("CLASSIFICATION", duration_ms=1234)
"""

from typing import Any, Dict, List, Optional, Union
import logging


# Box drawing characters
BOX_DOUBLE_H = "═"
BOX_SINGLE_H = "─"
BOX_CORNER_TL = "╔"
BOX_CORNER_TR = "╗"
BOX_CORNER_BL = "╚"
BOX_CORNER_BR = "╝"
BOX_VERT = "║"
BOX_T_RIGHT = "╠"
BOX_T_LEFT = "╣"

# Tree characters
TREE_BRANCH = "├─"
TREE_LAST = "└─"
TREE_VERT = "│"

# Emojis for quick visual scanning
EMOJI = {
    "start": "🚀",
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "time": "⏱️",
    "input": "📥",
    "output": "📤",
    "search": "🔍",
    "cache": "💾",
    "cache_hit": "🎯",
    "cache_miss": "💨",
    "llm": "🤖",
    "tool": "🔧",
    "data": "📊",
    "route": "🛤️",
    "classify": "🎯",
    "memory": "🧠",
    "context": "📋",
    "user": "👤",
    "session": "🔑",
    "complete": "✨",
    "processing": "⚙️",
    "stream": "📡",
}

# Phase colors (for terminals that support it)
PHASE_ICONS = {
    "REQUEST": "🚀",
    "CLASSIFICATION": "🎯",
    "ROUTING": "🛤️",
    "AGENT": "🤖",
    "TOOL": "🔧",
    "RESPONSE": "📤",
    "POST_PROCESSING": "💾",
    "COMPLETE": "✨",
}


class LogFormatter:
    """
    Enhanced logger wrapper that adds visual structure to logs.

    Does not change any logic - only formats log messages.
    """

    def __init__(self, logger: logging.Logger, width: int = 60):
        self.logger = logger
        self.width = width

    # =========================================================================
    # Phase Markers
    # =========================================================================

    def phase_start(
        self,
        phase_name: str,
        trace_id: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log the start of a major phase with visual separator"""
        icon = PHASE_ICONS.get(phase_name.upper(), "📊")

        self.logger.info("")
        self.logger.info(BOX_DOUBLE_H * self.width)
        self.logger.info(f"{icon} PHASE: {phase_name.upper()}" + (f" | Trace: {trace_id[:8]}..." if trace_id else ""))
        self.logger.info(BOX_DOUBLE_H * self.width)

        if extra_info:
            for key, value in extra_info.items():
                self.logger.info(f"  {TREE_BRANCH} {key}: {value}")

    def phase_end(
        self,
        phase_name: str,
        duration_ms: Optional[float] = None,
        success: bool = True
    ) -> None:
        """Log the end of a phase"""
        status = EMOJI["success"] if success else EMOJI["error"]
        time_str = f" ({duration_ms:.0f}ms)" if duration_ms else ""

        self.logger.info(f"  {TREE_LAST} {status} {phase_name} complete{time_str}")
        self.logger.info(BOX_SINGLE_H * self.width)

    def sub_phase_start(self, name: str) -> None:
        """Log start of a sub-phase"""
        self.logger.info("")
        self.logger.info(BOX_SINGLE_H * self.width)
        self.logger.info(f"📊 {name}")
        self.logger.info(BOX_SINGLE_H * self.width)

    # =========================================================================
    # Step Logging
    # =========================================================================

    def step(
        self,
        message: str,
        indent: int = 1,
        is_last: bool = False
    ) -> None:
        """Log a step within a phase"""
        prefix = "  " * indent
        branch = TREE_LAST if is_last else TREE_BRANCH
        self.logger.info(f"{prefix}{branch} {message}")

    def sub_step(
        self,
        message: str,
        indent: int = 2,
        is_last: bool = False
    ) -> None:
        """Log a sub-step (more indented)"""
        prefix = "  " * indent
        branch = TREE_LAST if is_last else TREE_BRANCH
        self.logger.info(f"{prefix}{branch} {message}")

    def detail(self, message: str, indent: int = 2) -> None:
        """Log a detail line (with vertical bar)"""
        prefix = "  " * indent
        self.logger.info(f"{prefix}{TREE_VERT}  {message}")

    # =========================================================================
    # Result Boxes
    # =========================================================================

    def result_box(
        self,
        title: str,
        data: Dict[str, Any],
        width: int = 45
    ) -> None:
        """Log a result in a box format"""
        self.logger.info(f"  ┌{'─' * (width - 2)}┐")
        self.logger.info(f"  │ {title:<{width - 4}} │")
        self.logger.info(f"  ├{'─' * (width - 2)}┤")

        for key, value in data.items():
            # Truncate long values
            str_value = str(value)
            if len(str_value) > width - len(key) - 8:
                str_value = str_value[:width - len(key) - 11] + "..."

            line = f"  {key}: {str_value}"
            self.logger.info(f"  │ {line:<{width - 4}} │")

        self.logger.info(f"  └{'─' * (width - 2)}┘")

    def classification_result(
        self,
        query_type: str,
        symbols: List[str],
        categories: List[str],
        requires_tools: bool,
        duration_ms: float,
        method: str = "llm"
    ) -> None:
        """Log classification result with structured format"""
        self.logger.info("")
        self.logger.info(f"  ┌{'─' * 43}┐")
        self.logger.info(f"  │ {EMOJI['classify']} CLASSIFICATION RESULT                  │")
        self.logger.info(f"  ├{'─' * 43}┤")
        self.logger.info(f"  │  Type: {query_type:<34}│")
        self.logger.info(f"  │  Symbols: {str(symbols):<31}│")
        self.logger.info(f"  │  Categories: {str(categories):<28}│")
        self.logger.info(f"  │  Requires Tools: {str(requires_tools):<23}│")
        self.logger.info(f"  │  Method: {method:<32}│")
        self.logger.info(f"  │  Time: {duration_ms:.0f}ms{' ' * 32}│"[:46] + "│")
        self.logger.info(f"  └{'─' * 43}┘")

    # =========================================================================
    # Specific Log Types
    # =========================================================================

    def cache_hit(self, key: str, data_type: str = "data") -> None:
        """Log cache hit"""
        self.logger.info(f"  {TREE_BRANCH} {EMOJI['cache_hit']} [CACHE HIT] {data_type}: {key[:20]}...")

    def cache_miss(self, key: str, data_type: str = "data") -> None:
        """Log cache miss"""
        self.logger.info(f"  {TREE_BRANCH} {EMOJI['cache_miss']} [CACHE MISS] {data_type}: {key[:20]}...")

    def llm_call(
        self,
        model: str,
        purpose: str = "classification"
    ) -> None:
        """Log LLM call start"""
        self.logger.info(f"  {TREE_BRANCH} {EMOJI['llm']} [LLM] Calling {model} for {purpose}...")

    def llm_response(self, summary: str, duration_ms: float) -> None:
        """Log LLM response"""
        self.logger.info(f"  {TREE_BRANCH} {EMOJI['success']} [LLM] Response: {summary} ({duration_ms:.0f}ms)")

    def tool_start(
        self,
        tool_name: str,
        params: Dict[str, Any]
    ) -> None:
        """Log tool execution start"""
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        self.logger.info(f"  ┌─ {EMOJI['tool']} TOOL: {tool_name}")
        self.logger.info(f"  │  Input: {{{params_str}}}")

    def tool_step(self, message: str) -> None:
        """Log a tool execution step"""
        self.logger.info(f"  │  {message}")

    def tool_success(
        self,
        tool_name: str,
        result_summary: str,
        duration_ms: float
    ) -> None:
        """Log tool success"""
        self.logger.info(f"  │  Result: {result_summary}")
        self.logger.info(f"  └─ {EMOJI['success']} SUCCESS ({duration_ms:.0f}ms)")

    def tool_error(
        self,
        tool_name: str,
        error: str,
        duration_ms: float
    ) -> None:
        """Log tool error"""
        self.logger.info(f"  │  Error: {error}")
        self.logger.info(f"  └─ {EMOJI['error']} FAILED ({duration_ms:.0f}ms)")

    def routing_decision(
        self,
        mode: str,
        reason: str,
        confidence: float = 0.0
    ) -> None:
        """Log routing decision"""
        self.logger.info(f"  {TREE_BRANCH} {EMOJI['route']} [ROUTE] Mode: {mode}")
        self.logger.info(f"  {TREE_BRANCH} Reason: {reason}")
        if confidence > 0:
            self.logger.info(f"  {TREE_LAST} Confidence: {confidence:.2f}")

    def context_loaded(
        self,
        history_count: int,
        symbols: List[str],
        tokens: int,
        compacted: bool = False
    ) -> None:
        """Log context loading result"""
        self.logger.info(f"  {TREE_BRANCH} {EMOJI['context']} [CONTEXT] Loaded:")
        self.logger.info(f"  │  History: {history_count} messages")
        self.logger.info(f"  │  Symbols: {symbols}")
        self.logger.info(f"  │  Tokens: {tokens}")
        self.logger.info(f"  {TREE_LAST} Compacted: {compacted}")

    def memory_update(
        self,
        operation: str,
        details: str
    ) -> None:
        """Log memory update"""
        self.logger.info(f"  {TREE_BRANCH} {EMOJI['memory']} [MEMORY] {operation}: {details}")

    # =========================================================================
    # Request/Response Boundaries
    # =========================================================================

    def request_start(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        query: str = ""
    ) -> None:
        """Log request start with clear visual boundary"""
        self.logger.info("")
        self.logger.info(BOX_DOUBLE_H * self.width)
        self.logger.info(f"{EMOJI['start']} REQUEST START | Session: {session_id[:8]}..." +
                        (f" | User: {user_id}" if user_id else ""))
        self.logger.info(BOX_DOUBLE_H * self.width)
        if query:
            display_query = query[:50] + "..." if len(query) > 50 else query
            self.logger.info(f"{EMOJI['input']} Query: \"{display_query}\"")

    def request_complete(
        self,
        total_time_ms: float,
        tool_calls: int = 0,
        turns: int = 0
    ) -> None:
        """Log request completion with summary"""
        self.logger.info("")
        self.logger.info(BOX_DOUBLE_H * self.width)
        self.logger.info(
            f"{EMOJI['complete']} REQUEST COMPLETE | "
            f"Total: {total_time_ms/1000:.2f}s | "
            f"Tools: {tool_calls} | "
            f"Turns: {turns}"
        )
        self.logger.info(BOX_DOUBLE_H * self.width)
        self.logger.info("")

    # =========================================================================
    # Turn Logging (for agent loops)
    # =========================================================================

    def turn_start(self, turn_number: int, max_turns: int) -> None:
        """Log agent turn start"""
        self.logger.info(f"  {TREE_BRANCH} [TURN {turn_number}/{max_turns}] Processing...")

    def turn_decision(self, decision: str, tools: List[str] = None) -> None:
        """Log turn decision"""
        if tools:
            self.logger.info(f"  │  Decision: {decision}")
            self.logger.info(f"  │  Tools: {tools}")
        else:
            self.logger.info(f"  │  Decision: {decision}")

    def turn_complete(self, turn_number: int, duration_ms: float) -> None:
        """Log turn completion"""
        self.logger.info(f"  {TREE_LAST} [TURN {turn_number}] Complete ({duration_ms:.0f}ms)")


# Convenience function to create a LogFormatter
def get_log_formatter(logger: logging.Logger, width: int = 60) -> LogFormatter:
    """Create a LogFormatter for the given logger"""
    return LogFormatter(logger, width)
