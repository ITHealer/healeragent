"""
Anti-Loop / Tool Validation for Agent Loop

Provides soft warnings when the agent exhibits loop-like behavior:
1. Tool usage counting - warns when same tool called too many times
2. Query similarity detection - warns when similar queries sent to same tool
3. Turn budget tracking - warns when approaching max turns

All warnings are SOFT (injected as system messages, never blocking).
This prevents the agent from wasting tokens on redundant tool calls.

Usage:
    validator = ToolValidator(max_tool_calls=3, similarity_threshold=0.7)

    # Before each tool execution
    warnings = validator.validate_tool_call(tool_name, arguments)
    for warning in warnings:
        messages.append({"role": "system", "content": warning})

    # After execution
    validator.record_tool_call(tool_name, arguments)
"""

import logging
from typing import Any, Dict, List, Optional, Set


# ============================================================================
# JACCARD SIMILARITY FOR QUERY COMPARISON
# ============================================================================

def _tokenize(text: str) -> Set[str]:
    """Simple word-level tokenization for similarity comparison."""
    return set(text.lower().split())


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Compute Jaccard similarity between two text strings.

    Returns value between 0.0 (no overlap) and 1.0 (identical).
    Uses word-level tokens for speed.
    """
    if not text_a or not text_b:
        return 0.0

    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    return len(intersection) / len(union) if union else 0.0


# ============================================================================
# TOOL VALIDATOR
# ============================================================================

class ToolValidator:
    """
    Validates tool calls and detects loop-like behavior.

    Tracks:
    - How many times each tool has been called
    - Query strings per tool for similarity detection
    - Total tool calls across all tools
    """

    # Default limits (soft - warnings only, never blocking)
    DEFAULT_MAX_SAME_TOOL = 3      # Warn after 3 calls to same tool
    DEFAULT_SIMILARITY_THRESHOLD = 0.7  # Warn when query similarity > 70%
    DEFAULT_MAX_TOTAL_CALLS = 15   # Warn when total calls exceed this

    def __init__(
        self,
        max_same_tool: int = DEFAULT_MAX_SAME_TOOL,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_total_calls: int = DEFAULT_MAX_TOTAL_CALLS,
    ):
        self.max_same_tool = max_same_tool
        self.similarity_threshold = similarity_threshold
        self.max_total_calls = max_total_calls

        # Tracking state
        self._tool_counts: Dict[str, int] = {}
        self._tool_queries: Dict[str, List[str]] = {}
        self._total_calls: int = 0

        self.logger = logging.getLogger("tool_validator")

    def validate_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Validate a batch of tool calls before execution.

        Args:
            tool_calls: List of dicts with 'name' and 'arguments' keys

        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []

        for tc in tool_calls:
            tc_warnings = self._validate_single(
                tool_name=tc.get("name", ""),
                arguments=tc.get("arguments", {}),
            )
            warnings.extend(tc_warnings)

        # Check total calls
        pending_total = self._total_calls + len(tool_calls)
        if pending_total > self.max_total_calls:
            warnings.append(
                f"High tool usage: {pending_total} total calls. "
                f"Focus on synthesizing available data rather than gathering more."
            )

        return warnings

    def _validate_single(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> List[str]:
        """Validate a single tool call."""
        warnings = []

        # Skip validation for meta-tools
        if tool_name in ("think", "tool_search", "invoke_workflow"):
            return warnings

        # 1. Tool usage count check
        current_count = self._tool_counts.get(tool_name, 0)
        if current_count >= self.max_same_tool:
            warnings.append(
                f"Tool '{tool_name}' already called {current_count} times. "
                f"Consider using a different tool or synthesizing existing data."
            )

        # 2. Query similarity check
        query_key = self._extract_query_key(arguments)
        if query_key and tool_name in self._tool_queries:
            for old_query in self._tool_queries[tool_name]:
                sim = jaccard_similarity(query_key, old_query)
                if sim > self.similarity_threshold:
                    warnings.append(
                        f"Similar query detected for '{tool_name}': "
                        f"'{query_key[:50]}' is {sim:.0%} similar to a previous call. "
                        f"Try different search terms or parameters."
                    )
                    break  # One warning per tool is enough

        return warnings

    def record_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> None:
        """
        Record tool calls after execution (updates tracking state).

        Args:
            tool_calls: List of dicts with 'name' and 'arguments' keys
        """
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            arguments = tc.get("arguments", {})

            # Update count
            self._tool_counts[tool_name] = self._tool_counts.get(tool_name, 0) + 1
            self._total_calls += 1

            # Update query tracker
            query_key = self._extract_query_key(arguments)
            if query_key:
                self._tool_queries.setdefault(tool_name, []).append(query_key)

    def get_usage_summary(self) -> str:
        """Get a summary of tool usage for logging."""
        if not self._tool_counts:
            return "No tools called yet"
        parts = [f"{name}={count}" for name, count in sorted(self._tool_counts.items())]
        return f"Tool usage: {', '.join(parts)} (total={self._total_calls})"

    @staticmethod
    def _extract_query_key(arguments: Dict[str, Any]) -> Optional[str]:
        """
        Extract the main query/search string from tool arguments.

        Looks for common parameter names used across financial tools.
        """
        # Priority order for query-like parameters
        query_params = ["query", "symbol", "ticker", "search_query", "keyword", "name"]
        for param in query_params:
            value = arguments.get(param)
            if value and isinstance(value, str):
                return value

        # For tools with multiple symbols
        symbols = arguments.get("symbols", [])
        if isinstance(symbols, list) and symbols:
            return " ".join(str(s) for s in symbols)

        return None
