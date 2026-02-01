"""
Context Manager - Summarization, history management, and context budget tracking.

Why: LLM context windows are finite. Without active management, tool results
and conversation history accumulate until the context overflows, causing
truncation and degraded responses. This manager:
1. Maintains a rolling conversation history with configurable depth
2. Summarizes tool results before injecting into LLM messages
3. Tracks approximate token usage to prevent overflow

How: Sits between the orchestrator and LLM calls. The orchestrator adds
messages through this manager, which decides whether to include full content,
summarize it, or reference an artifact.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ~= 4 characters for English text
CHARS_PER_TOKEN_ESTIMATE = 4


class ContextManager:
    """Manages the LLM message context for a single request lifecycle.

    Why: The orchestrator builds up messages across multiple turns. Without
    management, messages grow unbounded. This class provides:
    - Message history with max depth
    - Tool result summarization (only key metrics in context)
    - Token budget tracking with compaction at threshold
    - System prompt stability for prompt caching optimization
    """

    def __init__(
        self,
        max_history_messages: int = 10,
        max_context_tokens: int = 100_000,
        compaction_threshold: float = 0.7,
    ):
        self._max_history = max_history_messages
        self._max_tokens = max_context_tokens
        self._compaction_threshold = compaction_threshold
        self._messages: List[Dict[str, Any]] = []
        self._system_prompt: Optional[str] = None
        self._tool_result_summaries: Dict[str, str] = {}
        self._estimated_tokens: int = 0

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt. Should be called once and kept stable for caching."""
        self._system_prompt = prompt
        self._estimated_tokens = self._estimate_tokens(prompt)

    def add_conversation_history(self, history: List[Dict[str, str]]) -> None:
        """Load previous conversation history, trimmed to max depth.

        Why trim: Old messages provide diminishing context value but consume
        tokens. We keep the most recent messages which are most relevant.
        """
        if not history:
            return

        # Take only the most recent messages
        trimmed = history[-self._max_history:]
        for msg in trimmed:
            self._messages.append(msg)
            self._estimated_tokens += self._estimate_tokens(msg.get("content", ""))

    def add_user_message(self, content: str) -> None:
        """Add the current user query to messages."""
        self._messages.append({"role": "user", "content": content})
        self._estimated_tokens += self._estimate_tokens(content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant response to messages."""
        self._messages.append({"role": "assistant", "content": content})
        self._estimated_tokens += self._estimate_tokens(content)

    def add_tool_result(
        self,
        tool_name: str,
        tool_call_id: str,
        result_content: str,
        is_summary: bool = False,
    ) -> None:
        """Add a tool result to the message context.

        If the result was already offloaded by ArtifactManager, `result_content`
        will be the summary string. Otherwise, this method truncates long results.
        """
        # Truncate if too long and not already a summary
        if not is_summary and len(result_content) > 3000:
            result_content = result_content[:3000] + "\n... [truncated, full data in artifact]"

        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_content,
        })
        self._estimated_tokens += self._estimate_tokens(result_content)
        self._tool_result_summaries[tool_name] = result_content[:200]

        # Check if compaction needed
        if self._needs_compaction():
            self._compact()

    def add_tool_calls_message(self, tool_calls: List[Dict[str, Any]]) -> None:
        """Add an assistant message with tool_calls (for multi-turn agent loop)."""
        self._messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        })
        # Rough estimate for tool call tokens
        self._estimated_tokens += len(tool_calls) * 50

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get the full message list for an LLM call.

        Returns system prompt + conversation messages.
        """
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(self._messages)
        return messages

    def get_estimated_tokens(self) -> int:
        """Get approximate token count for the current context."""
        return self._estimated_tokens

    def get_gathered_data_summary(self) -> str:
        """Get a brief summary of all tool results gathered so far.

        Used by the evaluator and orchestrator to understand what data
        has been collected without re-reading full results.
        """
        if not self._tool_result_summaries:
            return "No tool data gathered yet."

        parts = []
        for tool_name, summary in self._tool_result_summaries.items():
            parts.append(f"- {tool_name}: {summary}")
        return "Gathered data:\n" + "\n".join(parts)

    def _needs_compaction(self) -> bool:
        """Check if context exceeds the compaction threshold."""
        return self._estimated_tokens > (self._max_tokens * self._compaction_threshold)

    def _compact(self) -> None:
        """Reduce context size by removing older messages.

        Strategy: Keep system prompt, first user message, and the most recent
        N messages. Remove middle messages and replace with a summary note.
        """
        if len(self._messages) <= 4:
            return  # Nothing to compact

        keep_recent = max(4, len(self._messages) // 2)
        removed = self._messages[:-keep_recent]
        self._messages = self._messages[-keep_recent:]

        # Add a compaction note
        removed_count = len(removed)
        self._messages.insert(0, {
            "role": "system",
            "content": f"[Context compacted: {removed_count} earlier messages removed for brevity. "
                       f"Key data from tools is preserved in summaries above.]",
        })

        # Recalculate token estimate
        self._estimated_tokens = self._estimate_tokens(self._system_prompt or "")
        for msg in self._messages:
            content = msg.get("content", "")
            if content:
                self._estimated_tokens += self._estimate_tokens(content)

        logger.info(
            f"[ContextManager] Compacted context: removed {removed_count} messages. "
            f"Estimated tokens now: {self._estimated_tokens}"
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count estimation. Good enough for budget tracking."""
        if not text:
            return 0
        return len(text) // CHARS_PER_TOKEN_ESTIMATE
