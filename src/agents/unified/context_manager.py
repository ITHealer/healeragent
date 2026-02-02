"""
Two-Tier Context Manager for Agent Loop

Implements a two-tier context strategy:
- ITERATION LOOP: Uses lightweight summaries of tool results (~100 tokens each)
  to keep context window small and focused for tool selection decisions.
- FINAL SYNTHESIS: Uses FULL tool result data to generate comprehensive,
  data-rich final responses without information loss.

This prevents the "lost in the middle" problem where LLMs forget early tool
results when context grows too large across multiple agent turns.

Architecture:
    Turn 1: tools execute → full results stored in ToolDataStore
                          → summaries appended to messages
    Turn 2: LLM sees summaries → decides next tools
    ...
    Final:  full results loaded from ToolDataStore → rich final answer
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.config import settings
from src.helpers.llm_helper import LLMGeneratorProvider


# ============================================================================
# CONSTANTS
# ============================================================================

# Token budget for final context (leave room for system prompt + response)
DEFAULT_FINAL_CONTEXT_MAX_TOKENS = 120_000

# Approximate chars per token (conservative for mixed content)
CHARS_PER_TOKEN = 3.5

# Max tokens for a single tool summary
MAX_SUMMARY_TOKENS = 150

# Model for fast summary generation
SUMMARY_MODEL = settings.SUMMARY_MODEL or "gpt-4.1-nano"
SUMMARY_PROVIDER = settings.SUMMARY_PROVIDER or "openai"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ToolResultEntry:
    """Stores both full result and summary for a single tool call."""
    tool_name: str
    tool_call_id: str
    arguments: Dict[str, Any]
    full_result: Dict[str, Any]  # Complete tool output
    summary: str                  # Condensed summary for iteration
    execution_ms: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def full_result_text(self) -> str:
        """Get text representation of full result for final context."""
        # Prefer formatted_context (LLM-ready), fall back to JSON
        fc = self.full_result.get("formatted_context", "")
        if fc and isinstance(fc, str) and len(fc) > 50:
            return fc
        # Fallback: serialize data
        data = self.full_result.get("data", self.full_result)
        if isinstance(data, dict):
            return json.dumps(data, ensure_ascii=False, default=str)
        return str(data)

    @property
    def estimated_tokens(self) -> int:
        """Estimate token count for full result."""
        return max(1, int(len(self.full_result_text) / CHARS_PER_TOKEN))


class ToolDataStore:
    """
    In-memory store for tool results during a single agent run.

    Keeps both full results (for final synthesis) and summaries
    (for iteration loop context). No file I/O needed.
    """

    def __init__(self):
        self.entries: List[ToolResultEntry] = []
        self.logger = logging.getLogger("context_manager.store")

    def add(self, entry: ToolResultEntry) -> None:
        """Add a tool result entry."""
        self.entries.append(entry)

    @property
    def total_entries(self) -> int:
        return len(self.entries)

    @property
    def total_estimated_tokens(self) -> int:
        """Estimate total tokens for all full results."""
        return sum(e.estimated_tokens for e in self.entries)

    def get_all_summaries(self) -> List[str]:
        """Get all summaries for iteration context."""
        return [e.summary for e in self.entries]

    def get_full_results_text(self) -> str:
        """Get all full results as formatted text."""
        parts = []
        for i, entry in enumerate(self.entries, 1):
            args_str = ", ".join(
                f"{k}={v}" for k, v in entry.arguments.items()
            ) if entry.arguments else ""
            parts.append(
                f"### [{i}] {entry.tool_name}({args_str})\n"
                f"{entry.full_result_text}"
            )
        return "\n\n".join(parts)


# ============================================================================
# SUMMARY GENERATION
# ============================================================================

async def generate_tool_summary(
    query: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    result: Dict[str, Any],
    llm_provider: LLMGeneratorProvider,
) -> str:
    """
    Generate a concise summary of a tool result using a fast/cheap model.

    Falls back to extractive summary if LLM call fails (no blocking).

    Args:
        query: Original user query (for relevance context)
        tool_name: Name of the tool that was called
        tool_args: Arguments passed to the tool
        result: Full tool result dict
        llm_provider: LLM provider for generation

    Returns:
        Summary string (~1-2 sentences with key data points)
    """
    # Get the text to summarize
    formatted_context = result.get("formatted_context", "")
    status = result.get("status", "unknown")

    # For errors/timeouts, return simple status
    if status in ("error", "timeout"):
        error_msg = result.get("error", "Unknown error")
        return f"{tool_name} → FAILED: {error_msg}"

    # For think tool, return thought directly
    if tool_name == "think":
        thought = result.get("data", {}).get("thought", "")
        return f"think → {thought[:200]}"

    # Build the content to summarize
    content_to_summarize = formatted_context
    if not content_to_summarize or len(content_to_summarize) < 50:
        data = result.get("data", {})
        if isinstance(data, dict):
            content_to_summarize = json.dumps(data, ensure_ascii=False, default=str)
        else:
            content_to_summarize = str(data)

    # If content is short enough, return directly with prefix
    if len(content_to_summarize) < 300:
        return f"{tool_name} → {content_to_summarize[:280]}"

    # Build summary prompt
    args_str = ", ".join(f"{k}={v}" for k, v in tool_args.items()) if tool_args else ""

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize this tool result in 1-2 sentences. "
                    "Include specific values (numbers, dates, prices) that are relevant. "
                    "Format: '[tool_call] → [key findings]'"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n"
                    f"Tool: {tool_name}({args_str})\n"
                    f"Result:\n{content_to_summarize[:3000]}"  # Cap input
                ),
            },
        ]

        response = await llm_provider.generate_response(
            model_name=SUMMARY_MODEL,
            messages=messages,
            provider_type=SUMMARY_PROVIDER,
            max_tokens=MAX_SUMMARY_TOKENS,
            temperature=0.0,
        )

        summary = response.get("content", "").strip()
        if summary:
            return summary

    except Exception as e:
        logging.getLogger("context_manager").warning(
            f"Summary generation failed for {tool_name}: {e}. Using extractive fallback."
        )

    # Extractive fallback: first 250 chars of formatted context
    return f"{tool_name}({args_str}) → {content_to_summarize[:250]}..."


# ============================================================================
# FINAL CONTEXT BUILDING
# ============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def build_final_context(
    tool_data_store: ToolDataStore,
    max_tokens: int = DEFAULT_FINAL_CONTEXT_MAX_TOKENS,
) -> str:
    """
    Build full context from tool data store for final synthesis.

    If all results fit within budget, returns everything.
    If over budget, includes full results for important tools and
    summaries for the rest (prioritizing by relevance).

    Args:
        tool_data_store: Store containing all tool results
        max_tokens: Maximum token budget for context

    Returns:
        Formatted context string with full tool data
    """
    entries = tool_data_store.entries
    if not entries:
        return "No tool data available."

    total_tokens = tool_data_store.total_estimated_tokens

    # Case 1: Everything fits - return all full results
    if total_tokens <= max_tokens:
        return tool_data_store.get_full_results_text()

    # Case 2: Over budget - prioritize and truncate
    # Strategy: Include all results but truncate the longest ones
    logger = logging.getLogger("context_manager")
    logger.info(
        f"Context over budget: {total_tokens:,} tokens > {max_tokens:,}. "
        f"Applying smart truncation."
    )

    # Sort entries by token cost (descending) for truncation
    sorted_entries = sorted(entries, key=lambda e: e.estimated_tokens, reverse=True)

    # Calculate how much we need to trim
    tokens_to_trim = total_tokens - max_tokens
    trimmed = 0

    # Truncation map: index -> max_chars
    truncation_map: Dict[int, int] = {}

    for entry in sorted_entries:
        if trimmed >= tokens_to_trim:
            break

        entry_tokens = entry.estimated_tokens
        # Don't truncate small results (< 500 tokens)
        if entry_tokens < 500:
            continue

        # Truncate to 40% of original size (keep the most important data)
        target_tokens = max(300, int(entry_tokens * 0.4))
        saved = entry_tokens - target_tokens
        truncation_map[id(entry)] = int(target_tokens * CHARS_PER_TOKEN)
        trimmed += saved

    # Build context with truncations applied
    parts = []
    for i, entry in enumerate(entries, 1):
        args_str = ", ".join(
            f"{k}={v}" for k, v in entry.arguments.items()
        ) if entry.arguments else ""

        text = entry.full_result_text
        max_chars = truncation_map.get(id(entry))
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated for token budget]"

        parts.append(
            f"### [{i}] {entry.tool_name}({args_str})\n{text}"
        )

    return "\n\n".join(parts)


async def select_relevant_results(
    query: str,
    tool_data_store: ToolDataStore,
    max_tokens: int,
    llm_provider: LLMGeneratorProvider,
) -> str:
    """
    When over budget, use LLM to select which results need full data.

    This is an advanced path for very large contexts (>120K tokens of tool data).
    The LLM reviews summaries and picks the most relevant results to include in full.

    Args:
        query: Original user query
        tool_data_store: Store with all results
        max_tokens: Token budget
        llm_provider: LLM provider

    Returns:
        Optimized context string
    """
    entries = tool_data_store.entries
    if not entries:
        return "No tool data available."

    # Build summary list for LLM
    summary_items = []
    for i, entry in enumerate(entries):
        summary_items.append({
            "index": i,
            "tool_name": entry.tool_name,
            "summary": entry.summary,
            "token_cost": entry.estimated_tokens,
        })

    summary_list = "\n".join(
        f"[{s['index']}] {s['tool_name']} (~{s['token_cost'] // 1000}k tokens): {s['summary']}"
        for s in summary_items
    )

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are selecting which tool results need full data for answering a query. "
                    "Return ONLY a JSON array of indices. Example: [0, 2, 5]"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Available tool results:\n{summary_list}\n\n"
                    f"Token budget: {max_tokens:,} tokens\n"
                    f"Select the results most essential for a comprehensive answer."
                ),
            },
        ]

        response = await llm_provider.generate_response(
            model_name=SUMMARY_MODEL,
            messages=messages,
            provider_type=SUMMARY_PROVIDER,
            max_tokens=200,
            temperature=0.0,
        )

        content = response.get("content", "").strip()
        # Parse JSON array
        selected_indices = json.loads(content)
        if not isinstance(selected_indices, list):
            selected_indices = list(range(len(entries)))

    except Exception as e:
        logging.getLogger("context_manager").warning(
            f"LLM selection failed: {e}. Including all with truncation."
        )
        return build_final_context(tool_data_store, max_tokens)

    # Build context: full data for selected, summaries for rest
    parts = []
    for i, entry in enumerate(entries):
        args_str = ", ".join(
            f"{k}={v}" for k, v in entry.arguments.items()
        ) if entry.arguments else ""

        if i in selected_indices:
            parts.append(
                f"### [{i + 1}] {entry.tool_name}({args_str}) [FULL DATA]\n"
                f"{entry.full_result_text}"
            )
        else:
            parts.append(
                f"### [{i + 1}] {entry.tool_name}({args_str}) [SUMMARY]\n"
                f"{entry.summary}"
            )

    return "\n\n".join(parts)
