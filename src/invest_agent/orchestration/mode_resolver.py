"""
Mode resolver: decides between Instant, Thinking, and Auto routing.

Why: The user can explicitly choose a mode, or let the system decide (Auto).
This module encapsulates ALL mode-selection logic in one place so the
orchestrator only needs to call `resolve()` and gets back a ready-to-use
ModeConfig.

How: Three resolution paths:
1. Explicit: User chose "instant" or "thinking" -> return corresponding config
2. Auto heuristic: Quick rules based on query length, symbol count, keywords
3. Auto with escalation: If Instant mode produces poor results, escalate to Thinking
"""

import logging
import re
from typing import List, Optional

from pydantic import BaseModel, Field

from src.invest_agent.core.config import (
    AgentMode,
    ModeConfig,
    INSTANT_MODE_CONFIG,
    THINKING_MODE_CONFIG,
)
from src.invest_agent.orchestration.intent_wrapper import ClassificationResult

logger = logging.getLogger(__name__)


class ModeDecision(BaseModel):
    """The output of mode resolution."""
    mode: AgentMode
    config: ModeConfig
    reason: str = Field(description="Human-readable reason for mode selection")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    was_auto_resolved: bool = False


class EscalationDecision(BaseModel):
    """Whether to escalate from Instant to Thinking mode."""
    should_escalate: bool = False
    reason: str = ""


class ModeResolver:
    """Resolves the operating mode for each request.

    Why: Centralizes mode selection so the orchestrator doesn't contain
    branching logic for mode decisions. Also handles escalation checks
    as a separate concern.

    How the Auto resolution works:
    1. Check query length: very short (< 10 words) + 0-1 symbols -> Instant
    2. Check complexity signals: multiple symbols, comparison keywords -> Thinking
    3. Use classification result if available: agent_loop complexity -> Thinking
    4. Default to Instant for speed when uncertain
    """

    def resolve(
        self,
        response_mode: str,
        enable_thinking: bool,
        query: str,
        classification: Optional[ClassificationResult] = None,
    ) -> ModeDecision:
        """Main entry point: resolve mode from user preference and query analysis.

        Args:
            response_mode: "instant", "thinking", or "auto"
            enable_thinking: Legacy flag; if False, forces Instant
            query: The user's question
            classification: Optional pre-computed classification result
        """
        # Legacy override: enable_thinking=False forces instant
        if not enable_thinking:
            return ModeDecision(
                mode=AgentMode.INSTANT,
                config=INSTANT_MODE_CONFIG,
                reason="enable_thinking=False (legacy override)",
            )

        # Explicit user selection
        if response_mode == "instant":
            return ModeDecision(
                mode=AgentMode.INSTANT,
                config=INSTANT_MODE_CONFIG,
                reason="explicit_user_selection",
            )

        if response_mode == "thinking":
            return ModeDecision(
                mode=AgentMode.THINKING,
                config=THINKING_MODE_CONFIG,
                reason="explicit_user_selection",
            )

        # Auto mode: use heuristics + classification
        return self._auto_resolve(query, classification)

    def _auto_resolve(
        self,
        query: str,
        classification: Optional[ClassificationResult] = None,
    ) -> ModeDecision:
        """Auto-resolve mode using heuristics and classification data."""
        words = query.split()
        word_count = len(words)

        # Extract potential symbols
        symbols = re.findall(r'\b[A-Z]{1,5}\b', query)
        common_words = {"I", "A", "THE", "AND", "OR", "NOT", "FOR", "IS", "IT", "IN", "TO", "OF", "AT", "ON", "BY"}
        symbols = [s for s in symbols if s not in common_words]

        # Use classification symbols if available (more accurate)
        if classification and classification.symbols:
            symbols = classification.symbols

        symbol_count = len(symbols)

        # Rule 1: Very short query with 0-1 symbols -> Instant
        if word_count < 10 and symbol_count <= 1:
            return ModeDecision(
                mode=AgentMode.INSTANT,
                config=INSTANT_MODE_CONFIG,
                reason="short_query_simple",
                confidence=0.85,
                was_auto_resolved=True,
            )

        # Rule 2: Multiple symbols -> Thinking (comparison likely)
        if symbol_count >= 2:
            return ModeDecision(
                mode=AgentMode.THINKING,
                config=THINKING_MODE_CONFIG,
                reason="multi_symbol_detected",
                confidence=0.90,
                was_auto_resolved=True,
            )

        # Rule 3: Complex keywords -> Thinking
        complex_keywords = {
            "compare", "comparison", "so sánh",
            "analysis", "phân tích", "toàn diện",
            "fundamental", "cơ bản",
            "portfolio", "danh mục",
            "backtest", "kiểm thử",
            "valuation", "định giá",
            "strategy", "chiến lược",
        }
        query_lower = query.lower()
        if any(kw in query_lower for kw in complex_keywords):
            return ModeDecision(
                mode=AgentMode.THINKING,
                config=THINKING_MODE_CONFIG,
                reason="complex_keywords_detected",
                confidence=0.80,
                was_auto_resolved=True,
            )

        # Rule 4: Classification says agent_loop -> Thinking
        if classification and classification.complexity == "agent_loop":
            return ModeDecision(
                mode=AgentMode.THINKING,
                config=THINKING_MODE_CONFIG,
                reason="classification_complex",
                confidence=classification.confidence,
                was_auto_resolved=True,
            )

        # Rule 5: Long query (likely detailed question) -> Thinking
        if word_count >= 20:
            return ModeDecision(
                mode=AgentMode.THINKING,
                config=THINKING_MODE_CONFIG,
                reason="long_query",
                confidence=0.70,
                was_auto_resolved=True,
            )

        # Default: Instant for speed
        return ModeDecision(
            mode=AgentMode.INSTANT,
            config=INSTANT_MODE_CONFIG,
            reason="default_fast",
            confidence=0.60,
            was_auto_resolved=True,
        )

    def check_escalation(
        self,
        tool_results: list,
        current_mode: AgentMode,
    ) -> EscalationDecision:
        """Check if Instant mode should escalate to Thinking mode.

        Escalation triggers:
        1. > 50% tool calls failed
        2. No meaningful data returned from any tool
        3. All tools returned empty results
        """
        if current_mode != AgentMode.INSTANT:
            return EscalationDecision(should_escalate=False, reason="not_instant_mode")

        if not tool_results:
            return EscalationDecision(
                should_escalate=True,
                reason="no_tools_executed",
            )

        total = len(tool_results)
        failed = sum(1 for r in tool_results if not r.get("success", False))
        has_data = any(
            r.get("data") or r.get("formatted_context")
            for r in tool_results
            if r.get("success", False)
        )

        if failed > 0 and (failed / total) > 0.5:
            return EscalationDecision(
                should_escalate=True,
                reason=f"tool_error_rate_high ({failed}/{total} failed)",
            )

        if not has_data:
            return EscalationDecision(
                should_escalate=True,
                reason="no_meaningful_data_returned",
            )

        return EscalationDecision(should_escalate=False, reason="results_adequate")
