"""
Lightweight Agent Scratchpad - Debug Trace for Production Issues

Per-query JSONL debug file that records:
- Tool calls with arguments, results, summaries, and execution time
- Agent thinking/reasoning steps
- Validation warnings (anti-loop)
- Final context building decisions

Controlled by ENABLE_SCRATCHPAD env var. Off by default in production.
Zero overhead when disabled (all methods are no-ops).

Usage:
    scratchpad = AgentScratchpad(
        query="Analyze AAPL",
        flow_id="UA-ALL-abc12345",
        enabled=os.getenv("ENABLE_SCRATCHPAD", "").lower() == "true",
    )
    scratchpad.log_tool_result("getStockPrice", {"symbol": "AAPL"}, result, summary, 150)
    scratchpad.log_thinking("Planning: Need technical + fundamental data")
    scratchpad.log_validation_warning("getStockPrice", "Called 4 times - soft limit")
    scratchpad.finalize(total_turns=3, total_tool_calls=8)
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class AgentScratchpad:
    """
    Lightweight per-query debug trace. Controlled by env flag.

    When disabled (default), all methods are no-ops with zero overhead.
    When enabled, writes JSONL entries to .logs/scratchpad/<flow_id>.jsonl.
    """

    def __init__(
        self,
        query: str,
        flow_id: str,
        enabled: Optional[bool] = None,
    ):
        # Auto-detect from env if not specified
        if enabled is None:
            enabled = os.getenv("ENABLE_SCRATCHPAD", "").lower() in ("true", "1", "yes")

        self.enabled = enabled
        self.flow_id = flow_id
        self._start_time = time.time()

        if not self.enabled:
            return

        self.logger = logging.getLogger("agent.scratchpad")

        try:
            self.filepath = Path(".logs/scratchpad") / f"{flow_id}.jsonl"
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            self._write({
                "type": "init",
                "query": query,
                "flow_id": flow_id,
                "ts": datetime.now().isoformat(),
            })
        except Exception as e:
            self.logger.warning(f"Scratchpad init failed: {e}. Disabling.")
            self.enabled = False

    def _write(self, entry: Dict[str, Any]) -> None:
        """Write a single JSONL entry."""
        if not self.enabled:
            return
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass  # Never block agent execution

    def log_tool_result(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Dict[str, Any],
        summary: str,
        exec_ms: int,
    ) -> None:
        """Log a tool execution result."""
        if not self.enabled:
            return
        self._write({
            "type": "tool_result",
            "tool": tool_name,
            "args": args,
            "status": result.get("status", "unknown"),
            "summary": summary,
            "exec_ms": exec_ms,
            "result_chars": len(json.dumps(result, default=str)),
            "ts": datetime.now().isoformat(),
        })

    def log_thinking(self, content: str) -> None:
        """Log agent thinking/reasoning."""
        if not self.enabled:
            return
        self._write({
            "type": "thinking",
            "content": content[:1000],
            "ts": datetime.now().isoformat(),
        })

    def log_validation_warning(self, tool_name: str, warning: str) -> None:
        """Log anti-loop validation warning."""
        if not self.enabled:
            return
        self._write({
            "type": "validation_warning",
            "tool": tool_name,
            "warning": warning,
            "ts": datetime.now().isoformat(),
        })

    def log_context_decision(
        self,
        total_tokens: int,
        budget_tokens: int,
        strategy: str,
        selected_indices: Optional[list] = None,
    ) -> None:
        """Log final context building decision."""
        if not self.enabled:
            return
        self._write({
            "type": "context_decision",
            "total_tokens": total_tokens,
            "budget_tokens": budget_tokens,
            "strategy": strategy,
            "selected_indices": selected_indices,
            "ts": datetime.now().isoformat(),
        })

    def finalize(self, total_turns: int, total_tool_calls: int) -> None:
        """Write final summary entry."""
        if not self.enabled:
            return
        elapsed_ms = int((time.time() - self._start_time) * 1000)
        self._write({
            "type": "finalize",
            "total_turns": total_turns,
            "total_tool_calls": total_tool_calls,
            "total_elapsed_ms": elapsed_ms,
            "ts": datetime.now().isoformat(),
        })
