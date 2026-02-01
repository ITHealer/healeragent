"""
SSE (Server-Sent Events) definitions for the invest_agent streaming protocol.

Why: A typed event system prevents typo-driven bugs in event names and ensures
the frontend contract is explicit. Each event type maps to a specific phase
in the orchestrator's state machine.
"""

import json
import time
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class SSEEventType(str, Enum):
    """All SSE event types emitted by the invest_agent pipeline."""

    # Session lifecycle
    SESSION_START = "session_start"
    DONE = "done"
    ERROR = "error"

    # Mode resolution
    MODE_SELECTING = "mode_selecting"
    MODE_SELECTED = "mode_selected"
    MODE_ESCALATED = "mode_escalated"

    # Classification
    CLASSIFYING = "classifying"
    CLASSIFIED = "classified"

    # Agent loop
    TURN_START = "turn_start"
    TOOL_CALLS = "tool_calls"
    TOOL_RESULTS = "tool_results"

    # Thinking mode specifics
    THINKING_STEP = "thinking_step"
    THINKING_SUMMARY = "thinking_summary"
    EVALUATION = "evaluation"

    # Response streaming
    CONTENT = "content"

    # Artifacts
    ARTIFACT_SAVED = "artifact_saved"


class SSEEvent(BaseModel):
    """A single SSE event ready for serialization.

    How: The router serializes this into the `data: {...}\n\n` SSE wire format.
    The `event` field maps to the SSE event type, `data` carries the payload.
    """
    event: SSEEventType
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

    def to_sse(self) -> str:
        """Serialize to SSE wire format: `event: <type>\ndata: <json>\n\n`."""
        payload = {"type": self.event.value, **self.data, "timestamp": self.timestamp}
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---- Factory helpers for common events ----

def event_session_start(session_id: str, version: str = "v3") -> SSEEvent:
    return SSEEvent(
        event=SSEEventType.SESSION_START,
        data={"session_id": session_id, "version": version},
    )

def event_mode_selecting(method: str) -> SSEEvent:
    return SSEEvent(
        event=SSEEventType.MODE_SELECTING,
        data={"method": method},
    )

def event_mode_selected(mode: str, reason: str, model: str, confidence: float = 1.0) -> SSEEvent:
    return SSEEvent(
        event=SSEEventType.MODE_SELECTED,
        data={"mode": mode, "reason": reason, "model": model, "confidence": confidence},
    )

def event_mode_escalated(from_mode: str, to_mode: str, reason: str) -> SSEEvent:
    return SSEEvent(
        event=SSEEventType.MODE_ESCALATED,
        data={"from": from_mode, "to": to_mode, "reason": reason},
    )

def event_classified(
    query_type: str,
    symbols: list,
    complexity: str,
    requires_tools: bool,
) -> SSEEvent:
    return SSEEvent(
        event=SSEEventType.CLASSIFIED,
        data={
            "query_type": query_type,
            "symbols": symbols,
            "complexity": complexity,
            "requires_tools": requires_tools,
        },
    )

def event_turn_start(turn: int, max_turns: int) -> SSEEvent:
    return SSEEvent(
        event=SSEEventType.TURN_START,
        data={"turn": turn, "max_turns": max_turns},
    )

def event_tool_calls(tools: list) -> SSEEvent:
    return SSEEvent(
        event=SSEEventType.TOOL_CALLS,
        data={"tools": tools},
    )

def event_tool_results(results: list) -> SSEEvent:
    return SSEEvent(
        event=SSEEventType.TOOL_RESULTS,
        data={"results": results},
    )

def event_thinking_step(phase: str, action: str, details: Optional[str] = None) -> SSEEvent:
    return SSEEvent(
        event=SSEEventType.THINKING_STEP,
        data={"phase": phase, "action": action, **({"details": details} if details else {})},
    )

def event_evaluation(iteration: int, sufficient: bool, missing: Optional[list] = None) -> SSEEvent:
    return SSEEvent(
        event=SSEEventType.EVALUATION,
        data={"iteration": iteration, "sufficient": sufficient, "missing": missing or []},
    )

def event_content(content: str) -> SSEEvent:
    return SSEEvent(
        event=SSEEventType.CONTENT,
        data={"content": content},
    )

def event_thinking_summary(total_duration_ms: int, steps: list) -> SSEEvent:
    return SSEEvent(
        event=SSEEventType.THINKING_SUMMARY,
        data={"total_duration_ms": total_duration_ms, "steps": steps},
    )

def event_done(
    total_turns: int,
    total_tool_calls: int,
    total_time_ms: int,
    mode: str,
) -> SSEEvent:
    return SSEEvent(
        event=SSEEventType.DONE,
        data={
            "total_turns": total_turns,
            "total_tool_calls": total_tool_calls,
            "total_time_ms": total_time_ms,
            "mode": mode,
        },
    )

def event_error(message: str, code: str = "internal_error") -> SSEEvent:
    return SSEEvent(
        event=SSEEventType.ERROR,
        data={"message": message, "code": code},
    )

def event_artifact_saved(artifact_id: str, tool_name: str, summary: str) -> SSEEvent:
    return SSEEvent(
        event=SSEEventType.ARTIFACT_SAVED,
        data={"artifact_id": artifact_id, "tool_name": tool_name, "summary": summary},
    )
