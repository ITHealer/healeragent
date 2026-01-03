"""
Streaming Event Service - Unified SSE Event Management

Provides standardized Server-Sent Events (SSE) for both Normal Mode and Deep Research Mode.
Implements heartbeat mechanism to keep connections alive.

Event Types:
- session: Session info at start
- context_loaded: Memory/context loaded
- classifying: Starting classification
- classified: Classification result
- tools_loading: Loading tools
- tools_loaded: Tools loaded with count
- turn_start: Agent turn started
- tool_calls: Tools being called (with names, args)
- tool_results: Tool execution results
- thinking: LLM is processing
- content: Response content chunk
- progress: Deep Research progress update
- error: Error occurred
- heartbeat: Keep-alive ping
- done: Processing complete

Usage:
    from src.services.streaming_event_service import (
        StreamEventEmitter,
        StreamEventType,
        create_sse_response,
    )

    emitter = StreamEventEmitter(session_id="abc123")

    async def generate():
        yield emitter.emit_session_start()
        yield emitter.emit_classified(classification)
        yield emitter.emit_tool_calls(tool_calls)
        yield emitter.emit_content("Response text...")
        yield emitter.emit_done()

    return StreamingResponse(generate(), media_type="text/event-stream")
"""

import json
import asyncio
from enum import Enum
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict

from src.utils.logger.custom_logging import LoggerMixin


# ============================================================================
# EVENT TYPES
# ============================================================================

class StreamEventType(str, Enum):
    """SSE event types for streaming responses"""

    # Session lifecycle
    SESSION = "session"
    DONE = "done"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

    # Context & Classification (Normal Mode)
    CONTEXT_LOADED = "context_loaded"
    CLASSIFYING = "classifying"
    CLASSIFIED = "classified"

    # Tool loading
    TOOLS_LOADING = "tools_loading"
    TOOLS_LOADED = "tools_loaded"

    # Agent loop
    TURN_START = "turn_start"
    TOOL_CALLS = "tool_calls"
    TOOL_RESULTS = "tool_results"
    THINKING = "thinking"
    CONTENT = "content"

    # Deep Research Mode
    PROGRESS = "progress"
    PLAN_UPDATE = "plan_update"
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"
    VERIFICATION = "verification"
    REPORT = "report"

    # Thinking/Reasoning Events (AI Agent)
    THINKING_START = "thinking_start"
    THINKING_DELTA = "thinking_delta"
    THINKING_END = "thinking_end"

    # LLM Decision Events (AI Agent Tree)
    LLM_THOUGHT = "llm_thought"
    LLM_DECISION = "llm_decision"
    LLM_ACTION = "llm_action"


# ============================================================================
# EVENT DATA MODELS
# ============================================================================

@dataclass
class StreamEvent:
    """Standard SSE event structure"""
    type: StreamEventType
    data: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": self.type.value,
            "timestamp": self.timestamp,
        }
        if self.data:
            result["data"] = self.data
        return result

    def to_sse(self) -> str:
        """Format as SSE data line"""
        return json.dumps(self.to_dict(), ensure_ascii=False) + "\n\n"


@dataclass
class ToolCallEvent:
    """Tool call event data"""
    id: str
    name: str
    arguments: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class ToolResultEvent:
    """Tool result event data"""
    id: str
    name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "name": self.name,
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
        }
        if self.success:
            # Truncate large results for streaming
            result_str = json.dumps(self.result) if not isinstance(self.result, str) else self.result
            if len(result_str) > 500:
                data["result_preview"] = result_str[:500] + "..."
                data["result_full_size"] = len(result_str)
            else:
                data["result"] = self.result
        else:
            data["error"] = self.error
        return data


# ============================================================================
# STREAM EVENT EMITTER
# ============================================================================

class StreamEventEmitter(LoggerMixin):
    """
    Emits standardized SSE events for streaming responses.

    Features:
    - Type-safe event emission
    - Automatic timestamp
    - Heartbeat support
    - JSON serialization
    - Logging for debugging
    """

    # Heartbeat interval in seconds
    HEARTBEAT_INTERVAL = 15.0

    def __init__(
        self,
        session_id: str,
        enable_logging: bool = True,
        log_content_preview: int = 100,  # chars to log from content
    ):
        super().__init__()
        self.session_id = session_id
        self.enable_logging = enable_logging
        self.log_content_preview = log_content_preview
        self._event_count = 0
        self._start_time = datetime.now()

    def _log_event(self, event_type: StreamEventType, details: str = "") -> None:
        """Log event for debugging"""
        if self.enable_logging:
            self._event_count += 1
            elapsed = (datetime.now() - self._start_time).total_seconds()
            self.logger.debug(
                f"[SSE:{self.session_id[:8]}] #{self._event_count} "
                f"{event_type.value} @ {elapsed:.1f}s {details}"
            )

    def _emit(
        self,
        event_type: StreamEventType,
        data: Optional[Dict[str, Any]] = None,
        log_details: str = "",
    ) -> str:
        """Emit a generic event"""
        event = StreamEvent(type=event_type, data=data)
        # self._log_event(event_type, log_details)
        return event.to_sse()

    # ========================================================================
    # SESSION LIFECYCLE EVENTS
    # ========================================================================

    def emit_session_start(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Emit session start event"""
        data = {"session_id": self.session_id}
        if metadata:
            data.update(metadata)
        return self._emit(StreamEventType.SESSION, data, f"session={self.session_id}")

    def emit_session(
        self,
        session_id: str,
        mode: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Emit session info event with mode and model"""
        data = {
            "session_id": session_id,
            "mode": mode,
            "model": model,
        }
        if metadata:
            data.update(metadata)
        return self._emit(StreamEventType.SESSION, data, f"session={session_id} mode={mode}")
    
    def emit_done(
        self,
        total_turns: int = 0,
        total_tool_calls: int = 0,
        total_time_ms: float = 0.0,
        charts: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Emit completion event"""
        elapsed = (datetime.now() - self._start_time).total_seconds() * 1000
        data = {
            "total_events": self._event_count,
            "total_turns": total_turns,
            "total_tool_calls": total_tool_calls,
            "processing_time_ms": total_time_ms or elapsed,
        }
        # Include charts if available
        if charts:
            data["charts"] = charts

        return self._emit(
            StreamEventType.DONE, data,
            f"turns={total_turns} tools={total_tool_calls} time={elapsed:.0f}ms charts={len(charts) if charts else 0}"
        )

    def emit_error(self, error_message: str, error_code: str = "UNKNOWN") -> str:
        """Emit error event"""
        data = {
            "code": error_code,
            "message": error_message,
        }
        return self._emit(StreamEventType.ERROR, data, f"code={error_code}")

    def emit_heartbeat(self) -> str:
        """Emit heartbeat to keep connection alive"""
        elapsed = (datetime.now() - self._start_time).total_seconds()
        data = {"elapsed_seconds": elapsed}
        return self._emit(StreamEventType.HEARTBEAT, data)

    # ========================================================================
    # CONTEXT & CLASSIFICATION EVENTS
    # ========================================================================

    def emit_context_loaded(
        self,
        token_count: int,
        sources: List[str],
        was_compacted: bool = False,
    ) -> str:
        """Emit context loaded event"""
        data = {
            "token_count": token_count,
            "sources": sources,
            "was_compacted": was_compacted,
        }
        return self._emit(
            StreamEventType.CONTEXT_LOADED, data,
            f"tokens={token_count} sources={sources}"
        )

    def emit_classifying(self) -> str:
        """Emit classification started event"""
        return self._emit(StreamEventType.CLASSIFYING, {"status": "in_progress"})

    def emit_classified(
        self,
        query_type: str,
        requires_tools: bool,
        symbols: List[str],
        categories: List[str],
        confidence: float,
        language: str,
        reasoning: Optional[str] = None,
        intent_summary: Optional[str] = None,
    ) -> str:
        """Emit classification result event"""
        data = {
            "query_type": query_type,
            "requires_tools": requires_tools,
            "symbols": symbols,
            "tool_categories": categories,
            "confidence": confidence,
            "response_language": language,
        }
        # Include AI reasoning if available
        if reasoning:
            data["reasoning"] = reasoning
        if intent_summary:
            data["intent_summary"] = intent_summary

        return self._emit(
            StreamEventType.CLASSIFIED, data,
            f"type={query_type} tools={requires_tools} symbols={symbols}"
        )

    # ========================================================================
    # TOOL LOADING EVENTS
    # ========================================================================

    def emit_tools_loading(self, categories: List[str]) -> str:
        """Emit tools loading event"""
        data = {"categories": categories}
        return self._emit(StreamEventType.TOOLS_LOADING, data)

    def emit_tools_loaded(
        self,
        count: int,
        method: str,
        categories: Optional[List[str]] = None,
    ) -> str:
        """Emit tools loaded event"""
        data = {
            "tool_count": count,
            "loading_method": method,
        }
        if categories:
            data["categories"] = categories
        return self._emit(
            StreamEventType.TOOLS_LOADED, data,
            f"count={count} method={method}"
        )

    # ========================================================================
    # AGENT LOOP EVENTS
    # ========================================================================

    def emit_turn_start(self, turn_number: int, max_turns: int) -> str:
        """Emit agent turn start event"""
        data = {
            "turn": turn_number,
            "max_turns": max_turns,
        }
        return self._emit(StreamEventType.TURN_START, data, f"turn={turn_number}")

    def emit_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> str:
        """Emit tool calls event"""
        # Format tool calls for streaming
        formatted = []
        names = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                formatted.append({
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", tc.get("name", "")),
                    "arguments": tc.get("function", {}).get("arguments", tc.get("arguments", {})),
                })
                names.append(tc.get("function", {}).get("name", tc.get("name", "")))
            else:
                # Object with attributes
                formatted.append({
                    "id": getattr(tc, "id", ""),
                    "name": getattr(tc.function, "name", "") if hasattr(tc, "function") else getattr(tc, "name", ""),
                    "arguments": getattr(tc.function, "arguments", "") if hasattr(tc, "function") else getattr(tc, "arguments", ""),
                })
                names.append(getattr(tc.function, "name", "") if hasattr(tc, "function") else getattr(tc, "name", ""))

        data = {
            "count": len(formatted),
            "tools": formatted,
        }
        return self._emit(
            StreamEventType.TOOL_CALLS, data,
            f"count={len(formatted)} names={names}"
        )

    def emit_tool_results(
        self,
        results: List[Union[ToolResultEvent, Dict[str, Any]]],
    ) -> str:
        """Emit tool results event"""
        formatted = []
        for r in results:
            if isinstance(r, ToolResultEvent):
                formatted.append(r.to_dict())
            elif isinstance(r, dict):
                formatted.append(r)

        success_count = sum(1 for r in formatted if r.get("success", True))
        data = {
            "count": len(formatted),
            "success_count": success_count,
            "results": formatted,
        }
        return self._emit(
            StreamEventType.TOOL_RESULTS, data,
            f"count={len(formatted)} success={success_count}"
        )

    def emit_thinking(self, message: str = "Processing...") -> str:
        """Emit thinking/processing event"""
        data = {"message": message}
        return self._emit(StreamEventType.THINKING, data)

    def emit_content(self, content: str, is_final: bool = False) -> str:
        """Emit content chunk event"""
        data = {
            "content": content,
            "is_final": is_final,
        }
        preview = content[:self.log_content_preview] if len(content) > self.log_content_preview else content
        return self._emit(
            StreamEventType.CONTENT, data,
            f"len={len(content)} preview=\"{preview}...\""
        )

    # ========================================================================
    # DEEP RESEARCH MODE EVENTS
    # ========================================================================

    def emit_progress(
        self,
        phase: str,
        progress_percent: float,
        message: str,
        step_id: Optional[str] = None,
        step_description: Optional[str] = None,
    ) -> str:
        """Emit progress update event (Deep Research)"""
        data = {
            "phase": phase,
            "progress": progress_percent,
            "message": message,
        }
        if step_id:
            data["step_id"] = step_id
        if step_description:
            data["step_description"] = step_description

        return self._emit(
            StreamEventType.PROGRESS, data,
            f"phase={phase} progress={progress_percent:.0f}%"
        )

    def emit_plan_update(self, added_steps: int, reason: str) -> str:
        """Emit plan update event (Deep Research)"""
        data = {
            "added_steps": added_steps,
            "reason": reason,
        }
        return self._emit(StreamEventType.PLAN_UPDATE, data)

    def emit_step_start(self, step_id: str, description: str) -> str:
        """Emit step start event (Deep Research)"""
        data = {
            "step_id": step_id,
            "description": description,
            "status": "in_progress",
        }
        return self._emit(StreamEventType.STEP_START, data, f"step={step_id}")

    def emit_step_complete(
        self,
        step_id: str,
        success: bool,
        execution_time_ms: float,
    ) -> str:
        """Emit step complete event (Deep Research)"""
        data = {
            "step_id": step_id,
            "success": success,
            "execution_time_ms": execution_time_ms,
            "status": "completed" if success else "failed",
        }
        return self._emit(
            StreamEventType.STEP_COMPLETE, data,
            f"step={step_id} success={success}"
        )

    def emit_report(self, content: str, path: Optional[str] = None) -> str:
        """Emit final report event (Deep Research)"""
        data = {
            "content": content,
            "content_length": len(content),
        }
        if path:
            data["file_path"] = path

        return self._emit(StreamEventType.REPORT, data, f"len={len(content)}")

    # ========================================================================
    # THINKING/REASONING EVENTS (AI Agent)
    # ========================================================================

    def emit_thinking_start(self, phase: str = "reasoning") -> str:
        """Emit thinking started event"""
        data = {
            "phase": phase,
            "status": "started",
        }
        return self._emit(StreamEventType.THINKING_START, data, f"phase={phase}")

    def emit_thinking_delta(
        self,
        content: str,
        phase: str = "reasoning",
        step: Optional[str] = None,
    ) -> str:
        """Emit thinking content chunk event"""
        data = {
            "content": content,
            "phase": phase,
        }
        if step:
            data["step"] = step

        preview = content[:50] if len(content) > 50 else content
        return self._emit(
            StreamEventType.THINKING_DELTA, data,
            f"phase={phase} content=\"{preview}...\""
        )

    def emit_thinking_end(
        self,
        phase: str = "reasoning",
        summary: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> str:
        """Emit thinking completed event"""
        data = {
            "phase": phase,
            "status": "completed",
            "duration_ms": duration_ms,
        }
        if summary:
            data["summary"] = summary

        return self._emit(
            StreamEventType.THINKING_END, data,
            f"phase={phase} duration={duration_ms:.0f}ms"
        )

    # ========================================================================
    # LLM DECISION EVENTS (AI Agent Tree)
    # ========================================================================

    def emit_llm_thought(
        self,
        thought: str,
        context: Optional[str] = None,
    ) -> str:
        """Emit LLM thought/reasoning event"""
        data = {
            "thought": thought,
        }
        if context:
            data["context"] = context

        preview = thought[:80] if len(thought) > 80 else thought
        return self._emit(StreamEventType.LLM_THOUGHT, data, f"\"{preview}...\"")

    def emit_llm_decision(
        self,
        decision: str,
        action: str,
        confidence: float = 0.0,
        alternatives: Optional[List[str]] = None,
    ) -> str:
        """Emit LLM decision event"""
        data = {
            "decision": decision,
            "action": action,
            "confidence": confidence,
        }
        if alternatives:
            data["alternatives"] = alternatives

        return self._emit(
            StreamEventType.LLM_DECISION, data,
            f"action={action} confidence={confidence:.2f}"
        )

    def emit_llm_action(
        self,
        action_type: str,
        action_name: str,
        reason: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Emit LLM action event"""
        data = {
            "action_type": action_type,
            "action_name": action_name,
            "reason": reason,
        }
        if params:
            data["params"] = params

        return self._emit(
            StreamEventType.LLM_ACTION, data,
            f"type={action_type} name={action_name}"
        )


# ============================================================================
# HEARTBEAT GENERATOR
# ============================================================================

async def heartbeat_generator(
    emitter: StreamEventEmitter,
    interval: float = StreamEventEmitter.HEARTBEAT_INTERVAL,
) -> AsyncGenerator[str, None]:
    """
    Generate heartbeat events at regular intervals.

    Use with asyncio.create_task() alongside main event stream.
    """
    while True:
        await asyncio.sleep(interval)
        yield emitter.emit_heartbeat()


async def with_heartbeat(
    event_generator: AsyncGenerator[str, None],
    emitter: StreamEventEmitter,
    heartbeat_interval: float = StreamEventEmitter.HEARTBEAT_INTERVAL,
) -> AsyncGenerator[str, None]:
    """
    Wrap an event generator with heartbeat support.

    Emits heartbeat if no event within interval.
    """
    last_event_time = datetime.now()

    async def heartbeat_task():
        nonlocal last_event_time
        while True:
            await asyncio.sleep(heartbeat_interval)
            elapsed = (datetime.now() - last_event_time).total_seconds()
            if elapsed >= heartbeat_interval:
                yield emitter.emit_heartbeat()
                last_event_time = datetime.now()

    # Use merge pattern with timeout
    try:
        async for event in event_generator:
            last_event_time = datetime.now()
            yield event
    except asyncio.CancelledError:
        pass


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_sse_event(event: Dict[str, Any]) -> str:
    """Format a dictionary as SSE data line"""
    return json.dumps(event, ensure_ascii=False) + "\n\n"


def format_done_marker() -> str:
    """Format the [DONE] marker for SSE stream end"""
    return "[DONE]\n\n"


def create_error_event(error_message: str, error_code: str = "UNKNOWN") -> str:
    """Create an error event string"""
    event = {
        "type": "error",
        "data": {
            "code": error_code,
            "message": error_message,
        },
        "timestamp": datetime.now().isoformat(),
    }
    return format_sse_event(event)