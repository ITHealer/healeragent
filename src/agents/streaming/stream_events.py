"""
Event Flow:
    START → THINKING_START → THINKING_DELTA* → PLANNING_COMPLETE
      ↓
    TOOL_START → TOOL_COMPLETE* → THINKING_END
      ↓
    TEXT_DELTA* → TEXT_COMPLETE → DONE
"""

import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid


class StreamEventType(str, Enum):
    """
    SSE Event Types
    """
    # Session events
    START = "start"

    # Thinking events (like Claude Extended Thinking)
    THINKING_START = "thinking_start"
    THINKING_DELTA = "thinking_delta"
    THINKING_END = "thinking_end"

    # Thinking Timeline Events (ChatGPT-style "Thought for Xs" display)
    THINKING_TIMELINE = "thinking_timeline"  # Individual timeline step
    THINKING_SUMMARY = "thinking_summary"    # Final "Thought for Xs" summary

    # Tool execution events
    TOOL_START = "tool_start"
    TOOL_PROGRESS = "tool_progress"
    TOOL_COMPLETE = "tool_complete"

    # Planning events
    PLANNING_START = "planning_start"
    PLANNING_PROGRESS = "planning_progress"
    PLANNING_COMPLETE = "planning_complete"

    # Context events
    CONTEXT_LOADING = "context_loading"
    CONTEXT_LOADED = "context_loaded"

    # Response events
    TEXT_DELTA = "text_delta"
    TEXT_COMPLETE = "text_complete"

    # Memory events
    MEMORY_UPDATE = "memory_update"

    # LLM Decision Events (for agent reasoning)
    LLM_THOUGHT = "llm_thought"
    LLM_DECISION = "llm_decision"
    LLM_ACTION = "llm_action"

    # Agent Tree Events
    AGENT_NODE = "agent_node"

    # Completion events
    DONE = "done"
    ERROR = "error"

    # Heartbeat for long connections
    HEARTBEAT = "heartbeat"


class ThinkingPhase(str, Enum):
    """
    Phases in the thinking timeline

    Used to track and display the agent's thought process
    in a ChatGPT-style "Thought for Xs" UI component
    """
    # Classification phase
    CLASSIFICATION = "classification"        # Analyzing query
    SYMBOL_DETECTION = "symbol_detection"    # Detecting stock symbols
    INTENT_ANALYSIS = "intent_analysis"      # Understanding intent

    # Tool selection phase
    TOOL_SELECTION = "tool_selection"        # Selecting tools
    TOOL_ROUTING = "tool_routing"            # Routing to tools

    # Execution phase
    TOOL_EXECUTION = "tool_execution"        # Executing tools
    DATA_GATHERING = "data_gathering"        # Gathering data

    # Synthesis phase
    SYNTHESIS = "synthesis"                  # Synthesizing response
    RESPONSE_GENERATION = "response_generation"  # Generating final response

    # Memory phase
    MEMORY_UPDATE = "memory_update"          # Updating memory
    LEARNING = "learning"                    # Learning from interaction


def generate_call_id(tool_name: str) -> str:
    """
    Generate unique call_id for tool correlation
    
    Format: {tool_name}_{uuid8}
    
    This allows frontend to correlate TOOL_START with TOOL_COMPLETE events
    """
    short_uuid = uuid.uuid4().hex[:8]
    safe_name = "".join(c for c in tool_name if c.isalnum())[:20]
    return f"{safe_name}_{short_uuid}"

def generate_node_id() -> str:
    """Generate unique node ID for agent tree tracking"""
    return f"node_{uuid.uuid4().hex[:12]}"

def truncate_with_cutoff(
    text: str,
    max_length: int = 500,
    generate_summary: bool = True
) -> Dict[str, Any]:
    """
    Truncate text and generate summary if needed
    """
    if len(text) <= max_length:
        return {
            "text": text,
            "cut_off": False,
            "original_length": len(text),
            "summaries": []
        }
    
    # Truncate at sentence boundary if possible
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    
    cut_point = max(last_period, last_newline)
    if cut_point > max_length * 0.7:  # At least 70% of max
        truncated = truncated[:cut_point + 1]
    else:
        truncated = truncated + "..."
    
    # Generate summaries if requested
    summaries = []
    if generate_summary:
        # Extract key points from original text
        sentences = text.split('.')
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
        
        summaries = [
            {"type": "key_point", "content": s[:100]}
            for s in key_sentences
        ]
    
    return {
        "text": truncated,
        "cut_off": True,
        "original_length": len(text),
        "summaries": summaries
    }


@dataclass
class StreamEvent:
    """
    Base class for all stream events
    
    SSE Format:
        event: {event_type}
        data: {json_payload}
    """
    event_type: StreamEventType
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    event_id: Optional[str] = None
    retry_ms: Optional[int] = None

    # Agent tree tracking
    node_id: Optional[str] = None
    parent_id: Optional[str] = None
    
    def to_sse(self) -> str:
        """Convert to SSE format string"""
        lines = []
        
        # Event type
        lines.append(f"event: {self.event_type.value}")
        
        # Event ID (optional)
        if self.event_id:
            lines.append(f"id: {self.event_id}")
        
        # Retry (optional)  
        if self.retry_ms:
            lines.append(f"retry: {self.retry_ms}")
        
        # Data payload - ensure JSON serializable
        payload = {
            "type": self.event_type.value,
            "timestamp": self.timestamp,
            **self.data
        }
        
        # Add tree tracking if present
        if self.node_id:
            payload["node_id"] = self.node_id
        if self.parent_id:
            payload["parent_id"] = self.parent_id

        # Remove any non-serializable objects
        safe_payload = self._make_json_safe(payload)
        lines.append(f"data: {json.dumps(safe_payload, ensure_ascii=False, default=str)}")
        
        # SSE requires double newline
        return "\n".join(lines) + "\n\n"
    
    def _make_json_safe(self, obj: Any) -> Any:
        """Make object JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items() 
                    if not k.startswith('_')}  # Skip private keys
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data
        }
        if self.node_id:
            result["node_id"] = self.node_id
        if self.parent_id:
            result["parent_id"] = self.parent_id
        return result


# ============================================================================
# SESSION EVENTS
# ============================================================================

@dataclass
class StartEvent(StreamEvent):
    """Session start event"""
    
    def __init__(
        self,
        session_id: str,
        flow_id: str,
        model_name: str = None,
        node_id: str = None
    ):
        # Generate root node_id if not provided
        _node_id = node_id or generate_node_id()

        super().__init__(
            event_type=StreamEventType.START,
            data={
                "session_id": session_id,
                "flow_id": flow_id,
                "model_name": model_name,
                "status": "initialized"
            },
            node_id=_node_id,
            parent_id=None  # Root node has no parent
        )


@dataclass
class ThinkingStartEvent(StreamEvent):
    """Start of thinking/reasoning phase"""
    
    def __init__(
        self, 
        phase: str, 
        message: str = "Analyzing query...",
        estimated_steps: int = None,
        node_id: str = None,
        parent_id: str = None
    ):
        super().__init__(
            event_type=StreamEventType.THINKING_START,
            data={
                "phase": phase,
                "message": message,
                "estimated_steps": estimated_steps,
                "status": "thinking"
            },
            node_id=node_id or generate_node_id(),
            parent_id=parent_id
        )


@dataclass
class ThinkingDeltaEvent(StreamEvent):
    """Progressive thinking update"""
    
    def __init__(
        self,
        phase: str,
        thought: str,
        progress: float = None,
        step_number: int = None,
        total_steps: int = None,
        max_thought_length: int = 2000, 
        node_id: str = None,
        parent_id: str = None
    ):
        
        # Apply cut_off + summaries
        truncation_result = truncate_with_cutoff(
            text=thought,
            max_length=max_thought_length,
            generate_summary=True
        )

        data = {
            "phase": phase,
            "thought": thought,
            "status": "thinking",
            "cut_off": truncation_result["cut_off"],
            "original_length": truncation_result["original_length"],
            "summaries": truncation_result["summaries"]
        }
        if progress is not None:
            data["progress"] = round(progress, 2)
        if step_number is not None:
            data["step_number"] = step_number
        if total_steps is not None:
            data["total_steps"] = total_steps
            
        super().__init__(
            event_type=StreamEventType.THINKING_DELTA,
            data=data,
            node_id=node_id,
            parent_id=parent_id
        )


@dataclass
class ThinkingEndEvent(StreamEvent):
    """End of thinking phase"""
    
    def __init__(
        self,
        phase: str,
        summary: str = None,
        duration_ms: int = None,
        node_id: str = None,
        parent_id: str = None
    ):
        super().__init__(
            event_type=StreamEventType.THINKING_END,
            data={
                "phase": phase,
                "summary": summary,
                "duration_ms": duration_ms,
                "status": "complete"
            },
            node_id=node_id,
            parent_id=parent_id
        )


@dataclass
class PlanningProgressEvent(StreamEvent):
    """Planning progress update"""
    
    def __init__(
        self,
        stage: int,
        total_stages: int,
        stage_name: str,
        details: str = None,
        node_id: str = None,
        parent_id: str = None
    ):
        super().__init__(
            event_type=StreamEventType.PLANNING_PROGRESS,
            data={
                "stage": stage,
                "total_stages": total_stages,
                "stage_name": stage_name,
                "details": details,
                "progress": round(stage / total_stages, 2)
            },
            node_id=node_id,
            parent_id=parent_id
        )


@dataclass
class PlanningCompleteEvent(StreamEvent):
    """Planning complete with task plan"""
    
    def __init__(
        self,
        task_count: int,
        strategy: str,
        symbols: List[str] = None,
        duration_ms: int = None,
        query_intent: str = None,
        node_id: str = None,
        parent_id: str = None
    ):
        super().__init__(
            event_type=StreamEventType.PLANNING_COMPLETE,
            data={
                "task_count": task_count,
                "strategy": strategy,
                "symbols": symbols or [],
                "duration_ms": duration_ms,
                "query_intent": query_intent,
                "status": "complete"
            },
            node_id=node_id,
            parent_id=parent_id
        )


@dataclass
class ToolStartEvent(StreamEvent):
    """Tool execution started"""
    
    def __init__(
        self,
        task_id: int,
        tool_name: str,
        tool_description: str = None,
        params: Dict[str, Any] = None,
        call_id: str = None,
        node_id: str = None,
        parent_id: str = None
    ):
        # Generate call_id if not provided
        _call_id = call_id or generate_call_id(tool_name)
        
        display_params = {}
        if params:
            for k, v in params.items():
                if k.lower() not in ['api_key', 'token', 'password', 'secret']:
                    display_params[k] = str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
        
        super().__init__(
            event_type=StreamEventType.TOOL_START,
            data={
                "task_id": task_id,
                "tool_name": tool_name,
                "tool_description": tool_description,
                "params": display_params,
                "call_id": _call_id,
                "status": "running"
            },
            node_id=node_id or generate_node_id(),
            parent_id=parent_id
        )

        # Store call_id as instance attribute for easy access
        self.call_id = _call_id


@dataclass
class ToolProgressEvent(StreamEvent):
    """Tool execution progress (for long-running tools)"""
    
    def __init__(
        self,
        task_id: int,
        tool_name: str,
        call_id: str, 
        progress: float,
        message: str = None,
        node_id: str = None,
        parent_id: str = None
    ):
        super().__init__(
            event_type=StreamEventType.TOOL_PROGRESS,
            data={
                "task_id": task_id,
                "tool_name": tool_name,
                "call_id": call_id,
                "progress": round(progress, 2),
                "message": message,
                "status": "running"
            },
            node_id=node_id,
            parent_id=parent_id
        )


@dataclass
class ToolCompleteEvent(StreamEvent):
    """Tool execution completed"""
    
    def __init__(
        self,
        task_id: int,
        tool_name: str,
        success: bool,
        call_id: str = None,
        preview: str = None,
        duration_ms: int = None,
        error: str = None,
        node_id: str = None,
        parent_id: str = None
    ):
        super().__init__(
            event_type=StreamEventType.TOOL_COMPLETE,
            data={
                "task_id": task_id,
                "tool_name": tool_name,
                "success": success,
                "call_id": call_id,
                "status": "success" if success else "failed",
                "preview": preview[:200] if preview else None,
                "duration_ms": duration_ms,
                "error": error
            },
            node_id=node_id,
            parent_id=parent_id
        )


@dataclass
class ContextLoadingEvent(StreamEvent):
    """Context loading progress"""
    
    def __init__(
        self,
        component: str,
        status: str,
        size_chars: int = None,
        display_name: str = None,
        node_id: str = None,
        parent_id: str = None
    ):
        super().__init__(
            event_type=StreamEventType.CONTEXT_LOADING,
            data={
                "component": component,
                "status": status,
                "size_chars": size_chars,
                "display_name": display_name
            },
            node_id=node_id,
            parent_id=parent_id
        )


@dataclass
class ContextLoadedEvent(StreamEvent):
    """All context loaded"""
    
    def __init__(
        self,
        total_tokens: int = None,
        context_usage_percent: float = None,
        components: Dict[str, int] = None,
        node_id: str = None,
        parent_id: str = None
    ):
        super().__init__(
            event_type=StreamEventType.CONTEXT_LOADED,
            data={
                "total_tokens": total_tokens,
                "context_usage_percent": context_usage_percent,
                "components": components,
                "status": "loaded"
            },
            node_id=node_id,
            parent_id=parent_id
        )


@dataclass
class TextDeltaEvent(StreamEvent):
    """Text chunk from LLM response"""
    
    def __init__(
        self,
        chunk: str,
        accumulated_length: int = None,
        node_id: str = None,
        parent_id: str = None
    ):
        super().__init__(
            event_type=StreamEventType.TEXT_DELTA,
            data={
                "chunk": chunk,
                "accumulated_length": accumulated_length
            },
            node_id=node_id,
            parent_id=parent_id
        )


@dataclass
class TextCompleteEvent(StreamEvent):
    """Full text response complete"""
    
    def __init__(
        self,
        total_length: int,
        duration_ms: int = None,
        node_id: str = None,
        parent_id: str = None
    ):
        super().__init__(
            event_type=StreamEventType.TEXT_COMPLETE,
            data={
                "total_length": total_length,
                "duration_ms": duration_ms,
                "status": "complete"
            },
            node_id=node_id,
            parent_id=parent_id
        )


@dataclass
class MemoryUpdateEvent(StreamEvent):
    """Memory update notification"""
    
    def __init__(
        self,
        action: str,
        memory_type: str,
        details: str = None,
        node_id: str = None,
        parent_id: str = None
    ):
        super().__init__(
            event_type=StreamEventType.MEMORY_UPDATE,
            data={
                "action": action,
                "memory_type": memory_type,
                "details": details
            },
            node_id=node_id,
            parent_id=parent_id
        )


# ============================================================================
# LLM DECISION EVENTS 
# ============================================================================

@dataclass
class LLMThoughtEvent(StreamEvent):
    """
    LLM internal reasoning step
    
    Shows what the LLM is "thinking" during decision making
    Similar to Claude's thinking blocks but more granular
    """
    
    def __init__(
        self,
        thought: str,
        thought_type: str = "reasoning",  # reasoning, analysis, consideration
        confidence: float = None,
        context: str = None,
        node_id: str = None,
        parent_id: str = None
    ):
        # Apply truncation with cut_off
        truncation = truncate_with_cutoff(thought, max_length=300)
        
        super().__init__(
            event_type=StreamEventType.LLM_THOUGHT,
            data={
                "thought": truncation["text"],
                "thought_type": thought_type,
                "confidence": round(confidence, 2) if confidence else None,
                "context": context,
                "cut_off": truncation["cut_off"],
                "summaries": truncation["summaries"]
            },
            node_id=node_id or generate_node_id(),
            parent_id=parent_id
        )


@dataclass
class LLMDecisionEvent(StreamEvent):
    """
    LLM decision point
    
    Shows when LLM makes a decision between options
    Useful for debugging and transparency
    """
    
    def __init__(
        self,
        decision: str,
        decision_type: str,  # tool_selection, category_selection, strategy_selection
        options_considered: List[str] = None,
        reasoning: str = None,
        confidence: float = None,
        node_id: str = None,
        parent_id: str = None
    ):
        super().__init__(
            event_type=StreamEventType.LLM_DECISION,
            data={
                "decision": decision,
                "decision_type": decision_type,
                "options_considered": options_considered or [],
                "reasoning": reasoning[:200] if reasoning else None,
                "confidence": round(confidence, 2) if confidence else None
            },
            node_id=node_id or generate_node_id(),
            parent_id=parent_id
        )


@dataclass
class LLMActionEvent(StreamEvent):
    """
    LLM action selection
    
    Shows what action the LLM decided to take
    """
    
    def __init__(
        self,
        action: str,  # call_tool, respond, request_info, skip
        action_target: str = None,  # Tool name or target of action
        parameters: Dict[str, Any] = None,
        reasoning: str = None,
        node_id: str = None,
        parent_id: str = None
    ):
        # Filter sensitive params
        safe_params = {}
        if parameters:
            for k, v in parameters.items():
                if k.lower() not in ['api_key', 'token', 'password', 'secret']:
                    safe_params[k] = v
        
        super().__init__(
            event_type=StreamEventType.LLM_ACTION,
            data={
                "action": action,
                "action_target": action_target,
                "parameters": safe_params,
                "reasoning": reasoning[:150] if reasoning else None
            },
            node_id=node_id or generate_node_id(),
            parent_id=parent_id
        )


# ============================================================================
# AGENT TREE EVENTS (Task 6 - NEW)
# ============================================================================

@dataclass
class AgentNodeEvent(StreamEvent):
    """
    Agent tree node event
    
    Tracks hierarchical agent execution for debugging
    
    Example tree structure:
    ROOT (query received)
    ├── PLANNING (planning phase)
    │   ├── CLASSIFICATION (classify query)
    │   └── TASK_CREATION (create tasks)
    ├── EXECUTION (execution phase)
    │   ├── TOOL_1 (first tool)
    │   └── TOOL_2 (second tool)
    └── GENERATION (response generation)
    """
    
    def __init__(
        self,
        node_type: str,  # root, planning, classification, execution, tool, generation
        node_name: str,
        status: str = "started",  # started, running, completed, failed
        metadata: Dict[str, Any] = None,
        duration_ms: int = None,
        error: str = None,
        node_id: str = None,
        parent_id: str = None
    ):
        super().__init__(
            event_type=StreamEventType.AGENT_NODE,
            data={
                "node_type": node_type,
                "node_name": node_name,
                "status": status,
                "metadata": metadata or {},
                "duration_ms": duration_ms,
                "error": error
            },
            node_id=node_id or generate_node_id(),
            parent_id=parent_id
        )


# ============================================================================
# COMPLETION EVENTS
# ============================================================================

@dataclass
class DoneEvent(StreamEvent):
    """Stream complete"""
    
    def __init__(
        self,
        session_id: str,
        flow_id: str,
        total_duration_ms: int = None,
        stats: Dict[str, Any] = None,
        agent_tree: Dict[str, Any] = None,
        node_id: str = None,
        parent_id: str = None
    ):
        safe_stats = {}
        if stats:
            for k, v in stats.items():
                if isinstance(v, (str, int, float, bool, type(None), list, dict)):
                    safe_stats[k] = v
                else:
                    safe_stats[k] = str(v)
        
        super().__init__(
            event_type=StreamEventType.DONE,
            data={
                "session_id": session_id,
                "flow_id": flow_id,
                "total_duration_ms": total_duration_ms,
                "stats": safe_stats,
                "agent_tree": agent_tree,
                "status": "complete"
            },
            node_id=node_id,
            parent_id=parent_id
        )


@dataclass
class ErrorEvent(StreamEvent):
    """Error occurred"""
    
    def __init__(
        self,
        error_message: str,
        error_type: str = None,
        phase: str = None,
        recoverable: bool = False,
        node_id: str = None,
        parent_id: str = None
    ):
        super().__init__(
            event_type=StreamEventType.ERROR,
            data={
                "error": error_message,
                "error_type": error_type,
                "phase": phase,
                "recoverable": recoverable,
                "status": "error"
            },
            node_id=node_id,
            parent_id=parent_id
        )


@dataclass
class HeartbeatEvent(StreamEvent):
    """Keep-alive heartbeat"""

    def __init__(self, elapsed_seconds: int = None):
        super().__init__(
            event_type=StreamEventType.HEARTBEAT,
            data={
                "elapsed_seconds": elapsed_seconds,
                "status": "alive"
            }
        )


# ============================================================================
# THINKING TIMELINE EVENTS (ChatGPT-style "Thought for Xs" Display)
# ============================================================================

@dataclass
class ThinkingTimelineStep:
    """
    Individual step in the thinking timeline

    Represents one line in the ChatGPT-style thinking display:
    ├── [0.3s] 🔍 LLM Call: Intent Classification

    Attributes:
        elapsed_ms: Time since thinking started (in milliseconds)
        phase: Phase of thinking (classification, tool_selection, etc.)
        action: Description of the action
        is_llm_call: Whether this step involves an LLM call (shows 🔍 indicator)
        is_tool_call: Whether this step involves a tool call (shows 🔧 indicator)
        details: Optional additional details (symbol, result preview, etc.)
        success: Whether the step completed successfully
    """
    elapsed_ms: int
    phase: str  # ThinkingPhase value
    action: str
    is_llm_call: bool = False
    is_tool_call: bool = False
    details: Optional[str] = None
    success: Optional[bool] = None
    step_index: int = 0

    def format_for_display(self) -> str:
        """
        Format step for display in UI

        Returns:
            Formatted string like "[0.3s] 🔍 LLM Call: Intent Classification"
        """
        elapsed_s = self.elapsed_ms / 1000
        indicator = ""
        if self.is_llm_call:
            indicator = "🔍 "
        elif self.is_tool_call:
            indicator = "🔧 "

        result = f"[{elapsed_s:.1f}s] {indicator}{self.action}"
        if self.details:
            result += f" → {self.details}"
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "elapsed_ms": self.elapsed_ms,
            "elapsed_s": round(self.elapsed_ms / 1000, 1),
            "phase": self.phase,
            "action": self.action,
            "is_llm_call": self.is_llm_call,
            "is_tool_call": self.is_tool_call,
            "details": self.details,
            "success": self.success,
            "step_index": self.step_index,
            "display": self.format_for_display()
        }


@dataclass
class ThinkingTimelineEvent(StreamEvent):
    """
    SSE Event for thinking timeline step

    Emitted for each step in the thinking process.
    Frontend can accumulate these to build the timeline display.

    Example SSE:
        event: thinking_timeline
        data: {
            "type": "thinking_timeline",
            "elapsed_ms": 300,
            "elapsed_s": 0.3,
            "phase": "classification",
            "action": "Intent Classification",
            "is_llm_call": true,
            "details": "Detected symbols: NVDA",
            "step_index": 2
        }
    """

    def __init__(
        self,
        elapsed_ms: int,
        phase: str,
        action: str,
        is_llm_call: bool = False,
        is_tool_call: bool = False,
        details: Optional[str] = None,
        success: Optional[bool] = None,
        step_index: int = 0,
        node_id: str = None,
        parent_id: str = None
    ):
        super().__init__(
            event_type=StreamEventType.THINKING_TIMELINE,
            data={
                "elapsed_ms": elapsed_ms,
                "elapsed_s": round(elapsed_ms / 1000, 1),
                "phase": phase,
                "action": action,
                "is_llm_call": is_llm_call,
                "is_tool_call": is_tool_call,
                "details": details,
                "success": success,
                "step_index": step_index,
                "display": self._format_display(elapsed_ms, action, is_llm_call, is_tool_call, details)
            },
            node_id=node_id,
            parent_id=parent_id
        )

    @staticmethod
    def _format_display(elapsed_ms: int, action: str, is_llm_call: bool, is_tool_call: bool, details: str) -> str:
        """Format for display"""
        elapsed_s = elapsed_ms / 1000
        indicator = ""
        if is_llm_call:
            indicator = "🔍 "
        elif is_tool_call:
            indicator = "🔧 "

        result = f"[{elapsed_s:.1f}s] {indicator}{action}"
        if details:
            result += f" → {details}"
        return result


@dataclass
class ThinkingSummaryEvent(StreamEvent):
    """
    Final thinking summary event - "Thought for Xs"

    Emitted when thinking phase completes.
    Provides the ChatGPT-style "Thought for X seconds" summary.

    Example SSE:
        event: thinking_summary
        data: {
            "type": "thinking_summary",
            "total_duration_ms": 2500,
            "total_duration_s": 2.5,
            "display": "Thought for 2.5s",
            "steps_count": 8,
            "llm_calls_count": 3,
            "tool_calls_count": 4,
            "timeline": [
                {"elapsed_s": 0.0, "action": "Analyzing query...", ...},
                {"elapsed_s": 0.3, "action": "Intent Classification", "is_llm_call": true, ...},
                ...
            ]
        }
    """

    def __init__(
        self,
        total_duration_ms: int,
        timeline_steps: List[ThinkingTimelineStep],
        node_id: str = None,
        parent_id: str = None
    ):
        # Calculate statistics
        llm_calls_count = sum(1 for s in timeline_steps if s.is_llm_call)
        tool_calls_count = sum(1 for s in timeline_steps if s.is_tool_call)

        super().__init__(
            event_type=StreamEventType.THINKING_SUMMARY,
            data={
                "total_duration_ms": total_duration_ms,
                "total_duration_s": round(total_duration_ms / 1000, 1),
                "display": f"Thought for {total_duration_ms / 1000:.1f}s",
                "steps_count": len(timeline_steps),
                "llm_calls_count": llm_calls_count,
                "tool_calls_count": tool_calls_count,
                "timeline": [step.to_dict() for step in timeline_steps]
            },
            node_id=node_id,
            parent_id=parent_id
        )


# ============================================================================
# THINKING TIMELINE TRACKER
# ============================================================================

class ThinkingTimeline:
    """
    Tracks thinking timeline for a request

    Usage:
        timeline = ThinkingTimeline()
        timeline.add_step("classification", "Analyzing query...", is_llm_call=True)
        timeline.add_step("symbol_detection", "Detected symbols", details="NVDA, AAPL")
        timeline.add_tool_step("getStockPrice", symbol="NVDA", success=True)

        # Get events for SSE streaming
        for event in timeline.get_pending_events():
            yield event.to_sse()

        # Get final summary
        summary_event = timeline.get_summary_event()
    """

    def __init__(self):
        self.start_time = datetime.utcnow()
        self.steps: List[ThinkingTimelineStep] = []
        self._pending_events: List[ThinkingTimelineEvent] = []
        self._step_index = 0
        self._parent_node_id: Optional[str] = None

    def set_parent_node(self, node_id: str):
        """Set parent node for tree tracking"""
        self._parent_node_id = node_id

    def _get_elapsed_ms(self) -> int:
        """Get elapsed time since start"""
        return int((datetime.utcnow() - self.start_time).total_seconds() * 1000)

    def add_step(
        self,
        phase: str,
        action: str,
        is_llm_call: bool = False,
        is_tool_call: bool = False,
        details: Optional[str] = None,
        success: Optional[bool] = None
    ) -> ThinkingTimelineStep:
        """
        Add a step to the timeline

        Args:
            phase: ThinkingPhase value
            action: Description of the action
            is_llm_call: Whether this is an LLM call
            is_tool_call: Whether this is a tool call
            details: Optional details
            success: Whether step succeeded

        Returns:
            Created ThinkingTimelineStep
        """
        elapsed_ms = self._get_elapsed_ms()
        step = ThinkingTimelineStep(
            elapsed_ms=elapsed_ms,
            phase=phase,
            action=action,
            is_llm_call=is_llm_call,
            is_tool_call=is_tool_call,
            details=details,
            success=success,
            step_index=self._step_index
        )
        self.steps.append(step)

        # Create SSE event
        event = ThinkingTimelineEvent(
            elapsed_ms=elapsed_ms,
            phase=phase,
            action=action,
            is_llm_call=is_llm_call,
            is_tool_call=is_tool_call,
            details=details,
            success=success,
            step_index=self._step_index,
            parent_id=self._parent_node_id
        )
        self._pending_events.append(event)
        self._step_index += 1

        return step

    def add_llm_step(self, action: str, details: Optional[str] = None) -> ThinkingTimelineStep:
        """Add an LLM call step"""
        phase = ThinkingPhase.CLASSIFICATION.value  # Default phase for LLM calls
        return self.add_step(
            phase=phase,
            action=f"LLM Call: {action}",
            is_llm_call=True,
            details=details
        )

    def add_tool_step(
        self,
        tool_name: str,
        symbol: Optional[str] = None,
        result_preview: Optional[str] = None,
        success: bool = True
    ) -> ThinkingTimelineStep:
        """Add a tool execution step"""
        action = f"Tool: {tool_name}"
        if symbol:
            action += f"({symbol})"

        return self.add_step(
            phase=ThinkingPhase.TOOL_EXECUTION.value,
            action=action,
            is_tool_call=True,
            details=result_preview,
            success=success
        )

    def get_pending_events(self) -> List[ThinkingTimelineEvent]:
        """Get and clear pending events"""
        events = self._pending_events
        self._pending_events = []
        return events

    def get_summary_event(self) -> ThinkingSummaryEvent:
        """Get final summary event"""
        return ThinkingSummaryEvent(
            total_duration_ms=self._get_elapsed_ms(),
            timeline_steps=self.steps,
            parent_id=self._parent_node_id
        )

    def get_elapsed_seconds(self) -> float:
        """Get elapsed time in seconds"""
        return self._get_elapsed_ms() / 1000

    def get_steps_count(self) -> int:
        """Get total steps count"""
        return len(self.steps)


# ============================================================================
# SSE Formatting Utilities
# ============================================================================

def format_sse_done() -> str:
    """Format the standard SSE done marker"""
    return "data: [DONE]\n\n"

def format_sse_comment(comment: str) -> str:
    """Format SSE comment (for keep-alive)"""
    return f": {comment}\n\n"


# ============================================================================
# Stream State Tracker
# ============================================================================

@dataclass
class StreamState:
    """Track stream state for a request"""
    session_id: str
    flow_id: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    
    # Root node for agent tree
    root_node_id: str = field(default_factory=generate_node_id)
    current_node_id: str = None

    # Phase tracking
    current_phase: str = "initializing"
    phase_start_time: datetime = None
    
    # Tool tracking
    total_tools: int = 0
    completed_tools: int = 0
    failed_tools: int = 0
    active_tool_calls: Dict[str, Dict] = field(default_factory=dict)  # call_id -> tool_info
    
    # Text tracking
    accumulated_text: str = ""
    text_chunks: int = 0
    
    # Performance tracking
    planning_duration_ms: int = 0
    tool_execution_duration_ms: int = 0
    response_generation_duration_ms: int = 0
    
    # Agent tree 
    agent_tree: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize agent tree with root node"""
        self.current_node_id = self.root_node_id
        self.agent_tree = {
            "root": self.root_node_id,
            "nodes": {
                self.root_node_id: {
                    "type": "root",
                    "name": "query_processing",
                    "parent": None,
                    "children": [],
                    "status": "started",
                    "start_time": self.started_at.isoformat()
                }
            }
        }
    
    def start_phase(self, phase: str, node_id: str = None) -> str:
        """Start a new phase"""
        self.current_phase = phase
        self.phase_start_time = datetime.utcnow()

        # Add to agent tree
        _node_id = node_id or generate_node_id()
        self.agent_tree["nodes"][_node_id] = {
            "type": "phase",
            "name": phase,
            "parent": self.current_node_id,
            "children": [],
            "status": "started",
            "start_time": datetime.utcnow().isoformat()
        }
        
        # Add as child to current node
        if self.current_node_id in self.agent_tree["nodes"]:
            self.agent_tree["nodes"][self.current_node_id]["children"].append(_node_id)
        
        self.current_node_id = _node_id
        return _node_id
    
    def end_phase(self) -> int:
        """End current phase, return duration in ms"""
        duration = 0
        if self.phase_start_time:
            duration = int((datetime.utcnow() - self.phase_start_time).total_seconds() * 1000)
        
        # Update agent tree
        if self.current_node_id in self.agent_tree["nodes"]:
            self.agent_tree["nodes"][self.current_node_id]["status"] = "completed"
            self.agent_tree["nodes"][self.current_node_id]["duration_ms"] = duration
            self.agent_tree["nodes"][self.current_node_id]["end_time"] = datetime.utcnow().isoformat()
            
            # Move back to parent
            parent_id = self.agent_tree["nodes"][self.current_node_id].get("parent")
            if parent_id:
                self.current_node_id = parent_id
        
        return duration
    
    def start_tool(self, call_id: str, tool_name: str, node_id: str = None) -> str:
        """
        Track tool start with call_id correlation
        Add to agent tree
        """
        _node_id = node_id or generate_node_id()
        
        self.active_tool_calls[call_id] = {
            "tool_name": tool_name,
            "node_id": _node_id,
            "start_time": datetime.utcnow()
        }
        self.total_tools += 1
        
        # Task 6: Add to agent tree
        self.agent_tree["nodes"][_node_id] = {
            "type": "tool",
            "name": tool_name,
            "call_id": call_id,
            "parent": self.current_node_id,
            "children": [],
            "status": "running",
            "start_time": datetime.utcnow().isoformat()
        }
        
        if self.current_node_id in self.agent_tree["nodes"]:
            self.agent_tree["nodes"][self.current_node_id]["children"].append(_node_id)
        
        return _node_id
    
    def end_tool(self, call_id: str, success: bool) -> int:
        """
        Track tool completion with call_id correlation
        Returns duration in ms
        """
        duration = 0
        
        if call_id in self.active_tool_calls:
            tool_info = self.active_tool_calls.pop(call_id)
            start_time = tool_info.get("start_time")
            node_id = tool_info.get("node_id")
            
            if start_time:
                duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Task 6: Update agent tree
            if node_id and node_id in self.agent_tree["nodes"]:
                self.agent_tree["nodes"][node_id]["status"] = "completed" if success else "failed"
                self.agent_tree["nodes"][node_id]["duration_ms"] = duration
                self.agent_tree["nodes"][node_id]["end_time"] = datetime.utcnow().isoformat()
        
        if success:
            self.completed_tools += 1
        else:
            self.failed_tools += 1
        
        return duration
    
    def add_tool_result(self, success: bool):
        """Track tool execution result"""
        if success:
            self.completed_tools += 1
        else:
            self.failed_tools += 1
    
    def add_text_chunk(self, chunk: str):
        """Track text generation"""
        self.accumulated_text += chunk
        self.text_chunks += 1
    
    def get_elapsed_ms(self) -> int:
        """Get total elapsed time in ms"""
        return int((datetime.utcnow() - self.started_at).total_seconds() * 1000)
    
    def get_agent_tree_summary(self) -> Dict[str, Any]:
        """
        Get summary of agent tree for debugging
        """
        nodes_count = len(self.agent_tree.get("nodes", {}))
        
        # Count by type
        type_counts = {}
        for node in self.agent_tree.get("nodes", {}).values():
            node_type = node.get("type", "unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        return {
            "root_node_id": self.root_node_id,
            "total_nodes": nodes_count,
            "nodes_by_type": type_counts,
            "tree": self.agent_tree
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics"""
        return {
            "total_duration_ms": self.get_elapsed_ms(),
            "planning_duration_ms": self.planning_duration_ms,
            "tool_execution_duration_ms": self.tool_execution_duration_ms,
            "response_generation_duration_ms": self.response_generation_duration_ms,
            "tools_total": self.total_tools,
            "tools_completed": self.completed_tools,
            "tools_failed": self.failed_tools,
            "text_chunks": self.text_chunks,
            "text_length": len(self.accumulated_text),
            "agent_tree_nodes": len(self.agent_tree.get("nodes", {}))
        }