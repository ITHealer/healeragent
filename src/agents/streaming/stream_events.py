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