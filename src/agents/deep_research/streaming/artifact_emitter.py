"""
Deep Research Artifact Emitter

Handles SSE (Server-Sent Events) streaming for deep research progress.
Provides transparency to users by streaming artifacts in real-time.

Event Types:
- research_init: Research session started
- clarification_request: Asking user questions
- plan_created: Research plan ready for confirmation
- worker_spawned: Worker agent started
- worker_progress: Worker progress update
- worker_artifact: Worker produced an artifact
- worker_completed: Worker finished
- synthesis_started: Report synthesis started
- synthesis_artifact: Synthesis produced content
- research_completed: All done

Each event follows SSE format:
    event: {event_type}
    data: {json_payload}

"""

import json
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, AsyncGenerator
from enum import Enum

from src.agents.deep_research.models import (
    AgentState,
    Artifact,
    ArtifactType,
    ResearchPlan,
    WorkerResult,
    DeepResearchEventType,
    ClarificationQuestion,
)


# ============================================================================
# SSE EVENT DATA CLASSES
# ============================================================================

@dataclass
class DeepResearchStreamEvent:
    """
    A single SSE event for deep research streaming.
    """
    event_type: DeepResearchEventType
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    event_id: Optional[str] = None

    def to_sse(self) -> str:
        """Convert to SSE format string."""
        lines = []

        # Event type
        lines.append(f"event: {self.event_type.value}")

        # Event ID (optional)
        if self.event_id:
            lines.append(f"id: {self.event_id}")

        # Data payload
        payload = {
            "type": self.event_type.value,
            "timestamp": self.timestamp,
            **self.data,
        }

        # Ensure JSON serializable
        safe_payload = self._make_json_safe(payload)
        lines.append(f"data: {json.dumps(safe_payload, ensure_ascii=False, default=str)}")

        # SSE requires double newline
        return "\n".join(lines) + "\n\n"

    def _make_json_safe(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
        }


# ============================================================================
# ARTIFACT EMITTER
# ============================================================================

class ArtifactEmitter:
    """
    Emitter for deep research artifacts and progress events.

    Provides a clean API for emitting various types of events
    during deep research execution.

    Usage:
        emitter = ArtifactEmitter(research_id="abc123")

        # Emit research start
        yield emitter.research_init(query="Analyze NVDA")

        # Emit plan
        yield emitter.plan_created(plan)

        # Emit worker events
        yield emitter.worker_spawned(worker_id="w1", role="market_analyst")
        yield emitter.worker_progress(worker_id="w1", progress=0.5)
        yield emitter.worker_artifact(artifact)
        yield emitter.worker_completed(worker_id="w1", result=result)

        # Emit completion
        yield emitter.research_completed(result)
    """

    def __init__(self, research_id: str):
        self.research_id = research_id
        self._event_counter = 0

    def _next_event_id(self) -> str:
        """Generate sequential event ID."""
        self._event_counter += 1
        return f"{self.research_id}_{self._event_counter}"

    # ========================================================================
    # LIFECYCLE EVENTS
    # ========================================================================

    def research_init(
        self,
        query: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> DeepResearchStreamEvent:
        """Emit research initialization event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.RESEARCH_INIT,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "query": query,
                "config": config or {},
                "status": "initialized",
            },
        )

    def research_completed(
        self,
        total_duration_ms: int,
        sections_completed: int,
        sources_count: int,
        final_report_length: int,
    ) -> DeepResearchStreamEvent:
        """Emit research completion event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.RESEARCH_COMPLETED,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "status": "completed",
                "total_duration_ms": total_duration_ms,
                "sections_completed": sections_completed,
                "sources_count": sources_count,
                "final_report_length": final_report_length,
            },
        )

    def research_failed(
        self,
        error: str,
        phase: str,
        duration_ms: int = 0,
    ) -> DeepResearchStreamEvent:
        """Emit research failure event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.RESEARCH_FAILED,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "status": "failed",
                "error": error,
                "phase": phase,
                "duration_ms": duration_ms,
            },
        )

    def research_cancelled(
        self,
        reason: str = "User cancelled",
    ) -> DeepResearchStreamEvent:
        """Emit research cancellation event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.RESEARCH_CANCELLED,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "status": "cancelled",
                "reason": reason,
            },
        )

    # ========================================================================
    # CLARIFICATION EVENTS
    # ========================================================================

    def clarification_request(
        self,
        questions: List[ClarificationQuestion],
        context: Optional[str] = None,
    ) -> DeepResearchStreamEvent:
        """Emit clarification request event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.CLARIFICATION_REQUEST,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "questions": [q.to_dict() for q in questions],
                "context": context,
                "status": "awaiting_clarification",
            },
        )

    def clarification_received(
        self,
        answers: Dict[str, str],
    ) -> DeepResearchStreamEvent:
        """Emit clarification received event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.CLARIFICATION_RECEIVED,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "answers": answers,
                "status": "clarification_received",
            },
        )

    # ========================================================================
    # PLAN EVENTS
    # ========================================================================

    def plan_created(
        self,
        plan: ResearchPlan,
    ) -> DeepResearchStreamEvent:
        """Emit plan created event with artifact."""
        artifact = Artifact.create(
            artifact_type=ArtifactType.PLAN,
            title=plan.title,
            content=plan.to_markdown(),
            metadata={"plan_id": plan.research_id},
        )

        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.PLAN_CREATED,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "plan": plan.to_dict(),
                "artifact": artifact.to_dict(),
                "status": "plan_ready",
            },
        )

    def plan_confirmed(
        self,
        plan_id: str,
        modifications: Optional[Dict[str, Any]] = None,
    ) -> DeepResearchStreamEvent:
        """Emit plan confirmation event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.PLAN_CONFIRMED,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "plan_id": plan_id,
                "modifications": modifications,
                "status": "plan_confirmed",
            },
        )

    # ========================================================================
    # WORKER EVENTS
    # ========================================================================

    def worker_spawned(
        self,
        worker_id: str,
        role: str,
        section_id: int,
        section_name: str,
        task_description: str,
    ) -> DeepResearchStreamEvent:
        """Emit worker spawned event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.WORKER_SPAWNED,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "worker_id": worker_id,
                "role": role,
                "section_id": section_id,
                "section_name": section_name,
                "task": task_description,
                "status": "running",
            },
        )

    def worker_progress(
        self,
        worker_id: str,
        progress: float,
        current_step: str,
        iteration: int = 0,
    ) -> DeepResearchStreamEvent:
        """Emit worker progress event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.WORKER_PROGRESS,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "worker_id": worker_id,
                "progress": round(progress, 2),
                "current_step": current_step,
                "iteration": iteration,
            },
        )

    def worker_artifact(
        self,
        worker_id: str,
        artifact: Artifact,
    ) -> DeepResearchStreamEvent:
        """Emit worker artifact event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.WORKER_ARTIFACT,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "worker_id": worker_id,
                "artifact": artifact.to_dict(),
            },
        )

    def worker_completed(
        self,
        worker_id: str,
        result: WorkerResult,
    ) -> DeepResearchStreamEvent:
        """Emit worker completed event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.WORKER_COMPLETED,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "worker_id": worker_id,
                "section_id": result.section_id,
                "success": result.success,
                "duration_ms": result.duration_ms,
                "findings_preview": result.findings[:500] if result.findings else "",
                "sources_count": len(result.sources),
            },
        )

    def worker_failed(
        self,
        worker_id: str,
        section_id: int,
        error: str,
        duration_ms: int = 0,
    ) -> DeepResearchStreamEvent:
        """Emit worker failed event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.WORKER_FAILED,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "worker_id": worker_id,
                "section_id": section_id,
                "error": error,
                "duration_ms": duration_ms,
            },
        )

    # ========================================================================
    # SYNTHESIS EVENTS
    # ========================================================================

    def synthesis_started(
        self,
        sections_to_combine: List[int],
    ) -> DeepResearchStreamEvent:
        """Emit synthesis started event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.SYNTHESIS_STARTED,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "sections_to_combine": sections_to_combine,
                "status": "synthesizing",
            },
        )

    def synthesis_progress(
        self,
        current_section: str,
        progress: float,
    ) -> DeepResearchStreamEvent:
        """Emit synthesis progress event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.SYNTHESIS_PROGRESS,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "current_section": current_section,
                "progress": round(progress, 2),
            },
        )

    def synthesis_artifact(
        self,
        artifact: Artifact,
    ) -> DeepResearchStreamEvent:
        """Emit synthesis artifact event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.SYNTHESIS_ARTIFACT,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "artifact": artifact.to_dict(),
            },
        )

    # ========================================================================
    # GENERAL EVENTS
    # ========================================================================

    def artifact(
        self,
        artifact: Artifact,
    ) -> DeepResearchStreamEvent:
        """Emit generic artifact event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.ARTIFACT,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "artifact": artifact.to_dict(),
            },
        )

    def progress(
        self,
        phase: str,
        progress: float,
        message: str,
    ) -> DeepResearchStreamEvent:
        """Emit general progress event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.PROGRESS,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "phase": phase,
                "progress": round(progress, 2),
                "message": message,
            },
        )

    def error(
        self,
        error: str,
        phase: str,
        recoverable: bool = False,
    ) -> DeepResearchStreamEvent:
        """Emit error event."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.ERROR,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "error": error,
                "phase": phase,
                "recoverable": recoverable,
            },
        )

    def heartbeat(
        self,
        elapsed_seconds: int,
    ) -> DeepResearchStreamEvent:
        """Emit heartbeat event for long-running operations."""
        return DeepResearchStreamEvent(
            event_type=DeepResearchEventType.HEARTBEAT,
            event_id=self._next_event_id(),
            data={
                "research_id": self.research_id,
                "elapsed_seconds": elapsed_seconds,
                "status": "alive",
            },
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_sse_done() -> str:
    """Format the standard SSE done marker."""
    return "data: [DONE]\n\n"


def format_sse_comment(comment: str) -> str:
    """Format SSE comment (for keep-alive)."""
    return f": {comment}\n\n"


async def heartbeat_generator(
    emitter: ArtifactEmitter,
    interval_seconds: int = 15,
    max_duration_seconds: int = 600,
) -> AsyncGenerator[str, None]:
    """
    Generate periodic heartbeat events.

    Use this in parallel with main research to keep connection alive.
    """
    import asyncio

    elapsed = 0
    while elapsed < max_duration_seconds:
        await asyncio.sleep(interval_seconds)
        elapsed += interval_seconds
        yield emitter.heartbeat(elapsed).to_sse()
