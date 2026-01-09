"""
Deep Research Mode - Data Models

Defines all data structures used in the Deep Research system:
- AgentState: State machine enum for tracking agent progress
- Research planning models: ResearchPlan, ResearchSection
- Worker models: WorkerTask, WorkerResult
- Artifact models: For streaming transparency to users
- Clarification models: For user confirmation flow

Design principles:
- Immutable where possible (use frozen=True for dataclasses)
- Clear serialization to dict/JSON for SSE streaming
- Type hints for all fields
"""

import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime


# ============================================================================
# STATE MACHINE (OpenManus Pattern)
# ============================================================================

class AgentState(str, Enum):
    """
    State machine for Deep Research agents.

    Inspired by OpenManus pattern for tracking agent lifecycle.
    """
    IDLE = "idle"                      # Agent not started
    CLARIFYING = "clarifying"          # Asking user clarification questions
    WAITING_CONFIRMATION = "waiting_confirmation"  # Waiting for user to confirm plan
    PLANNING = "planning"              # Creating research plan
    EXECUTING = "executing"            # Workers running
    SYNTHESIZING = "synthesizing"      # Combining results into report
    STUCK = "stuck"                    # Agent detected stuck condition
    COMPLETED = "completed"            # Successfully finished
    FAILED = "failed"                  # Failed with error
    CANCELLED = "cancelled"            # User cancelled


class ResearchSessionStatus(str, Enum):
    """
    Status enum for research sessions in the router.

    Provides type-safe status management with valid transitions.
    """
    STARTED = "started"
    ANALYZING = "analyzing"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    EXECUTING = "executing"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


# Valid state transitions for session management
VALID_SESSION_TRANSITIONS: Dict[ResearchSessionStatus, List[ResearchSessionStatus]] = {
    ResearchSessionStatus.STARTED: [
        ResearchSessionStatus.ANALYZING,
        ResearchSessionStatus.CANCELLED,
    ],
    ResearchSessionStatus.ANALYZING: [
        ResearchSessionStatus.AWAITING_CLARIFICATION,
        ResearchSessionStatus.AWAITING_CONFIRMATION,
        ResearchSessionStatus.FAILED,
    ],
    ResearchSessionStatus.AWAITING_CLARIFICATION: [
        ResearchSessionStatus.AWAITING_CONFIRMATION,
        ResearchSessionStatus.CANCELLED,
        ResearchSessionStatus.FAILED,
    ],
    ResearchSessionStatus.AWAITING_CONFIRMATION: [
        ResearchSessionStatus.EXECUTING,
        ResearchSessionStatus.CANCELLED,
        ResearchSessionStatus.FAILED,
    ],
    ResearchSessionStatus.EXECUTING: [
        ResearchSessionStatus.SYNTHESIZING,
        ResearchSessionStatus.COMPLETED,
        ResearchSessionStatus.FAILED,
        ResearchSessionStatus.CANCELLED,
    ],
    ResearchSessionStatus.SYNTHESIZING: [
        ResearchSessionStatus.COMPLETED,
        ResearchSessionStatus.FAILED,
    ],
    ResearchSessionStatus.COMPLETED: [],  # Terminal state
    ResearchSessionStatus.CANCELLED: [],  # Terminal state
    ResearchSessionStatus.FAILED: [],     # Terminal state
}


def validate_session_transition(
    current: ResearchSessionStatus,
    target: ResearchSessionStatus,
) -> bool:
    """Check if a session state transition is valid."""
    valid_targets = VALID_SESSION_TRANSITIONS.get(current, [])
    return target in valid_targets


class WorkerRole(str, Enum):
    """
    Specialized roles for research workers.

    Each role has specific tools and prompts optimized for their domain.
    """
    MARKET_ANALYST = "market_analyst"           # Market position, trends
    FINANCIAL_ANALYST = "financial_analyst"     # Financials, ratios
    TECHNICAL_ANALYST = "technical_analyst"     # Charts, patterns
    WEB_RESEARCHER = "web_researcher"           # Web search, news
    INDUSTRY_EXPERT = "industry_expert"         # Competitive analysis
    RISK_ANALYST = "risk_analyst"               # Risk assessment


class ArtifactType(str, Enum):
    """
    Types of artifacts emitted during research.

    Frontend uses these to render appropriate UI components.
    """
    CLARIFICATION = "clarification"     # Questions for user
    PLAN = "plan"                       # Research plan
    FINDING = "finding"                 # Worker finding/insight
    DATA = "data"                       # Raw data (charts, tables)
    SECTION = "section"                 # Report section
    SUMMARY = "summary"                 # Executive summary
    SOURCE = "source"                   # Source reference
    PROGRESS = "progress"               # Progress update
    ERROR = "error"                     # Error information


# ============================================================================
# CLARIFICATION MODELS
# ============================================================================

@dataclass
class ClarificationQuestion:
    """
    A question to clarify user intent before research.

    Example:
        ClarificationQuestion(
            question_id="q1",
            question="What time horizon are you considering?",
            question_type="single_choice",
            options=["Short-term (< 6 months)", "Medium-term (6-24 months)", "Long-term (> 2 years)"],
            default="Medium-term (6-24 months)"
        )
    """
    question_id: str
    question: str
    question_type: str = "single_choice"  # single_choice, multi_choice, text, boolean
    options: List[str] = field(default_factory=list)
    default: Optional[str] = None
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "question_type": self.question_type,
            "options": self.options,
            "default": self.default,
            "required": self.required,
        }


@dataclass
class ClarificationResponse:
    """
    User's response to clarification questions.
    """
    research_id: str
    answers: Dict[str, str]  # question_id -> answer
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ============================================================================
# RESEARCH PLANNING MODELS
# ============================================================================

@dataclass
class ResearchSection:
    """
    A section of the research report.

    Each section is assigned to a worker agent for execution.
    """
    id: int
    name: str
    description: str
    worker_role: WorkerRole
    tools_needed: List[str] = field(default_factory=list)
    estimated_duration_sec: int = 60
    priority: int = 1  # 1 = highest priority
    dependencies: List[int] = field(default_factory=list)  # section IDs this depends on
    status: str = "pending"  # pending, running, completed, failed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "worker_role": self.worker_role.value if isinstance(self.worker_role, WorkerRole) else self.worker_role,
            "tools_needed": self.tools_needed,
            "estimated_duration_sec": self.estimated_duration_sec,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "status": self.status,
        }


@dataclass
class ResearchPlan:
    """
    Complete research plan generated by the orchestrator.

    Shown to user for confirmation before execution.
    """
    research_id: str
    title: str
    query: str
    objective: str
    sections: List[ResearchSection] = field(default_factory=list)
    estimated_duration_min: int = 5
    symbols: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    confirmed: bool = False
    confirmed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "research_id": self.research_id,
            "title": self.title,
            "query": self.query,
            "objective": self.objective,
            "sections": [s.to_dict() for s in self.sections],
            "estimated_duration_min": self.estimated_duration_min,
            "symbols": self.symbols,
            "created_at": self.created_at.isoformat(),
            "confirmed": self.confirmed,
            "confirmed_at": self.confirmed_at.isoformat() if self.confirmed_at else None,
        }

    def to_markdown(self) -> str:
        """Generate markdown representation for user display."""
        lines = [
            f"# {self.title}",
            "",
            f"**Objective:** {self.objective}",
            "",
            f"**Symbols:** {', '.join(self.symbols) if self.symbols else 'N/A'}",
            "",
            f"**Estimated Duration:** {self.estimated_duration_min} minutes",
            "",
            "## Research Sections",
            "",
        ]

        for section in self.sections:
            role_name = section.worker_role.value if isinstance(section.worker_role, WorkerRole) else section.worker_role
            lines.extend([
                f"### {section.id}. {section.name}",
                f"- **Description:** {section.description}",
                f"- **Analyst:** {role_name.replace('_', ' ').title()}",
                f"- **Tools:** {', '.join(section.tools_needed) if section.tools_needed else 'Standard toolkit'}",
                "",
            ])

        return "\n".join(lines)


# ============================================================================
# WORKER MODELS
# ============================================================================

@dataclass
class WorkerTask:
    """
    Task assigned to a worker agent.

    Contains everything the worker needs to execute their research.
    """
    worker_id: str
    section_id: int
    section_name: str
    role: WorkerRole
    prompt: str
    symbols: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context from other workers
    max_iterations: int = 5
    timeout_sec: int = 120

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "section_id": self.section_id,
            "section_name": self.section_name,
            "role": self.role.value if isinstance(self.role, WorkerRole) else self.role,
            "prompt": self.prompt,
            "symbols": self.symbols,
            "tools": self.tools,
            "max_iterations": self.max_iterations,
            "timeout_sec": self.timeout_sec,
        }


@dataclass
class WorkerResult:
    """
    Result from a worker agent's execution.
    """
    worker_id: str
    section_id: int
    section_name: str
    success: bool
    findings: str = ""  # Markdown content
    data: Dict[str, Any] = field(default_factory=dict)  # Structured data
    sources: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List['Artifact'] = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: int = 0
    iterations_used: int = 0
    tools_called: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "section_id": self.section_id,
            "section_name": self.section_name,
            "success": self.success,
            "findings": self.findings,
            "data": self.data,
            "sources": self.sources,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "error": self.error,
            "duration_ms": self.duration_ms,
            "iterations_used": self.iterations_used,
            "tools_called": self.tools_called,
        }


# ============================================================================
# ARTIFACT MODELS
# ============================================================================

@dataclass
class Artifact:
    """
    An artifact emitted during research for user transparency.

    Artifacts are streamed to the frontend in real-time so users can
    see the research progress and intermediate findings.
    """
    artifact_id: str
    artifact_type: ArtifactType
    title: str
    content: str  # Markdown or JSON string
    worker_id: Optional[str] = None
    section_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(
        cls,
        artifact_type: ArtifactType,
        title: str,
        content: str,
        worker_id: Optional[str] = None,
        section_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'Artifact':
        """Factory method to create artifact with auto-generated ID."""
        return cls(
            artifact_id=f"art_{uuid.uuid4().hex[:12]}",
            artifact_type=artifact_type,
            title=title,
            content=content,
            worker_id=worker_id,
            section_id=section_id,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value if isinstance(self.artifact_type, ArtifactType) else self.artifact_type,
            "title": self.title,
            "content": self.content,
            "worker_id": self.worker_id,
            "section_id": self.section_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# FINAL RESULT MODEL
# ============================================================================

@dataclass
class DeepResearchResult:
    """
    Complete result of a deep research session.
    """
    research_id: str
    query: str
    plan: ResearchPlan
    worker_results: List[WorkerResult] = field(default_factory=list)
    final_report: str = ""  # Full markdown report
    executive_summary: str = ""  # Brief summary
    key_findings: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    total_duration_ms: int = 0
    state: AgentState = AgentState.COMPLETED
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "research_id": self.research_id,
            "query": self.query,
            "plan": self.plan.to_dict(),
            "worker_results": [w.to_dict() for w in self.worker_results],
            "final_report": self.final_report,
            "executive_summary": self.executive_summary,
            "key_findings": self.key_findings,
            "sources": self.sources,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "total_duration_ms": self.total_duration_ms,
            "state": self.state.value,
            "error": self.error,
        }


# ============================================================================
# CONFIGURATION MODELS
# ============================================================================

@dataclass
class DeepResearchConfig:
    """
    Configuration for deep research execution.
    """
    # Model configuration
    model_tier: str = "budget"  # budget, standard, premium (default: budget for cost savings)
    lead_model: str = "gpt-4o-mini"
    worker_model: str = "gpt-4o-mini"
    synthesis_model: str = "gpt-4o-mini"

    # Execution limits
    max_workers: int = 4
    max_iterations_per_worker: int = 5
    worker_timeout_sec: int = 120
    total_timeout_sec: int = 600  # 10 minutes

    # Features
    skip_clarification: bool = False
    auto_confirm_plan: bool = False
    stream_artifacts: bool = True
    include_sources: bool = True

    # Quality settings
    min_sources_per_section: int = 2
    require_data_validation: bool = True

    def get_model_for_tier(self, role: str) -> str:
        """Get model based on tier and role."""
        tier_config = {
            "budget": {
                "lead": "gpt-4o-mini",
                "worker": "gpt-4o-mini",
                "synthesis": "gpt-4o-mini",
            },
            "standard": {
                "lead": "gpt-4o",
                "worker": "gpt-4o-mini",
                "synthesis": "gpt-4o",
            },
            "premium": {
                "lead": "gpt-4o",
                "worker": "gpt-4o",
                "synthesis": "gpt-4o",
            },
            "enterprise": {
                "lead": "o1",
                "worker": "gpt-4o",
                "synthesis": "o1",
            },
        }

        return tier_config.get(self.model_tier, tier_config["standard"]).get(role, "gpt-4o-mini")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_tier": self.model_tier,
            "lead_model": self.lead_model,
            "worker_model": self.worker_model,
            "synthesis_model": self.synthesis_model,
            "max_workers": self.max_workers,
            "max_iterations_per_worker": self.max_iterations_per_worker,
            "worker_timeout_sec": self.worker_timeout_sec,
            "total_timeout_sec": self.total_timeout_sec,
            "skip_clarification": self.skip_clarification,
            "auto_confirm_plan": self.auto_confirm_plan,
            "stream_artifacts": self.stream_artifacts,
        }


# ============================================================================
# SSE EVENT TYPES (for streaming)
# ============================================================================

class DeepResearchEventType(str, Enum):
    """
    SSE event types for deep research streaming.
    """
    # Lifecycle events
    RESEARCH_INIT = "research_init"
    RESEARCH_COMPLETED = "research_completed"
    RESEARCH_FAILED = "research_failed"
    RESEARCH_CANCELLED = "research_cancelled"

    # Clarification flow
    CLARIFICATION_REQUEST = "clarification_request"
    CLARIFICATION_RECEIVED = "clarification_received"

    # Planning flow
    PLAN_CREATED = "plan_created"
    PLAN_CONFIRMED = "plan_confirmed"
    PLAN_MODIFIED = "plan_modified"

    # Worker events
    WORKER_SPAWNED = "worker_spawned"
    WORKER_PROGRESS = "worker_progress"
    WORKER_ARTIFACT = "worker_artifact"
    WORKER_COMPLETED = "worker_completed"
    WORKER_FAILED = "worker_failed"

    # Synthesis events
    SYNTHESIS_STARTED = "synthesis_started"
    SYNTHESIS_PROGRESS = "synthesis_progress"
    SYNTHESIS_ARTIFACT = "synthesis_artifact"

    # General events
    ARTIFACT = "artifact"
    PROGRESS = "progress"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
