"""
Deep Research Mode - Multi-Agent Research System

Implements a multi-agent architecture for comprehensive research tasks:
- Orchestrator (Lead Agent): Coordinates the entire research process
- Workers: Specialized agents for different research aspects
- Synthesis: Combines worker outputs into coherent reports

Architecture inspired by OpenManus patterns:
- think()/act() separation for clarity
- State machine for tracking progress
- Stuck detection to avoid infinite loops
- Artifact streaming for transparency

Usage:
    from src.agents.deep_research import DeepResearchOrchestrator

    orchestrator = DeepResearchOrchestrator()
    async for event in orchestrator.run_research(
        query="Analyze NVDA for long-term investment",
        user_id=123,
    ):
        yield event
"""

from src.agents.deep_research.models import (
    AgentState,
    ResearchPlan,
    ResearchSection,
    WorkerTask,
    WorkerResult,
    Artifact,
    ArtifactType,
    DeepResearchResult,
    ClarificationQuestion,
    ClarificationResponse,
)

__all__ = [
    "AgentState",
    "ResearchPlan",
    "ResearchSection",
    "WorkerTask",
    "WorkerResult",
    "Artifact",
    "ArtifactType",
    "DeepResearchResult",
    "ClarificationQuestion",
    "ClarificationResponse",
]
