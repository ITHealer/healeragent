"""
SKILL.md Workflow System

Provides workflow-level guidance for complex financial analysis tasks.
Works ON TOP of the existing Python skill system (StockSkill, CryptoSkill, etc.),
adding step-by-step methodology for specific analysis types.

Architecture:
    - Python Skills (skill_base.py): Domain-level prompts (stock vs crypto)
    - SKILL.md Workflows: Task-level methodology (DCF, portfolio optimization, etc.)

Two integration points:
    1. Auto-inject: IntentClassifier detects analysis type → inject SKILL.md into system prompt
    2. Tool-invoke: Agent calls invoke_workflow tool mid-conversation when needed
"""

from src.agents.skills.workflows.registry import (
    WorkflowRegistry,
    get_workflow_registry,
)
from src.agents.skills.workflows.loader import (
    WorkflowDefinition,
    load_workflow_from_path,
    extract_workflow_metadata,
)

__all__ = [
    "WorkflowRegistry",
    "get_workflow_registry",
    "WorkflowDefinition",
    "load_workflow_from_path",
    "extract_workflow_metadata",
]
