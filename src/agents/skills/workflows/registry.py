"""
Workflow Registry - Discover, Cache, and Match SKILL.md Workflows

Scans builtin workflow directories for SKILL.md files, caches metadata,
and provides matching logic to find relevant workflows for a given query.

Matching strategies:
1. Trigger-based: Check if query contains any trigger keywords
2. Analysis-type: Map intent classifier's analysis_type to workflow name
3. Tool-based: invoke_workflow tool call (explicit agent decision)

Thread Safety:
    Registry uses lazy initialization with cached metadata.
    Workflow instructions are loaded on-demand (not at startup).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.agents.skills.workflows.loader import (
    WorkflowMetadata,
    WorkflowDefinition,
    extract_workflow_metadata,
    load_workflow_from_path,
)


logger = logging.getLogger("workflow.registry")


# ============================================================================
# ANALYSIS TYPE → WORKFLOW MAPPING
# ============================================================================

# Maps IntentClassifier analysis_type to workflow names
ANALYSIS_TYPE_WORKFLOW_MAP: Dict[str, str] = {
    "valuation": "dcf-valuation",
    "portfolio": "portfolio-analysis",
    "technical_deep": "technical-deep-dive",
    # crypto_fundamental mapped separately based on market_type
}


class WorkflowRegistry:
    """
    Singleton registry for SKILL.md workflows.

    Scans builtin directories at initialization, caches metadata,
    and loads full instructions on demand.
    """

    _instance: Optional["WorkflowRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "WorkflowRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if WorkflowRegistry._initialized:
            return

        self.logger = logging.getLogger("workflow.registry")
        self._metadata_cache: Dict[str, WorkflowMetadata] = {}
        self._definition_cache: Dict[str, WorkflowDefinition] = {}

        # Scan builtin workflows
        self._scan_builtin_workflows()

        WorkflowRegistry._initialized = True

    def _scan_builtin_workflows(self) -> None:
        """Scan builtin workflow directories for SKILL.md files."""
        builtin_dir = Path(__file__).parent / "builtin"
        if not builtin_dir.exists():
            self.logger.warning(f"Builtin workflows directory not found: {builtin_dir}")
            return

        count = 0
        for skill_dir in builtin_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue

            metadata = extract_workflow_metadata(str(skill_file))
            if metadata:
                self._metadata_cache[metadata.name] = metadata
                count += 1
                self.logger.debug(f"Discovered workflow: {metadata.name}")

        self.logger.info(
            f"[WORKFLOW_REGISTRY] Discovered {count} builtin workflows: "
            f"{list(self._metadata_cache.keys())}"
        )

    def get_workflow(self, name: str) -> Optional[WorkflowDefinition]:
        """
        Get full workflow definition by name (lazy-loads instructions).

        Args:
            name: Workflow name (e.g., "dcf-valuation")

        Returns:
            WorkflowDefinition or None if not found
        """
        # Check definition cache first
        if name in self._definition_cache:
            return self._definition_cache[name]

        # Check metadata cache
        metadata = self._metadata_cache.get(name)
        if not metadata:
            return None

        # Load full definition
        definition = load_workflow_from_path(metadata.path)
        if definition:
            self._definition_cache[name] = definition
        return definition

    def match_query(
        self,
        query: str,
        analysis_type: Optional[str] = None,
        market_type: Optional[str] = None,
    ) -> Optional[WorkflowDefinition]:
        """
        Find the best matching workflow for a query.

        Matching priority:
        1. Analysis type mapping (from IntentClassifier)
        2. Trigger keyword matching in query text

        Args:
            query: User query text
            analysis_type: From IntentClassifier (e.g., "valuation", "portfolio")
            market_type: Market type (e.g., "stock", "crypto")

        Returns:
            Best matching WorkflowDefinition or None
        """
        # Strategy 1: Analysis type mapping
        if analysis_type:
            analysis_lower = analysis_type.lower().strip()

            # Special case: crypto + fundamental → crypto-fundamental workflow
            if market_type == "crypto" and analysis_lower in ("fundamental", "basic"):
                workflow = self.get_workflow("crypto-fundamental")
                if workflow:
                    self.logger.info(
                        f"[WORKFLOW_MATCH] Matched via analysis_type={analysis_lower} "
                        f"+ market={market_type} → {workflow.name}"
                    )
                    return workflow

            # Standard analysis type mapping
            workflow_name = ANALYSIS_TYPE_WORKFLOW_MAP.get(analysis_lower)
            if workflow_name:
                workflow = self.get_workflow(workflow_name)
                if workflow:
                    self.logger.info(
                        f"[WORKFLOW_MATCH] Matched via analysis_type={analysis_lower} "
                        f"→ {workflow.name}"
                    )
                    return workflow

        # Strategy 2: Trigger keyword matching
        query_lower = query.lower()
        best_match: Optional[WorkflowMetadata] = None
        best_score = 0

        for metadata in self._metadata_cache.values():
            score = 0
            for trigger in metadata.triggers:
                if trigger.lower() in query_lower:
                    score += 1

            if score > best_score:
                best_score = score
                best_match = metadata

        # Require at least 1 trigger match
        if best_match and best_score >= 1:
            workflow = self.get_workflow(best_match.name)
            if workflow:
                self.logger.info(
                    f"[WORKFLOW_MATCH] Matched via triggers (score={best_score}) "
                    f"→ {workflow.name}"
                )
                return workflow

        return None

    def get_all_metadata(self) -> List[WorkflowMetadata]:
        """Get metadata for all discovered workflows."""
        return list(self._metadata_cache.values())

    def get_workflow_names(self) -> List[str]:
        """Get names of all available workflows."""
        return list(self._metadata_cache.keys())

    def build_workflow_list_for_prompt(self) -> str:
        """
        Build a compact list of available workflows for system prompt injection.

        Returns:
            Formatted string listing workflow names and descriptions
        """
        if not self._metadata_cache:
            return ""

        lines = ["## Available Workflows (invoke with invoke_workflow tool)", ""]
        for metadata in self._metadata_cache.values():
            lines.append(f"- **{metadata.name}**: {metadata.description}")
        return "\n".join(lines)

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
        cls._initialized = False


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_registry_instance: Optional[WorkflowRegistry] = None


def get_workflow_registry() -> WorkflowRegistry:
    """Get singleton WorkflowRegistry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = WorkflowRegistry()
    return _registry_instance
