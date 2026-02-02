"""
SKILL.md Loader - Parse workflow definition files

Parses YAML frontmatter + Markdown body from SKILL.md files.

SKILL.md format:
    ---
    name: dcf-valuation
    description: Performs DCF valuation analysis...
    triggers:
      - fair value
      - intrinsic value
      - DCF
      - valuation
    tools_hint:
      - getCashFlow
      - getFinancialRatios
    max_tokens: 2000
    ---

    # DCF Valuation Workflow

    ## Step 1: Gather Financial Data
    ...
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger("workflow.loader")


@dataclass
class WorkflowMetadata:
    """Lightweight metadata from SKILL.md frontmatter (for discovery)."""
    name: str
    description: str
    path: str  # Absolute path to SKILL.md
    triggers: List[str] = field(default_factory=list)
    tools_hint: List[str] = field(default_factory=list)
    max_tokens: int = 2000


@dataclass
class WorkflowDefinition:
    """Full workflow definition including instructions."""
    name: str
    description: str
    path: str
    triggers: List[str] = field(default_factory=list)
    tools_hint: List[str] = field(default_factory=list)
    instructions: str = ""  # Full markdown body
    max_tokens: int = 2000
    # Supplementary files (e.g., sector_wacc.md)
    supplements: Dict[str, str] = field(default_factory=dict)


def _parse_frontmatter(content: str) -> tuple:
    """
    Parse YAML-like frontmatter from SKILL.md content.

    Returns (metadata_dict, body_text).
    Uses simple regex parsing instead of PyYAML to avoid dependency.
    """
    # Match frontmatter block
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
    if not match:
        # No frontmatter - treat entire content as body
        return {}, content

    frontmatter_str = match.group(1)
    body = match.group(2)

    # Simple YAML parser for flat key-value pairs and lists
    metadata: Dict[str, Any] = {}
    current_key = None
    current_list: Optional[List[str]] = None

    for line in frontmatter_str.split("\n"):
        line = line.rstrip()

        # List item (indented with -)
        if line.strip().startswith("- ") and current_key:
            value = line.strip()[2:].strip()
            if current_list is not None:
                current_list.append(value)
            continue

        # Key-value pair
        kv_match = re.match(r'^(\w+)\s*:\s*(.*)', line)
        if kv_match:
            key = kv_match.group(1)
            value = kv_match.group(2).strip()

            if value:
                # Inline value
                metadata[key] = value
                current_key = key
                current_list = None
            else:
                # List follows
                current_key = key
                current_list = []
                metadata[key] = current_list

    return metadata, body


def extract_workflow_metadata(filepath: str) -> Optional[WorkflowMetadata]:
    """
    Extract metadata from a SKILL.md file without loading full instructions.

    Args:
        filepath: Path to SKILL.md file

    Returns:
        WorkflowMetadata or None if parsing fails
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return None

        content = path.read_text(encoding="utf-8")
        metadata, _ = _parse_frontmatter(content)

        name = metadata.get("name", path.parent.name)
        description = metadata.get("description", "")

        if not name:
            return None

        triggers = metadata.get("triggers", [])
        if isinstance(triggers, str):
            triggers = [t.strip() for t in triggers.split(",")]

        tools_hint = metadata.get("tools_hint", [])
        if isinstance(tools_hint, str):
            tools_hint = [t.strip() for t in tools_hint.split(",")]

        max_tokens = int(metadata.get("max_tokens", 2000))

        return WorkflowMetadata(
            name=name,
            description=description,
            path=str(path.absolute()),
            triggers=triggers,
            tools_hint=tools_hint,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.warning(f"Failed to parse workflow metadata from {filepath}: {e}")
        return None


def load_workflow_from_path(filepath: str) -> Optional[WorkflowDefinition]:
    """
    Load full workflow definition from a SKILL.md file.

    Also loads supplementary .md files from the same directory.

    Args:
        filepath: Path to SKILL.md file

    Returns:
        WorkflowDefinition or None if loading fails
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return None

        content = path.read_text(encoding="utf-8")
        metadata, body = _parse_frontmatter(content)

        name = metadata.get("name", path.parent.name)
        description = metadata.get("description", "")

        triggers = metadata.get("triggers", [])
        if isinstance(triggers, str):
            triggers = [t.strip() for t in triggers.split(",")]

        tools_hint = metadata.get("tools_hint", [])
        if isinstance(tools_hint, str):
            tools_hint = [t.strip() for t in tools_hint.split(",")]

        max_tokens = int(metadata.get("max_tokens", 2000))

        # Load supplementary files (other .md files in same directory)
        supplements = {}
        parent_dir = path.parent
        for md_file in parent_dir.glob("*.md"):
            if md_file.name.lower() != "skill.md":
                try:
                    sup_content = md_file.read_text(encoding="utf-8")
                    supplements[md_file.stem] = sup_content
                except Exception:
                    pass

        # Inline supplement references in body
        # Replace [ref](filename.md) with actual content
        for sup_name, sup_content in supplements.items():
            # Replace markdown links to supplements with inline content
            pattern = rf'\[.*?\]\({re.escape(sup_name)}\.md\)'
            if re.search(pattern, body):
                body = re.sub(
                    pattern,
                    f"(see {sup_name} section below)",
                    body,
                )
                body += f"\n\n## {sup_name.replace('_', ' ').title()}\n\n{sup_content}"

        return WorkflowDefinition(
            name=name,
            description=description,
            path=str(path.absolute()),
            triggers=triggers,
            tools_hint=tools_hint,
            instructions=body.strip(),
            max_tokens=max_tokens,
            supplements=supplements,
        )
    except Exception as e:
        logger.error(f"Failed to load workflow from {filepath}: {e}")
        return None
