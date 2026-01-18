"""
Deep Research Prompts Module

Contains all prompt templates for deep research agents:
- Clarification prompts
- Planning prompts
- Worker prompts
- Synthesis prompts
"""

from src.agents.deep_research.prompts.clarification import (
    CLARIFICATION_SYSTEM_PROMPT,
    generate_clarification_prompt,
)
from src.agents.deep_research.prompts.planning import (
    PLANNING_SYSTEM_PROMPT,
    generate_planning_prompt,
)
from src.agents.deep_research.prompts.synthesis import (
    SYNTHESIS_SYSTEM_PROMPT,
    generate_synthesis_prompt,
)

__all__ = [
    "CLARIFICATION_SYSTEM_PROMPT",
    "generate_clarification_prompt",
    "PLANNING_SYSTEM_PROMPT",
    "generate_planning_prompt",
    "SYNTHESIS_SYSTEM_PROMPT",
    "generate_synthesis_prompt",
]
