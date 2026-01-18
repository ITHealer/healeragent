"""
Character Router

Routes requests to appropriate character personas and builds
character-injected system prompts for the Agent Loop.

This is a simple, stateless router that:
1. Looks up character personas
2. Builds system prompts with character personality injected
3. Provides character metadata for API responses

The actual conversation execution is handled by the existing Agent Loop.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.agents.characters.personas import (
    CHARACTER_PERSONAS,
    CharacterPersona,
    get_persona,
    list_personas,
    list_persona_ids,
)
from src.utils.logger.custom_logging import LoggerMixin


@dataclass
class CharacterInfo:
    """Simplified character info for API responses."""
    character_id: str
    name: str
    title: str
    description: str
    avatar_url: str
    investment_style: str
    specialties: List[str]
    metric_focus: List[str]
    time_horizon: str
    risk_tolerance: str


class CharacterRouter(LoggerMixin):
    """
    Routes requests to character personas and builds system prompts.

    This router is stateless and simply:
    1. Provides character lookup functionality
    2. Builds character-injected system prompts
    3. Returns character metadata for responses

    Usage:
        router = CharacterRouter()

        # Get character info
        character = router.get_character("warren_buffett")

        # Build system prompt with character persona
        system_prompt = router.build_system_prompt(
            character_id="warren_buffett",
            base_prompt="You are a helpful financial assistant."
        )
    """

    def __init__(self):
        super().__init__()
        self._personas = CHARACTER_PERSONAS
        self.logger.info(
            f"CharacterRouter initialized with {len(self._personas)} characters: "
            f"{list(self._personas.keys())}"
        )

    def get_character(self, character_id: str) -> Optional[CharacterPersona]:
        """
        Get a character persona by ID.

        Args:
            character_id: The character identifier (e.g., "warren_buffett")

        Returns:
            CharacterPersona if found, None otherwise
        """
        return self._personas.get(character_id)

    def get_character_info(self, character_id: str) -> Optional[CharacterInfo]:
        """
        Get simplified character info for API responses.

        Args:
            character_id: The character identifier

        Returns:
            CharacterInfo if found, None otherwise
        """
        persona = self.get_character(character_id)
        if not persona:
            return None

        return CharacterInfo(
            character_id=persona.character_id,
            name=persona.name,
            title=persona.title,
            description=persona.description,
            avatar_url=persona.avatar_url,
            investment_style=persona.investment_style.value,
            specialties=persona.specialties,
            metric_focus=persona.metric_focus,
            time_horizon=persona.time_horizon,
            risk_tolerance=persona.risk_tolerance,
        )

    def list_characters(self) -> List[CharacterInfo]:
        """
        List all available characters with their info.

        Returns:
            List of CharacterInfo objects
        """
        characters = []
        for character_id in self._personas:
            info = self.get_character_info(character_id)
            if info:
                characters.append(info)
        return characters

    def list_character_ids(self) -> List[str]:
        """
        List all available character IDs.

        Returns:
            List of character ID strings
        """
        return list(self._personas.keys())

    def character_exists(self, character_id: str) -> bool:
        """
        Check if a character exists.

        Args:
            character_id: The character identifier

        Returns:
            True if character exists, False otherwise
        """
        return character_id in self._personas

    def build_system_prompt(
        self,
        character_id: str,
        base_prompt: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> str:
        """
        Build a complete system prompt with character persona injected.

        The resulting prompt structure:
        1. Character persona (personality, philosophy, metrics focus)
        2. Base prompt (if provided)
        3. Additional context (if provided)

        Args:
            character_id: The character to use
            base_prompt: Optional base system prompt to append
            additional_context: Optional additional context to append

        Returns:
            Complete system prompt string

        Raises:
            ValueError: If character_id is not found
        """
        persona = self.get_character(character_id)
        if not persona:
            raise ValueError(f"Character '{character_id}' not found")

        # Build the prompt parts
        parts = []

        # 1. Character persona (main content)
        parts.append(persona.system_prompt)

        # 2. Base prompt (optional)
        if base_prompt:
            parts.append(f"\n## ADDITIONAL INSTRUCTIONS\n{base_prompt}")

        # 3. Additional context (optional)
        if additional_context:
            parts.append(f"\n## CONTEXT\n{additional_context}")

        return "\n".join(parts)

    def get_character_metadata(self, character_id: str) -> Dict[str, Any]:
        """
        Get character metadata for storing in memory/responses.

        Args:
            character_id: The character identifier

        Returns:
            Dictionary with character metadata
        """
        persona = self.get_character(character_id)
        if not persona:
            return {"character_id": character_id, "error": "Character not found"}

        return {
            "character_id": persona.character_id,
            "character_name": persona.name,
            "character_title": persona.title,
            "investment_style": persona.investment_style.value,
            "metric_focus": persona.metric_focus,
            "time_horizon": persona.time_horizon,
        }

    def get_metric_focus(self, character_id: str) -> List[str]:
        """
        Get the metrics this character focuses on.

        Useful for understanding what data points the character
        will emphasize in their analysis.

        Args:
            character_id: The character identifier

        Returns:
            List of metric names the character focuses on
        """
        persona = self.get_character(character_id)
        if not persona:
            return []
        return persona.metric_focus

    def get_famous_quotes(self, character_id: str) -> List[str]:
        """
        Get famous quotes for a character.

        Args:
            character_id: The character identifier

        Returns:
            List of famous quotes
        """
        persona = self.get_character(character_id)
        if not persona:
            return []
        return persona.famous_quotes


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_character_router: Optional[CharacterRouter] = None


def get_character_router() -> CharacterRouter:
    """
    Get the singleton CharacterRouter instance.

    Returns:
        CharacterRouter singleton instance
    """
    global _character_router
    if _character_router is None:
        _character_router = CharacterRouter()
    return _character_router
