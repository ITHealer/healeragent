"""
Character Agent System

This module provides investment character personas that can be injected into
the Agent Loop to provide personality-driven investment analysis.

The system is designed to be simple and maintainable:
- Characters are defined as prompts with specific metric focuses
- The existing Agent Loop handles all tool calling and execution
- Memory system tracks character-specific context

Usage:
    from src.agents.characters import CharacterRouter, get_character_router

    router = get_character_router()
    character = router.get_character("warren_buffett")
    system_prompt = router.build_system_prompt("warren_buffett", base_prompt)
"""

from src.agents.characters.personas import (
    CHARACTER_PERSONAS,
    CharacterPersona,
    InvestmentStyle,
)
from src.agents.characters.router import (
    CharacterRouter,
    get_character_router,
)

__all__ = [
    # Personas
    "CHARACTER_PERSONAS",
    "CharacterPersona",
    "InvestmentStyle",
    # Router
    "CharacterRouter",
    "get_character_router",
]
