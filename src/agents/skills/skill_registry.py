"""
Skill Registry - Central Management for Domain Skills

This module provides a singleton registry for managing domain-specific skills
and selecting the appropriate skill based on classification results.

Architecture:
    Classification.market_type → SkillRegistry → Selected Skill → Domain Prompt

Thread Safety:
    The registry uses a singleton pattern with lazy initialization.
    Skills are immutable after creation.
"""

import logging
from typing import Any, Dict, List, Optional

from src.agents.skills.skill_base import BaseSkill, SkillContext
from src.utils.logger.custom_logging import LoggerMixin


class SkillRegistry(LoggerMixin):
    """
    Singleton registry for domain-specific skills.

    Manages skill lifecycle and provides selection based on market type.
    Follows the Service Locator pattern for skill discovery.

    Usage:
        # Get singleton instance
        registry = SkillRegistry.get_instance()

        # Select skill based on market type
        skill = registry.select_skill("stock")

        # Get prompt for selected skill
        prompt = skill.get_full_prompt(context)

    Thread Safety:
        Singleton creation is not thread-safe by default.
        For concurrent access, ensure initialization happens at startup.
    """

    _instance: Optional["SkillRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "SkillRegistry":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initialize registry with available skills.

        Only runs once due to _initialized flag.
        """
        if SkillRegistry._initialized:
            return

        super().__init__()
        self.logger = logging.getLogger("skill.registry")

        # Import skills here to avoid circular imports
        from src.agents.skills.stock_skill import StockSkill
        from src.agents.skills.crypto_skill import CryptoSkill
        from src.agents.skills.mixed_skill import MixedSkill

        # Initialize skill instances
        self._skills: Dict[str, BaseSkill] = {
            "stock": StockSkill(),
            "crypto": CryptoSkill(),
            "mixed": MixedSkill(),
            "both": MixedSkill(),  # Alias for mixed
        }

        # Default skill when market_type is unknown
        self._default_skill_type = "stock"

        SkillRegistry._initialized = True

        self.logger.info(
            f"[SKILL_REGISTRY] Initialized with {len(self._skills)} skills: "
            f"{list(self._skills.keys())}"
        )

    @classmethod
    def get_instance(cls) -> "SkillRegistry":
        """
        Get singleton instance of SkillRegistry.

        Returns:
            SkillRegistry singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset singleton instance (for testing purposes).

        Warning: Not thread-safe. Only use in test setup/teardown.
        """
        cls._instance = None
        cls._initialized = False

    def select_skill(self, market_type: str) -> BaseSkill:
        """
        Select appropriate skill based on market type.

        Args:
            market_type: Market type from classification ("stock", "crypto", "mixed", "both")

        Returns:
            Selected BaseSkill instance

        Example:
            skill = registry.select_skill("crypto")
            # Returns CryptoSkill instance
        """
        market_type_lower = (market_type or "").lower().strip()

        if market_type_lower in self._skills:
            skill = self._skills[market_type_lower]
            self.logger.debug(
                f"[SKILL_REGISTRY] Selected {skill.name} for market_type={market_type}"
            )
            return skill

        # Fallback to default
        self.logger.warning(
            f"[SKILL_REGISTRY] Unknown market_type '{market_type}', "
            f"falling back to {self._default_skill_type}"
        )
        return self._skills[self._default_skill_type]

    def select_skill_from_classification(
        self,
        classification: Any
    ) -> BaseSkill:
        """
        Select skill directly from classification result.

        Args:
            classification: Classification result object or dict

        Returns:
            Selected BaseSkill instance
        """
        # Extract market_type from classification
        if isinstance(classification, dict):
            market_type = classification.get("market_type", "")
        else:
            market_type = getattr(classification, "market_type", "")

        return self.select_skill(market_type)

    def get_skill_prompt(
        self,
        market_type: str,
        context: Optional[SkillContext] = None
    ) -> str:
        """
        Get complete prompt for a market type.

        Args:
            market_type: Target market type
            context: Optional runtime context for dynamic generation

        Returns:
            Combined system prompt + analysis framework
        """
        skill = self.select_skill(market_type)
        return skill.get_full_prompt(context)

    def get_available_skills(self) -> List[str]:
        """
        Get list of available skill market types.

        Returns:
            List of registered market types
        """
        # Return unique skill names (exclude aliases)
        unique_skills = set()
        for skill in self._skills.values():
            unique_skills.add(skill.market_type)
        return sorted(list(unique_skills))

    def get_skill_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered skills.

        Returns:
            Dict mapping market_type to skill info
        """
        info = {}
        seen_skills = set()

        for market_type, skill in self._skills.items():
            # Skip aliases
            if skill.name in seen_skills:
                continue
            seen_skills.add(skill.name)

            info[market_type] = {
                "name": skill.name,
                "description": skill.config.description,
                "preferred_tools": skill.config.preferred_tools,
                "version": skill.config.version,
            }

        return info

    def register_skill(self, market_type: str, skill: BaseSkill) -> None:
        """
        Register a new skill or replace existing one.

        Args:
            market_type: Market type key for the skill
            skill: Skill instance to register

        Raises:
            TypeError: If skill is not a BaseSkill instance
        """
        if not isinstance(skill, BaseSkill):
            raise TypeError(
                f"Expected BaseSkill instance, got {type(skill).__name__}"
            )

        self._skills[market_type.lower()] = skill
        self.logger.info(
            f"[SKILL_REGISTRY] Registered skill {skill.name} "
            f"for market_type={market_type}"
        )

    def __repr__(self) -> str:
        return (
            f"<SkillRegistry skills={list(self._skills.keys())} "
            f"default={self._default_skill_type}>"
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def get_skill_registry() -> SkillRegistry:
    """
    Get the global SkillRegistry instance.

    Convenience function for accessing the singleton.

    Returns:
        SkillRegistry singleton instance
    """
    return SkillRegistry.get_instance()


def get_skill_for_market(market_type: str) -> BaseSkill:
    """
    Get skill for a specific market type.

    Convenience function for quick skill lookup.

    Args:
        market_type: Target market type

    Returns:
        Selected BaseSkill instance
    """
    return get_skill_registry().select_skill(market_type)


def get_domain_prompt(
    market_type: str,
    context: Optional[SkillContext] = None
) -> str:
    """
    Get domain-specific prompt for a market type.

    Convenience function for getting prompts.

    Args:
        market_type: Target market type
        context: Optional runtime context

    Returns:
        Combined domain prompt
    """
    return get_skill_registry().get_skill_prompt(market_type, context)


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    """Test skill registry functionality."""

    print("=" * 60)
    print("Testing SkillRegistry")
    print("=" * 60)

    # Get registry instance
    registry = SkillRegistry.get_instance()
    print(f"\nRegistry: {registry}")

    # Test skill selection
    print("\n--- Skill Selection ---")
    for market_type in ["stock", "crypto", "mixed", "both", "unknown"]:
        skill = registry.select_skill(market_type)
        print(f"  {market_type:10} → {skill.name}")

    # Test skill info
    print("\n--- Skill Info ---")
    info = registry.get_skill_info()
    for market_type, skill_info in info.items():
        print(f"\n  [{market_type}]")
        print(f"    Name: {skill_info['name']}")
        print(f"    Tools: {len(skill_info['preferred_tools'])} preferred")

    # Test prompt generation
    print("\n--- Prompt Generation ---")
    stock_skill = registry.select_skill("stock")
    prompt = stock_skill.get_system_prompt()
    print(f"  Stock prompt length: {len(prompt)} chars")

    crypto_skill = registry.select_skill("crypto")
    prompt = crypto_skill.get_system_prompt()
    print(f"  Crypto prompt length: {len(prompt)} chars")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
