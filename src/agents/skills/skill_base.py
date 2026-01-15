"""
Skill Base Module - Foundation for Domain-Specific Skills

This module provides the base classes and interfaces for implementing
domain-specific skills in the HealerAgent system.

Design Principles:
    1. Single Responsibility: Each skill handles one domain
    2. Open/Closed: Easy to add new skills without modifying existing code
    3. Dependency Inversion: Skills depend on abstractions, not concretions
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.logger.custom_logging import LoggerMixin


@dataclass
class SkillConfig:
    """
    Configuration for a domain skill.

    Attributes:
        name: Unique identifier for the skill (e.g., "STOCK_ANALYST")
        description: Brief description of the skill's expertise
        market_type: Target market type ("stock", "crypto", "mixed")
        preferred_tools: List of tools commonly used by this skill
        supported_languages: Languages this skill can respond in
        version: Skill version for tracking updates
    """

    name: str
    description: str
    market_type: str  # "stock", "crypto", "mixed"
    preferred_tools: List[str] = field(default_factory=list)
    supported_languages: List[str] = field(
        default_factory=lambda: ["vi", "en", "zh"]
    )
    version: str = "1.0.0"

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_market_types = {"stock", "crypto", "mixed"}
        if self.market_type not in valid_market_types:
            raise ValueError(
                f"Invalid market_type: {self.market_type}. "
                f"Must be one of {valid_market_types}"
            )


@dataclass
class SkillContext:
    """
    Runtime context passed to skill for dynamic prompt generation.

    Attributes:
        symbols: List of symbols being analyzed
        query: Original user query
        categories: Data categories available (price, technical, etc.)
        language: Detected response language
        user_preferences: User-specific preferences from memory
        current_date: Current date for time-sensitive analysis
    """

    symbols: List[str] = field(default_factory=list)
    query: str = ""
    categories: List[str] = field(default_factory=list)
    language: str = "vi"
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    current_date: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d")
    )

    @classmethod
    def from_classification(cls, classification: Any) -> "SkillContext":
        """Create context from classification result."""
        return cls(
            symbols=getattr(classification, "symbols", []),
            query="",  # Set separately
            categories=getattr(classification, "tool_categories", []),
            language=getattr(classification, "response_language", "vi"),
        )


class BaseSkill(ABC, LoggerMixin):
    """
    Abstract base class for all domain-specific skills.

    A Skill encapsulates domain expertise through:
    1. System prompt with domain-specific guidance
    2. Analysis framework tailored to the domain
    3. Few-shot examples for the domain
    4. Context enhancement with domain hints

    Subclasses must implement:
        - get_system_prompt(): Domain-specific system instructions
        - get_analysis_framework(): Structured analysis template
        - get_few_shot_examples(): Domain-specific examples

    Example:
        class StockSkill(BaseSkill):
            def get_system_prompt(self) -> str:
                return "You are a Senior Equity Analyst..."
    """

    def __init__(self, config: SkillConfig):
        """
        Initialize skill with configuration.

        Args:
            config: SkillConfig instance with skill metadata
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(f"skill.{config.name.lower()}")

    @property
    def name(self) -> str:
        """Get skill name."""
        return self.config.name

    @property
    def market_type(self) -> str:
        """Get target market type."""
        return self.config.market_type

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get domain-specific system prompt.

        Returns:
            String containing the system prompt for this domain.
            Should include:
            - Role definition
            - Expertise areas
            - Domain-specific terminology
            - Analysis principles
        """
        pass

    @abstractmethod
    def get_analysis_framework(self) -> str:
        """
        Get domain-specific analysis framework.

        Returns:
            String containing structured analysis template.
            Should define:
            - Required analysis sections
            - Output format guidelines
            - Domain-specific metrics to include
        """
        pass

    @abstractmethod
    def get_few_shot_examples(self) -> List[Dict[str, str]]:
        """
        Get domain-specific few-shot examples.

        Returns:
            List of example dicts with 'query' and 'response' keys.
            Used for in-context learning.
        """
        pass

    def get_terminology(self) -> Dict[str, Dict[str, str]]:
        """
        Get domain-specific terminology translations.

        Returns:
            Dict mapping English terms to translations in supported languages.
            Default implementation returns empty dict - override in subclasses.
        """
        return {}

    def get_full_prompt(self, context: Optional[SkillContext] = None) -> str:
        """
        Get complete prompt combining system prompt and framework.

        Args:
            context: Optional runtime context for dynamic generation

        Returns:
            Combined prompt ready for use in LLM call
        """
        parts = [
            self.get_system_prompt(),
            "\n---\n",
            self.get_analysis_framework(),
        ]

        # Add context-specific hints if available
        if context:
            context_hints = self._build_context_hints(context)
            if context_hints:
                parts.extend(["\n---\n", context_hints])

        return "\n".join(parts)

    def _build_context_hints(self, context: SkillContext) -> str:
        """
        Build context-specific hints for the prompt.

        Args:
            context: Runtime context

        Returns:
            String with context hints or empty string
        """
        hints = []

        if context.symbols:
            hints.append(f"Symbols to analyze: {', '.join(context.symbols)}")

        if context.categories:
            hints.append(f"Available data: {', '.join(context.categories)}")

        hints.append(f"Current date: {context.current_date}")
        hints.append(f"Response language: {context.language.upper()}")

        if hints:
            return "CURRENT CONTEXT:\n" + "\n".join(f"- {h}" for h in hints)
        return ""

    def enhance_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance execution context with skill-specific hints.

        Args:
            context: Original execution context

        Returns:
            Enhanced context with skill metadata
        """
        enhanced = context.copy()
        enhanced["skill_name"] = self.config.name
        enhanced["skill_market_type"] = self.config.market_type
        enhanced["preferred_tools"] = self.config.preferred_tools
        return enhanced

    def validate_tools(self, selected_tools: List[str]) -> List[str]:
        """
        Validate and potentially adjust tool selection for this skill.

        Args:
            selected_tools: Tools selected by router

        Returns:
            Validated/adjusted tool list
        """
        # Default: return as-is. Subclasses can override for domain logic
        return selected_tools

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"name={self.config.name} "
            f"market_type={self.config.market_type}>"
        )
