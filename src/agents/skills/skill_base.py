"""
Skill Base Module - Foundation for Domain-Specific Skills

This module provides the base classes and interfaces for implementing
domain-specific skills in the HealerAgent system.

Design Principles:
    1. Single Responsibility: Each skill handles one domain
    2. Open/Closed: Easy to add new skills without modifying existing code
    3. Dependency Inversion: Skills depend on abstractions, not concretions
    4. Hierarchical Synthesis: Phase-level intermediate synthesis to prevent
       information loss in large context windows (research-validated pattern)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


from src.utils.logger.custom_logging import LoggerMixin


# ============================================================================
# Phase & Plan Data Models (Hierarchical Synthesis Architecture)
# ============================================================================


class PhaseType(str, Enum):
    """Standard phase types for skill execution.

    Skills can define custom phases, but these are the common ones
    used across stock, crypto, and mixed analysis.
    """
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    CONTEXT = "context"          # news, sentiment, analyst ratings
    MARKET_STRUCTURE = "market_structure"  # crypto-specific
    COMPARISON = "comparison"    # mixed/cross-asset
    CUSTOM = "custom"


@dataclass
class Phase:
    """A single execution phase within a skill's tool plan.

    Each phase groups related tools and defines how their outputs
    should be synthesized into an intermediate summary.

    Attributes:
        name: Unique phase identifier (e.g., "technical", "fundamental")
        display_name: Human-readable name for logging/UI
        phase_type: Categorization for the phase
        tools: List of tool names to execute in this phase (parallel)
        synthesis_focus: Brief description of what this phase should summarize
        max_summary_tokens: Target length for intermediate synthesis
        priority: Execution order (lower = first). Same priority = parallel phases
        required: If False, phase can be skipped when tools return no data
    """
    name: str
    display_name: str
    phase_type: PhaseType
    tools: List[str] = field(default_factory=list)
    synthesis_focus: str = ""
    max_summary_tokens: int = 500
    priority: int = 1
    required: bool = True

    def __post_init__(self):
        if not self.name:
            raise ValueError("Phase name cannot be empty")


@dataclass
class ToolPlan:
    """Execution plan produced by a skill's plan_execution() method.

    Contains ordered phases with their tools and synthesis instructions.
    The orchestrator iterates through phases, executes tools in parallel
    within each phase, then calls intermediate synthesis before proceeding.

    Attributes:
        skill_name: Name of the skill that created this plan
        phases: Ordered list of execution phases
        final_synthesis_sections: Sections expected in the final report
        metadata: Additional plan-level metadata
    """
    skill_name: str
    phases: List[Phase] = field(default_factory=list)
    final_synthesis_sections: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tools(self) -> int:
        return sum(len(p.tools) for p in self.phases)

    @property
    def phase_count(self) -> int:
        return len(self.phases)

    def get_all_tools(self) -> List[str]:
        """Get flat list of all tools across all phases."""
        tools = []
        for phase in self.phases:
            tools.extend(phase.tools)
        return tools

    def get_phases_by_priority(self) -> Dict[int, List[Phase]]:
        """Group phases by priority for execution ordering."""
        groups: Dict[int, List[Phase]] = {}
        for phase in self.phases:
            groups.setdefault(phase.priority, []).append(phase)
        return dict(sorted(groups.items()))


@dataclass
class PhaseSummary:
    """Result of intermediate synthesis for a single phase.

    Stored in state and passed to final synthesis to prevent
    information loss from large context windows.

    Attributes:
        phase_name: Which phase produced this summary
        phase_type: Phase categorization
        summary: The synthesized text (target: 300-500 tokens)
        tool_count: Number of tools that contributed data
        tools_used: Names of tools that returned data
        token_estimate: Approximate token count of the summary
        raw_token_estimate: Approximate tokens of raw tool outputs (before synthesis)
        success: Whether synthesis completed successfully
    """
    phase_name: str
    phase_type: PhaseType
    summary: str
    tool_count: int = 0
    tools_used: List[str] = field(default_factory=list)
    token_estimate: int = 0
    raw_token_estimate: int = 0
    success: bool = True

    def __post_init__(self):
        if not self.token_estimate and self.summary:
            # Rough estimate: 1 token ~= 4 chars
            self.token_estimate = len(self.summary) // 4


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
        enable_hierarchical_synthesis: Whether to use phase-level synthesis
    """

    name: str
    description: str
    market_type: str  # "stock", "crypto", "mixed"
    preferred_tools: List[str] = field(default_factory=list)
    supported_languages: List[str] = field(
        default_factory=lambda: ["vi", "en", "zh"]
    )
    version: str = "1.0.0"
    enable_hierarchical_synthesis: bool = True

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
        analysis_type: Type of analysis requested (basic, technical, etc.)
    """

    symbols: List[str] = field(default_factory=list)
    query: str = ""
    categories: List[str] = field(default_factory=list)
    language: str = "vi"
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    current_date: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d")
    )
    analysis_type: str = "general"

    @classmethod
    def from_classification(cls, classification: Any) -> "SkillContext":
        """Create context from classification result."""
        return cls(
            symbols=getattr(classification, "symbols", []),
            query="",  # Set separately
            categories=getattr(classification, "tool_categories", []),
            language=getattr(classification, "response_language", "vi"),
            analysis_type=getattr(classification, "analysis_type", "general"),
        )


class BaseSkill(ABC, LoggerMixin):
    """
    Abstract base class for all domain-specific skills.

    A Skill encapsulates domain expertise through:
    1. System prompt with domain-specific guidance
    2. Analysis framework tailored to the domain
    3. Few-shot examples for the domain
    4. Context enhancement with domain hints
    5. **Hierarchical synthesis**: Phase-level tool plans and intermediate
       synthesis prompts to prevent information loss

    Subclasses must implement:
        - get_system_prompt(): Domain-specific system instructions
        - get_analysis_framework(): Structured analysis template
        - get_few_shot_examples(): Domain-specific examples

    Subclasses should override for hierarchical synthesis:
        - get_phases(): Define execution phases with tools
        - get_phase_synthesis_prompt(): Intermediate synthesis prompt per phase
        - get_final_synthesis_prompt(): Final synthesis combining phase summaries

    Example:
        class StockSkill(BaseSkill):
            def get_system_prompt(self) -> str:
                return "You are a Senior Equity Analyst..."

            def get_phases(self, context) -> List[Phase]:
                return [
                    Phase(name="technical", ...),
                    Phase(name="fundamental", ...),
                ]
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

    @property
    def supports_hierarchical_synthesis(self) -> bool:
        """Check if this skill supports hierarchical synthesis."""
        return self.config.enable_hierarchical_synthesis

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

    # ========================================================================
    # Hierarchical Synthesis Interface
    # ========================================================================

    def get_phases(self, context: Optional[SkillContext] = None) -> List[Phase]:
        """
        Define execution phases with their tools for hierarchical synthesis.

        Override in subclasses to define domain-specific phases.
        Default returns empty list (no hierarchical synthesis).

        Args:
            context: Runtime context for dynamic phase construction

        Returns:
            Ordered list of Phase objects
        """
        return []

    def plan_execution(self, context: Optional[SkillContext] = None) -> ToolPlan:
        """
        Create a complete execution plan with phases and tools.

        Uses get_phases() to build the plan. Override get_phases() in
        subclasses to customize.

        Args:
            context: Runtime context for dynamic plan construction

        Returns:
            ToolPlan with phases, tools, and synthesis instructions
        """
        phases = self.get_phases(context)

        if not phases:
            # Fallback: single phase with all preferred tools
            phases = [
                Phase(
                    name="default",
                    display_name="Data Collection",
                    phase_type=PhaseType.CUSTOM,
                    tools=self.config.preferred_tools,
                    synthesis_focus="Synthesize all gathered data",
                    priority=1,
                )
            ]

        return ToolPlan(
            skill_name=self.config.name,
            phases=phases,
            final_synthesis_sections=self._get_final_sections(context),
            metadata={
                "market_type": self.config.market_type,
                "version": self.config.version,
            },
        )

    def get_phase_synthesis_prompt(
        self,
        phase: Phase,
        context: Optional[SkillContext] = None,
    ) -> str:
        """
        Get the intermediate synthesis prompt for a specific phase.

        This prompt instructs the LLM to create a focused, structured summary
        from the tool outputs of a single phase. Override in subclasses for
        domain-specific synthesis instructions.

        Args:
            phase: The phase being synthesized
            context: Runtime context

        Returns:
            Synthesis prompt string for the intermediate LLM call
        """
        language_hint = ""
        if context and context.language:
            language_hint = f"\nRespond in {context.language.upper()} language."

        symbols_hint = ""
        if context and context.symbols:
            symbols_hint = f"\nSymbols being analyzed: {', '.join(context.symbols)}"

        return (
            f"You are a {self.config.description}.\n\n"
            f"Synthesize ONLY the {phase.display_name} data from the tool outputs below.\n"
            f"Create a STRUCTURED summary with key metrics and insights.\n"
            f"Focus: {phase.synthesis_focus}\n"
            f"\nRules:\n"
            f"- Use bullet points for metrics with exact numbers\n"
            f"- Maximum ~{phase.max_summary_tokens} tokens\n"
            f"- Include ALL important data points - do not skip any\n"
            f"- Preserve numerical precision (don't round excessively)\n"
            f"- If data includes tables with multiple periods, preserve the comparison\n"
            f"- End with a 1-sentence outlook/assessment"
            f"{symbols_hint}"
            f"{language_hint}"
        )

    def get_final_synthesis_prompt(
        self,
        phase_summaries: Dict[str, PhaseSummary],
        context: Optional[SkillContext] = None,
    ) -> str:
        """
        Get the final synthesis prompt that combines all phase summaries.

        This prompt receives only the compressed phase summaries (~1500 tokens total)
        instead of raw tool outputs (~140K tokens), preventing information loss.

        Args:
            phase_summaries: Dict mapping phase_name to PhaseSummary
            context: Runtime context

        Returns:
            Final synthesis prompt string
        """
        language_hint = ""
        if context and context.language:
            language_hint = f"Language: {context.language.upper()}"

        query_hint = ""
        if context and context.query:
            query_hint = f"User's question: {context.query}"

        # Build summaries section
        summaries_text = ""
        for phase_name, summary in phase_summaries.items():
            summaries_text += (
                f"\n### {summary.phase_name.upper()} SUMMARY\n"
                f"{summary.summary}\n"
            )

        # Build sections instruction
        sections = self._get_final_sections(context)
        sections_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sections))

        return (
            f"You are a {self.config.description}.\n\n"
            f"{query_hint}\n"
            f"{language_hint}\n\n"
            f"Create a comprehensive analysis using these data summaries:\n"
            f"{summaries_text}\n\n"
            f"Structure your response with these sections:\n"
            f"{sections_text}\n\n"
            f"Rules:\n"
            f"- Use ALL data from every summary - do not skip any section\n"
            f"- Cite exact numbers from the summaries\n"
            f"- Be specific with metrics, not vague\n"
            f"- If some data was unavailable, acknowledge it briefly\n"
            f"- NEVER mention internal tool names\n"
            f"- Present data naturally: 'AAPL is at $259' not 'The tool returned...'\n"
        )

    def _get_final_sections(self, context: Optional[SkillContext] = None) -> List[str]:
        """
        Get the expected sections for the final synthesis report.

        Override in subclasses for domain-specific sections.

        Returns:
            List of section names/descriptions
        """
        return [
            "Executive Summary (2-3 sentences, key takeaway + verdict)",
            "Detailed Analysis (expand on each data summary)",
            "Risk Assessment (key risks and mitigations)",
            "Recommendation (clear action with rationale)",
        ]

    # ========================================================================
    # Existing Interface (Backward Compatible)
    # ========================================================================

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
