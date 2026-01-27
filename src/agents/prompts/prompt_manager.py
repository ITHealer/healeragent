"""
Prompt Manager - Model-Aware Prompt Selection

This module provides a centralized manager for selecting and composing
prompts based on the LLM provider and model being used.

Usage:
    manager = PromptManager()

    # Get system prompt for a specific provider/model
    prompt = manager.get_system_prompt(
        provider="openai",
        model="gpt-5",
        context=PromptContext(...)
    )

    # Get synthesis prompt for combining tool results
    synthesis = manager.get_synthesis_prompt(
        provider="gemini",
        model="gemini-2.5-flash",
        context=PromptContext(...),
        tool_results="...",
        web_citations=[...]
    )
"""

from typing import Dict, List, Optional, Type, Any
from enum import Enum

from .base_prompt import BasePromptTemplate, PromptContext, ResponseStyle
from .openai_prompt import OpenAIPromptTemplate
from .gemini_prompt import GeminiPromptTemplate
from src.utils.logger.custom_logging import LoggerMixin


class ProviderType(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"  # Future support


class PromptManager(LoggerMixin):
    """
    Centralized manager for model-specific prompt selection and composition.

    Supports:
    - OpenAI: GPT-5, GPT-5-mini, GPT-5.1-mini, GPT-4o series
    - Gemini: 2.5-pro, 2.5-flash, 3-flash, 2.0-flash

    Features:
    - Automatic provider/model detection
    - Context-aware prompt composition
    - Domain-specific prompt injection
    - Synthesis prompt generation
    """

    # Registry of prompt templates by provider
    _TEMPLATES: Dict[str, Type[BasePromptTemplate]] = {
        "openai": OpenAIPromptTemplate,
        "gemini": GeminiPromptTemplate,
    }

    # Model to provider mapping for auto-detection
    _MODEL_PROVIDER_MAP: Dict[str, str] = {
        # OpenAI models
        "gpt-5": "openai",
        "gpt-5-mini": "openai",
        "gpt-5.1-mini": "openai",
        "gpt-4o": "openai",
        "gpt-4o-mini": "openai",
        "gpt-4-turbo": "openai",
        "gpt-4": "openai",
        "gpt-3.5-turbo": "openai",
        # Gemini models
        "gemini-3-flash": "gemini",
        "gemini-2.5-pro": "gemini",
        "gemini-2.5-flash": "gemini",
        "gemini-2.0-flash": "gemini",
        "gemini-pro": "gemini",
        "gemini-1.5-pro": "gemini",
        "gemini-1.5-flash": "gemini",
    }

    def __init__(self):
        """Initialize prompt manager with template instances"""
        self._template_instances: Dict[str, BasePromptTemplate] = {}

    def _get_template(self, provider: str) -> BasePromptTemplate:
        """
        Get or create template instance for a provider.

        Args:
            provider: Provider name (openai, gemini)

        Returns:
            Template instance for the provider

        Raises:
            ValueError: If provider is not supported
        """
        provider = provider.lower()

        if provider not in self._TEMPLATES:
            self.logger.warning(
                f"[PromptManager] Unknown provider '{provider}', falling back to OpenAI"
            )
            provider = "openai"

        if provider not in self._template_instances:
            template_class = self._TEMPLATES[provider]
            self._template_instances[provider] = template_class()
            self.logger.debug(f"[PromptManager] Created template instance for {provider}")

        return self._template_instances[provider]

    def detect_provider(self, model_name: str) -> str:
        """
        Auto-detect provider from model name.

        Args:
            model_name: Model name (e.g., "gpt-5", "gemini-2.5-flash")

        Returns:
            Provider name (openai, gemini)
        """
        model_lower = model_name.lower()

        # Direct lookup
        if model_lower in self._MODEL_PROVIDER_MAP:
            return self._MODEL_PROVIDER_MAP[model_lower]

        # Pattern matching
        if "gpt" in model_lower or "openai" in model_lower:
            return "openai"
        if "gemini" in model_lower or "google" in model_lower:
            return "gemini"
        if "claude" in model_lower or "anthropic" in model_lower:
            return "anthropic"

        # Default to OpenAI
        self.logger.warning(
            f"[PromptManager] Could not detect provider for '{model_name}', defaulting to OpenAI"
        )
        return "openai"

    def get_system_prompt(
        self,
        provider: str,
        model: str,
        context: PromptContext,
        domain_prompt: str = "",
        analysis_framework: str = ""
    ) -> str:
        """
        Get complete system prompt for a provider/model combination.

        Args:
            provider: LLM provider (openai, gemini)
            model: Model name
            context: Prompt context with all required information
            domain_prompt: Optional domain-specific prompt (e.g., stock analysis)
            analysis_framework: Optional analysis framework guidelines

        Returns:
            Complete system prompt string
        """
        template = self._get_template(provider)

        # Store model name in context for template use
        context.user_preferences["model_name"] = model

        prompt = template.get_system_prompt(
            context=context,
            model_name=model,
            domain_prompt=domain_prompt,
            analysis_framework=analysis_framework
        )

        self.logger.debug(
            f"[PromptManager] Generated system prompt for {provider}/{model} "
            f"(length={len(prompt)}, web_search={context.enable_web_search})"
        )

        return prompt

    def get_synthesis_prompt(
        self,
        provider: str,
        model: str,
        context: PromptContext,
        tool_results: str,
        web_citations: List[dict] = None
    ) -> str:
        """
        Get synthesis prompt for combining tool results.

        Args:
            provider: LLM provider
            model: Model name
            context: Prompt context
            tool_results: Formatted tool results string
            web_citations: List of web citation dicts (title, url)

        Returns:
            Synthesis prompt string
        """
        template = self._get_template(provider)

        return template.get_synthesis_prompt(
            context=context,
            tool_results=tool_results,
            web_citations=web_citations
        )

    def get_domain_prompt(
        self,
        provider: str,
        market_type: str = "stock"
    ) -> str:
        """
        Get domain-specific prompt for a market type.

        Args:
            provider: LLM provider
            market_type: Market type (stock, crypto, mixed)

        Returns:
            Domain prompt string
        """
        template = self._get_template(provider)

        if hasattr(template, 'get_finance_domain_prompt'):
            return template.get_finance_domain_prompt(market_type)

        # Fallback to base implementation
        return ""

    def create_context(
        self,
        language: str = "vi",
        symbols: List[str] = None,
        market_type: str = "stock",
        analysis_type: str = "general",
        enable_web_search: bool = False,
        enable_tool_search: bool = False,
        enable_think_tool: bool = False,
        response_style: str = "detailed",
        min_words: int = 500,
        max_words: int = 1500,
        user_profile: str = None,
        conversation_summary: str = None,
        **kwargs
    ) -> PromptContext:
        """
        Create a PromptContext with common defaults.

        Args:
            language: Response language (vi, en, zh)
            symbols: List of symbols being analyzed
            market_type: Market type (stock, crypto, mixed)
            analysis_type: Analysis type (basic, technical, fundamental, etc.)
            enable_web_search: Whether web search is enabled
            enable_tool_search: Whether tool search mode is enabled
            enable_think_tool: Whether think tool is enabled
            response_style: Response style (concise, detailed, narrative, technical)
            min_words: Minimum response length
            max_words: Maximum response length
            user_profile: User profile string
            conversation_summary: Conversation summary string
            **kwargs: Additional context parameters

        Returns:
            Configured PromptContext
        """
        # Map string to ResponseStyle enum
        style_map = {
            "concise": ResponseStyle.CONCISE,
            "detailed": ResponseStyle.DETAILED,
            "narrative": ResponseStyle.NARRATIVE,
            "technical": ResponseStyle.TECHNICAL,
        }
        style = style_map.get(response_style.lower(), ResponseStyle.DETAILED)

        return PromptContext(
            language=language,
            symbols=symbols or [],
            market_type=market_type,
            analysis_type=analysis_type,
            enable_web_search=enable_web_search,
            enable_tool_search=enable_tool_search,
            enable_think_tool=enable_think_tool,
            response_style=style,
            min_words=min_words,
            max_words=max_words,
            user_profile=user_profile,
            conversation_summary=conversation_summary,
            user_preferences=kwargs
        )


# Convenience functions for direct import
def get_prompt_manager() -> PromptManager:
    """Get singleton PromptManager instance"""
    if not hasattr(get_prompt_manager, '_instance'):
        get_prompt_manager._instance = PromptManager()
    return get_prompt_manager._instance


def get_system_prompt(
    provider: str,
    model: str,
    context: PromptContext,
    **kwargs
) -> str:
    """
    Convenience function to get system prompt.

    Example:
        prompt = get_system_prompt(
            provider="openai",
            model="gpt-5",
            context=PromptContext(language="vi", enable_web_search=True)
        )
    """
    return get_prompt_manager().get_system_prompt(
        provider=provider,
        model=model,
        context=context,
        **kwargs
    )


def detect_provider_from_model(model_name: str) -> str:
    """
    Convenience function to detect provider from model name.

    Example:
        provider = detect_provider_from_model("gpt-5")  # Returns "openai"
        provider = detect_provider_from_model("gemini-2.5-flash")  # Returns "gemini"
    """
    return get_prompt_manager().detect_provider(model_name)
