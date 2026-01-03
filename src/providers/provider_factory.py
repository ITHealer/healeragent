from enum import Enum
from typing import Optional, Dict, List

from src.providers.base_provider import ModelProvider
from src.providers.openai_provider import OpenAIModelProvider
from src.providers.gemini_provider import GeminiModelProvider
from src.providers.ollama_provider import OllamaModelProvider
from src.providers.openrouter_provider import OpenRouterModelProvider
from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    GEMINI = "gemini"

    @classmethod
    def list(cls) -> List[str]:
        """Return all provider type values."""
        return [p.value for p in cls]

    @classmethod
    def get_display_names(cls) -> Dict[str, str]:
        """Return provider display names for UI."""
        return {
            cls.OLLAMA.value: "Ollama (Local)",
            cls.OPENAI.value: "OpenAI",
            cls.OPENROUTER.value: "OpenRouter (Multi-Provider)",
            cls.GEMINI.value: "Google Gemini",
        }

class ModelProviderFactory(LoggerMixin):
    """Factory for creating LLM model providers."""

    @staticmethod
    def create_provider(
        provider_type: str,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> ModelProvider:
        """
        Create a model provider instance.

        Args:
            provider_type: Provider type (ollama, openai, openrouter, gemini)
            model_name: Model identifier for the provider
            api_key: API key (not required for Ollama)
            **kwargs: Provider-specific options (site_url, site_name for OpenRouter)

        Returns:
            Configured ModelProvider instance

        Raises:
            ValueError: If provider type is unsupported or API key is missing
        """
        logger = LoggerMixin().logger
        ptype = provider_type.lower() if isinstance(provider_type, str) else provider_type

        if ptype == ProviderType.OLLAMA or ptype == "ollama":
            logger.debug(f"[FACTORY] Creating Ollama provider | model={model_name}")
            return OllamaModelProvider(model_name=model_name)

        if ptype == ProviderType.OPENAI or ptype == "openai":
            if not api_key:
                raise ValueError("API key required for OpenAI. Set OPENAI_API_KEY or pass api_key.")
            logger.debug(f"[FACTORY] Creating OpenAI provider | model={model_name}")
            return OpenAIModelProvider(api_key=api_key, model_name=model_name)

        if ptype == ProviderType.OPENROUTER or ptype == "openrouter":
            if not api_key:
                raise ValueError("API key required for OpenRouter. Set OPENROUTER_API_KEY or pass api_key.")
            logger.debug(f"[FACTORY] Creating OpenRouter provider | model={model_name}")
            return OpenRouterModelProvider(
                api_key=api_key,
                model_name=model_name,
                site_url=kwargs.get("site_url"),
                site_name=kwargs.get("site_name", "CODEMIND Agent"),
            )

        if ptype == ProviderType.GEMINI or ptype == "gemini":
            if not api_key:
                raise ValueError("API key required for Gemini. Set GEMINI_API_KEY or pass api_key.")
            logger.debug(f"[FACTORY] Creating Gemini provider | model={model_name}")
            return GeminiModelProvider(api_key=api_key, model_name=model_name)

        raise ValueError(f"Unsupported provider: {provider_type}. Supported: {ProviderType.list()}")
    
    @staticmethod
    def _get_api_key(provider_type: str) -> Optional[str]:
        """Get API key from environment for the specified provider."""
        ptype = provider_type.lower() if isinstance(provider_type, str) else provider_type

        key_map = {
            ProviderType.OPENAI: settings.OPENAI_API_KEY,
            ProviderType.OPENROUTER: settings.OPENROUTER_API_KEY,
            ProviderType.GEMINI: settings.GEMINI_API_KEY,
            ProviderType.OLLAMA: settings.OLLAMA_ENDPOINT,  # Ollama uses endpoint, not key
        }
        return key_map.get(ptype)

    @staticmethod
    def is_api_key_configured(provider_type: str) -> bool:
        """Check if API key/endpoint is configured for a provider."""
        api_key = ModelProviderFactory._get_api_key(provider_type)
        return bool(api_key)

    @staticmethod
    def get_available_providers() -> List[str]:
        """Return list of providers with configured API keys."""
        return [
            p.value for p in ProviderType
            if ModelProviderFactory.is_api_key_configured(p.value)
        ]
    
    @staticmethod
    def get_provider_info(provider_type: str) -> Dict[str, any]:
        """Return metadata about a provider (name, description, example models, docs)."""
        provider_info = {
            ProviderType.OLLAMA.value: {
                "name": "Ollama",
                "description": "Local/self-hosted LLM models",
                "api_key_required": False,
                "example_models": ["llama3:8b", "codellama:7b", "mistral:7b"],
                "docs_url": "https://ollama.ai/",
            },
            ProviderType.OPENAI.value: {
                "name": "OpenAI",
                "description": "OpenAI's GPT models",
                "api_key_required": True,
                "api_key_env": "OPENAI_API_KEY",
                "example_models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                "docs_url": "https://platform.openai.com/docs",
            },
            ProviderType.OPENROUTER.value: {
                "name": "OpenRouter",
                "description": "Unified API gateway for 100+ models",
                "api_key_required": True,
                "api_key_env": "OPENROUTER_API_KEY",
                "example_models": [
                    "openai/gpt-4o",
                    "anthropic/claude-3.5-sonnet",
                    "google/gemini-pro-1.5",
                    "meta-llama/llama-3.1-70b-instruct",
                ],
                "docs_url": "https://openrouter.ai/docs",
            },
            ProviderType.GEMINI.value: {
                "name": "Google Gemini",
                "description": "Google's Gemini AI models",
                "api_key_required": True,
                "api_key_env": "GEMINI_API_KEY",
                "example_models": ["gemini-pro", "gemini-1.5-flash", "gemini-1.5-pro"],
                "docs_url": "https://ai.google.dev/docs",
            },
        }

        ptype = provider_type.lower() if isinstance(provider_type, str) else provider_type
        return provider_info.get(ptype, {"name": provider_type, "description": "Unknown provider"})