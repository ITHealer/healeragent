from enum import Enum
from typing import Optional

from src.providers.base_provider import ModelProvider
from src.providers.openai_provider import OpenAIModelProvider
from src.providers.gemini_provider import GeminiModelProvider
from src.providers.ollama_provider import OllamaModelProvider
from src.providers.openrouter_provider import OpenRouterModelProvider
from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings

class ProviderType(str, Enum):
    """
    Supported LLM provider types.
    
    Each provider has different capabilities, pricing, and model availability:
    
    - OLLAMA: Local/self-hosted models (free, private)
    - OPENAI: OpenAI's models (GPT-4, GPT-3.5, etc.)
    - OPENROUTER: Unified API for 100+ models from multiple providers
    - GEMINI: Google's Gemini models
    """
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    GEMINI = "gemini"
    
    @classmethod
    def list(cls):
        """Get list of all provider type values."""
        return list(map(lambda c: c.value, cls))
    
    @classmethod
    def get_display_names(cls) -> dict:
        """Get provider display names for UI."""
        return {
            cls.OLLAMA.value: "Ollama (Local)",
            cls.OPENAI.value: "OpenAI",
            cls.OPENROUTER.value: "OpenRouter (Multi-Provider)",
            cls.GEMINI.value: "Google Gemini",
        }

class ModelProviderFactory(LoggerMixin):
    """
    Factory for creating model providers.
    
    Centralizes provider creation logic and handles:
    - Provider instantiation
    - API key validation
    - Configuration from environment
    """
    
    @staticmethod
    def create_provider(
        provider_type: str,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> ModelProvider:
        """
        Create appropriate provider based on type.
        
        Args:
            provider_type: One of ProviderType values (ollama, openai, openrouter, gemini)
            model_name: Model identifier
                - OpenAI: "gpt-4o-mini", "gpt-4-turbo"
                - OpenRouter: "provider/model" format, e.g., "anthropic/claude-3.5-sonnet"
                - Ollama: Local model name, e.g., "llama3:8b"
                - Gemini: "gemini-pro", "gemini-1.5-flash"
            api_key: API key for the provider (not needed for Ollama)
            **kwargs: Additional provider-specific arguments
                - site_url: For OpenRouter analytics
                - site_name: For OpenRouter analytics
                
        Returns:
            ModelProvider: Configured provider instance
            
        Raises:
            ValueError: If provider type is unsupported or API key is missing
        """
        logger = LoggerMixin().logger

        # Normalize provider type
        provider_type_lower = provider_type.lower() if isinstance(provider_type, str) else provider_type
        
        if provider_type_lower == ProviderType.OLLAMA:
            logger.debug(f"[FACTORY] Creating Ollama provider: {model_name}")
            return OllamaModelProvider(model_name=model_name)
        elif provider_type == ProviderType.OPENAI:
            if not api_key:
                raise ValueError(
                    "API key is required for OpenAI provider. "
                    "Set OPENAI_API_KEY environment variable or pass api_key parameter."
                )
            logger.debug(f"[FACTORY] Creating OpenAI provider: {model_name}")
            return OpenAIModelProvider(api_key=api_key, model_name=model_name)
        elif provider_type_lower == ProviderType.OPENROUTER:
            if not api_key:
                raise ValueError(
                    "API key is required for OpenRouter provider. "
                    "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
                )
            
            # Extract OpenRouter-specific kwargs
            site_url = kwargs.get('site_url')
            site_name = kwargs.get('site_name', 'CODEMIND Agent')
            
            logger.debug(f"[FACTORY] Creating OpenRouter provider: {model_name}")
            return OpenRouterModelProvider(
                api_key=api_key,
                model_name=model_name,
                site_url=site_url,
                site_name=site_name
            )
        elif provider_type == ProviderType.GEMINI:
            if not api_key:
                raise ValueError(
                    "API key is required for Gemini provider. "
                    "Set GEMINI_API_KEY environment variable or pass api_key parameter."
                )
            logger.debug(f"[FACTORY] Creating Gemini provider: {model_name}")
            return GeminiModelProvider(api_key=api_key, model_name=model_name)
        else:
            raise ValueError(
                f"Unsupported provider type: {provider_type}. "
                f"Supported providers: {ProviderType.list()}"
            )
    
    @staticmethod
    def _get_api_key(provider_type: str) -> Optional[str]:
        """
        Get API key for a provider from environment variables.
        
        Args:
            provider_type: Provider type (ollama, openai, openrouter, gemini)
            
        Returns:
            Optional[str]: API key for the provider, or None/endpoint for Ollama
            
        Note:
            Environment variables used:
            - OPENAI_API_KEY for OpenAI
            - OPENROUTER_API_KEY for OpenRouter
            - GEMINI_API_KEY for Gemini
            - OLLAMA_ENDPOINT for Ollama (returns endpoint, not key)
        """
        provider_type_lower = provider_type.lower() if isinstance(provider_type, str) else provider_type
        
        if provider_type_lower == ProviderType.OPENAI:
            return settings.OPENAI_API_KEY
        
        elif provider_type_lower == ProviderType.OPENROUTER:
            return settings.OPENROUTER_API_KEY
        
        elif provider_type_lower == ProviderType.GEMINI:
            return settings.GEMINI_API_KEY
        
        elif provider_type_lower == ProviderType.OLLAMA:
            # Ollama doesn't need API key, return endpoint instead
            return settings.OLLAMA_ENDPOINT
        
        return None
    
    @staticmethod
    def is_api_key_configured(provider_type: str) -> bool:
        """
        Check if API key is configured for a provider.
        
        Args:
            provider_type: Provider type to check
            
        Returns:
            bool: True if API key is available
        """
        api_key = ModelProviderFactory._get_api_key(provider_type)
        
        # Ollama doesn't need API key
        if provider_type.lower() == ProviderType.OLLAMA:
            return bool(api_key)  # Just check endpoint is set
        
        return bool(api_key and len(api_key) > 0)
    
    @staticmethod
    def get_available_providers() -> list:
        """
        Get list of providers with configured API keys.
        
        Returns:
            list: Provider types that are ready to use
        """
        available = []
        for provider in ProviderType:
            if ModelProviderFactory.is_api_key_configured(provider.value):
                available.append(provider.value)
        return available
    
    @staticmethod
    def get_provider_info(provider_type: str) -> dict:
        """
        Get information about a provider.
        
        Args:
            provider_type: Provider type
            
        Returns:
            dict: Provider information
        """
        info = {
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
                "description": "Unified API gateway for 100+ models from multiple providers",
                "api_key_required": True,
                "api_key_env": "OPENROUTER_API_KEY",
                "example_models": [
                    "openai/gpt-4o",
                    "anthropic/claude-3.5-sonnet",
                    "google/gemini-pro-1.5",
                    "meta-llama/llama-3.1-70b-instruct",
                    "mistralai/mistral-large",
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
        
        return info.get(provider_type.lower(), {"name": provider_type, "description": "Unknown provider"})