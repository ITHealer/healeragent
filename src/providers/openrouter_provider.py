import openai
from openai import AsyncOpenAI
from typing import Dict, Any, List, Optional, AsyncGenerator

from src.providers.base_provider import ModelProvider
from src.utils.logger.custom_logging import LoggerMixin


# OpenRouter API base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Popular models available on OpenRouter
# Format: provider/model-name
OPENROUTER_MODELS = {
    # OpenAI Models
    "openai/gpt-5-nano": "OpenAI GPT-5 Nano",
    "openai/gpt-4.1-nano": "OpenAI GPT-4.1 Nano",
    "openai/gpt-4o": "OpenAI GPT-4o",
    "openai/gpt-4o-mini": "OpenAI GPT-4o Mini",
    
    # Google Models
    "google/gemma-3-27b-it:free": "Gemma 3 27B It:Free",
    
    # Meta Models
    "meta-llama/llama-3.1-70b-instruct": "Llama 3.1 70B",
}


class OpenRouterModelProvider(ModelProvider, LoggerMixin):
    """
    Provider for OpenRouter API - unified access to multiple LLM providers.
    
    OpenRouter uses OpenAI-compatible API format, making it easy to switch
    between providers without code changes.
    
    Attributes:
        api_key: OpenRouter API key (starts with sk-or-)
        model_name: Model identifier in format "provider/model" (e.g., "openai/gpt-4o")
        site_url: Optional site URL for ranking/analytics
        site_name: Optional site name for ranking/analytics
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "openai/gpt-4.1-nano",
        site_url: Optional[str] = None,
        site_name: Optional[str] = ""
    ):
        """
        Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key
            model_name: Model in format "provider/model-name"
            site_url: Optional URL for OpenRouter analytics
            site_name: Optional app name for OpenRouter analytics
        """
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.site_url = site_url
        self.site_name = site_name
        self.client: Optional[AsyncOpenAI] = None
        
        # Validate model name format
        if "/" not in model_name:
            self.logger.warning(
                f"Model name '{model_name}' doesn't follow provider/model format. "
                "Consider using format like 'openai/gpt-4o-mini'"
            )
    
    async def initialize(self) -> None:
        """Initialize the OpenRouter client using OpenAI SDK with custom base URL."""
        try:
            # Build default headers for OpenRouter
            default_headers = {}
            
            if self.site_url:
                default_headers["HTTP-Referer"] = self.site_url
            
            if self.site_name:
                default_headers["X-Title"] = self.site_name
            
            # Initialize OpenAI client with OpenRouter base URL
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=OPENROUTER_BASE_URL,
                default_headers=default_headers if default_headers else None
            )
            
            self.logger.info(
                f"[OPENROUTER] Initialized provider with model: {self.model_name}"
            )
            
        except Exception as e:
            self.logger.error(f"[OPENROUTER] Failed to initialize: {str(e)}")
            raise
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text completion using OpenRouter.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Dict containing response content and metadata
        """
        if not self.client:
            await self.initialize()
        
        try:
            # Filter kwargs to only include supported parameters
            filtered_kwargs = self._filter_params(kwargs)
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **filtered_kwargs
            )
            
            return self._format_response(response)
            
        except openai.APIError as e:
            self.logger.error(f"[OPENROUTER] API error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"[OPENROUTER] Error generating completion: {str(e)}")
            raise
    
    async def stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream response chunks from OpenRouter.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters
            
        Yields:
            str: Response text chunks
        """
        if not self.client:
            await self.initialize()
        
        try:
            # Filter kwargs to only include supported parameters
            filtered_kwargs = self._filter_params(kwargs)
            
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                **filtered_kwargs
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except openai.APIError as e:
            self.logger.error(f"[OPENROUTER] Stream API error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"[OPENROUTER] Error streaming: {str(e)}")
            raise
    
    def supports_feature(self, feature_name: str) -> bool:
        """
        Check if provider supports a specific feature.
        
        Feature support varies by underlying model. This provides
        general guidance based on common capabilities.
        
        Args:
            feature_name: Feature to check (thinking_mode, vision, function_calling, json_mode)
            
        Returns:
            bool: Whether feature is likely supported
        """
        # Feature support depends on the underlying model
        model_lower = self.model_name.lower()
        
        if feature_name == "thinking_mode":
            # o1 models support reasoning/thinking
            return "o1" in model_lower
        
        elif feature_name == "vision":
            # Vision-capable models
            vision_models = [
                "gpt-4o", "gpt-4-turbo", "gpt-4-vision",
                "claude-3", "gemini",
                "llama-3.2-90b-vision", "llama-3.2-11b-vision"
            ]
            return any(vm in model_lower for vm in vision_models)
        
        elif feature_name == "function_calling":
            # Most modern models support function calling
            return True
        
        elif feature_name == "json_mode":
            # JSON mode support
            json_capable = [
                "gpt-4", "gpt-3.5-turbo",
                "claude-3", "gemini",
                "mistral", "mixtral"
            ]
            return any(jc in model_lower for jc in json_capable)
        
        return False
    
    def _filter_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter parameters to only include OpenRouter-supported ones.
        
        Args:
            params: Raw parameters dict
            
        Returns:
            Filtered parameters dict
        """
        # Standard OpenAI-compatible parameters
        supported_params = {
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "n",
            "logprobs",
            "top_logprobs",
            "response_format",
            "seed",
            "tools",
            "tool_choice",
            "user",
            # OpenRouter-specific
            "transforms",  # For prompt transformations
            "models",      # For fallback models
            "route",       # For routing preference
            "provider",    # For provider preferences
        }
        
        filtered = {
            k: v for k, v in params.items()
            if k in supported_params and v is not None
        }
        
        # Log filtered out params for debugging
        removed = set(params.keys()) - set(filtered.keys())
        if removed:
            self.logger.debug(f"[OPENROUTER] Filtered params: {removed}")
        
        return filtered
    
    def _format_response(self, response) -> Dict[str, Any]:
        """
        Format OpenRouter response to common structure.
        
        Args:
            response: Raw API response
            
        Returns:
            Standardized response dict
        """
        choice = response.choices[0] if response.choices else None
        
        return {
            "content": choice.message.content if choice else "",
            "role": choice.message.role if choice else "assistant",
            "model": response.model,
            "id": response.id,
            "finish_reason": choice.finish_reason if choice else None,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            } if response.usage else None,
            "raw_response": response
        }
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from OpenRouter.
        
        Returns:
            List of model info dicts
        """
        if not self.client:
            await self.initialize()
        
        try:
            # OpenRouter supports the models endpoint
            response = await self.client.models.list()
            
            models = []
            for model in response.data:
                models.append({
                    "id": model.id,
                    "name": getattr(model, "name", model.id),
                    "context_length": getattr(model, "context_length", None),
                    "pricing": getattr(model, "pricing", None),
                })
            
            return models
            
        except Exception as e:
            self.logger.error(f"[OPENROUTER] Failed to list models: {e}")
            # Return static list as fallback
            return [
                {"id": k, "name": v}
                for k, v in OPENROUTER_MODELS.items()
            ]
    
    async def get_model_info(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_id: Model identifier (uses self.model_name if not provided)
            
        Returns:
            Model information dict
        """
        model_id = model_id or self.model_name
        
        # Check static list first
        if model_id in OPENROUTER_MODELS:
            return {
                "id": model_id,
                "name": OPENROUTER_MODELS[model_id],
                "provider": model_id.split("/")[0] if "/" in model_id else "unknown"
            }
        
        # Try to fetch from API
        try:
            models = await self.list_available_models()
            for model in models:
                if model["id"] == model_id:
                    return model
        except Exception:
            pass
        
        return {"id": model_id, "name": model_id}


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_openrouter_models() -> Dict[str, str]:
    """Get dict of popular OpenRouter models."""
    return OPENROUTER_MODELS.copy()


def validate_openrouter_model(model_name: str) -> bool:
    """
    Validate if model name follows OpenRouter format.
    
    Args:
        model_name: Model identifier to validate
        
    Returns:
        bool: True if valid format
    """
    if "/" not in model_name:
        return False
    
    parts = model_name.split("/")
    if len(parts) != 2:
        return False
    
    provider, model = parts
    return len(provider) > 0 and len(model) > 0