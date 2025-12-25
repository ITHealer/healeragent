import openai
from openai import AsyncOpenAI
from typing import Dict, Any, List, Optional, AsyncGenerator

from src.providers.base_provider import ModelProvider
from src.utils.logger.custom_logging import LoggerMixin


# ============================================================================
# MODEL CAPABILITY CONSTANTS
# ============================================================================

# Models that require max_completion_tokens instead of max_tokens
NEW_API_MODELS = [
    "o1",
    "o1-mini",
    "o1-preview",
    "o3",
    "o3-mini",
    "gpt-5",
    "gpt-5-nano",
    "gpt-5-mini",
]

# Models that don't support temperature parameter (only default=1)
MODELS_WITHOUT_TEMPERATURE = [
    "gpt-5-nano",
    "gpt-4.1-nano",
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_new_api_model(model_name: str) -> bool:
    """
    Check if model uses new API format (max_completion_tokens)
    
    Args:
        model_name: Full model name 
        
    Returns:
        True if model requires max_completion_tokens
    """
    model_lower = model_name.lower()
    
    for prefix in NEW_API_MODELS:
        if model_lower.startswith(prefix):
            return True
    
    return False


def model_supports_temperature(model_name: str) -> bool:
    """
    Check if model supports temperature parameter
    
    Args:
        model_name: Full model name
        
    Returns:
        True if model supports temperature, False otherwise
    """
    model_lower = model_name.lower()
    
    for prefix in MODELS_WITHOUT_TEMPERATURE:
        if model_lower.startswith(prefix.lower()):
            return False
    
    return True


def convert_params_for_model(model_name: str, kwargs: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """
    Convert and filter parameters based on model requirements
    
    Handles:
    - max_tokens → max_completion_tokens for new models (o1, o3, gpt-5)
    - Removes temperature for models that don't support it (gpt-5-nano, gpt-4.1-nano)
    
    Args:
        model_name: Model name
        kwargs: Original parameters
        logger: Optional logger for debug messages
        
    Returns:
        Converted and filtered parameters
    """
    params = kwargs.copy()
    
    # Convert max_tokens to max_completion_tokens for new models
    if is_new_api_model(model_name):
        if "max_tokens" in params:
            max_tokens = params.pop("max_tokens")
            params["max_completion_tokens"] = max_tokens
    
    # Remove temperature for models that don't support it
    if not model_supports_temperature(model_name):
        if "temperature" in params:
            if logger:
                logger.debug(
                    f"[OpenAI] Removing temperature for {model_name} (only default=1 supported)"
                )
            del params["temperature"]
    
    return params


# ============================================================================
# OPENAI PROVIDER CLASS
# ============================================================================

class OpenAIModelProvider(ModelProvider, LoggerMixin):
    """
    Provider for OpenAI models using official SDK
    
    Supports:
    - GPT-4.x models (gpt-4, gpt-4o, gpt-4-turbo, etc.)
    - GPT-4.1 models (gpt-4.1-nano, gpt-4.1-mini, etc.)
    - GPT-5 models (gpt-5-nano, etc.) - uses max_completion_tokens
    - O1/O3 reasoning models - uses max_completion_tokens
    
    Auto-handles:
    - Parameter conversion (max_tokens → max_completion_tokens)
    - Temperature filtering for unsupported models
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-4.1-nano"):
        """
        Initialize OpenAI provider
        
        Args:
            api_key: OpenAI API key
            model_name: Default model name
        """
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        
    async def initialize(self) -> None:
        """Initialize the OpenAI client"""
        try:
            self.client = AsyncOpenAI(api_key=self.api_key)
            self.logger.info(f"Initialized OpenAI provider with model {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI provider: {str(e)}")
            raise
            
    async def generate(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text completion (non-streaming)
        
        Args:
            messages: List of message dicts with role and content
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Formatted response dict
        """
        if not self.client:
            await self.initialize()
        
        # Get model from kwargs or use default
        model = kwargs.pop("model", self.model_name)
        
        # Convert and filter parameters for model compatibility
        converted_kwargs = convert_params_for_model(model, kwargs, self.logger)
        
        try:
            self.logger.debug(
                f"[OpenAI] Generating with model={model}, "
                f"params={list(converted_kwargs.keys())}"
            )
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                **converted_kwargs
            )
            return self._format_response(response)
            
        except openai.BadRequestError as e:
            # Handle specific parameter errors
            error_msg = str(e)
            self.logger.error(f"OpenAI BadRequest: {error_msg}")
            
            # Retry with parameter conversion if it's a max_tokens issue
            if "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
                self.logger.warning(
                    f"[OpenAI] Retrying with max_completion_tokens for model {model}"
                )
                if "max_tokens" in converted_kwargs:
                    max_val = converted_kwargs.pop("max_tokens")
                    converted_kwargs["max_completion_tokens"] = max_val
                    
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **converted_kwargs
                    )
                    return self._format_response(response)
            
            raise
            
        except Exception as e:
            self.logger.error(f"Error generating OpenAI completion: {str(e)}")
            raise
            
    async def stream(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream response chunks
        
        Args:
            messages: List of message dicts
            **kwargs: Additional parameters
            
        Yields:
            Response text chunks
        """
        if not self.client:
            await self.initialize()
        
        # Get model from kwargs or use default
        model = kwargs.pop("model", self.model_name)
        
        # Convert and filter parameters for model compatibility
        converted_kwargs = convert_params_for_model(model, kwargs, self.logger)
        
        try:
            self.logger.debug(
                f"[OpenAI] Streaming with model={model}, "
                f"params={list(converted_kwargs.keys())}"
            )
            
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **converted_kwargs
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except openai.BadRequestError as e:
            error_msg = str(e)
            self.logger.error(f"OpenAI Stream BadRequest: {error_msg}")
            
            # Retry with parameter conversion if needed
            if "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
                self.logger.warning(
                    f"[OpenAI] Retrying stream with max_completion_tokens for model {model}"
                )
                if "max_tokens" in converted_kwargs:
                    max_val = converted_kwargs.pop("max_tokens")
                    converted_kwargs["max_completion_tokens"] = max_val
                    
                    stream = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=True,
                        **converted_kwargs
                    )
                    
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                    return
            
            raise
            
        except Exception as e:
            self.logger.error(f"Error streaming OpenAI completion: {str(e)}")
            raise
    
    def supports_feature(self, feature_name: str) -> bool:
        """
        Check if provider supports a specific feature
        
        Args:
            feature_name: Name of feature to check
            
        Returns:
            True if feature is supported
        """
        feature_support = {
            "thinking_mode": self.model_name.startswith(("o1", "o3")),
            "vision": "vision" in self.model_name or self.model_name in [
                "gpt-4-vision", 
                "gpt-4-vision-preview",
                "gpt-4o",
                "gpt-4o-mini"
            ],
            "function_calling": True,
            "json_mode": True,
            "translation": True,
            "max_completion_tokens": is_new_api_model(self.model_name),
            "temperature": model_supports_temperature(self.model_name)
        }
        return feature_support.get(feature_name, False)
        
        
    def _format_response(self, response) -> Dict[str, Any]:
        """
        Format OpenAI response to common structure
        
        Args:
            response: Raw OpenAI response
            
        Returns:
            Standardized response dict
        """
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "id": response.id,
            "finish_reason": response.choices[0].finish_reason,
            "raw_response": response
        }