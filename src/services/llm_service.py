import logging
from typing import Dict, Any, List, Optional
from src.helpers.llm_helper import LLMGeneratorProvider, ProviderType

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.llm_provider = LLMGeneratorProvider()
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        provider_type: str = ProviderType.OPENAI,  # Default to OpenAI
        model_name: str = "gpt-5-nano-2025-08-07",  # Default model
        api_key: Optional[str] = None
    ) -> str:
        """
        Generate response using LLM
        
        Args:
            prompt: The prompt to send to LLM
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            provider_type: Provider type (openai, ollama, gemini)
            model_name: Model name to use
            api_key: API key for the provider
            
        Returns:
            Generated text response
        """
        try:
            messages = [
                {"role": "system", "content": "You are a helpful stock market assistant."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm_provider.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=False  # For simple generation, disable thinking
            )
            
            # Extract content from response
            if isinstance(response, dict):
                if "content" in response:
                    return response["content"]
                elif "choices" in response:  # OpenAI format
                    return response["choices"][0]["message"]["content"]
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Error generating with LLM: {e}")
            raise

    
    async def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        provider_type: str = ProviderType.OPENAI,  # Default to OpenAI
        model_name: str = "gpt-5-nano-2025-08-07",  # Default model
        api_key: Optional[str] = None
    ) -> str:
        """
        Generate response using LLM
        
        Args:
            prompt: The prompt to send to LLM
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            provider_type: Provider type (openai, ollama, gemini)
            model_name: Model name to use
            api_key: API key for the provider
            
        Returns:
            Generated text response
        """
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that creates concise titles."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm_provider.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=False  # For simple generation, disable thinking
            )
            
            # Extract content from response
            if isinstance(response, dict):
                if "content" in response:
                    return response["content"]
                elif "choices" in response:  # OpenAI format
                    return response["choices"][0]["message"]["content"]
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Error generating with LLM: {e}")
            raise