import google.generativeai as genai
from typing import Dict, Any, List, Optional, AsyncGenerator

from src.providers.base_provider import ModelProvider
from src.utils.logger.custom_logging import LoggerMixin

class GeminiModelProvider(ModelProvider, LoggerMixin):
    """Provider for Google Gemini models"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        
    async def initialize(self) -> None:
        """Initialize Gemini client"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.logger.info(f"Initialized Gemini provider with model {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini provider: {str(e)}")
            raise
            
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate text completion"""
        if not self.model:
            await self.initialize()
            
        try:
            # Convert messages from OpenAI format to Gemini format
            gemini_messages = self._convert_messages(messages)
            
            response = await self.model.generate_content_async(
                gemini_messages,
                generation_config=self._convert_params(kwargs)
            )
            
            return self._format_response(response)
        except Exception as e:
            self.logger.error(f"Error generating Gemini completion: {str(e)}")
            raise
            
    async def stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream response chunks"""
        if not self.model:
            await self.initialize()
            
        try:
            gemini_messages = self._convert_messages(messages)
            
            stream = await self.model.generate_content_async(
                gemini_messages,
                generation_config=self._convert_params(kwargs),
                stream=True
            )
            
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            self.logger.error(f"Error streaming Gemini completion: {str(e)}")
            raise
    
    def supports_feature(self, feature_name: str) -> bool:
        """Check if provider supports a specific feature"""
        feature_support = {
            "thinking_mode": False,
            "vision": "vision" in self.model_name or self.model_name == "gemini-pro-vision",
            "function_calling": True,
            "json_mode": True,
            "translation": True
        }
        return feature_support.get(feature_name, False)
    
    def _convert_messages(self, openai_messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to Gemini format"""
        gemini_messages = []
        
        for msg in openai_messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                # Gemini không có system role nên chuyển thành user message đầu tiên
                gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [content]})
                
        return gemini_messages
    
    def _convert_params(self, openai_params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI parameters to Gemini parameters"""
        gemini_params = {}
        
        # Map parameters
        if "temperature" in openai_params:
            gemini_params["temperature"] = openai_params["temperature"]
        if "top_p" in openai_params:
            gemini_params["top_p"] = openai_params["top_p"]
        if "max_tokens" in openai_params:
            gemini_params["max_output_tokens"] = openai_params["max_tokens"]
            
        return gemini_params
        
    def _format_response(self, response) -> Dict[str, Any]:
        """Format Gemini response to common structure"""
        # Get actual finish_reason from Gemini response
        finish_reason = "stop"
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    # Map Gemini finish_reason to OpenAI-style
                    gemini_reason = str(candidate.finish_reason)
                    if "MAX_TOKENS" in gemini_reason:
                        finish_reason = "length"
                        self.logger.warning(f"[GEMINI] Response truncated due to max_tokens limit")
                    elif "STOP" in gemini_reason:
                        finish_reason = "stop"
                    elif "SAFETY" in gemini_reason:
                        finish_reason = "content_filter"
                        self.logger.warning(f"[GEMINI] Response filtered due to safety: {gemini_reason}")
                    else:
                        finish_reason = gemini_reason.lower()
        except Exception as e:
            self.logger.debug(f"[GEMINI] Could not extract finish_reason: {e}")

        return {
            "content": response.text,
            "model": self.model_name,
            "id": getattr(response, "response_id", None),
            "finish_reason": finish_reason,
            "raw_response": response
        }