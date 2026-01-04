import aiohttp
import json
import re
from typing import Dict, Any, List, Optional, AsyncGenerator

from src.providers.base_provider import ModelProvider
from src.utils.config import settings
from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.model_manager import model_manager


# Vision-capable Ollama models
VISION_MODELS = [
    r"^llava",       # llava, llava:7b, llava:13b, etc.
    r"^bakllava",    # bakllava
    r"^llama3\.2-vision",  # llama3.2-vision:11b, etc.
    r"^moondream",   # moondream
    r"^minicpm-v",   # minicpm-v
]


def _is_vision_model(model_name: str) -> bool:
    """Check if model supports vision input."""
    model_lower = model_name.lower()
    for pattern in VISION_MODELS:
        if re.match(pattern, model_lower):
            return True
    return False


class OllamaModelProvider(ModelProvider, LoggerMixin):
    """Provider for Ollama models with vision support"""

    def __init__(self, model_name: str, base_url: str = settings.OLLAMA_ENDPOINT):
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url
        self._supports_vision = _is_vision_model(model_name)
        
    async def initialize(self) -> None:
        """Initialize model by ensuring it's loaded"""
        try:
            if self.model_name not in model_manager.loaded_models:
                self.logger.info(f"Model {self.model_name} is not loaded yet, loading...")
                await model_manager.load_model(self.model_name)
                
            self.logger.info(f"Initialized Ollama provider with model {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama provider: {str(e)}")
            raise
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate text completion"""
        await self.initialize()

        # Preprocess messages for vision if model supports it
        processed_messages = self._preprocess_messages_for_vision(messages)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": processed_messages,
                        "stream": False,
                        **self._convert_params(kwargs)
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API returned {response.status}: {error_text}")

                    result = await response.json()
                    return self._format_response(result)

        except Exception as e:
            self.logger.error(f"Error generating Ollama completion: {str(e)}")
            raise
    
    async def ainvoke(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Invoke the model asynchronously - compatibility layer for LangChain interface."""
        result = await self.generate(messages, **kwargs)
        return result

    async def stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream response chunks"""
        await self.initialize()

        # Preprocess messages for vision if model supports it
        processed_messages = self._preprocess_messages_for_vision(messages)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": processed_messages,
                        "stream": True,
                        **self._convert_params(kwargs)
                    },
                    headers={"Accept": "application/x-ndjson"}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API returned {response.status}: {error_text}")

                    # Stream JSON lines
                    async for line in response.content:
                        if line:
                            try:
                                chunk = json.loads(line)
                                if "message" in chunk and "content" in chunk["message"]:
                                    yield chunk["message"]["content"]
                            except json.JSONDecodeError:
                                self.logger.warning(f"Failed to parse chunk: {line}")

        except Exception as e:
            self.logger.error(f"Error streaming Ollama completion: {str(e)}")
            raise
    
    def supports_feature(self, feature_name: str) -> bool:
        """Check if provider supports a specific feature"""
        feature_support = {
            "thinking_mode": True,
            "vision": self._supports_vision,
            "function_calling": False,
            "json_mode": False,
            "translation": True
        }
        return feature_support.get(feature_name, False)

    def _preprocess_messages_for_vision(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Preprocess messages to convert multimodal content to Ollama format.

        Ollama expects: {"role": "user", "content": "text", "images": ["base64..."]}
        OpenAI format: {"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}]}
        """
        result = []

        for msg in messages:
            content = msg.get("content")
            role = msg.get("role", "user")

            # If content is already a string, pass through
            if isinstance(content, str):
                result.append({"role": role, "content": content})
                continue

            # If content is a list (multimodal format), convert to Ollama format
            if isinstance(content, list):
                texts = []
                images = []

                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "text")

                        if block_type == "text":
                            texts.append(block.get("text", ""))

                        elif block_type == "image_url":
                            # OpenAI format - extract base64 from data URL
                            url = block.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                parts = url.split(",", 1)
                                if len(parts) == 2:
                                    images.append(parts[1])  # base64 data
                            else:
                                # URL images not supported in Ollama, skip
                                self.logger.warning(f"[OLLAMA] Skipping URL image: {url[:50]}...")

                        elif block_type == "image":
                            # Anthropic format
                            source = block.get("source", {})
                            if source.get("type") == "base64":
                                images.append(source.get("data", ""))

                ollama_msg = {
                    "role": role,
                    "content": "\n".join(texts),
                }
                if images:
                    ollama_msg["images"] = images

                result.append(ollama_msg)
            else:
                # Unknown format, pass through
                result.append(msg)

        return result

    def _convert_params(self, openai_params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI parameters to Ollama parameters"""
        ollama_params = {}
        
        # Map parameters
        if "temperature" in openai_params:
            ollama_params["temperature"] = openai_params["temperature"]
        if "top_p" in openai_params:
            ollama_params["top_p"] = openai_params["top_p"]
        if "max_tokens" in openai_params:
            ollama_params["num_predict"] = openai_params["max_tokens"]
        
        return ollama_params
        
    def _format_response(self, response) -> Dict[str, Any]:
        """Format Ollama response to common structure"""
        return {
            "content": response.get("message", {}).get("content", ""),
            "model": response.get("model", self.model_name),
            "id": None,  # Ollama doesn't provide response IDs
            "finish_reason": "stop",  # Ollama doesn't have finish_reason
            "raw_response": response
        }