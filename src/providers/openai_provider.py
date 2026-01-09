import openai
from openai import AsyncOpenAI
from typing import Dict, Any, List, Optional, AsyncGenerator
import re

from src.providers.base_provider import ModelProvider
from src.utils.logger.custom_logging import LoggerMixin


MODEL_CONFIG = {
    # Models requiring max_completion_tokens instead of max_tokens
    "use_max_completion_tokens": [
        r"^o1",           # o1, o1-mini, o1-preview, o1-pro, etc.
        r"^o3",           # o3, o3-mini, etc.
        r"^o4",           # Future o4 models
        r"^gpt-5",        # gpt-5, gpt-5-nano, gpt-5-mini, etc.
        r"^gpt-6",        # Future gpt-6 models
    ],
    # Models that don't support temperature parameter (fixed at 1.0)
    "no_temperature": [
        r"^o1",           # o1 series uses fixed temperature
        r"^o3",           # o3 series uses fixed temperature
        r"^o4",           # Future o4 models
        r"^gpt-5",        # gpt-5 series
        r"^gpt-6",        # Future gpt-6 models
        r"^gpt-4\.1",     # gpt-4.1-nano, gpt-4.1-mini, etc.
    ],
    # Models with reasoning/thinking capabilities
    "reasoning_models": [
        r"^o1",
        r"^o3",
        r"^o4",
    ],
    # Models with vision capabilities
    "vision_models": [
        r"^gpt-4o",       # gpt-4o, gpt-4o-mini
        r"^gpt-4-vision",
        r"^gpt-5",        # Assuming gpt-5 supports vision
    ],
}


def _matches_any_pattern(model_name: str, patterns: List[str]) -> bool:
    """Check if model name matches any pattern in the list."""
    model_lower = model_name.lower()
    for pattern in patterns:
        if re.match(pattern, model_lower):
            return True
    return False


def uses_max_completion_tokens(model_name: str) -> bool:
    """Check if model uses max_completion_tokens instead of max_tokens."""
    return _matches_any_pattern(model_name, MODEL_CONFIG["use_max_completion_tokens"])


def supports_temperature(model_name: str) -> bool:
    """Check if model supports temperature parameter."""
    return not _matches_any_pattern(model_name, MODEL_CONFIG["no_temperature"])


def is_reasoning_model(model_name: str) -> bool:
    """Check if model has reasoning/thinking capabilities."""
    return _matches_any_pattern(model_name, MODEL_CONFIG["reasoning_models"])


def has_vision_capability(model_name: str) -> bool:
    """Check if model supports vision/image input."""
    return _matches_any_pattern(model_name, MODEL_CONFIG["vision_models"])


def convert_params_for_model(
    model_name: str,
    params: Dict[str, Any],
    logger=None
) -> Dict[str, Any]:
    """
    Convert and filter parameters based on model requirements.

    Handles:
    - max_tokens -> max_completion_tokens for newer models
    - Removes temperature for models that don't support it
    """
    result = params.copy()

    # Convert max_tokens to max_completion_tokens if needed
    if uses_max_completion_tokens(model_name) and "max_tokens" in result:
        result["max_completion_tokens"] = result.pop("max_tokens")

    # Remove temperature for models that don't support it
    if not supports_temperature(model_name) and "temperature" in result:
        if logger:
            logger.debug(f"[OPENAI] Removing unsupported temperature param for {model_name}")
        del result["temperature"]

    return result


class OpenAIModelProvider(ModelProvider, LoggerMixin):
    """
    OpenAI model provider with automatic parameter handling.

    Automatically handles:
    - Parameter conversion (max_tokens -> max_completion_tokens for newer models)
    - Temperature filtering for models that don't support it
    - Feature detection based on model capabilities
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4.1-nano"):
        super().__init__()
        self._api_key = api_key
        self._model_name = model_name
        self._client: Optional[AsyncOpenAI] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def client(self) -> Optional[AsyncOpenAI]:
        return self._client

    async def initialize(self) -> None:
        """Initialize the async OpenAI client."""
        if self._client is not None:
            return

        try:
            self._client = AsyncOpenAI(api_key=self._api_key)
            self.logger.info(f"[OPENAI] Initialized provider | model={self._model_name}")
        except Exception as e:
            self.logger.error(f"[OPENAI] Failed to initialize: {e}")
            raise
            
    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a chat completion (non-streaming)."""
        import time

        if self._client is None:
            await self.initialize()

        model = kwargs.pop("model", self._model_name)
        params = convert_params_for_model(model, kwargs, self.logger)

        try:
            msg_count = len(messages)
            total_chars = sum(len(m.get('content', '')) for m in messages)
            self.logger.debug(
                f"[OPENAI] Generate | model={model} | msgs={msg_count} | "
                f"~{total_chars} chars | params={list(params.keys())}"
            )

            start_time = time.time()
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                **params
            )
            elapsed = time.time() - start_time
            self.logger.debug(f"[OPENAI] Generate done | {elapsed:.2f}s | model={model}")

            return self._format_response(response)

        except openai.BadRequestError as e:
            error_msg = str(e)
            self.logger.error(f"[OPENAI] BadRequest: {error_msg}")

            # Auto-retry with max_completion_tokens if that was the issue
            if "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
                if "max_tokens" in params:
                    self.logger.warning(f"[OPENAI] Retrying with max_completion_tokens | model={model}")
                    params["max_completion_tokens"] = params.pop("max_tokens")

                    response = await self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **params
                    )
                    return self._format_response(response)
            raise

        except Exception as e:
            self.logger.error(f"[OPENAI] Generate failed: {e}")
            raise
            
    async def stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion chunks."""
        import time

        if self._client is None:
            await self.initialize()

        model = kwargs.pop("model", self._model_name)
        params = convert_params_for_model(model, kwargs, self.logger)

        try:
            msg_count = len(messages)
            total_chars = sum(len(m.get('content', '')) for m in messages)
            self.logger.debug(
                f"[OPENAI] Stream | model={model} | msgs={msg_count} | "
                f"~{total_chars} chars | params={list(params.keys())}"
            )

            stream_start = time.time()
            response_stream = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **params
            )

            first_chunk = True
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_chunk:
                        ttft = time.time() - stream_start
                        self.logger.debug(f"[OPENAI] First chunk | TTFT={ttft:.2f}s | model={model}")
                        first_chunk = False
                    yield chunk.choices[0].delta.content

        except openai.BadRequestError as e:
            error_msg = str(e)
            self.logger.error(f"[OPENAI] Stream BadRequest: {error_msg}")

            # Auto-retry with max_completion_tokens if that was the issue
            if "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
                if "max_tokens" in params:
                    self.logger.warning(f"[OPENAI] Retrying stream with max_completion_tokens | model={model}")
                    params["max_completion_tokens"] = params.pop("max_tokens")

                    response_stream = await self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=True,
                        **params
                    )

                    async for chunk in response_stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                    return
            raise

        except Exception as e:
            self.logger.error(f"[OPENAI] Stream failed: {e}")
            raise
    
    def supports_feature(self, feature_name: str) -> bool:
        """Check if the current model supports a specific feature."""
        feature_map = {
            "thinking_mode": is_reasoning_model(self._model_name),
            "vision": has_vision_capability(self._model_name),
            "function_calling": True,
            "json_mode": True,
            "translation": True,
            "max_completion_tokens": uses_max_completion_tokens(self._model_name),
            "temperature": supports_temperature(self._model_name),
        }
        return feature_map.get(feature_name, False)

    def _format_response(self, response) -> Dict[str, Any]:
        """Format OpenAI response to standard structure."""

        message = response.choices[0].message

        result = {
            "content": response.choices[0].message.content,
            "model": response.model,
            "id": response.id,
            "finish_reason": response.choices[0].finish_reason,
            "raw_response": response,
        }
        
        # Extract tool_calls if present (for function calling)
        if hasattr(message, "tool_calls") and message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in message.tool_calls
            ]

        return result
    