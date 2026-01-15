import json
import google.generativeai as genai
from typing import Dict, Any, List, Optional, AsyncGenerator

from src.providers.base_provider import ModelProvider
from src.utils.logger.custom_logging import LoggerMixin


class GeminiModelProvider(ModelProvider, LoggerMixin):
    """
    Provider for Google Gemini models with function calling support.

    Converts OpenAI-style tool definitions to Gemini format.
    """

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
        """
        Generate text completion with optional tool/function calling.

        Args:
            messages: OpenAI-format messages
            **kwargs: Additional params including 'tools' for function calling

        Returns:
            Response dict with content and optional tool_calls
        """
        if not self.model:
            await self.initialize()

        try:
            # Convert messages from OpenAI format to Gemini format
            gemini_messages = self._convert_messages(messages)

            # Build generation config
            generation_config = self._convert_params(kwargs)

            # Convert OpenAI tools to Gemini format if provided
            gemini_tools = None
            openai_tools = kwargs.get("tools")
            if openai_tools:
                gemini_tools = self._convert_tools_to_gemini(openai_tools)
                if gemini_tools:
                    self.logger.debug(
                        f"[GEMINI] Converted {len(openai_tools)} tools to Gemini format"
                    )

            # Make the API call
            if gemini_tools:
                response = await self.model.generate_content_async(
                    gemini_messages,
                    generation_config=generation_config,
                    tools=gemini_tools,
                )
            else:
                response = await self.model.generate_content_async(
                    gemini_messages,
                    generation_config=generation_config,
                )

            return self._format_response(response)

        except Exception as e:
            self.logger.error(f"Error generating Gemini completion: {str(e)}")
            raise

    async def stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream response chunks (text only, no tool calling in stream mode)"""
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

    def _convert_tools_to_gemini(self, openai_tools: List[Dict[str, Any]]) -> Optional[List[Dict]]:
        """
        Convert OpenAI-style tools to Gemini function_declarations format.

        Uses dict format which is more compatible across SDK versions.

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }

        Gemini format (dict):
        {
            "function_declarations": [
                {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {...}
                }
            ]
        }
        """
        function_declarations = []

        for tool in openai_tools:
            if tool.get("type") != "function":
                continue

            func = tool.get("function", {})
            name = func.get("name", "")
            description = func.get("description", "")
            parameters = func.get("parameters", {})

            if not name:
                continue

            # Build function declaration as dict
            func_decl = {
                "name": name,
                "description": description or f"Function: {name}",
            }

            # Clean and add parameters if present
            cleaned_params = self._clean_parameters_for_gemini(parameters)
            if cleaned_params:
                func_decl["parameters"] = cleaned_params

            function_declarations.append(func_decl)

        if not function_declarations:
            return None

        # Return as list of tool dicts
        return [{"function_declarations": function_declarations}]

    def _clean_parameters_for_gemini(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Clean OpenAI parameter schema for Gemini compatibility.

        Gemini is stricter about JSON schema - remove unsupported fields.
        """
        if not params:
            return None

        properties = params.get("properties", {})
        if not properties:
            return None

        cleaned = {
            "type": "object",
            "properties": {},
        }

        # Clean each property
        for prop_name, prop_def in properties.items():
            cleaned_prop = {}

            # Only keep supported fields
            if "type" in prop_def:
                cleaned_prop["type"] = prop_def["type"]
            if "description" in prop_def:
                cleaned_prop["description"] = prop_def["description"]
            if "enum" in prop_def:
                cleaned_prop["enum"] = prop_def["enum"]
            if "items" in prop_def:
                # For array types
                cleaned_prop["items"] = {"type": prop_def["items"].get("type", "string")}

            if cleaned_prop:
                cleaned["properties"][prop_name] = cleaned_prop

        # Add required fields if present
        if "required" in params:
            cleaned["required"] = params["required"]

        return cleaned if cleaned["properties"] else None

    def _convert_messages(self, openai_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style messages to Gemini format.

        Handles:
        - system → prepend to first user message (Gemini doesn't have system role)
        - assistant → model
        - tool messages → function_response
        """
        gemini_messages = []
        system_content = None

        for msg in openai_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""

            if role == "system":
                # Gemini doesn't have system role, prepend to first user message
                system_content = content

            elif role == "user":
                # Add system context if present
                if system_content:
                    content = f"{system_content}\n\n---\n\n{content}"
                    system_content = None
                gemini_messages.append({"role": "user", "parts": [content]})

            elif role == "assistant":
                parts = []
                if content:
                    parts.append(content)

                # Handle tool_calls from assistant
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    if tc.get("type") == "function":
                        func = tc.get("function", {})
                        try:
                            args = json.loads(func.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            args = {}
                        # Add function call as a part
                        parts.append({
                            "function_call": {
                                "name": func.get("name", ""),
                                "args": args,
                            }
                        })

                if parts:
                    gemini_messages.append({"role": "model", "parts": parts})

            elif role == "tool":
                # Tool response - add as function_response
                tool_call_id = msg.get("tool_call_id", "")
                func_name = self._find_function_name(tool_call_id, openai_messages)

                # Parse content if it's JSON
                try:
                    result = json.loads(content) if content else {}
                except json.JSONDecodeError:
                    result = {"result": content}

                gemini_messages.append({
                    "role": "user",
                    "parts": [{
                        "function_response": {
                            "name": func_name,
                            "response": result,
                        }
                    }]
                })

        # If only system message was provided, add it as user
        if system_content and not gemini_messages:
            gemini_messages.append({"role": "user", "parts": [system_content]})

        return gemini_messages

    def _find_function_name(self, tool_call_id: str, openai_messages: List[Dict]) -> str:
        """Find function name from tool_call_id in original messages"""
        for msg in openai_messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    if tc.get("id") == tool_call_id:
                        return tc.get("function", {}).get("name", "unknown")
        return "unknown"

    def _convert_params(self, openai_params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI parameters to Gemini parameters"""
        gemini_params = {}

        if "temperature" in openai_params:
            gemini_params["temperature"] = openai_params["temperature"]
        if "top_p" in openai_params:
            gemini_params["top_p"] = openai_params["top_p"]
        if "max_tokens" in openai_params:
            gemini_params["max_output_tokens"] = openai_params["max_tokens"]

        return gemini_params

    def _format_response(self, response) -> Dict[str, Any]:
        """
        Format Gemini response to OpenAI-compatible structure.

        Handles both text responses and function calls.
        """
        finish_reason = "stop"
        content = ""
        tool_calls = []

        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]

                # Extract finish_reason
                if hasattr(candidate, 'finish_reason'):
                    gemini_reason = str(candidate.finish_reason)
                    if "MAX_TOKENS" in gemini_reason:
                        finish_reason = "length"
                        self.logger.warning("[GEMINI] Response truncated due to max_tokens limit")
                    elif "STOP" in gemini_reason:
                        finish_reason = "stop"
                    elif "SAFETY" in gemini_reason:
                        finish_reason = "content_filter"
                        self.logger.warning(f"[GEMINI] Response filtered due to safety: {gemini_reason}")
                    else:
                        finish_reason = gemini_reason.lower()

                # Extract content and function calls from parts
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        # Text content
                        if hasattr(part, 'text') and part.text:
                            content += part.text

                        # Function call
                        if hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            tool_call = {
                                "id": f"call_{fc.name}_{len(tool_calls)}",
                                "type": "function",
                                "function": {
                                    "name": fc.name,
                                    "arguments": json.dumps(dict(fc.args)) if fc.args else "{}",
                                }
                            }
                            tool_calls.append(tool_call)
                            finish_reason = "tool_calls"
                            self.logger.info(f"[GEMINI] Function call: {fc.name}")

        except Exception as e:
            self.logger.debug(f"[GEMINI] Could not extract response details: {e}")
            # Fallback to basic text extraction
            try:
                content = response.text
            except:
                content = ""

        result = {
            "content": content,
            "model": self.model_name,
            "id": getattr(response, "response_id", None),
            "finish_reason": finish_reason,
            "raw_response": response,
        }

        # Add tool_calls if present (OpenAI-compatible format)
        if tool_calls:
            result["tool_calls"] = tool_calls
            self.logger.info(f"[GEMINI] Response contains {len(tool_calls)} tool calls")

        return result
