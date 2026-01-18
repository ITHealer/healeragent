import json
import base64
import google.generativeai as genai
from google.generativeai import protos
from google.protobuf import struct_pb2
from google.protobuf import json_format
from typing import Dict, Any, List, Optional, AsyncGenerator, Union

from src.providers.base_provider import ModelProvider
from src.utils.logger.custom_logging import LoggerMixin


# Check if protos.Part supports thought_signature field
def _check_thought_signature_support():
    """Check if the installed SDK supports thought_signature field on Part."""
    try:
        part = protos.Part()
        descriptor = part.DESCRIPTOR
        field_names = [f.name for f in descriptor.fields]
        return 'thought_signature' in field_names
    except Exception:
        return False


# Cache the check result
_THOUGHT_SIGNATURE_SUPPORTED = _check_thought_signature_support()


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
                # Handle chunks that may contain FunctionCall instead of text
                # chunk.text raises ValueError if no valid text Part exists
                try:
                    if hasattr(chunk, 'text') and chunk.text:
                        yield chunk.text
                except ValueError:
                    # This happens when response contains FunctionCall instead of text
                    # For Gemini 2.5+, finish_reason=10 means invalid FunctionCall
                    # Log and continue - the caller should handle function calls separately
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        candidate = chunk.candidates[0]
                        finish_reason = getattr(candidate, 'finish_reason', None)
                        if finish_reason:
                            self.logger.warning(
                                f"[GEMINI] Stream chunk has no text (finish_reason={finish_reason}). "
                                f"Response may contain FunctionCall instead of text."
                            )
                    continue

        except Exception as e:
            self.logger.error(f"Error streaming Gemini completion: {str(e)}")
            raise

    def supports_feature(self, feature_name: str) -> bool:
        """Check if provider supports a specific feature"""
        # Gemini 3+ models (preview) support thinking mode with thought_signatures
        is_thinking_model = any(x in self.model_name.lower() for x in [
            "gemini-3", "gemini-2.5", "preview", "flash-preview", "pro-preview"
        ])

        feature_support = {
            "thinking_mode": is_thinking_model,
            "thought_signatures": is_thinking_model,  # Requires special handling
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

        For Gemini 3+ function calling, uses proper protobuf types (protos.Part)
        to correctly include thought_signature. Raw dicts don't work because
        the SDK's to_part() doesn't recognize 'function_call' as a dict key.
        See: https://ai.google.dev/gemini-api/docs/thought-signatures
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
                # For Gemini 3+, thought_signature MUST be preserved
                # See: https://ai.google.dev/gemini-api/docs/thought-signatures
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    if tc.get("type") == "function":
                        func = tc.get("function", {})
                        try:
                            args = json.loads(func.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            args = {}

                        # Build function call part
                        # Priority: Use original proto bytes (preserves thought_signature)
                        fc_part = self._build_function_call_part(
                            name=func.get("name", ""),
                            args=args,
                            thought_signature=tc.get("thought_signature"),
                            part_proto_bytes=tc.get("_part_proto_bytes")  # Original Part proto
                        )
                        parts.append(fc_part)

                if parts:
                    gemini_messages.append({"role": "model", "parts": parts})

            elif role == "tool":
                # Tool response - add as function_response using protobuf types
                tool_call_id = msg.get("tool_call_id", "")
                func_name = self._find_function_name(tool_call_id, openai_messages)

                # Parse content if it's JSON
                try:
                    result = json.loads(content) if content else {}
                except json.JSONDecodeError:
                    result = {"result": content}

                # Build function response part using protobuf types
                fr_part = self._build_function_response_part(
                    name=func_name,
                    response=result
                )
                gemini_messages.append({
                    "role": "user",
                    "parts": [fr_part]
                })

        # If only system message was provided, add it as user
        if system_content and not gemini_messages:
            gemini_messages.append({"role": "user", "parts": [system_content]})

        return gemini_messages

    def _build_function_call_part(
        self,
        name: str,
        args: Dict[str, Any],
        thought_signature: Optional[str] = None,
        part_proto_bytes: Optional[str] = None
    ) -> protos.Part:
        """
        Build a Gemini Part with FunctionCall using protobuf types.

        For Gemini 3+ models, thought_signature MUST be included on the first
        function call part in each step, otherwise the API returns 400 error.

        Args:
            name: Function name
            args: Function arguments as dict
            thought_signature: Encrypted thought signature from previous response
            part_proto_bytes: Base64-encoded original Part proto bytes (preserves thought_signature)

        Returns:
            protos.Part with function_call set
        """
        # Method 1 (BEST): Deserialize original Part proto bytes
        # This preserves thought_signature even if SDK doesn't expose it
        if part_proto_bytes:
            try:
                proto_bytes = base64.b64decode(part_proto_bytes)
                part = protos.Part()
                # Try to parse from bytes (preserves all fields including thought_signature)
                if hasattr(part, 'ParseFromString'):
                    part.ParseFromString(proto_bytes)
                elif hasattr(part, '_pb'):
                    part._pb.ParseFromString(proto_bytes)
                self.logger.info(f"[GEMINI] Restored Part from proto bytes for {name}")
                return part
            except Exception as e:
                self.logger.warning(f"[GEMINI] Could not restore Part from proto bytes: {e}")
                # Fall through to other methods

        # Method 2: Try json_format.ParseDict if SDK supports thought_signature
        if thought_signature and thought_signature not in ["__FROM_PROTO_BYTES__", "skip_thought_signature_validator"]:
            if _THOUGHT_SIGNATURE_SUPPORTED:
                try:
                    part_dict = {
                        "functionCall": {
                            "name": name,
                            "args": args
                        },
                        "thoughtSignature": thought_signature
                    }
                    part = protos.Part()
                    json_format.ParseDict(part_dict, part, ignore_unknown_fields=False)
                    self.logger.debug(f"[GEMINI] Built Part with thought_signature via json_format for {name}")
                    return part
                except Exception as e:
                    self.logger.debug(f"[GEMINI] json_format.ParseDict failed: {e}")

        # Method 3: Build Part using protobuf types directly
        args_struct = struct_pb2.Struct()
        args_struct.update(args)

        fc = protos.FunctionCall(
            name=name,
            args=args_struct
        )

        part = protos.Part(function_call=fc)

        # Try to set thought_signature if present and valid
        if thought_signature and thought_signature not in ["__FROM_PROTO_BYTES__"]:
            sig_to_use = thought_signature
            if thought_signature == "skip_thought_signature_validator":
                # Use the skip validator value
                sig_to_use = "skip_thought_signature_validator"

            sig_set = False

            # Try direct attribute
            try:
                part.thought_signature = sig_to_use
                sig_set = True
                self.logger.debug(f"[GEMINI] Set thought_signature directly for {name}")
            except (AttributeError, TypeError):
                pass

            # Try via _pb
            if not sig_set:
                try:
                    if hasattr(part, '_pb') and hasattr(part._pb, 'thought_signature'):
                        part._pb.thought_signature = sig_to_use
                        sig_set = True
                        self.logger.debug(f"[GEMINI] Set thought_signature via _pb for {name}")
                except Exception:
                    pass

            if not sig_set:
                self.logger.warning(
                    f"[GEMINI] Could not set thought_signature for {name}. "
                    f"Gemini 3+ function calling may fail with 400 error."
                )

        return part

    def _build_function_response_part(
        self,
        name: str,
        response: Dict[str, Any]
    ) -> protos.Part:
        """
        Build a Gemini Part with FunctionResponse using protobuf types.

        Args:
            name: Function name
            response: Function response as dict

        Returns:
            protos.Part with function_response set
        """
        # Convert response dict to Struct protobuf
        response_struct = struct_pb2.Struct()
        response_struct.update(response)

        # Create FunctionResponse protobuf
        fr = protos.FunctionResponse(
            name=name,
            response=response_struct
        )

        # Create Part with function_response
        return protos.Part(function_response=fr)

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

        Handles text responses, function calls, and thought_signatures.

        For Gemini 3+ models with thinking enabled:
        - thought_signature MUST be preserved and sent back for function calling to work
        - thinking_content contains the model's reasoning (for streaming/display)
        See: https://ai.google.dev/gemini-api/docs/thought-signatures
        """
        finish_reason = "stop"
        content = ""
        tool_calls = []
        thinking_content = ""  # Model's reasoning/thinking process

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

                # Extract content, thinking, and function calls from parts
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for idx, part in enumerate(candidate.content.parts):
                        # Text content
                        if hasattr(part, 'text') and part.text:
                            content += part.text

                        # Thinking content (Gemini's chain-of-thought)
                        # This is the model's internal reasoning process
                        if hasattr(part, 'thought') and part.thought:
                            thinking_content += part.thought
                            self.logger.debug(f"[GEMINI] Thinking: {part.thought[:100]}...")

                        # Function call with thought_signature
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

                            # CRITICAL: Store the original Part proto as base64 bytes
                            # This preserves thought_signature even if SDK doesn't expose it
                            # When sending back, we deserialize and reuse the original Part
                            try:
                                # Serialize the Part proto to bytes (preserves all fields including unknown)
                                part_bytes = part.SerializeToString() if hasattr(part, 'SerializeToString') else None
                                if not part_bytes and hasattr(part, '_pb'):
                                    part_bytes = part._pb.SerializeToString()
                                if part_bytes:
                                    tool_call["_part_proto_bytes"] = base64.b64encode(part_bytes).decode('ascii')
                                    self.logger.debug(f"[GEMINI] Stored Part proto bytes for {fc.name} (size={len(part_bytes)})")
                            except Exception as e:
                                self.logger.debug(f"[GEMINI] Could not serialize Part proto: {e}")

                            # Also try to extract thought_signature for logging/debugging
                            thought_sig = None

                            # Pattern 1: part.thought_signature (snake_case)
                            if hasattr(part, 'thought_signature') and part.thought_signature:
                                thought_sig = part.thought_signature
                                self.logger.debug(f"[GEMINI] Found thought_signature on part (snake_case)")

                            # Pattern 2: Check _pb attribute for protobuf access
                            if not thought_sig:
                                try:
                                    if hasattr(part, '_pb') and hasattr(part._pb, 'thought_signature'):
                                        thought_sig = part._pb.thought_signature
                                        self.logger.debug(f"[GEMINI] Found thought_signature via protobuf")
                                except Exception:
                                    pass

                            # For Gemini 3+ models without thought_signature, use skip validator
                            is_gemini3_model = any(x in self.model_name.lower() for x in [
                                "gemini-3", "flash-preview", "pro-preview", "gemini-2.5"
                            ])

                            if not thought_sig and is_gemini3_model and idx == 0:
                                # If we have proto bytes, the thought_signature is preserved there
                                if "_part_proto_bytes" in tool_call:
                                    thought_sig = "__FROM_PROTO_BYTES__"  # Marker to use bytes
                                    self.logger.info(f"[GEMINI] Function call {fc.name} - will use preserved proto bytes")
                                else:
                                    thought_sig = "skip_thought_signature_validator"
                                    self.logger.warning(
                                        f"[GEMINI] Using skip_thought_signature_validator for {fc.name} "
                                        f"(Gemini 3+ fallback - may affect performance)"
                                    )

                            if thought_sig:
                                tool_call["thought_signature"] = thought_sig

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

        # Add thinking content if present (for UI display)
        if thinking_content:
            result["thinking_content"] = thinking_content

        # Add tool_calls if present (OpenAI-compatible format with thought_signature)
        if tool_calls:
            result["tool_calls"] = tool_calls
            self.logger.info(f"[GEMINI] Response contains {len(tool_calls)} tool calls")

        return result
