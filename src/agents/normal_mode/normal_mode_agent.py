"""
Normal Mode Agent - OpenAI Runner.run() Style Agent Loop

Implements a simplified agent loop for 90% of queries (Normal Mode).
The LLM decides which tools to call inline, without upfront planning.

Key Features:
- Single LLM call to both reason and select tools
- Parallel tool execution within each turn
- 2-level tool loading: summaries for selection, full schema for execution
- Exit when LLM returns no tool_calls or max turns reached
- Streaming support for real-time responses

Flow:
1. Build prompt with tool summaries (~50-100 tokens each)
2. LLM decides: respond directly OR call tools
3. If tools: execute in parallel, add results to context
4. Repeat until LLM returns final response (no tools)
5. Max 10 turns to prevent infinite loops

This replaces the Planning → Execution → Synthesis 3-stage pipeline
for simple queries, reducing from 4+ LLM calls to 2-3.
"""

import json
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from dataclasses import dataclass, field

from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.agents.tools.registry import ToolRegistry
from src.agents.tools.tool_loader import get_registry
from src.agents.tools.base import ToolOutput
from src.agents.classification.models import UnifiedClassificationResult, ClassifierContext


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ToolCall:
    """Represents a single tool call from LLM"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class AgentTurn:
    """Represents one turn in the agent loop"""
    turn_number: int
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    assistant_message: str = ""
    execution_time_ms: int = 0


@dataclass
class AgentResult:
    """Final result from the agent"""
    success: bool
    response: str
    turns: List[AgentTurn] = field(default_factory=list)
    total_turns: int = 0
    total_tool_calls: int = 0
    total_execution_time_ms: int = 0
    classification: Optional[UnifiedClassificationResult] = None
    error: Optional[str] = None


# ============================================================================
# NORMAL MODE AGENT
# ============================================================================

class NormalModeAgent(LoggerMixin):
    """
    Normal Mode Agent with OpenAI Runner.run() style loop.

    The agent loop continues until:
    1. LLM returns a response without tool calls (done)
    2. Max turns reached (safety limit)
    3. Error occurs

    Usage:
        agent = NormalModeAgent()
        result = await agent.run(
            query="What is AAPL's current price?",
            classification=classification_result,
            conversation_history=[...],
        )
    """

    # Maximum turns to prevent infinite loops
    MAX_TURNS = 10

    # Default model configuration (from settings, with fallback)
    @property
    def default_model(self):
        return settings.AGENT_MODEL or "gpt-4o-mini"

    @property
    def default_provider(self):
        return settings.AGENT_PROVIDER or ProviderType.OPENAI

    def __init__(
        self,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None,
        max_turns: int = MAX_TURNS,
        registry: Optional[ToolRegistry] = None,
    ):
        """
        Initialize NormalModeAgent.

        Args:
            model_name: LLM model to use
            provider_type: Provider type (openai, gemini, etc.)
            max_turns: Maximum agent turns
            registry: Tool registry (uses singleton if not provided)
        """
        super().__init__()

        self.model_name = model_name or settings.AGENT_MODEL or "gpt-4o-mini"
        self.provider_type = provider_type or settings.AGENT_PROVIDER or ProviderType.OPENAI
        self.max_turns = max_turns

        # Get tool registry
        self.registry = registry or get_registry()

        # LLM provider
        self.llm_provider = LLMGeneratorProvider()
        self.api_key = ModelProviderFactory._get_api_key(self.provider_type)

        self.logger.info(
            f"[NORMAL_MODE] Initialized: model={self.model_name}, "
            f"max_turns={self.max_turns}, tools={len(self.registry.get_all_tools())}"
        )

    # ========================================================================
    # MAIN ENTRY POINT
    # ========================================================================

    async def run(
        self,
        query: str,
        classification: Optional[UnifiedClassificationResult] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_language: str = "en",
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> AgentResult:
        """
        Run the agent loop.

        Args:
            query: User query
            classification: Pre-computed classification (optional)
            conversation_history: Previous conversation turns
            system_language: Response language
            user_id: User identifier
            session_id: Session identifier

        Returns:
            AgentResult with response and execution details
        """
        flow_id = f"NM-{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()

        self.logger.info(f"[{flow_id}] ========================================")
        self.logger.info(f"[{flow_id}] NORMAL MODE AGENT START")
        self.logger.info(f"[{flow_id}] Query: {query[:100]}...")
        self.logger.info(f"[{flow_id}] ========================================")

        try:
            # Check if tools are needed
            if classification and not classification.requires_tools:
                self.logger.info(f"[{flow_id}] No tools required, responding directly")
                return await self._respond_without_tools(
                    query=query,
                    classification=classification,
                    conversation_history=conversation_history,
                    system_language=system_language,
                    flow_id=flow_id,
                )

            # Get relevant tools based on classification
            tools = self._get_relevant_tools(classification)

            if not tools:
                self.logger.warning(f"[{flow_id}] No tools available, responding directly")
                return await self._respond_without_tools(
                    query=query,
                    classification=classification,
                    conversation_history=conversation_history,
                    system_language=system_language,
                    flow_id=flow_id,
                )

            # Build initial messages
            messages = self._build_initial_messages(
                query=query,
                classification=classification,
                conversation_history=conversation_history,
                system_language=system_language,
                tools=tools,
                user_id=user_id,
            )

            # Run agent loop
            turns: List[AgentTurn] = []
            total_tool_calls = 0

            for turn_num in range(1, self.max_turns + 1):
                self.logger.info(f"[{flow_id}] --- Turn {turn_num}/{self.max_turns} ---")

                turn_start = datetime.now()

                # Call LLM with tools
                response = await self._call_llm_with_tools(
                    messages=messages,
                    tools=tools,
                    flow_id=flow_id,
                )

                # Parse tool calls from response
                tool_calls = self._parse_tool_calls(response)
                assistant_content = response.get("content") or ""

                turn = AgentTurn(
                    turn_number=turn_num,
                    tool_calls=tool_calls,
                    assistant_message=assistant_content or "",
                )

                # If no tool calls, we're done
                if not tool_calls:
                    turn.execution_time_ms = int(
                        (datetime.now() - turn_start).total_seconds() * 1000
                    )
                    turns.append(turn)

                    self.logger.info(
                        f"[{flow_id}] Turn {turn_num}: No tool calls, done. "
                        f"Response length: {len(assistant_content)}"
                    )

                    total_time_ms = int(
                        (datetime.now() - start_time).total_seconds() * 1000
                    )

                    return AgentResult(
                        success=True,
                        response=assistant_content,
                        turns=turns,
                        total_turns=turn_num,
                        total_tool_calls=total_tool_calls,
                        total_execution_time_ms=total_time_ms,
                        classification=classification,
                    )

                # Execute tool calls in parallel
                self.logger.info(
                    f"[{flow_id}] Turn {turn_num}: {len(tool_calls)} tool calls"
                )

                tool_results = await self._execute_tools_parallel(
                    tool_calls=tool_calls,
                    flow_id=flow_id,
                    user_id=user_id,
                    session_id=session_id,
                )

                turn.tool_results = tool_results
                turn.execution_time_ms = int(
                    (datetime.now() - turn_start).total_seconds() * 1000
                )
                turns.append(turn)
                total_tool_calls += len(tool_calls)

                # Add assistant message with tool calls to history
                messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            }
                        }
                        for tc in tool_calls
                    ]
                })

                # Add tool results to history
                for tc, result in zip(tool_calls, tool_results):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

            # Max turns reached
            self.logger.warning(f"[{flow_id}] Max turns ({self.max_turns}) reached")

            total_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            # Generate final response after max turns
            final_response = await self._generate_final_response(
                messages=messages,
                flow_id=flow_id,
            )

            return AgentResult(
                success=True,
                response=final_response,
                turns=turns,
                total_turns=self.max_turns,
                total_tool_calls=total_tool_calls,
                total_execution_time_ms=total_time_ms,
                classification=classification,
            )

        except Exception as e:
            self.logger.error(f"[{flow_id}] Agent error: {e}", exc_info=True)

            total_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return AgentResult(
                success=False,
                response="",
                error=str(e),
                total_execution_time_ms=total_time_ms,
                classification=classification,
            )

    # ========================================================================
    # STREAMING SUPPORT
    # ========================================================================

    async def run_stream(
        self,
        query: str,
        classification: Optional[UnifiedClassificationResult] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_language: str = "en",
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
        core_memory: Optional[str] = None,
        conversation_summary: Optional[str] = None,
        enable_thinking: bool = True,
        enable_llm_events: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the agent loop with streaming and thinking/reasoning output.

        Yields events for:
        - turn_start: New turn started
        - thinking: AI reasoning/thinking process
        - llm_thought: LLM thought events
        - tool_calls: Tools being called
        - tool_results: Tool execution results
        - content: Streaming content chunks
        - done: Agent completed
        - error: Error occurred

        Args:
            query: User query
            classification: Pre-computed classification
            conversation_history: Previous conversation
            system_language: Response language
            user_id: User identifier
            session_id: Session identifier
            core_memory: User profile/preferences from CoreMemory
            conversation_summary: Summary of older conversation
            enable_thinking: Enable thinking events
            enable_llm_events: Enable LLM decision events

        Yields:
            Event dictionaries with type and data
        """
        flow_id = f"NM-{uuid.uuid4().hex[:8]}"

        # Store user_id for tool execution
        self._current_user_id = user_id
        self._current_session_id = session_id

        try:
            # Emit initial thinking - analyzing query
            if enable_thinking:
                yield {
                    "type": "thinking",
                    "content": f"Analyzing query: {query[:100]}...",
                    "phase": "query_analysis",
                }

            # Check if tools are needed
            if classification and not classification.requires_tools:
                if enable_thinking:
                    yield {
                        "type": "thinking",
                        "content": "No tools required - generating direct response",
                        "phase": "routing",
                    }

                yield {"type": "turn_start", "turn": 1}

                async for chunk in self._stream_response_without_tools(
                    query=query,
                    classification=classification,
                    conversation_history=conversation_history,
                    system_language=system_language,
                    flow_id=flow_id,
                    core_memory=core_memory,
                    conversation_summary=conversation_summary,
                ):
                    yield {"type": "content", "content": chunk}

                yield {"type": "done", "total_turns": 1, "total_tool_calls": 0}
                return

            # Get relevant tools
            tools = self._get_relevant_tools(classification)

            if enable_thinking:
                tool_names = [t.get("function", {}).get("name", "") for t in tools[:5]]
                yield {
                    "type": "thinking",
                    "content": f"Selected {len(tools)} relevant tools: {', '.join(tool_names)}...",
                    "phase": "tool_selection",
                }

            if not tools:
                yield {"type": "turn_start", "turn": 1}

                async for chunk in self._stream_response_without_tools(
                    query=query,
                    classification=classification,
                    conversation_history=conversation_history,
                    system_language=system_language,
                    flow_id=flow_id,
                    core_memory=core_memory,
                    conversation_summary=conversation_summary,
                ):
                    yield {"type": "content", "content": chunk}

                yield {"type": "done", "total_turns": 1, "total_tool_calls": 0}
                return

            # Build initial messages with full context
            messages = self._build_initial_messages(
                query=query,
                classification=classification,
                conversation_history=conversation_history,
                system_language=system_language,
                tools=tools,
                core_memory=core_memory,
                conversation_summary=conversation_summary,
                user_id=user_id,
            )

            # Run agent loop
            total_tool_calls = 0

            for turn_num in range(1, self.max_turns + 1):
                yield {"type": "turn_start", "turn": turn_num}

                if enable_thinking:
                    yield {
                        "type": "thinking",
                        "content": f"Turn {turn_num}: Deciding whether to use tools or respond directly",
                        "phase": f"turn_{turn_num}_decision",
                    }

                # Call LLM with tools
                response = await self._call_llm_with_tools(
                    messages=messages,
                    tools=tools,
                    flow_id=flow_id,
                )

                tool_calls = self._parse_tool_calls(response)
                assistant_content = response.get("content") or ""

                # Emit LLM thought if we have reasoning content
                if enable_llm_events and assistant_content:
                    yield {
                        "type": "llm_thought",
                        "thought": assistant_content[:200],
                        "context": f"turn_{turn_num}",
                    }

                # If no tool calls, stream the final response from LLM
                if not tool_calls:
                    if enable_thinking:
                        yield {
                            "type": "thinking",
                            "content": "All information gathered - generating final response",
                            "phase": "synthesis",
                        }

                    # If assistant provided content directly, stream it
                    if assistant_content:
                        # Stream final response using LLM (TRUE STREAMING)
                        async for chunk in self._stream_final_response(
                            messages=messages,
                            flow_id=flow_id,
                        ):
                            yield {"type": "content", "content": chunk}

                    yield {
                        "type": "done",
                        "total_turns": turn_num,
                        "total_tool_calls": total_tool_calls,
                    }
                    return

                # Emit thinking about tool selection
                if enable_thinking:
                    tool_names = [tc.name for tc in tool_calls]
                    yield {
                        "type": "thinking",
                        "content": f"Decided to call {len(tool_calls)} tools: {', '.join(tool_names)}",
                        "phase": "tool_decision",
                    }

                # Yield tool calls info
                yield {
                    "type": "tool_calls",
                    "tools": [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in tool_calls
                    ]
                }

                # Execute tools
                tool_results = await self._execute_tools_parallel(
                    tool_calls=tool_calls,
                    flow_id=flow_id,
                    user_id=user_id,
                    session_id=session_id,
                )

                total_tool_calls += len(tool_calls)

                # Emit thinking about tool results
                if enable_thinking:
                    success_count = sum(
                        1 for r in tool_results
                        if r.get("status") in ["success", "200"]
                    )
                    yield {
                        "type": "thinking",
                        "content": f"Received {len(tool_results)} results ({success_count} successful)",
                        "phase": "tool_analysis",
                    }

                # Yield tool results
                yield {
                    "type": "tool_results",
                    "results": [
                        {"tool": tc.name, "success": r.get("status") in ["success", "200"]}
                        for tc, r in zip(tool_calls, tool_results)
                    ]
                }

                # Update messages for next turn
                messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            }
                        }
                        for tc in tool_calls
                    ]
                })

                for tc, result in zip(tool_calls, tool_results):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

            # Max turns reached - stream final response
            yield {"type": "max_turns_reached", "turns": self.max_turns}

            if enable_thinking:
                yield {
                    "type": "thinking",
                    "content": f"Max turns ({self.max_turns}) reached - synthesizing final response",
                    "phase": "max_turns_synthesis",
                }

            # Stream final response (TRUE STREAMING)
            async for chunk in self._stream_final_response(
                messages=messages,
                flow_id=flow_id,
            ):
                yield {"type": "content", "content": chunk}

            yield {
                "type": "done",
                "total_turns": self.max_turns,
                "total_tool_calls": total_tool_calls,
            }

        except Exception as e:
            self.logger.error(f"[{flow_id}] Stream error: {e}", exc_info=True)
            yield {"type": "error", "error": str(e)}

    # ========================================================================
    # TOOL MANAGEMENT
    # ========================================================================

    def _get_relevant_tools(
        self,
        classification: Optional[UnifiedClassificationResult] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get relevant tools based on classification.

        Uses 2-level loading:
        - Returns tool summaries for LLM selection (~50-100 tokens each)
        - Full schema is loaded only during execution

        Args:
            classification: Query classification result

        Returns:
            List of tool definitions in OpenAI format
        """
        if not classification or not classification.tool_categories:
            # Return all tools if no classification
            all_schemas = self.registry.get_all_schemas()
            return [
                schema.to_openai_function()
                for schema in all_schemas.values()
            ]

        # Get tools by categories
        tools = []
        seen_names = set()

        for category in classification.tool_categories:
            category_tools = self.registry.get_tools_by_category(category)

            for tool_name, tool in category_tools.items():
                if tool_name not in seen_names:
                    schema = tool.get_schema()
                    tools.append(schema.to_openai_function())
                    seen_names.add(tool_name)

        self.logger.info(
            f"[TOOLS] Selected {len(tools)} tools for categories: "
            f"{classification.tool_categories}"
        )

        return tools

    # ========================================================================
    # LLM CALLS
    # ========================================================================

    async def _call_llm_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        flow_id: str,
    ) -> Dict[str, Any]:
        """
        Call LLM with tool definitions.

        Uses OpenAI function calling format.
        """
        try:
            params = {
                "model_name": self.model_name,
                "messages": messages,
                "provider_type": self.provider_type,
                "api_key": self.api_key,
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": 4000,
                "temperature": 0.1,
            }

            response = await self.llm_provider.generate_response(**params)

            return response if isinstance(response, dict) else {"content": str(response)}

        except Exception as e:
            self.logger.error(f"[{flow_id}] LLM call failed: {e}")
            raise

    async def _respond_without_tools(
        self,
        query: str,
        classification: Optional[UnifiedClassificationResult],
        conversation_history: Optional[List[Dict[str, str]]],
        system_language: str,
        flow_id: str,
        core_memory: Optional[str] = None,
        conversation_summary: Optional[str] = None,
    ) -> AgentResult:
        """Generate response without tools (conversational/general knowledge)."""
        start_time = datetime.now()

        messages = self._build_no_tools_messages(
            query=query,
            classification=classification,
            conversation_history=conversation_history,
            system_language=system_language,
            core_memory=core_memory,
            conversation_summary=conversation_summary,
        )

        try:
            response = await self.llm_provider.generate_response(
                model_name=self.model_name,
                messages=messages,
                provider_type=self.provider_type,
                api_key=self.api_key,
                max_tokens=2000,
                temperature=0.7,
            )

            content = (response.get("content") or "") if isinstance(response, dict) else str(response)

            total_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return AgentResult(
                success=True,
                response=content,
                turns=[AgentTurn(turn_number=1, assistant_message=content)],
                total_turns=1,
                total_tool_calls=0,
                total_execution_time_ms=total_time_ms,
                classification=classification,
            )

        except Exception as e:
            self.logger.error(f"[{flow_id}] No-tools response failed: {e}")
            raise

    async def _stream_response_without_tools(
        self,
        query: str,
        classification: Optional[UnifiedClassificationResult],
        conversation_history: Optional[List[Dict[str, str]]],
        system_language: str,
        flow_id: str,
        core_memory: Optional[str] = None,
        conversation_summary: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response without tools."""
        messages = self._build_no_tools_messages(
            query=query,
            classification=classification,
            conversation_history=conversation_history,
            system_language=system_language,
            core_memory=core_memory,
            conversation_summary=conversation_summary,
        )

        async for chunk in self.llm_provider.stream_response(
            model_name=self.model_name,
            messages=messages,
            provider_type=self.provider_type,
            api_key=self.api_key,
            max_tokens=2000,
            temperature=0.7,
        ):
            yield chunk

    async def _generate_final_response(
        self,
        messages: List[Dict[str, Any]],
        flow_id: str,
    ) -> str:
        """Generate final response after max turns or when synthesis needed."""
        # Add instruction to synthesize
        messages.append({
            "role": "user",
            "content": (
                "Based on all the information gathered above, please provide "
                "a comprehensive response to the original question."
            )
        })

        try:
            response = await self.llm_provider.generate_response(
                model_name=self.model_name,
                messages=messages,
                provider_type=self.provider_type,
                api_key=self.api_key,
                max_tokens=4000,
                temperature=0.3,
            )

            return (response.get("content") or "") if isinstance(response, dict) else str(response)

        except Exception as e:
            self.logger.error(f"[{flow_id}] Final response failed: {e}")
            return "I apologize, but I encountered an error generating the response."

    async def _stream_final_response(
        self,
        messages: List[Dict[str, Any]],
        flow_id: str,
    ) -> AsyncGenerator[str, None]:
        """
        Stream final response after tools have been executed.

        This enables TRUE streaming from the LLM instead of
        returning the complete response at once.
        """
        # Add instruction to synthesize if needed
        last_msg = messages[-1] if messages else {}
        if last_msg.get("role") == "tool":
            messages.append({
                "role": "user",
                "content": (
                    "Based on all the information gathered above, please provide "
                    "a comprehensive response to the original question."
                )
            })

        try:
            async for chunk in self.llm_provider.stream_response(
                model_name=self.model_name,
                messages=messages,
                provider_type=self.provider_type,
                api_key=self.api_key,
                max_tokens=4000,
                temperature=0.3,
            ):
                yield chunk

        except Exception as e:
            self.logger.error(f"[{flow_id}] Stream final response failed: {e}")
            yield "I apologize, but I encountered an error generating the response."

    # ========================================================================
    # TOOL EXECUTION
    # ========================================================================

    async def _execute_tools_parallel(
        self,
        tool_calls: List[ToolCall],
        flow_id: str,
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tools in parallel.

        Args:
            tool_calls: List of tool calls to execute
            flow_id: Flow identifier for logging
            user_id: User identifier
            session_id: Session identifier

        Returns:
            List of tool results in same order as input
        """
        async def execute_single(tc: ToolCall) -> Dict[str, Any]:
            try:
                self.logger.debug(
                    f"[{flow_id}] Executing {tc.name}: {tc.arguments}"
                )

                result = await self.registry.execute_tool(
                    tool_name=tc.name,
                    params=tc.arguments,
                )

                # Convert ToolOutput to dict
                if isinstance(result, ToolOutput):
                    return {
                        "tool_name": result.tool_name,
                        "status": result.status,
                        "data": result.data,
                        "error": result.error,
                        "formatted_context": result.formatted_context,
                    }

                return result if isinstance(result, dict) else {"data": result}

            except Exception as e:
                self.logger.error(f"[{flow_id}] Tool {tc.name} failed: {e}")
                return {
                    "tool_name": tc.name,
                    "status": "error",
                    "error": str(e),
                }

        # Execute all tools in parallel
        results = await asyncio.gather(
            *[execute_single(tc) for tc in tool_calls],
            return_exceptions=True
        )

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "tool_name": tool_calls[i].name,
                    "status": "error",
                    "error": str(result),
                })
            else:
                processed_results.append(result)

        # Log summary
        success_count = sum(
            1 for r in processed_results
            if r.get("status") in ["success", "200"]
        )
        self.logger.info(
            f"[{flow_id}] Tools executed: {success_count}/{len(tool_calls)} success"
        )

        return processed_results

    def _parse_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """
        Parse tool calls from LLM response.

        Handles OpenAI function calling format.
        """
        tool_calls = []

        raw_calls = response.get("tool_calls", [])

        for call in raw_calls:
            try:
                if isinstance(call, dict):
                    call_id = call.get("id", f"call_{uuid.uuid4().hex[:8]}")

                    function = call.get("function", {})
                    name = function.get("name", "")

                    arguments_str = function.get("arguments", "{}")
                    if isinstance(arguments_str, str):
                        arguments = json.loads(arguments_str)
                    else:
                        arguments = arguments_str

                    if name:
                        tool_calls.append(ToolCall(
                            id=call_id,
                            name=name,
                            arguments=arguments,
                        ))

            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Failed to parse tool call: {e}")
                continue

        return tool_calls

    # ========================================================================
    # MESSAGE BUILDING
    # ========================================================================

    def _build_initial_messages(
        self,
        query: str,
        classification: Optional[UnifiedClassificationResult],
        conversation_history: Optional[List[Dict[str, str]]],
        system_language: str,
        tools: List[Dict[str, Any]],
        core_memory: Optional[str] = None,
        conversation_summary: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Build initial messages for agent loop."""
        # System prompt
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        symbols_hint = ""
        if classification and classification.symbols:
            symbols_hint = f"\nSymbols of interest: {', '.join(classification.symbols)}"

        intent_hint = ""
        if classification and classification.intent_summary:
            intent_hint = f"\nUser intent: {classification.intent_summary}"

        # Special hint for real-time info queries - MUST use web search
        real_time_hint = ""
        if classification and classification.query_type.value == "real_time_info":
            real_time_hint = """

⚠️ IMPORTANT: This is a REAL-TIME INFORMATION query.
You MUST use the webSearch tool to get current, up-to-date information.
DO NOT answer from your training data as it may be outdated.
The user is asking about something that changes over time (current leaders, recent events, latest news).
ALWAYS call webSearch first before responding."""

        # Build user context section
        user_context = ""
        if core_memory:
            user_context += f"\n\n<USER_PROFILE>\n{core_memory}\n</USER_PROFILE>"
        if conversation_summary:
            user_context += f"\n\n<CONVERSATION_SUMMARY>\n{conversation_summary}\n</CONVERSATION_SUMMARY>"

        # User ID for tool calls
        user_id_hint = ""
        if user_id:
            user_id_hint = f"\nUser ID: {user_id} (use this for memory/profile tool calls)"

        system_prompt = f"""You are a professional financial analyst assistant with access to real-time market data tools.

Current Date: {current_date}
Response Language: {system_language.upper()}
{symbols_hint}
{intent_hint}
{user_id_hint}
{real_time_hint}
{user_context}

INSTRUCTIONS:
1. Analyze the user's query and determine what information is needed
2. Use the available tools to gather real-time data
3. You can call multiple tools in parallel if they are independent
4. After gathering data, provide a comprehensive analysis
5. If the user asks about specific stocks, always get current prices first
6. Be concise but thorough in your analysis
7. IMPORTANT: When calling memory/profile tools, use the User ID provided above

TOOL USAGE GUIDELINES:
- Call tools when you need real-time data (prices, indicators, news)
- Don't call tools for static knowledge questions (definitions, concepts that never change)
- EXCEPTION: For real-time info queries (current leaders, latest events), ALWAYS use webSearch
- If a tool fails, try an alternative approach or explain the limitation
- For memory tools (searchConversationHistory, memoryUserEdits), ALWAYS use the User ID: {user_id or 'default'}

WEB SEARCH GUIDELINES (webSearch tool):
- ALWAYS use webSearch for real-time information queries (current leaders, recent events, latest news)
- Use webSearch when financial data tools cannot provide sufficient information
- Good use cases: current political leaders, CEO positions, recent company announcements, market-moving events, regulatory changes
- Do NOT use for: basic stock prices, financials, technical analysis (use dedicated FMP tools instead)
- Limit to 3-5 results to avoid overwhelming context
- For real-time info queries, webSearch is MANDATORY - do not answer from training data

RESPONSE FORMAT:
- Respond in {system_language.upper()} language
- Use clear formatting with sections and bullet points
- Include relevant numbers and percentages
- Provide actionable insights when appropriate
"""

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 turns
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

        # Add current query
        messages.append({"role": "user", "content": query})

        return messages

    def _build_no_tools_messages(
        self,
        query: str,
        classification: Optional[UnifiedClassificationResult],
        conversation_history: Optional[List[Dict[str, str]]],
        system_language: str,
        core_memory: Optional[str] = None,
        conversation_summary: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build messages for no-tools response."""
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        query_type = ""
        if classification:
            query_type = classification.query_type.value

        # Build user context section
        user_context = ""
        if core_memory:
            user_context += f"\n\n<USER_PROFILE>\n{core_memory}\n</USER_PROFILE>"
        if conversation_summary:
            user_context += f"\n\n<CONVERSATION_SUMMARY>\n{conversation_summary}\n</CONVERSATION_SUMMARY>"

        system_prompt = f"""You are a friendly financial assistant.

Current Date: {current_date}
Response Language: {system_language.upper()}
Query Type: {query_type}
{user_context}

Respond naturally and helpfully. Be conversational for greetings,
and educational for general knowledge questions.
When answering questions about user's profile or past conversations,
use the USER_PROFILE and CONVERSATION_SUMMARY provided above."""

        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            for msg in conversation_history[-5:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": query})

        return messages


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

_agent_instance: Optional[NormalModeAgent] = None


def get_normal_mode_agent(
    model_name: Optional[str] = None,
    provider_type: Optional[str] = None,
) -> NormalModeAgent:
    """
    Get singleton NormalModeAgent instance.

    Args:
        model_name: Override model name
        provider_type: Override provider type

    Returns:
        NormalModeAgent instance
    """
    global _agent_instance

    if _agent_instance is None:
        _agent_instance = NormalModeAgent(
            model_name=model_name,
            provider_type=provider_type,
        )

    return _agent_instance


def reset_agent():
    """Reset singleton instance (for testing)."""
    global _agent_instance
    _agent_instance = None