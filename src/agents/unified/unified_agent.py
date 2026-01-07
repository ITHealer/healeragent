"""
Unified Agent - Complexity-Based Execution

Merges Normal Mode Agent and Deep Research into a single agent that
adapts its execution strategy based on query complexity.

Strategies:
- SIMPLE (1-2 tools): Direct execution, single LLM call, max 2 turns
- MEDIUM (3-5 tools): Iterative agent loop, max 4 turns
- COMPLEX (6+ tools): Planning + parallel execution, max 6 turns

This replaces:
- src/agents/normal_mode/normal_mode_agent.py (for SIMPLE/MEDIUM)
- src/agents/streaming/streaming_chat_handler.py (for COMPLEX)

Usage:
    agent = UnifiedAgent()

    # With Router decision
    async for event in agent.run_stream(
        query="Analyze NVDA",
        router_decision=router_decision,
        classification=classification,
        ...
    ):
        yield emitter.emit(event)
"""

import json
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field

from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ModelProviderFactory, ProviderType

from src.agents.tools.tool_catalog import ToolCatalog, get_tool_catalog
from src.agents.tools.registry import ToolRegistry
from src.agents.tools.tool_loader import get_registry
from src.agents.tools.base import ToolOutput
from src.agents.router.llm_tool_router import (
    RouterDecision,
    Complexity,
    ExecutionStrategy,
)
from src.agents.classification.models import UnifiedClassificationResult

# Skill System for domain-specific prompts
from src.agents.skills.skill_registry import SkillRegistry, get_skill_registry
from src.agents.skills.skill_base import SkillContext


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ToolCall:
    """Represents a single tool call from LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class AgentTurn:
    """Represents one turn in the agent loop."""
    turn_number: int
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    assistant_message: str = ""
    execution_time_ms: int = 0


@dataclass
class UnifiedAgentResult:
    """Final result from the unified agent."""
    success: bool
    response: str
    turns: List[AgentTurn] = field(default_factory=list)
    total_turns: int = 0
    total_tool_calls: int = 0
    total_execution_time_ms: int = 0
    complexity: Optional[Complexity] = None
    strategy: Optional[ExecutionStrategy] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "response": self.response,
            "total_turns": self.total_turns,
            "total_tool_calls": self.total_tool_calls,
            "total_execution_time_ms": self.total_execution_time_ms,
            "complexity": self.complexity.value if self.complexity else None,
            "strategy": self.strategy.value if self.strategy else None,
            "error": self.error,
        }


# ============================================================================
# UNIFIED AGENT
# ============================================================================

class UnifiedAgent(LoggerMixin):
    """
    Unified Agent with complexity-based execution strategies.

    Adapts behavior based on RouterDecision:
    - SIMPLE: Fast direct execution
    - MEDIUM: Iterative agent loop
    - COMPLEX: Planning + parallel execution

    Features:
    - Single codebase for all query types
    - Streaming support for all strategies
    - Per-tool timeout (5s default)
    - Partial success handling
    - Memory integration (Working Memory, Core Memory)
    """

    # Default tool timeout
    DEFAULT_TOOL_TIMEOUT = 5.0

    def __init__(
        self,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None,
        catalog: Optional[ToolCatalog] = None,
        registry: Optional[ToolRegistry] = None,
    ):
        """
        Initialize unified agent.

        Args:
            model_name: LLM model for agent
            provider_type: Provider type
            catalog: Tool catalog (uses singleton if not provided)
            registry: Tool registry (uses singleton if not provided)
        """
        super().__init__()

        self.model_name = model_name or settings.AGENT_MODEL or "gpt-4o-mini"
        self.provider_type = provider_type or settings.AGENT_PROVIDER or ProviderType.OPENAI

        # Get tool catalog and registry
        self.catalog = catalog or get_tool_catalog()
        self.registry = registry or get_registry()

        # Skill registry for domain-specific prompts
        self.skill_registry = get_skill_registry()

        # LLM provider
        self.llm_provider = LLMGeneratorProvider()
        self.api_key = ModelProviderFactory._get_api_key(self.provider_type)

        self.logger.info(
            f"[UNIFIED_AGENT] Initialized: model={self.model_name}, "
            f"tools={len(self.catalog.get_tool_names())}, "
            f"skills={len(self.skill_registry.get_available_skills())}"
        )

    # =========================================================================
    # MAIN ENTRY POINTS
    # =========================================================================

    async def run(
        self,
        query: str,
        router_decision: RouterDecision,
        classification: Optional[UnifiedClassificationResult] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_language: str = "en",
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
        core_memory: Optional[str] = None,
        conversation_summary: Optional[str] = None,
    ) -> UnifiedAgentResult:
        """
        Run agent (non-streaming).

        Routes to appropriate strategy based on complexity.
        """
        flow_id = f"UA-{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()

        self.logger.info(f"[{flow_id}] ========================================")
        self.logger.info(f"[{flow_id}] UNIFIED AGENT START")
        self.logger.info(f"[{flow_id}] Complexity: {router_decision.complexity.value}")
        self.logger.info(f"[{flow_id}] Strategy: {router_decision.execution_strategy.value}")
        self.logger.info(f"[{flow_id}] Tools: {router_decision.selected_tools}")
        self.logger.info(f"[{flow_id}] ========================================")

        try:
            # No tools selected - direct response
            if not router_decision.selected_tools:
                return await self._respond_without_tools(
                    query=query,
                    classification=classification,
                    conversation_history=conversation_history,
                    system_language=system_language,
                    flow_id=flow_id,
                    core_memory=core_memory,
                    conversation_summary=conversation_summary,
                )

            # Route to strategy
            if router_decision.complexity == Complexity.SIMPLE:
                return await self._execute_simple(
                    query=query,
                    router_decision=router_decision,
                    classification=classification,
                    conversation_history=conversation_history,
                    system_language=system_language,
                    flow_id=flow_id,
                    user_id=user_id,
                    session_id=session_id,
                    core_memory=core_memory,
                    conversation_summary=conversation_summary,
                )

            elif router_decision.complexity == Complexity.MEDIUM:
                return await self._execute_iterative(
                    query=query,
                    router_decision=router_decision,
                    classification=classification,
                    conversation_history=conversation_history,
                    system_language=system_language,
                    flow_id=flow_id,
                    user_id=user_id,
                    session_id=session_id,
                    core_memory=core_memory,
                    conversation_summary=conversation_summary,
                )

            else:  # COMPLEX
                return await self._execute_complex(
                    query=query,
                    router_decision=router_decision,
                    classification=classification,
                    conversation_history=conversation_history,
                    system_language=system_language,
                    flow_id=flow_id,
                    user_id=user_id,
                    session_id=session_id,
                    core_memory=core_memory,
                    conversation_summary=conversation_summary,
                )

        except Exception as e:
            self.logger.error(f"[{flow_id}] Agent error: {e}", exc_info=True)
            total_time = int((datetime.now() - start_time).total_seconds() * 1000)

            return UnifiedAgentResult(
                success=False,
                response="",
                error=str(e),
                total_execution_time_ms=total_time,
                complexity=router_decision.complexity,
                strategy=router_decision.execution_strategy,
            )

    async def run_stream(
        self,
        query: str,
        router_decision: RouterDecision,
        classification: Optional[UnifiedClassificationResult] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_language: str = "en",
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
        core_memory: Optional[str] = None,
        conversation_summary: Optional[str] = None,
        enable_reasoning: bool = True,
        images: Optional[List[Any]] = None,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run agent with streaming.

        Yields events:
        - reasoning: Agent thought process
        - tool_calls: Tools being called
        - tool_results: Tool execution results
        - content: Response content chunks
        - done: Completion
        - error: Error occurred

        Args:
            model_name: Override model for final response (uses instance default if not provided)
            provider_type: Override provider (uses instance default if not provided)
        """
        flow_id = f"UA-{uuid.uuid4().hex[:8]}"

        # Resolve effective model/provider from user input or instance defaults
        effective_model = model_name or self.model_name
        effective_provider = provider_type or self.provider_type

        self.logger.debug(
            f"[{flow_id}] Using model={effective_model}, provider={effective_provider}"
        )

        try:
            # Emit routing decision
            if enable_reasoning:
                yield {
                    "type": "reasoning",
                    "phase": "routing",
                    "action": "decision",
                    "content": (
                        f"Selected {len(router_decision.selected_tools)} tools: "
                        f"{', '.join(router_decision.selected_tools[:5])}"
                    ),
                    "metadata": {
                        "complexity": router_decision.complexity.value,
                        "strategy": router_decision.execution_strategy.value,
                        "tools": router_decision.selected_tools,
                    },
                }

            # No tools - stream direct response
            if not router_decision.selected_tools:
                if enable_reasoning:
                    yield {
                        "type": "reasoning",
                        "phase": "execution",
                        "action": "thought",
                        "content": "No tools needed - generating direct response",
                    }

                async for chunk in self._stream_response_without_tools(
                    query=query,
                    classification=classification,
                    conversation_history=conversation_history,
                    system_language=system_language,
                    flow_id=flow_id,
                    core_memory=core_memory,
                    conversation_summary=conversation_summary,
                    images=images,
                    model_name=effective_model,
                    provider_type=effective_provider,
                ):
                    yield {"type": "content", "content": chunk}

                yield {"type": "done", "total_turns": 1, "total_tool_calls": 0}
                return

            # Route to streaming strategy (pass effective model/provider)
            if router_decision.complexity == Complexity.SIMPLE:
                async for event in self._stream_simple(
                    query=query,
                    router_decision=router_decision,
                    classification=classification,
                    conversation_history=conversation_history,
                    system_language=system_language,
                    flow_id=flow_id,
                    user_id=user_id,
                    session_id=session_id,
                    core_memory=core_memory,
                    conversation_summary=conversation_summary,
                    enable_reasoning=enable_reasoning,
                    images=images,
                    model_name=effective_model,
                    provider_type=effective_provider,
                ):
                    yield event

            elif router_decision.complexity == Complexity.MEDIUM:
                async for event in self._stream_iterative(
                    query=query,
                    router_decision=router_decision,
                    classification=classification,
                    conversation_history=conversation_history,
                    system_language=system_language,
                    flow_id=flow_id,
                    user_id=user_id,
                    session_id=session_id,
                    core_memory=core_memory,
                    conversation_summary=conversation_summary,
                    enable_reasoning=enable_reasoning,
                    images=images,
                    model_name=effective_model,
                    provider_type=effective_provider,
                ):
                    yield event

            else:  # COMPLEX
                async for event in self._stream_complex(
                    query=query,
                    router_decision=router_decision,
                    classification=classification,
                    conversation_history=conversation_history,
                    system_language=system_language,
                    flow_id=flow_id,
                    user_id=user_id,
                    session_id=session_id,
                    core_memory=core_memory,
                    conversation_summary=conversation_summary,
                    enable_reasoning=enable_reasoning,
                    images=images,
                    model_name=effective_model,
                    provider_type=effective_provider,
                ):
                    yield event

        except Exception as e:
            self.logger.error(f"[{flow_id}] Stream error: {e}", exc_info=True)
            yield {"type": "error", "error": str(e)}

    # =========================================================================
    # SIMPLE STRATEGY (1-2 tools, direct execution)
    # =========================================================================

    async def _execute_simple(
        self,
        query: str,
        router_decision: RouterDecision,
        classification: Optional[UnifiedClassificationResult],
        conversation_history: Optional[List[Dict[str, str]]],
        system_language: str,
        flow_id: str,
        user_id: Optional[int],
        session_id: Optional[str],
        core_memory: Optional[str],
        conversation_summary: Optional[str],
    ) -> UnifiedAgentResult:
        """
        Simple strategy: Execute all tools once, synthesize response.

        Best for: Price checks, simple lookups
        """
        start_time = datetime.now()
        self.logger.info(f"[{flow_id}] SIMPLE strategy: {router_decision.selected_tools}")

        # Execute all tools in parallel
        tool_results = await self._execute_tools_parallel(
            tool_names=router_decision.selected_tools,
            query=query,
            symbols=classification.symbols if classification else [],
            flow_id=flow_id,
            user_id=user_id,
            session_id=session_id,
        )

        # Build synthesis prompt
        messages = self._build_synthesis_messages(
            query=query,
            tool_results=tool_results,
            classification=classification,
            conversation_history=conversation_history,
            system_language=system_language,
            core_memory=core_memory,
            conversation_summary=conversation_summary,
        )

        # Generate response
        response = await self.llm_provider.generate_response(
            model_name=self.model_name,
            messages=messages,
            provider_type=self.provider_type,
            api_key=self.api_key,
            max_tokens=2000,
            temperature=0.3,
        )

        content = (
            response.get("content", "")
            if isinstance(response, dict)
            else str(response)
        )

        total_time = int((datetime.now() - start_time).total_seconds() * 1000)

        # Build tool calls from actual results (expanded for multiple symbols)
        actual_tool_calls = []
        for i, result in enumerate(tool_results):
            tool_name = result.get("tool_name", "unknown")
            symbol = result.get("symbol")
            args = {"symbol": symbol} if symbol else {}
            actual_tool_calls.append(ToolCall(id=f"call_{i}", name=tool_name, arguments=args))

        return UnifiedAgentResult(
            success=True,
            response=content,
            turns=[AgentTurn(
                turn_number=1,
                tool_calls=actual_tool_calls,
                tool_results=tool_results,
            )],
            total_turns=1,
            total_tool_calls=len(tool_results),  # Count actual tool calls (expanded)
            total_execution_time_ms=total_time,
            complexity=Complexity.SIMPLE,
            strategy=ExecutionStrategy.DIRECT,
        )

    async def _stream_simple(
        self,
        query: str,
        router_decision: RouterDecision,
        classification: Optional[UnifiedClassificationResult],
        conversation_history: Optional[List[Dict[str, str]]],
        system_language: str,
        flow_id: str,
        user_id: Optional[int],
        session_id: Optional[str],
        core_memory: Optional[str],
        conversation_summary: Optional[str],
        enable_reasoning: bool,
        images: Optional[List[Any]],
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream simple strategy."""
        effective_model = model_name or self.model_name
        effective_provider = provider_type or self.provider_type
        symbols = classification.symbols if classification else []
        num_symbols = len(symbols)

        # Calculate expected tool calls (expanded for multiple symbols)
        expected_calls = 0
        for tool_name in router_decision.selected_tools:
            schema = self.catalog.get_full_schema(tool_name)
            requires_symbol = False
            if schema and schema.parameters:
                for param in schema.parameters:
                    if param.get("name") == "symbol" and param.get("required", False):
                        requires_symbol = True
                        break
            if requires_symbol and num_symbols > 1:
                expected_calls += num_symbols
            else:
                expected_calls += 1

        if enable_reasoning:
            content = f"Executing {expected_calls} tool calls"
            if num_symbols > 1:
                content += f" ({len(router_decision.selected_tools)} tools × {num_symbols} symbols)"
            yield {
                "type": "reasoning",
                "phase": "tool_execution",
                "action": "start",
                "content": content,
            }

        # Execute tools (this will expand for multiple symbols)
        tool_results = await self._execute_tools_parallel(
            tool_names=router_decision.selected_tools,
            query=query,
            symbols=symbols,
            flow_id=flow_id,
            user_id=user_id,
            session_id=session_id,
        )

        # Emit tool calls (from actual results, showing expansion)
        yield {
            "type": "tool_calls",
            "tools": [
                {
                    "name": r.get("tool_name", "unknown"),
                    "arguments": {"symbol": r.get("symbol")} if r.get("symbol") else {},
                }
                for r in tool_results
            ],
        }

        # Emit results
        success_count = sum(1 for r in tool_results if r.get("status") in ["success", "200"])
        yield {
            "type": "tool_results",
            "results": [
                {
                    "tool": r.get("tool_name", ""),
                    "symbol": r.get("symbol"),
                    "success": r.get("status") in ["success", "200"],
                }
                for r in tool_results
            ],
        }

        if enable_reasoning:
            yield {
                "type": "reasoning",
                "phase": "tool_execution",
                "action": "complete",
                "content": f"Received {success_count}/{len(tool_results)} successful results",
            }

        # Stream synthesis
        if enable_reasoning:
            yield {
                "type": "reasoning",
                "phase": "synthesis",
                "action": "start",
                "content": "Generating response from tool data",
            }

        messages = self._build_synthesis_messages(
            query=query,
            tool_results=tool_results,
            classification=classification,
            conversation_history=conversation_history,
            system_language=system_language,
            core_memory=core_memory,
            conversation_summary=conversation_summary,
        )

        async for chunk in self.llm_provider.stream_response(
            model_name=effective_model,
            messages=messages,
            provider_type=effective_provider,
            api_key=self.api_key,
            max_tokens=2000,
            temperature=0.3,
        ):
            yield {"type": "content", "content": chunk}

        yield {
            "type": "done",
            "total_turns": 1,
            "total_tool_calls": len(tool_results),  # Use actual count (expanded)
        }

    # =========================================================================
    # MEDIUM/ITERATIVE STRATEGY (3-5 tools, agent loop)
    # =========================================================================

    async def _execute_iterative(
        self,
        query: str,
        router_decision: RouterDecision,
        classification: Optional[UnifiedClassificationResult],
        conversation_history: Optional[List[Dict[str, str]]],
        system_language: str,
        flow_id: str,
        user_id: Optional[int],
        session_id: Optional[str],
        core_memory: Optional[str],
        conversation_summary: Optional[str],
    ) -> UnifiedAgentResult:
        """
        Iterative strategy: Agent loop with tool calling.

        Best for: Analysis queries, multi-step lookups
        """
        start_time = datetime.now()
        max_turns = router_decision.suggested_max_turns

        self.logger.info(f"[{flow_id}] ITERATIVE strategy: max_turns={max_turns}")

        # Get full schemas for selected tools
        tools = self.catalog.get_openai_functions(router_decision.selected_tools)

        # Build initial messages
        messages = self._build_agent_messages(
            query=query,
            classification=classification,
            conversation_history=conversation_history,
            system_language=system_language,
            core_memory=core_memory,
            conversation_summary=conversation_summary,
            tools=tools,
            user_id=user_id,
            complexity=router_decision.complexity,
        )

        turns = []
        total_tool_calls = 0

        for turn_num in range(1, max_turns + 1):
            self.logger.info(f"[{flow_id}] Turn {turn_num}/{max_turns}")

            # Call LLM with tools
            response = await self._call_llm_with_tools(messages, tools, flow_id)

            tool_calls = self._parse_tool_calls(response)
            assistant_content = response.get("content") or ""

            turn = AgentTurn(
                turn_number=turn_num,
                tool_calls=tool_calls,
                assistant_message=assistant_content,
            )

            # No tool calls - done
            if not tool_calls:
                turns.append(turn)
                total_time = int((datetime.now() - start_time).total_seconds() * 1000)

                return UnifiedAgentResult(
                    success=True,
                    response=assistant_content,
                    turns=turns,
                    total_turns=turn_num,
                    total_tool_calls=total_tool_calls,
                    total_execution_time_ms=total_time,
                    complexity=Complexity.MEDIUM,
                    strategy=ExecutionStrategy.ITERATIVE,
                )

            # Execute tools
            tool_results = await self._execute_tool_calls(
                tool_calls=tool_calls,
                flow_id=flow_id,
                user_id=user_id,
                session_id=session_id,
            )

            turn.tool_results = tool_results
            turns.append(turn)
            total_tool_calls += len(tool_calls)

            # Update messages
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
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tc, result in zip(tool_calls, tool_results):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        # Max turns reached - generate final response
        self.logger.warning(f"[{flow_id}] Max turns ({max_turns}) reached")

        messages.append({
            "role": "user",
            "content": "Please provide your final response based on all the information gathered.",
        })

        response = await self.llm_provider.generate_response(
            model_name=self.model_name,
            messages=messages,
            provider_type=self.provider_type,
            api_key=self.api_key,
            max_tokens=4000,
            temperature=0.3,
        )

        content = response.get("content", "") if isinstance(response, dict) else str(response)
        total_time = int((datetime.now() - start_time).total_seconds() * 1000)

        return UnifiedAgentResult(
            success=True,
            response=content,
            turns=turns,
            total_turns=max_turns,
            total_tool_calls=total_tool_calls,
            total_execution_time_ms=total_time,
            complexity=Complexity.MEDIUM,
            strategy=ExecutionStrategy.ITERATIVE,
        )

    async def _stream_iterative(
        self,
        query: str,
        router_decision: RouterDecision,
        classification: Optional[UnifiedClassificationResult],
        conversation_history: Optional[List[Dict[str, str]]],
        system_language: str,
        flow_id: str,
        user_id: Optional[int],
        session_id: Optional[str],
        core_memory: Optional[str],
        conversation_summary: Optional[str],
        enable_reasoning: bool,
        images: Optional[List[Any]],
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream iterative strategy."""
        effective_model = model_name or self.model_name
        effective_provider = provider_type or self.provider_type
        max_turns = router_decision.suggested_max_turns

        if enable_reasoning:
            yield {
                "type": "reasoning",
                "phase": "agent_loop",
                "action": "start",
                "content": f"Starting iterative agent loop (max {max_turns} turns)",
            }

        # Get tools
        tools = self.catalog.get_openai_functions(router_decision.selected_tools)

        # Build messages
        messages = self._build_agent_messages(
            query=query,
            classification=classification,
            conversation_history=conversation_history,
            system_language=system_language,
            core_memory=core_memory,
            conversation_summary=conversation_summary,
            tools=tools,
            user_id=user_id,
            images=images,
            complexity=router_decision.complexity,
        )

        total_tool_calls = 0

        for turn_num in range(1, max_turns + 1):
            yield {"type": "turn_start", "turn": turn_num}

            if enable_reasoning:
                yield {
                    "type": "reasoning",
                    "phase": f"turn_{turn_num}",
                    "action": "thought",
                    "content": f"Turn {turn_num}: Deciding next action",
                }

            # Call LLM (using effective model)
            response = await self._call_llm_with_tools(
                messages, tools, flow_id,
                model_name=effective_model,
                provider_type=effective_provider,
            )

            tool_calls = self._parse_tool_calls(response)
            assistant_content = response.get("content") or ""

            # No tool calls - stream final response
            if not tool_calls:
                if enable_reasoning:
                    yield {
                        "type": "reasoning",
                        "phase": "synthesis",
                        "action": "start",
                        "content": "All data gathered - generating response",
                    }

                async for chunk in self.llm_provider.stream_response(
                    model_name=effective_model,
                    messages=messages,
                    provider_type=effective_provider,
                    api_key=self.api_key,
                    max_tokens=4000,
                    temperature=0.3,
                ):
                    yield {"type": "content", "content": chunk}

                yield {
                    "type": "done",
                    "total_turns": turn_num,
                    "total_tool_calls": total_tool_calls,
                }
                return

            # Emit tool calls
            if enable_reasoning:
                yield {
                    "type": "reasoning",
                    "phase": "tool_selection",
                    "action": "decision",
                    "content": f"Calling {len(tool_calls)} tools: {', '.join(tc.name for tc in tool_calls)}",
                }

            yield {
                "type": "tool_calls",
                "tools": [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in tool_calls
                ],
            }

            # Execute tools
            tool_results = await self._execute_tool_calls(
                tool_calls=tool_calls,
                flow_id=flow_id,
                user_id=user_id,
                session_id=session_id,
            )

            total_tool_calls += len(tool_calls)

            # Emit results
            yield {
                "type": "tool_results",
                "results": [
                    {"tool": tc.name, "success": r.get("status") in ["success", "200"]}
                    for tc, r in zip(tool_calls, tool_results)
                ],
            }

            # Update messages
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
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tc, result in zip(tool_calls, tool_results):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        # Max turns - stream final
        yield {"type": "max_turns_reached", "turns": max_turns}

        if enable_reasoning:
            yield {
                "type": "reasoning",
                "phase": "max_turns",
                "action": "synthesis",
                "content": f"Max turns ({max_turns}) reached - synthesizing final response",
            }

        messages.append({
            "role": "user",
            "content": "Please provide your final response based on all the information gathered.",
        })

        async for chunk in self.llm_provider.stream_response(
            model_name=effective_model,
            messages=messages,
            provider_type=effective_provider,
            api_key=self.api_key,
            max_tokens=4000,
            temperature=0.3,
        ):
            yield {"type": "content", "content": chunk}

        yield {
            "type": "done",
            "total_turns": max_turns,
            "total_tool_calls": total_tool_calls,
        }

    # =========================================================================
    # COMPLEX STRATEGY (6+ tools, planning + parallel)
    # =========================================================================

    async def _execute_complex(
        self,
        query: str,
        router_decision: RouterDecision,
        classification: Optional[UnifiedClassificationResult],
        conversation_history: Optional[List[Dict[str, str]]],
        system_language: str,
        flow_id: str,
        user_id: Optional[int],
        session_id: Optional[str],
        core_memory: Optional[str],
        conversation_summary: Optional[str],
    ) -> UnifiedAgentResult:
        """
        Complex strategy: Planning + parallel execution.

        Best for: Comparisons, comprehensive analysis, multi-symbol queries
        """
        start_time = datetime.now()
        self.logger.info(f"[{flow_id}] COMPLEX strategy: {len(router_decision.selected_tools)} tools")

        # For complex queries, use iterative with higher max_turns
        # This is a simplified version - full implementation would include
        # planning phase similar to PlanningAgent
        return await self._execute_iterative(
            query=query,
            router_decision=router_decision,
            classification=classification,
            conversation_history=conversation_history,
            system_language=system_language,
            flow_id=flow_id,
            user_id=user_id,
            session_id=session_id,
            core_memory=core_memory,
            conversation_summary=conversation_summary,
        )

    async def _stream_complex(
        self,
        query: str,
        router_decision: RouterDecision,
        classification: Optional[UnifiedClassificationResult],
        conversation_history: Optional[List[Dict[str, str]]],
        system_language: str,
        flow_id: str,
        user_id: Optional[int],
        session_id: Optional[str],
        core_memory: Optional[str],
        conversation_summary: Optional[str],
        enable_reasoning: bool,
        images: Optional[List[Any]],
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream complex strategy."""
        if enable_reasoning:
            yield {
                "type": "reasoning",
                "phase": "planning",
                "action": "start",
                "content": f"Complex query - planning execution for {len(router_decision.selected_tools)} tools",
            }

        # For now, delegate to iterative strategy with more turns
        # Full implementation would include proper planning phase
        async for event in self._stream_iterative(
            query=query,
            router_decision=router_decision,
            classification=classification,
            conversation_history=conversation_history,
            system_language=system_language,
            flow_id=flow_id,
            user_id=user_id,
            session_id=session_id,
            core_memory=core_memory,
            conversation_summary=conversation_summary,
            enable_reasoning=enable_reasoning,
            images=images,
            model_name=model_name,
            provider_type=provider_type,
        ):
            yield event

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    async def _respond_without_tools(
        self,
        query: str,
        classification: Optional[UnifiedClassificationResult],
        conversation_history: Optional[List[Dict[str, str]]],
        system_language: str,
        flow_id: str,
        core_memory: Optional[str],
        conversation_summary: Optional[str],
    ) -> UnifiedAgentResult:
        """Generate response without tools."""
        start_time = datetime.now()

        messages = self._build_no_tools_messages(
            query=query,
            classification=classification,
            conversation_history=conversation_history,
            system_language=system_language,
            core_memory=core_memory,
            conversation_summary=conversation_summary,
        )

        response = await self.llm_provider.generate_response(
            model_name=self.model_name,
            messages=messages,
            provider_type=self.provider_type,
            api_key=self.api_key,
            max_tokens=2000,
            temperature=0.7,
        )

        content = response.get("content", "") if isinstance(response, dict) else str(response)
        total_time = int((datetime.now() - start_time).total_seconds() * 1000)

        return UnifiedAgentResult(
            success=True,
            response=content,
            turns=[AgentTurn(turn_number=1, assistant_message=content)],
            total_turns=1,
            total_tool_calls=0,
            total_execution_time_ms=total_time,
            complexity=Complexity.SIMPLE,
            strategy=ExecutionStrategy.DIRECT,
        )

    async def _stream_response_without_tools(
        self,
        query: str,
        classification: Optional[UnifiedClassificationResult],
        conversation_history: Optional[List[Dict[str, str]]],
        system_language: str,
        flow_id: str,
        core_memory: Optional[str],
        conversation_summary: Optional[str],
        images: Optional[List[Any]] = None,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response without tools."""
        effective_model = model_name or self.model_name
        effective_provider = provider_type or self.provider_type

        messages = self._build_no_tools_messages(
            query=query,
            classification=classification,
            conversation_history=conversation_history,
            system_language=system_language,
            core_memory=core_memory,
            conversation_summary=conversation_summary,
            images=images,
        )

        async for chunk in self.llm_provider.stream_response(
            model_name=effective_model,
            messages=messages,
            provider_type=effective_provider,
            api_key=self.api_key,
            max_tokens=2000,
            temperature=0.7,
        ):
            yield chunk

    async def _execute_tools_parallel(
        self,
        tool_names: List[str],
        query: str,
        symbols: List[str],
        flow_id: str,
        user_id: Optional[int],
        session_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Execute tools in parallel with inferred arguments.

        For tools requiring a 'symbol' parameter:
        - Expands to one call per symbol (e.g., getStockPrice for NVDA and AAPL)
        - Executes all calls in parallel

        Args:
            tool_names: List of tool names to execute
            symbols: List of symbols from classification (may have multiple)
        """

        async def execute_single(
            tool_name: str,
            symbol: Optional[str] = None,
        ) -> Dict[str, Any]:
            try:
                # Infer arguments, using specific symbol if provided
                args = self._infer_tool_arguments(
                    tool_name=tool_name,
                    query=query,
                    symbols=[symbol] if symbol else symbols,
                )

                self.logger.debug(
                    f"[{flow_id}] Executing {tool_name} with args: {args}"
                )

                result = await asyncio.wait_for(
                    self.registry.execute_tool(tool_name=tool_name, params=args),
                    timeout=self.DEFAULT_TOOL_TIMEOUT,
                )

                if isinstance(result, ToolOutput):
                    return {
                        "tool_name": result.tool_name,
                        "status": result.status,
                        "data": result.data,
                        "error": result.error,
                        "formatted_context": result.formatted_context,
                        "symbol": symbol,  # Track which symbol this result is for
                    }

                return result if isinstance(result, dict) else {"data": result}

            except asyncio.TimeoutError:
                return {
                    "tool_name": tool_name,
                    "status": "timeout",
                    "error": f"Tool timed out after {self.DEFAULT_TOOL_TIMEOUT}s",
                    "symbol": symbol,
                }
            except Exception as e:
                return {
                    "tool_name": tool_name,
                    "status": "error",
                    "error": str(e),
                    "symbol": symbol,
                }

        # Build list of (tool_name, symbol) pairs to execute
        # For tools requiring symbol, expand to one call per symbol
        execution_tasks = []

        for tool_name in tool_names:
            # Check if tool requires symbol parameter
            schema = self.catalog.get_full_schema(tool_name)
            requires_symbol = False

            if schema and schema.parameters:
                for param in schema.parameters:
                    if param.get("name") == "symbol" and param.get("required", False):
                        requires_symbol = True
                        break

            if requires_symbol and len(symbols) > 1:
                # Expand: one call per symbol
                for symbol in symbols:
                    execution_tasks.append((tool_name, symbol))
                    self.logger.info(
                        f"[{flow_id}] Expanding {tool_name} for symbol: {symbol}"
                    )
            else:
                # Single call (use first symbol if any)
                execution_tasks.append((tool_name, symbols[0] if symbols else None))

        # Execute all tasks in parallel
        results = await asyncio.gather(
            *[execute_single(name, sym) for name, sym in execution_tasks],
            return_exceptions=True,
        )

        # Process results
        processed = []
        for i, result in enumerate(results):
            tool_name, symbol = execution_tasks[i]
            if isinstance(result, Exception):
                processed.append({
                    "tool_name": tool_name,
                    "status": "error",
                    "error": str(result),
                    "symbol": symbol,
                })
            else:
                processed.append(result)

        self.logger.info(
            f"[{flow_id}] Executed {len(processed)} tool calls for {len(tool_names)} tools, {len(symbols)} symbols"
        )

        return processed

    async def _execute_tool_calls(
        self,
        tool_calls: List[ToolCall],
        flow_id: str,
        user_id: Optional[int],
        session_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Execute tool calls from LLM."""

        async def execute_single(tc: ToolCall) -> Dict[str, Any]:
            try:
                result = await asyncio.wait_for(
                    self.registry.execute_tool(tool_name=tc.name, params=tc.arguments),
                    timeout=self.DEFAULT_TOOL_TIMEOUT,
                )

                if isinstance(result, ToolOutput):
                    return {
                        "tool_name": result.tool_name,
                        "status": result.status,
                        "data": result.data,
                        "error": result.error,
                        "formatted_context": result.formatted_context,
                    }

                return result if isinstance(result, dict) else {"data": result}

            except asyncio.TimeoutError:
                return {
                    "tool_name": tc.name,
                    "status": "timeout",
                    "error": f"Timeout after {self.DEFAULT_TOOL_TIMEOUT}s",
                }
            except Exception as e:
                return {
                    "tool_name": tc.name,
                    "status": "error",
                    "error": str(e),
                }

        results = await asyncio.gather(
            *[execute_single(tc) for tc in tool_calls],
            return_exceptions=True,
        )

        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed.append({
                    "tool_name": tool_calls[i].name,
                    "status": "error",
                    "error": str(result),
                })
            else:
                processed.append(result)

        return processed

    def _infer_tool_arguments(
        self,
        tool_name: str,
        query: str,
        symbols: List[str],
    ) -> Dict[str, Any]:
        """Infer tool arguments from query and symbols."""
        args = {}

        # Get schema
        schema = self.catalog.get_full_schema(tool_name)
        if not schema:
            return args

        for param in schema.parameters:
            param_name = param.get("name", "")

            # Symbol parameter
            if param_name == "symbol" and symbols:
                args["symbol"] = symbols[0]

            # Period/timeframe defaults
            elif param_name == "period":
                args["period"] = "1y"
            elif param_name == "timeframe":
                args["timeframe"] = "daily"

            # Limit defaults
            elif param_name == "limit":
                args["limit"] = 10

        return args

    async def _call_llm_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        flow_id: str,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call LLM with tool definitions."""
        effective_model = model_name or self.model_name
        effective_provider = provider_type or self.provider_type

        try:
            response = await self.llm_provider.generate_response(
                model_name=effective_model,
                messages=messages,
                provider_type=effective_provider,
                api_key=self.api_key,
                tools=tools,
                tool_choice="auto",
                max_tokens=4000,
                temperature=0.1,
            )

            return response if isinstance(response, dict) else {"content": str(response)}

        except Exception as e:
            self.logger.error(f"[{flow_id}] LLM call failed: {e}")
            raise

    def _parse_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """Parse tool calls from LLM response."""
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

    def _build_agent_messages(
        self,
        query: str,
        classification: Optional[UnifiedClassificationResult],
        conversation_history: Optional[List[Dict[str, str]]],
        system_language: str,
        core_memory: Optional[str],
        conversation_summary: Optional[str],
        tools: List[Dict[str, Any]],
        user_id: Optional[int],
        images: Optional[List[Any]] = None,
        complexity: Optional[Complexity] = None,
    ) -> List[Dict[str, Any]]:
        """Build messages for agent with tools using domain-specific skill prompts."""
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        # Get domain-specific skill based on classification
        market_type = "stock"  # Default
        if classification:
            market_type = getattr(classification, "market_type", "stock") or "stock"

        # Select skill and get domain prompt
        skill = self.skill_registry.select_skill(market_type)
        skill_prompt = skill.get_full_prompt()

        self.logger.debug(
            f"[UNIFIED_AGENT] Using skill: {skill.name} for market_type={market_type}"
        )

        # Build context hints
        symbols_hint = ""
        if classification and classification.symbols:
            symbols_hint = f"Symbols to analyze: {', '.join(classification.symbols)}"

        categories_hint = ""
        if classification and hasattr(classification, "tool_categories"):
            categories_hint = f"Data categories: {', '.join(classification.tool_categories or [])}"

        user_context = ""
        if core_memory:
            user_context += f"\n\n<USER_PROFILE>\n{core_memory}\n</USER_PROFILE>"
        if conversation_summary:
            user_context += f"\n\n<CONVERSATION_SUMMARY>\n{conversation_summary}\n</CONVERSATION_SUMMARY>"

        # Get list of tool names for instruction
        tool_names = [t.get("function", {}).get("name", "") for t in tools if t.get("function")]
        tool_list_str = ", ".join(tool_names) if tool_names else "none"

        # Count symbols to determine data volume
        num_symbols = len(classification.symbols) if classification and classification.symbols else 0
        is_high_volume = complexity == Complexity.COMPLEX or (num_symbols > 5 and len(tool_names) > 4)

        # Build execution guidelines based on complexity
        if is_high_volume:
            # COMPLEX query with many symbols - use incremental approach
            execution_guidelines = f"""## TOOL EXECUTION GUIDELINES

**IMPORTANT: This is a COMPLEX query with {num_symbols} symbols and {len(tool_names)} tools.**
**To avoid context overflow, use INCREMENTAL data gathering across multiple turns.**

Tools available: [{tool_list_str}]

**Execution Strategy (Incremental):**
1. **Turn 1**: Start with essential data (price, basic indicators) for ALL symbols
2. **Turn 2**: Add detailed analysis (patterns, support/resistance) for TOP symbols
3. **Turn 3+**: Get fundamentals/news only for most relevant symbols
4. Do NOT call all tools for all symbols at once - this will overflow context

**Priority Order:**
1. getStockPrice → Basic price snapshot for all
2. getTechnicalIndicators → Key indicators (RSI, MACD) for all
3. detectChartPatterns, getSupportResistance → Only for interesting symbols
4. getIncomeStatement, getBalanceSheet, getCashFlow → Only for deep-dive symbols
5. getStockNews, webSearch → Only if specifically needed

**After each turn:**
- Evaluate which symbols need more analysis
- Focus on most actionable insights
- Synthesize when you have enough data for a useful response"""
        else:
            # SIMPLE/MEDIUM query - can call all tools in first turn
            execution_guidelines = f"""## TOOL EXECUTION GUIDELINES

**IMPORTANT: The router has pre-selected these tools as needed for this query:**
Tools to use: [{tool_list_str}]

**Execution Rules:**
1. You MUST call ALL the pre-selected tools above in your first turn to gather comprehensive data
2. Execute multiple tools in parallel when possible
3. If a tool fails, note the limitation but still proceed
4. Only skip a tool if it's truly redundant with another tool's data
5. After gathering all data, synthesize into a comprehensive response

**Evaluation after tool calls:**
- Check if you have enough data to fully answer the user's question
- If missing critical information, call additional tools
- If data is sufficient, proceed to final response"""

        # Combine skill prompt with runtime context
        system_prompt = f"""{skill_prompt}

---
## RUNTIME CONTEXT

Current Date: {current_date}
Response Language: {system_language.upper()}
{symbols_hint}
{categories_hint}
{user_context}

{execution_guidelines}

**Response Quality:**
- Explain technical terms briefly (e.g., "RSI = 72 (quá mua)")
- End with 2-3 follow-up questions for user engagement
- Use friendly, advisor-like tone
"""

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-5:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

        # Add current query (with images if present)
        if images:
            user_message = self._format_user_message_with_images(query, images)
            messages.append(user_message)
        else:
            messages.append({"role": "user", "content": query})

        return messages

    def _build_synthesis_messages(
        self,
        query: str,
        tool_results: List[Dict[str, Any]],
        classification: Optional[UnifiedClassificationResult],
        conversation_history: Optional[List[Dict[str, str]]],
        system_language: str,
        core_memory: Optional[str],
        conversation_summary: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Build messages for synthesis after tool execution using domain skill."""
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        # Get domain-specific skill based on classification
        market_type = "stock"  # Default
        if classification:
            market_type = getattr(classification, "market_type", "stock") or "stock"

        # Select skill and get analysis framework
        skill = self.skill_registry.select_skill(market_type)
        analysis_framework = skill.get_analysis_framework()

        # Check if web search results are present for source citation
        has_web_search = any(
            r.get("tool_name", "").lower() == "websearch"
            for r in tool_results
            if r.get("status") in ["success", "200"]
        )

        # Format tool results
        results_text = "\n\n".join([
            f"**{r.get('tool_name', 'Unknown')}**:\n{r.get('formatted_context') or json.dumps(r.get('data', {}), indent=2)}"
            for r in tool_results
            if r.get("status") in ["success", "200"]
        ])

        # Source citation instructions when web search is used
        source_citation_instructions = ""
        if has_web_search:
            source_citation_instructions = """
## SOURCE CITATION REQUIREMENTS

When using information from webSearch results:
- MUST include a "## Nguồn tham khảo / Sources" section at the END of your response
- List all relevant sources as clickable markdown links: [Title](URL)
- Only cite sources you actually used in your analysis
- Format example:
  ## Nguồn tham khảo
  - [Article Title 1](https://example.com/article1)
  - [Article Title 2](https://example.com/article2)
"""

        system_prompt = f"""You are a {skill.config.description}.

Current Date: {current_date}
Response Language: {system_language.upper()}

{analysis_framework}
{source_citation_instructions}
---
## TOOL RESULTS TO SYNTHESIZE

<tool_results>
{results_text}
</tool_results>

## YOUR TASK

Synthesize the above tool results into a comprehensive, user-friendly response:

**Content Requirements:**
- Use the data from tools to support your analysis with specific numbers
- Follow the structured format from your expertise framework
- Highlight key insights and actionable recommendations

**Engagement Requirements:**
- Write in a friendly, conversational tone (like a helpful advisor, not a formal report)
- Explain technical terms briefly for beginners (e.g., "RSI = 72 (quá mua, có thể điều chỉnh)")
- End with 2-3 follow-up questions to help user explore further

**Language:**
- Default to Vietnamese for Vietnamese queries
- Be natural and engaging, avoid robotic/template language
{"- Include source citations at the end if web search was used" if has_web_search else ""}
"""

        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            for msg in conversation_history[-3:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": query})

        return messages

    def _build_no_tools_messages(
        self,
        query: str,
        classification: Optional[UnifiedClassificationResult],
        conversation_history: Optional[List[Dict[str, str]]],
        system_language: str,
        core_memory: Optional[str],
        conversation_summary: Optional[str],
        images: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Build messages for no-tools response."""
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        user_context = ""
        if core_memory:
            user_context += f"\n\n<USER_PROFILE>\n{core_memory}\n</USER_PROFILE>"
        if conversation_summary:
            user_context += f"\n\n<CONVERSATION_SUMMARY>\n{conversation_summary}\n</CONVERSATION_SUMMARY>"

        system_prompt = f"""You are a friendly financial assistant.

Current Date: {current_date}
Response Language: {system_language.upper()}
{user_context}

Respond naturally and helpfully."""

        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            for msg in conversation_history[-5:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

        if images:
            user_message = self._format_user_message_with_images(query, images)
            messages.append(user_message)
        else:
            messages.append({"role": "user", "content": query})

        return messages

    def _format_user_message_with_images(
        self,
        query: str,
        images: Optional[List[Any]],
    ) -> Dict[str, Any]:
        """Format user message with images."""
        if not images:
            return {"role": "user", "content": query}

        try:
            from src.utils.image import build_multimodal_message
            return build_multimodal_message(
                role="user",
                text=query,
                images=images,
                provider=self.provider_type,
            )
        except Exception as e:
            self.logger.warning(f"Failed to format images: {e}")
            return {"role": "user", "content": query}


# ============================================================================
# SINGLETON
# ============================================================================

_agent_instance: Optional[UnifiedAgent] = None


def get_unified_agent(
    model_name: Optional[str] = None,
    provider_type: Optional[str] = None,
) -> UnifiedAgent:
    """Get singleton UnifiedAgent instance."""
    global _agent_instance

    if _agent_instance is None:
        _agent_instance = UnifiedAgent(
            model_name=model_name,
            provider_type=provider_type,
        )

    return _agent_instance


def reset_unified_agent():
    """Reset singleton (for testing)."""
    global _agent_instance
    _agent_instance = None
