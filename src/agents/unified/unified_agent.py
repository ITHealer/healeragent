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

# Tool Search Service for semantic tool discovery
from src.services.tool_search_service import (
    get_tool_search_service,
    ToolSearchService,
    TOOL_SEARCH_DEFINITION,
)

# Graceful degradation for parallel tool execution
from src.utils.graceful_degradation import (
    execute_with_degradation,
    DegradationConfig,
    DegradationStrategy,
    DegradationResult,
    synthesize_partial_response,
)

# Web Search Tool (OpenAI primary, Tavily fallback)
from src.agents.tools.web.web_search import WebSearchTool

# News tools that should trigger auto web search
NEWS_TOOL_NAMES = {
    "getStockNews",
    "getMarketNews",
    "getCompanyEvents",
    "getEarningsCalendar",
}


# ============================================================================
# THINK TOOL DEFINITION
# ============================================================================
# This is added to the tool list to allow LLM to externalize its reasoning.
# Think tool does NOT cost extra API calls - it's just a placeholder for
# the LLM to output structured reasoning that can be displayed to users.

THINK_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "think",
        "description": """Use this tool to think through complex problems step by step.

WHEN TO USE:
- BEFORE calling other tools: Plan your approach and analyze what data you need
- AFTER receiving tool results: Analyze the data, identify key insights, decide next steps
- Before giving final response: Synthesize all information, formulate recommendations

This tool does NOT fetch new data - it's for organizing your reasoning.

Example:
{"thought": "User asks about NVDA investment. Need to check: 1) Valuation (P/E, P/S), 2) Growth rates, 3) Technical indicators. User profile shows growth investor preference, so emphasize growth metrics.", "reasoning_type": "planning"}""",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your reasoning, analysis, or planning. Be specific about what you're thinking and why.",
                },
                "reasoning_type": {
                    "type": "string",
                    "enum": ["planning", "analyzing", "deciding", "reflecting"],
                    "description": "Type of thinking: planning (before action), analyzing (after data), deciding (before response), reflecting (on errors/gaps)",
                    "default": "analyzing",
                },
            },
            "required": ["thought"],
        },
    },
}


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

        # Get tools + tool_search meta-tool
        tools = self.catalog.get_openai_functions(router_decision.selected_tools)
        tools.append(TOOL_SEARCH_DEFINITION)  # Enable semantic tool discovery

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
                        "content": "All data gathered - generating comprehensive response",
                    }

                # Fixed max_tokens as safety cutoff - output length controlled by system prompt
                # Per OpenAI: "max_tokens is a hard cutoff limit, not a length control"
                MAX_RESPONSE_TOKENS = 12000

                async for chunk in self.llm_provider.stream_response(
                    model_name=effective_model,
                    messages=messages,
                    provider_type=effective_provider,
                    api_key=self.api_key,
                    max_tokens=MAX_RESPONSE_TOKENS,
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
            "content": """Please provide your COMPREHENSIVE final response based on all the information gathered.

IMPORTANT:
- Include ALL important data points and numbers from tool results
- For each symbol, provide: price, technical signals (RSI, MACD), fundamental metrics (P/E, ROE)
- Give clear insights and actionable recommendations
- Don't truncate or summarize - be as detailed as needed for a thorough analysis
- End with 2-3 follow-up questions""",
        })

        # Fixed max_tokens as safety cutoff - length controlled by system prompt
        MAX_RESPONSE_TOKENS = 12000

        async for chunk in self.llm_provider.stream_response(
            model_name=effective_model,
            messages=messages,
            provider_type=effective_provider,
            api_key=self.api_key,
            max_tokens=MAX_RESPONSE_TOKENS,
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
        min_required: int = 1,
        use_degradation: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Execute tools in parallel with inferred arguments and graceful degradation.

        For tools requiring a 'symbol' parameter:
        - Expands to one call per symbol (e.g., getStockPrice for NVDA and AAPL)
        - Executes all calls in parallel
        - Continues with partial results if some tools fail (graceful degradation)

        Args:
            tool_names: List of tool names to execute
            symbols: List of symbols from classification (may have multiple)
            min_required: Minimum number of successful results required to proceed
            use_degradation: Whether to use graceful degradation pattern

        Returns:
            List of tool results with degradation metadata if enabled
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

        # Process results with graceful degradation
        processed = []
        successful_count = 0
        failed_count = 0
        failed_tools = []

        for i, result in enumerate(results):
            tool_name, symbol = execution_tasks[i]
            if isinstance(result, Exception):
                processed.append({
                    "tool_name": tool_name,
                    "status": "error",
                    "error": str(result),
                    "symbol": symbol,
                })
                failed_count += 1
                failed_tools.append(f"{tool_name}({symbol})" if symbol else tool_name)
            else:
                # Check if result indicates an error status
                status = result.get("status", "success") if isinstance(result, dict) else "success"
                if status in ("error", "timeout"):
                    failed_count += 1
                    failed_tools.append(f"{tool_name}({symbol})" if symbol else tool_name)
                else:
                    successful_count += 1
                processed.append(result)

        total_count = len(processed)

        # Log execution summary
        self.logger.info(
            f"[{flow_id}] Tool execution: {successful_count}/{total_count} succeeded, "
            f"{failed_count} failed"
        )

        # Apply graceful degradation logic
        if use_degradation and failed_count > 0:
            # Check if we have enough successful results
            if successful_count >= min_required:
                # Partial success - add degradation metadata
                self.logger.info(
                    f"[{flow_id}] ✅ Graceful degradation: proceeding with "
                    f"{successful_count}/{total_count} results (min_required={min_required})"
                )

                # Add degradation metadata to first result
                if processed:
                    processed[0]["_degradation"] = {
                        "is_partial": True,
                        "success_count": successful_count,
                        "total_count": total_count,
                        "failed_tools": failed_tools,
                        "message": (
                            f"⚠️ Some data sources unavailable ({failed_count}/{total_count}). "
                            f"Showing partial results."
                        ),
                    }
            else:
                # Not enough successful results
                self.logger.warning(
                    f"[{flow_id}] ⚠️ Graceful degradation: only {successful_count}/{total_count} "
                    f"succeeded, below min_required={min_required}"
                )

                # Add fallback metadata
                if processed:
                    processed[0]["_degradation"] = {
                        "is_partial": False,
                        "should_fallback": True,
                        "success_count": successful_count,
                        "total_count": total_count,
                        "min_required": min_required,
                        "failed_tools": failed_tools,
                        "message": (
                            f"Unable to retrieve sufficient data. "
                            f"Only {successful_count}/{total_count} sources available, "
                            f"need at least {min_required}."
                        ),
                    }

        self.logger.info(
            f"[{flow_id}] Executed {len(processed)} tool calls for "
            f"{len(tool_names)} tools, {len(symbols)} symbols"
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
                # ============================================================
                # SPECIAL HANDLING: tool_search meta-tool
                # ============================================================
                if tc.name == "tool_search":
                    return await self._execute_tool_search(tc, flow_id)

                # ============================================================
                # REGULAR TOOL EXECUTION
                # ============================================================
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

        # ============================================================
        # DEBUG: Log tool results before sending to LLM
        # ============================================================
        for res in processed:
            tool_name = res.get("tool_name", "unknown")
            status = res.get("status", "unknown")
            self.logger.info(f"[{flow_id}] 📊 TOOL RESULT: {tool_name} | status={status}")

            # Log key data from financial tools
            if status == "success" and "data" in res:
                data = res["data"]
                if isinstance(data, dict):
                    # Income Statement specific logging
                    if "revenue" in data:
                        rev = data.get('revenue')
                        net = data.get('net_income')
                        eps = data.get('eps')
                        rev_str = f"{rev:,.0f}" if isinstance(rev, (int, float)) else str(rev)
                        net_str = f"{net:,.0f}" if isinstance(net, (int, float)) else str(net)
                        self.logger.info(
                            f"[{flow_id}]   └─ revenue={rev_str} | "
                            f"net_income={net_str} | eps={eps}"
                        )
                    # Log first statement if available
                    if "statements" in data and data["statements"]:
                        stmt = data["statements"][0]
                        stmt_rev = stmt.get('revenue')
                        rev_str = f"{stmt_rev:,.0f}" if isinstance(stmt_rev, (int, float)) else "N/A"
                        self.logger.info(
                            f"[{flow_id}]   └─ Latest: date={stmt.get('date')} | "
                            f"period={stmt.get('period')} | revenue={rev_str}"
                        )

        return processed

    async def _execute_tool_search(
        self,
        tc: ToolCall,
        flow_id: str,
    ) -> Dict[str, Any]:
        """
        Execute tool_search meta-tool for semantic tool discovery.

        When GPT calls tool_search, we:
        1. Search for relevant tools using embeddings
        2. Return the search results as formatted text
        3. GPT can then understand what tools are available

        Args:
            tc: ToolCall with query and optional top_k
            flow_id: Flow ID for logging

        Returns:
            Dict with tool_name, status, and search results
        """
        try:
            query = tc.arguments.get("query", "")
            top_k = tc.arguments.get("top_k", 5)

            if not query:
                return {
                    "tool_name": "tool_search",
                    "status": "error",
                    "error": "Missing required parameter: query",
                }

            self.logger.info(
                f"[{flow_id}] tool_search called: '{query[:50]}...' top_k={top_k}"
            )

            # Use ToolSearchService for semantic search
            search_service = get_tool_search_service()
            response = await search_service.search(query, top_k=top_k)

            # Format results for LLM
            formatted_result = response.format_for_llm()

            self.logger.info(
                f"[{flow_id}] tool_search found {len(response.results)} tools: "
                f"{response.tool_names[:3]}{'...' if len(response.results) > 3 else ''}"
            )

            return {
                "tool_name": "tool_search",
                "status": "success",
                "data": {
                    "found_tools": response.tool_names,
                    "search_time_ms": response.search_time_ms,
                    "formatted_result": formatted_result,
                },
                "formatted_context": formatted_result,
            }

        except Exception as e:
            self.logger.error(f"[{flow_id}] tool_search error: {e}")
            return {
                "tool_name": "tool_search",
                "status": "error",
                "error": str(e),
            }

    async def _execute_auto_web_search(
        self,
        query: str,
        symbols: List[str],
        flow_id: str,
        enable_streaming: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute web search with SSE streaming events.

        Uses WebSearchTool which has:
        - PRIMARY: OpenAI Responses API with web_search (gpt-5-mini)
        - FALLBACK: Tavily API (when OpenAI fails)

        This method:
        1. Builds a search query from user query and symbols
        2. Calls WebSearchTool.execute_streaming()
        3. Yields SSE events for UI progress display
        4. Returns search results with citations

        Args:
            query: Original user query
            symbols: Stock symbols for context
            flow_id: Flow ID for logging
            enable_streaming: Whether to emit SSE events

        Yields:
            Dict events: web_search_start, web_search_progress, web_search_complete
        """
        try:
            # Initialize Web Search Tool (OpenAI primary, Tavily fallback)
            web_search_tool = WebSearchTool()

            # Build search query with symbols context
            if symbols:
                symbol_str = ", ".join(symbols[:3])  # Limit to 3 symbols
                search_query = f"{query} {symbol_str} stock news latest"
            else:
                search_query = f"{query} latest news"

            self.logger.info(
                f"[{flow_id}] 🌐 Auto Web Search: '{search_query[:60]}...'"
            )

            # Use streaming execution for progress events
            # Don't filter domains - let the model decide sources based on query
            search_data = None
            async for event in web_search_tool.execute_streaming(
                query=search_query,
                max_results=5,
                use_finance_domains=False,  # Let model decide sources
            ):
                event_type = event.get("type", "")

                # Forward SSE events for UI
                if event_type == "web_search_start":
                    self.logger.info(f"[{flow_id}] 🔍 Web search started")
                    if enable_streaming:
                        yield event

                elif event_type == "web_search_progress":
                    action = event.get("action", "searching")
                    self.logger.info(f"[{flow_id}] 🔍 Web search: {action}")
                    if enable_streaming:
                        yield event

                elif event_type == "web_search_complete":
                    data = event.get("data", {})
                    citation_count = data.get("citation_count", 0)
                    source = data.get("source", "unknown")
                    exec_time = event.get("execution_time_ms", 0)

                    self.logger.info(
                        f"[{flow_id}] ✅ Web search complete: {citation_count} citations "
                        f"(source={source}, {int(exec_time)}ms)"
                    )
                    if enable_streaming:
                        yield event

                elif event_type == "web_search_error":
                    error = event.get("error", "Unknown error")
                    self.logger.error(f"[{flow_id}] ❌ Web search error: {error}")
                    if enable_streaming:
                        yield event

                elif event_type == "web_search_result":
                    # Final result - capture for LLM context
                    search_data = event
                    yield event

        except Exception as e:
            self.logger.error(f"[{flow_id}] Auto web search error: {e}", exc_info=True)

            if enable_streaming:
                yield {
                    "type": "web_search_error",
                    "error": str(e),
                }

            yield {
                "type": "web_search_result",
                "tool_name": "webSearch",
                "status": "error",
                "error": str(e),
            }

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

## RESPONSE QUALITY REQUIREMENTS (CRITICAL)

**1. DATA PRESENTATION FORMAT (MANDATORY):**
Your response MUST follow this structure:

```
### 📊 Dữ liệu từ [Tool Name] | Nguồn: [Data Source]
| Chỉ tiêu | Giá trị | Ghi chú |
|----------|---------|---------|
| ...      | ...     | ...     |

### 💡 Phân tích & Nhận định
[Your analysis based on the data above]
```

**2. RAW DATA FIRST - ALWAYS SHOW COMPLETE DATA:**
- Display ALL raw data from tool results in TABLE format
- NEVER skip or summarize data - users need exact numbers for verification
- Include data timestamp/period clearly (e.g., "Q4 FY2025 (Oct-Jan)" or "Q4 CY2025 (Oct-Dec)")
- For financial data, always specify: Fiscal Year (FY) vs Calendar Year (CY)
  - NVDA fiscal year ends in January: Q4 FY2025 = Oct 2024 - Jan 2025
  - AAPL fiscal year ends in September: Q1 FY2025 = Oct-Dec 2024

**3. DATA SOURCE CITATION (REQUIRED):**
- ALWAYS cite data source: "Nguồn: FMP API" or "Nguồn: Yahoo Finance"
- Include timestamp when available

**4. STRUCTURED TABLE FORMATS:**

For Financial Data:
| Chỉ tiêu | Giá trị | YoY Change |
|----------|---------|------------|
| Revenue | $XX.XX B | +XX.X% |
| Net Income | $XX.XX B | +XX.X% |
| EPS | $X.XX | +XX.X% |

For Technical Analysis:
| Indicator | Value | Signal |
|-----------|-------|--------|
| RSI(14) | XX.X | Bullish/Bearish |
| MACD | XX.X | Bullish/Bearish |

**5. ANALYSIS AFTER DATA:**
- Insights come AFTER showing raw data in tables
- Reference specific numbers from tables
- Explain what the numbers mean
- Include actionable recommendations

**6. DATA INTEGRITY:**
- NEVER make up numbers - only use tool results
- If data is missing: "⚠️ Không có dữ liệu cho [item]"
- Clarify fiscal year vs calendar year when relevant
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

        # Count symbols for adaptive response length guidance
        num_symbols = sum(1 for r in tool_results if r.get("status") in ["success", "200"])

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

Synthesize the above tool results into a **COMPREHENSIVE, DETAILED** response:

**DATA SYNTHESIS REQUIREMENTS (CRITICAL):**
- Include **ALL important numbers** from tool results (prices, percentages, ratios, indicators)
- For EACH symbol, provide: current price, 52-week position, technical signals, fundamental metrics
- **DON'T skip or summarize data** - users want specific numbers and detailed analysis!
- If P/E, ROE, debt ratios are available, explain what they mean for the stock

**ADAPTIVE RESPONSE LENGTH:**
- You have data for approximately {num_symbols} data sources
- Simple query (1 symbol) → Medium response (300-500 words)
- Complex query (3+ symbols) → Detailed response (800-1500 words)
- Comparison queries → Include comparison table + detailed per-symbol analysis
- **NEVER truncate just to be brief** - be as detailed as the data requires

**STRUCTURED FORMAT:**
For each symbol analyzed, include:
1. Price & Position: Current price, 52-week high/low, support/resistance levels
2. Technical Signals: RSI, MACD, moving averages with interpretation
3. Fundamental Health: P/E, PEG, ROE, debt ratios with meaning
4. Key Insight: Overall assessment - bullish/bearish/neutral? Why?
5. Risk Factors: What could go wrong?

**ENGAGEMENT STYLE:**
- Write like a knowledgeable friend/advisor, not a formal report
- Explain technical terms briefly (e.g., "RSI = 72 (overbought - có thể điều chỉnh)")
- Highlight actionable insights and recommendations
- End with 2-3 relevant follow-up questions

**LANGUAGE:**
- Respond in {system_language.upper()}
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


    # =========================================================================
    # NEW ARCHITECTURE: Agent sees ALL tools
    # =========================================================================

    async def run_stream_with_all_tools(
        self,
        query: str,
        intent_result: Any,  # IntentResult from IntentClassifier
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
        max_turns: int = 6,
        enable_tool_search_mode: bool = False,
        working_memory_symbols: Optional[List[str]] = None,
        enable_think_tool: bool = False,
        enable_web_search: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run agent with ALL tools available (new architecture).

        Key difference from run_stream():
        - Agent sees ALL tools from catalog (not pre-filtered by Router)

        - Agent decides which tools to call (ChatGPT-style)
        - Uses IntentResult instead of RouterDecision

        Args:
            query: User query
            intent_result: IntentResult from IntentClassifier
            max_turns: Maximum turns for agent loop (default 6)
            enable_tool_search_mode: If True, start with ONLY tool_search meta-tool.
                Agent must call tool_search to discover available tools.
                Token savings: ~85% (from ~15K to ~500 tokens)
            working_memory_symbols: Symbols from previous turns (Working Memory).
                These are MERGED with intent_result.validated_symbols to provide
                context continuity for follow-up questions.
            enable_think_tool: If True, add STRONG instruction to always use think tool
                for explicit reasoning before and after tool calls.
            enable_web_search: If True, FORCE inject webSearch/serpSearch tools
                and add instruction to search for latest news/information.

        Yields:
            Stream events: reasoning, tool_calls, tool_results, content, done
        """
        flow_id = f"UA-ALL-{uuid.uuid4().hex[:8]}"

        # Resolve effective model/provider
        effective_model = model_name or self.model_name
        effective_provider = provider_type or self.provider_type

        # =====================================================================
        # SYMBOL RESOLUTION: intent_result OR working_memory (NOT both!)
        # - If intent has symbols → use ONLY those (explicit user request)
        # - If intent has NO symbols → use working_memory as FALLBACK
        # Merging causes confusion: agent sees NVDA,GRAB,TSLA and picks wrong one
        # =====================================================================
        intent_symbols = getattr(intent_result, 'validated_symbols', []) or []
        wm_symbols = working_memory_symbols or []

        # Use intent symbols if available, otherwise fallback to working_memory
        if intent_symbols:
            # User explicitly mentioned symbols - use only those
            merged_symbols = list(intent_symbols)
            self.logger.info(
                f"[{flow_id}] Using intent symbols (explicit): {merged_symbols}"
            )
        elif wm_symbols:
            # No explicit symbols - use working_memory as fallback for follow-ups
            merged_symbols = list(wm_symbols)
            self.logger.info(
                f"[{flow_id}] 📝 Using working_memory symbols (fallback): {merged_symbols}"
            )
        else:
            merged_symbols = []

        self.logger.info("─" * 50)
        self.logger.info(f"[{flow_id}] 🚀 AGENT WITH ALL TOOLS")
        self.logger.info("─" * 50)
        self.logger.info(f"  ├─ Symbols (intent): {intent_symbols}")
        self.logger.info(f"  ├─ Symbols (working_memory): {wm_symbols}")
        self.logger.info(f"  ├─ Symbols (final): {merged_symbols}")
        self.logger.info(f"  ├─ Market: {getattr(intent_result, 'market_type', 'unknown')}")
        self.logger.info(f"  ├─ Tools: ALL ({len(self.catalog.get_tool_names())})")
        self.logger.info(f"  ├─ Model: {effective_model}")
        self.logger.info(f"  └─ Max Turns: {max_turns}")

        try:
            # Check if tools are needed
            requires_tools = getattr(intent_result, 'requires_tools', True)

            if not requires_tools:
                # Direct response without tools
                if enable_reasoning:
                    yield {
                        "type": "reasoning",
                        "phase": "execution",
                        "action": "thought",
                        "content": "No tools needed - generating direct response",
                    }

                async for chunk in self._stream_response_without_tools(
                    query=query,
                    classification=None,
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

            # Agent loop with ALL tools
            if enable_reasoning:
                yield {
                    "type": "reasoning",
                    "phase": "agent_loop",
                    "action": "start",
                    "content": f"Starting agent loop with ALL {len(self.catalog.get_tool_names())} tools available",
                }

            # ================================================================
            # TOOL LOADING STRATEGY
            # ================================================================
            if enable_tool_search_mode:
                # TOOL SEARCH MODE: Start with ONLY tool_search + think for token savings
                # GPT must call tool_search to discover available tools
                tools = [TOOL_SEARCH_DEFINITION, THINK_TOOL_DEFINITION]
                discovered_tool_names = set()  # No tools discovered yet

                # FORCE INJECT: Web search tools when enabled
                if enable_web_search:
                    web_search_tools = self.catalog.get_openai_functions(["webSearch", "serpSearch"])
                    tools.extend(web_search_tools)
                    discovered_tool_names.update(["webSearch", "serpSearch"])
                    self.logger.info(f"[{flow_id}] 🌐 FORCE INJECTED: webSearch, serpSearch (enable_web_search=True)")

                self.logger.info(
                    f"[{flow_id}] 🔍 TOOL SEARCH MODE: Starting with tool_search + think "
                    f"(~600 tokens vs ~{len(self.catalog.get_tool_names()) * 400:,} tokens)"
                )

                if enable_reasoning:
                    yield {
                        "type": "reasoning",
                        "phase": "tool_search_mode",
                        "action": "start",
                        "content": (
                            "Tool Search Mode enabled. Agent will discover tools dynamically "
                            "using semantic search for maximum token efficiency."
                        ),
                    }
            else:
                # STANDARD MODE: Load ALL tools from catalog
                all_tool_names = self.catalog.get_tool_names()
                tools = self.catalog.get_openai_functions(all_tool_names)
                tools.append(TOOL_SEARCH_DEFINITION)  # Also add tool_search
                tools.append(THINK_TOOL_DEFINITION)   # Add think tool for explicit reasoning
                discovered_tool_names = set(all_tool_names)

                self.logger.info(f"[{flow_id}] Agent sees {len(tools)} tools (including tool_search, think)")

                # Log when special modes are enabled
                if enable_web_search:
                    self.logger.info(f"[{flow_id}] 🌐 WEB SEARCH ENABLED: Strong instruction added to system prompt")
                if enable_think_tool:
                    self.logger.info(f"[{flow_id}] 🧠 THINK TOOL ENABLED: Strong instruction added to system prompt")

            # Build messages with all tools
            messages = self._build_agent_messages_all_tools(
                query=query,
                intent_result=intent_result,
                conversation_history=conversation_history,
                system_language=system_language,
                core_memory=core_memory,
                conversation_summary=conversation_summary,
                tools=tools,
                user_id=user_id,
                images=images,
                merged_symbols=merged_symbols,  # Pass merged symbols for context continuity
                enable_think_tool=enable_think_tool,
                enable_web_search=enable_web_search,
            )

            total_tool_calls = 0

            for turn_num in range(1, max_turns + 1):
                yield {"type": "turn_start", "turn": turn_num}

                if enable_reasoning:
                    yield {
                        "type": "reasoning",
                        "phase": f"turn_{turn_num}",
                        "action": "thought",
                        "content": f"Turn {turn_num}: Analyzing query and deciding which tools to call",
                    }

                # Call LLM with ALL tools
                self.logger.info(f"[{flow_id}] Turn {turn_num}: Calling LLM with {len(tools)} tools")
                response = await self._call_llm_with_tools(
                    messages, tools, flow_id,
                    model_name=effective_model,
                    provider_type=effective_provider,
                )
                self.logger.info(f"[{flow_id}] Turn {turn_num}: LLM responded")

                tool_calls = self._parse_tool_calls(response)
                assistant_content = response.get("content") or ""
                self.logger.info(
                    f"[{flow_id}] Turn {turn_num}: tool_calls={len(tool_calls)}, "
                    f"content_len={len(assistant_content)}"
                )

                # No tool calls - stream final response
                if not tool_calls:
                    if enable_reasoning:
                        yield {
                            "type": "reasoning",
                            "phase": "synthesis",
                            "action": "start",
                            "content": "Agent decided no more tools needed - generating final response",
                        }

                    # Add assistant message to get final response
                    messages.append({"role": "assistant", "content": assistant_content})

                    # Fixed max_tokens as safety cutoff - length controlled by system prompt
                    # Per OpenAI: "max_tokens is a hard cutoff limit, not a length control"
                    MAX_RESPONSE_TOKENS = 12000

                    self.logger.info(f"[{flow_id}] Streaming final response (max_tokens={MAX_RESPONSE_TOKENS})")
                    content_chunks = 0
                    async for chunk in self.llm_provider.stream_response(
                        model_name=effective_model,
                        messages=messages,
                        provider_type=effective_provider,
                        api_key=self.api_key,
                        max_tokens=MAX_RESPONSE_TOKENS,
                        temperature=0.3,
                    ):
                        content_chunks += 1
                        yield {"type": "content", "content": chunk}

                    self.logger.info(f"[{flow_id}] Final response streamed: {content_chunks} chunks")
                    yield {
                        "type": "done",
                        "total_turns": turn_num,
                        "total_tool_calls": total_tool_calls,
                    }
                    return

                # ============================================================
                # SEPARATE THINK TOOL CALLS FROM DATA TOOL CALLS
                # Think tool is handled inline (no actual execution needed)
                # ============================================================
                think_calls = [tc for tc in tool_calls if tc.name == "think"]
                data_tool_calls = [tc for tc in tool_calls if tc.name != "think"]

                # Process think tool calls FIRST - emit thinking events
                think_results = []
                for tc in think_calls:
                    thought = tc.arguments.get("thought", "")
                    reasoning_type = tc.arguments.get("reasoning_type", "analyzing")

                    # Log the thought
                    self.logger.info(f"[{flow_id}] 💭 THINK [{reasoning_type}]: {thought[:100]}...")

                    # Emit thinking event for frontend
                    yield {
                        "type": "thinking",
                        "phase": reasoning_type,
                        "content": thought,
                        "tool_call_id": tc.id,
                    }

                    # Create simple result (no actual execution needed)
                    think_results.append({
                        "status": "success",
                        "data": {
                            "thought": thought,
                            "reasoning_type": reasoning_type,
                            "acknowledged": True,
                        },
                        "message": f"Thought recorded ({reasoning_type})",
                    })

                # Emit tool calls (data tools only for cleaner display)
                if data_tool_calls:
                    if enable_reasoning:
                        yield {
                            "type": "reasoning",
                            "phase": "tool_selection",
                            "action": "decision",
                            "content": f"Calling {len(data_tool_calls)} tools: {', '.join(tc.name for tc in data_tool_calls)}",
                        }

                    yield {
                        "type": "tool_calls",
                        "tools": [
                            {"name": tc.name, "arguments": tc.arguments}
                            for tc in data_tool_calls
                        ],
                    }

                # Execute data tools only
                if data_tool_calls:
                    data_tool_results = await self._execute_tool_calls(
                        tool_calls=data_tool_calls,
                        flow_id=flow_id,
                        user_id=user_id,
                        session_id=session_id,
                    )
                else:
                    data_tool_results = []

                # Combine results in original order
                tool_results = []
                think_idx = 0
                data_idx = 0
                for tc in tool_calls:
                    if tc.name == "think":
                        tool_results.append(think_results[think_idx])
                        think_idx += 1
                    else:
                        tool_results.append(data_tool_results[data_idx])
                        data_idx += 1

                total_tool_calls += len(tool_calls)

                # ============================================================
                # DYNAMIC TOOL INJECTION from tool_search results
                # ============================================================
                for tc, result in zip(tool_calls, tool_results):
                    if tc.name == "tool_search" and result.get("status") == "success":
                        found_tools = result.get("data", {}).get("found_tools", [])
                        for tool_name in found_tools:
                            if tool_name not in discovered_tool_names:
                                # Get full tool definition and inject
                                tool_def = self.catalog.get_full_schema(tool_name)
                                if tool_def:
                                    tools.append(tool_def.to_openai_function())
                                    discovered_tool_names.add(tool_name)
                                    self.logger.info(
                                        f"[{flow_id}] 🔧 INJECTED tool: {tool_name}"
                                    )

                        if found_tools:
                            self.logger.info(
                                f"[{flow_id}] tool_search discovered {len(found_tools)} tools, "
                                f"total available: {len(tools)}"
                            )

                # ============================================================
                # AUTO WEB SEARCH: When news tools are called OR enable_web_search=True
                # Uses WebSearchTool (OpenAI primary, Tavily fallback)
                # ============================================================
                called_tool_names = {tc.name for tc in data_tool_calls}
                news_tools_called = called_tool_names & NEWS_TOOL_NAMES
                web_search_already_called = "webSearch" in called_tool_names or "serpSearch" in called_tool_names

                should_auto_search = (
                    (enable_web_search or news_tools_called) and
                    not web_search_already_called and
                    turn_num == 1  # Only auto-search on first turn
                )

                if should_auto_search:
                    self.logger.info(
                        f"[{flow_id}] 🌐 AUTO WEB SEARCH triggered: "
                        f"enable_web_search={enable_web_search} | "
                        f"news_tools={news_tools_called}"
                    )

                    # Execute web search with SSE events (OpenAI primary, Tavily fallback)
                    web_search_result = None
                    async for ws_event in self._execute_auto_web_search(
                        query=query,
                        symbols=merged_symbols,
                        flow_id=flow_id,
                        enable_streaming=enable_reasoning,
                    ):
                        ws_type = ws_event.get("type", "")

                        # Forward SSE events for UI
                        if ws_type in ["web_search_start", "web_search_progress", "web_search_complete"]:
                            yield ws_event

                        # Capture final result
                        elif ws_type == "web_search_result":
                            web_search_result = ws_event

                    # Add web search result to tool results for LLM context
                    if web_search_result and web_search_result.get("status") == "success":
                        # Create a virtual tool call for the web search
                        web_search_tc_id = f"ws_{uuid.uuid4().hex[:8]}"

                        # Add to messages as a tool result
                        messages.append({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": web_search_tc_id,
                                "type": "function",
                                "function": {
                                    "name": "webSearch",
                                    "arguments": json.dumps({"query": query}),
                                },
                            }],
                        })

                        messages.append({
                            "role": "tool",
                            "tool_call_id": web_search_tc_id,
                            "content": json.dumps({
                                "tool_name": "webSearch",
                                "status": "success",
                                "data": web_search_result.get("data", {}),
                                "formatted_context": web_search_result.get("formatted_context", ""),
                            }, ensure_ascii=False),
                        })

                        total_tool_calls += 1
                        self.logger.info(
                            f"[{flow_id}] ✅ Web search result added to context"
                        )

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

            # Max turns reached
            self.logger.info(f"[{flow_id}] Max turns ({max_turns}) reached - generating final response")
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
                "content": """Please provide your COMPREHENSIVE final response based on all the information gathered.

IMPORTANT:
- Include ALL important data points and numbers from tool results
- For each symbol, provide: price, technical signals (RSI, MACD), fundamental metrics (P/E, ROE)
- Give clear insights and actionable recommendations
- Don't truncate or summarize - be as detailed as needed for a thorough analysis
- End with 2-3 follow-up questions""",
            })

            # Fixed max_tokens as safety cutoff - length controlled by system prompt
            MAX_RESPONSE_TOKENS = 12000

            self.logger.info(f"[{flow_id}] Streaming final response after max_turns (max_tokens={MAX_RESPONSE_TOKENS})")
            content_chunks = 0
            async for chunk in self.llm_provider.stream_response(
                model_name=effective_model,
                messages=messages,
                provider_type=effective_provider,
                api_key=self.api_key,
                max_tokens=MAX_RESPONSE_TOKENS,
                temperature=0.3,
            ):
                content_chunks += 1
                yield {"type": "content", "content": chunk}

            self.logger.info(f"[{flow_id}] Final response after max_turns: {content_chunks} chunks")
            yield {
                "type": "done",
                "total_turns": max_turns,
                "total_tool_calls": total_tool_calls,
            }

        except Exception as e:
            self.logger.error(f"[{flow_id}] Error: {e}", exc_info=True)
            yield {"type": "error", "error": str(e)}

    def _build_agent_messages_all_tools(
        self,
        query: str,
        intent_result: Any,
        conversation_history: Optional[List[Dict[str, str]]],
        system_language: str,
        core_memory: Optional[str],
        conversation_summary: Optional[str],
        tools: List[Dict[str, Any]],
        user_id: Optional[int],
        images: Optional[List[Any]] = None,
        merged_symbols: Optional[List[str]] = None,
        enable_think_tool: bool = False,
        enable_web_search: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Build messages for agent with ALL tools.

        Key difference: Agent sees ALL tools, prompt encourages smart selection.

        Args:
            merged_symbols: Pre-merged symbols from intent_result + working_memory.
                            If provided, these override intent_result.validated_symbols.
            enable_think_tool: If True, add STRONG instruction to ALWAYS use think tool.
            enable_web_search: If True, add STRONG instruction to ALWAYS search web.
        """
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        # Get market type and symbols from intent result
        market_type = getattr(intent_result, 'market_type', 'stock')
        if hasattr(market_type, 'value'):
            market_type = market_type.value

        # Use merged_symbols if provided, else fallback to intent_result
        validated_symbols = merged_symbols if merged_symbols is not None else getattr(intent_result, 'validated_symbols', [])
        intent_summary = getattr(intent_result, 'intent_summary', '')

        # Select skill for domain-specific prompt
        skill = self.skill_registry.select_skill(market_type)
        skill_prompt = skill.get_full_prompt()

        # Build context hints - be EXPLICIT about which symbols to analyze
        symbols_hint = ""
        if validated_symbols:
            if len(validated_symbols) == 1:
                symbols_hint = f"**PRIMARY SYMBOL TO ANALYZE: {validated_symbols[0]}** - Focus your analysis on this symbol only."
            else:
                symbols_hint = f"**SYMBOLS TO ANALYZE: {', '.join(validated_symbols)}** - Focus on these specific symbols."

        user_context = ""
        if core_memory:
            user_context += f"\n\n<USER_PROFILE>\n{core_memory}\n</USER_PROFILE>"
        if conversation_summary:
            user_context += f"\n\n<CONVERSATION_SUMMARY>\n{conversation_summary}\n</CONVERSATION_SUMMARY>"

        # Build tool list for reference
        tool_names = [t.get("function", {}).get("name", "") for t in tools if t.get("function")]

        system_prompt = f"""{skill_prompt}

---
## RUNTIME CONTEXT

Current Date: {current_date}
Response Language: {system_language.upper()}
{symbols_hint}
Intent: {intent_summary}
{user_context}

## TOOL EXECUTION GUIDELINES

**You have access to {len(tool_names)} tools including a special "think" tool.**

**IMPORTANT - ReAct Pattern (Reasoning + Acting):**
1. **THINK FIRST**: Use the `think` tool to analyze the query and plan your approach
2. **ACT**: Call the most relevant data tools based on your plan
3. **THINK AGAIN**: After receiving data, use `think` to analyze results and decide next steps
4. **REPEAT**: Continue until you have enough data
5. **RESPOND**: Generate your final response with data-backed insights

**🧠 THE THINK TOOL - USE IT!**

The `think` tool lets you reason explicitly. Use it:
- **BEFORE calling tools**: Plan what data you need and why
- **AFTER receiving data**: Analyze results, identify patterns, decide if more data needed
- **BEFORE final response**: Synthesize insights, formulate recommendations

Example flow for "So sánh NVDA và AMD để đầu tư":
```
Turn 1: think(planning) → "User wants investment comparison. Need: valuation (P/E),
        growth rates, technical momentum. User profile: growth investor."
        + getKeyMetrics(NVDA, AMD)

Turn 2: think(analyzing) → "NVDA P/E=65 vs AMD P/E=45. Higher valuation needs
        growth justification. Will check revenue growth."
        + getFinancialGrowth(NVDA, AMD)

Turn 3: think(deciding) → "NVDA 120% growth justifies premium. For growth investor,
        NVDA better fit. Check RSI for entry timing."
        + getTechnicalIndicators(NVDA)

Turn 4: Final response with data-backed, personalized recommendation
```

**Tool Selection Strategy - MANDATORY for Symbol Queries:**

⚠️ **ALWAYS CALL THESE TOOLS WHEN SYMBOLS ARE DETECTED:**
1. **getStockPrice/getCryptoPrice** - Get real-time price data (ALWAYS FIRST)
2. **getStockNews** - Get latest news for context
3. **webSearch/serpSearch** - Search for recent news, events, analyst opinions

Then add based on query type:
- Technical analysis: getTechnicalIndicators, getSupportResistance
- Fundamental analysis: getFinancialRatios, getIncomeStatement, getKeyMetrics
- Comparison: Get same tools for all symbols

**CRITICAL - Always Include External Context:**
- Use `webSearch` for latest market sentiment, analyst ratings, and recent news
- webSearch uses OpenAI (primary) or Tavily (fallback) automatically
- Include source URLs and publication dates in your response with proper citations
- Web search results include verified citations - display them to user
- NOTE: When news tools are called, web search is AUTO-triggered for additional context

## RESPONSE QUALITY REQUIREMENTS (CRITICAL)

**1. DATA PRESENTATION FORMAT (MANDATORY):**
Your response MUST follow this structure:

```
### 📊 Dữ liệu từ [Tool Name] | Nguồn: [Data Source]
| Chỉ tiêu | Giá trị | Ghi chú |
|----------|---------|---------|
| ...      | ...     | ...     |

### 💡 Phân tích & Nhận định
[Your analysis based on the data above]
```

**2. RAW DATA FIRST - ALWAYS SHOW COMPLETE DATA:**
- Display ALL raw data from tool results in TABLE format
- NEVER skip or summarize data - users need exact numbers for verification
- Include data timestamp/period clearly (e.g., "Q4 FY2025 (Oct-Jan)" or "Q4 CY2025 (Oct-Dec)")
- For financial data, always specify: Fiscal Year (FY) vs Calendar Year (CY)
  - Example: "NVDA Q4 FY2025 = kết thúc 26/01/2025" (NVDA fiscal year ends in January)
  - Example: "AAPL Q1 FY2025 = Oct-Dec 2024" (AAPL fiscal year starts in October)

**3. DATA SOURCE CITATION (REQUIRED):**
- ALWAYS cite the data source for each table: FMP API, Yahoo Finance, TradingView, etc.
- Include the data retrieval timestamp when available
- Format: "Nguồn: [Source] | Thời điểm: [Timestamp]"

**4. STRUCTURED TABLE FORMATS:**

For Income Statement:
| Chỉ tiêu | Giá trị | So với kỳ trước |
|----------|---------|-----------------|
| Doanh thu | $X.XX tỷ | +X.X% |
| Lợi nhuận gộp | $X.XX tỷ | +X.X% |
| Chi phí hoạt động | $X.XX tỷ | +X.X% |
| EBITDA | $X.XX tỷ | +X.X% |
| Lợi nhuận ròng | $X.XX tỷ | +X.X% |
| EPS | $X.XX | +X.X% |

For Technical Analysis:
| Chỉ báo | Giá trị | Tín hiệu |
|---------|---------|----------|
| RSI(14) | XX.X | Bullish/Bearish/Neutral |
| MACD | XX.X | Bullish/Bearish |
| MA(50) | $XXX.XX | Trên/Dưới giá |
| MA(200) | $XXX.XX | Trên/Dưới giá |

**5. ANALYSIS AFTER DATA:**
- Analysis and insights come AFTER showing raw data
- Reference specific numbers from the tables above
- Explain what the numbers mean in context
- Include actionable recommendations

**6. DATA INTEGRITY:**
- NEVER make up numbers - only use data from tool results
- If a tool failed or data is missing, explicitly state: "⚠️ Không có dữ liệu cho [item]"
- Be honest about data limitations or conflicting signals
- If fiscal year differs from calendar year, ALWAYS clarify

## OUTPUT VERBOSITY & STRUCTURE CONTROL

<output_verbosity_spec>
Adapt response length to query complexity - let content dictate length naturally:

- **Simple queries** (giá hiện tại, 1 chỉ báo): 2-4 câu với data table
- **Standard analysis** (1 symbol, technical/fundamental): Đầy đủ data + phân tích ngắn gọn
- **Complex analysis** (so sánh, multi-symbol, comprehensive):
  - Structured sections với headers
  - Tables cho data comparison
  - Detailed analysis với reasoning
- **Research queries** (tin tức, sự kiện, xu hướng): Include web search results với citations

Avoid long narrative paragraphs - prefer compact bullets, tables, and short focused sections.
Do not rephrase the user's request unless it changes semantics.
</output_verbosity_spec>

## WEB SEARCH & RESEARCH GUIDELINES

<web_research_rules>
**Web Search Integration:**
- Use `webSearch` tool for latest news, events, and market information
- webSearch uses OpenAI web search (primary) with Tavily fallback
- Returns verified citations with URLs - always include them in your response
- Web search is AUTO-triggered when news tools are called for additional context
- When `enable_web_search=true`, web search runs automatically on first turn

**When using web search tools:**
- Default to comprehensive, well-structured answers grounded in reliable sources
- Include citations (source name + URL) for ALL web-derived information
- Research until you have sufficient information for accurate, comprehensive answers
- For time-sensitive topics: explicitly compare publish dates and event dates
- Prioritize primary sources: Bloomberg, Reuters, official company sites, SEC filings
</web_research_rules>

<citation_format>
When citing web sources, use this format:
- Inline: "Theo [Source Name], ... [URL]"
- Or footnote style: "... tin tức mới nhất.¹" with sources listed at end
- ALWAYS include publication date when available
- For OpenAI web search citations: display title + URL from the citations array
</citation_format>

## RESPONSE STRUCTURE (Adapt to query complexity)

**For Comprehensive Analysis:**
1. **📊 Tổng quan** - Price, key metrics, market context
2. **📈 Phân tích kỹ thuật** - Indicators, signals, levels (if relevant)
3. **💰 Phân tích cơ bản** - Financials, valuation, growth (if relevant)
4. **📰 Tin tức & Context** - Recent news with sources and dates
5. **⚠️ Rủi ro & Cơ hội** - Key factors to consider
6. **💡 Kết luận** - Summary and actionable recommendation

**For Simple Queries:**
- Direct answer with supporting data
- Skip irrelevant sections
- Use tables for numeric data

## HANDLING UNCERTAINTY

<uncertainty_and_ambiguity>
- If query is ambiguous: state your interpretation, then answer comprehensively
- When external facts may have changed: answer in general terms, note details may have changed
- Never fabricate exact figures or citations when uncertain
- Prefer language like "Dựa trên dữ liệu có sẵn..." instead of absolute claims
- If data is missing or tool failed: explicitly state "⚠️ Không có dữ liệu cho [item]"
</uncertainty_and_ambiguity>

## QUALITY CHECKLIST (Before responding)

Before finalizing, verify:
1. Did you answer ALL parts of the query?
2. Did you include concrete data/numbers from tools?
3. Did you cite sources for web-derived information?
4. Are there any unstated assumptions or unverifiable claims?
5. Is the response length appropriate for query complexity?
"""

        # ====================================================================
        # DYNAMIC INSTRUCTIONS BASED ON FLAGS
        # ====================================================================

        # Add STRONG think tool instructions when enabled
        if enable_think_tool:
            think_tool_instruction = """

## 🧠 MANDATORY: USE THE THINK TOOL (enable_think_tool=True)

**⚠️ CRITICAL REQUIREMENT - YOU MUST USE THE THINK TOOL!**

The `think` tool is MANDATORY for this conversation. You MUST:

1. **ALWAYS START with think()**: Before calling ANY data tool, call `think` first to:
   - Analyze what the user is asking
   - Plan which tools you need to call
   - Identify what data will answer the question

2. **ALWAYS THINK AFTER DATA**: After receiving tool results, call `think` to:
   - Analyze the data you received
   - Identify key insights and patterns
   - Decide if you need more data or can respond

3. **THINK BEFORE RESPONDING**: Before generating your final response, call `think` to:
   - Synthesize all the information
   - Formulate your recommendations
   - Structure your response

**Example MANDATORY Pattern:**
```
Turn 1: think("Planning: User asks X. I need to call Y, Z tools.") → getStockPrice()
Turn 2: think("Data received: Price is $X. Now analyzing...") → getTechnicalIndicators()
Turn 3: think("Synthesizing: Price + indicators suggest...") → Final response
```

**DO NOT skip think() calls. This is not optional!**
"""
            system_prompt += think_tool_instruction

        # Add STRONG web search instructions when enabled
        if enable_web_search:
            web_search_instruction = """

## 🌐 MANDATORY: USE WEB SEARCH (enable_web_search=True)

**⚠️ CRITICAL REQUIREMENT - YOU MUST SEARCH THE WEB!**

Web search is MANDATORY for this conversation. You MUST:

1. **ALWAYS call webSearch or serpSearch** in your first turn (or early turns):
   - Search for latest news about the symbols/topics
   - Search for recent analyst opinions and ratings
   - Search for any recent events that affect the analysis

2. **Include web sources in your response**:
   - Cite the source name and URL for all web-derived information
   - Include publication dates when available
   - Prioritize recent and authoritative sources

3. **Do NOT skip web search**: Even if you have financial data from other tools,
   web search provides crucial context about recent events, sentiment, and news.

**Example Search Queries:**
- "NVDA stock news today January 2025"
- "NVIDIA earnings announcement 2025"
- "AMD vs NVDA analyst ratings 2025"

**Both webSearch (Tavily) and serpSearch (Google) are available - use either or both.**
"""
            system_prompt += web_search_instruction

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
