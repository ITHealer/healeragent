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
    # Gemini 3+ requires thought_signature to be preserved and sent back
    # See: https://ai.google.dev/gemini-api/docs/thought-signatures
    thought_signature: Optional[str] = None
    # Original Part proto bytes - preserves thought_signature even if SDK doesn't expose it
    # This is CRITICAL for Gemini 3+ function calling to work
    _part_proto_bytes: Optional[str] = None


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

    # Default tool timeout (increased for stability)
    DEFAULT_TOOL_TIMEOUT = 90.0

    # Slow tools with longer timeouts (seconds)
    SLOW_TOOL_TIMEOUTS = {
        "webSearch": 180.0,     # Web search can take 60+ seconds with OpenAI Responses API
        "serpSearch": 120.0,    # SerpAPI search
    }

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
                    yield {"type": "content", "content": chunk, "is_final": False}

                # Signal end of content stream
                yield {"type": "content", "content": "", "is_final": True}
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

        # Log tool results for data integrity verification
        self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════════════")
        self.logger.info(f"[{flow_id}] TOOL RESULTS → LLM SYNTHESIS ({len(tool_results)} tools)")
        self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════════════")
        for i, result in enumerate(tool_results):
            tool_name = result.get("tool_name", "unknown")
            status = result.get("status", "unknown")
            symbol = result.get("symbol", "N/A")
            formatted_context = result.get("formatted_context")
            data_keys = list(result.get("data", {}).keys()) if result.get("data") else []

            # Log tool info
            self.logger.info(f"[{flow_id}] [{i+1}] {tool_name} | symbol={symbol} | status={status}")

            # Log formatted_context status with length
            if formatted_context:
                self.logger.info(f"[{flow_id}]     ✅ formatted_context: {len(formatted_context)} chars")
                self.logger.info(f"[{flow_id}]     Preview: {formatted_context[:300]}...")
            else:
                self.logger.warning(f"[{flow_id}]     ⚠️ formatted_context: MISSING (will use json.dumps of data)")
                self.logger.info(f"[{flow_id}]     data keys: {data_keys[:10]}")  # First 10 keys
        self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════════════")

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
            max_tokens=16000,  # Allow longer responses
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

        # Log tool results for data integrity verification
        self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════════════")
        self.logger.info(f"[{flow_id}] TOOL RESULTS → LLM SYNTHESIS ({len(tool_results)} tools)")
        self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════════════")
        for i, result in enumerate(tool_results):
            tool_name = result.get("tool_name", "unknown")
            status = result.get("status", "unknown")
            symbol = result.get("symbol", "N/A")
            formatted_context = result.get("formatted_context")
            data_keys = list(result.get("data", {}).keys()) if result.get("data") else []

            # Log tool info
            self.logger.info(f"[{flow_id}] [{i+1}] {tool_name} | symbol={symbol} | status={status}")

            # Log formatted_context status with length
            if formatted_context:
                self.logger.info(f"[{flow_id}]     ✅ formatted_context: {len(formatted_context)} chars")
                self.logger.info(f"[{flow_id}]     Preview: {formatted_context[:300]}...")
            else:
                self.logger.warning(f"[{flow_id}]     ⚠️ formatted_context: MISSING (will use json.dumps of data)")
                self.logger.info(f"[{flow_id}]     data keys: {data_keys[:10]}")  # First 10 keys
        self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════════════")

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
            max_tokens=16000,  # Allow longer responses
            temperature=0.3,
        ):
            yield {"type": "content", "content": chunk, "is_final": False}

        # Signal end of content stream
        yield {"type": "content", "content": "", "is_final": True}
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

            # ============================================================
            # DETAILED LOGGING: Show tool results being sent to LLM
            # ============================================================
            self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════════════")
            self.logger.info(f"[{flow_id}] TOOL RESULTS → LLM (Turn {turn_num})")
            self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════════════")
            for i, (tc, result) in enumerate(zip(tool_calls, tool_results)):
                tool_name = tc.name
                status = result.get("status", "unknown")
                formatted_context = result.get("formatted_context")
                data_keys = list(result.get("data", {}).keys()) if result.get("data") else []

                self.logger.info(f"[{flow_id}] [{i+1}] {tool_name} | status={status}")

                if formatted_context:
                    self.logger.info(f"[{flow_id}]     ✅ formatted_context: {len(formatted_context)} chars")
                    self.logger.info(f"[{flow_id}]     Preview: {formatted_context[:400]}...")
                else:
                    self.logger.warning(f"[{flow_id}]     ⚠️ formatted_context: MISSING")
                    self.logger.info(f"[{flow_id}]     data keys: {data_keys[:10]}")

                if tool_name == "getTechnicalIndicators" and result.get("data"):
                    data = result["data"]
                    outlook = data.get("outlook", {})
                    rec = data.get("trading_recommendation", {})
                    self.logger.info(f"[{flow_id}]     📊 outlook={outlook.get('outlook')} | action={rec.get('overall_action')}")
            self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════════════")

            # Update messages (preserves thought_signature for Gemini 3+)
            messages.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": [self._build_tool_call_dict(tc) for tc in tool_calls],
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
            "content": """Please provide your COMPREHENSIVE final response based on all the information gathered.

IMPORTANT:
- Include ALL important data points and numbers from tool results
- For each symbol, provide: price, technical signals (RSI, MACD), fundamental metrics (P/E, ROE)
- Give clear insights and actionable recommendations
- Don't truncate or summarize - be as detailed as needed for a thorough analysis
- End with 2-3 follow-up questions""",
        })

        response = await self.llm_provider.generate_response(
            model_name=self.model_name,
            messages=messages,
            provider_type=self.provider_type,
            api_key=self.api_key,
            max_tokens=16000,  # Allow longer responses
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
                MAX_RESPONSE_TOKENS = 32000  # High limit - let model generate full responses

                async for chunk in self.llm_provider.stream_response(
                    model_name=effective_model,
                    messages=messages,
                    provider_type=effective_provider,
                    api_key=self.api_key,
                    max_tokens=MAX_RESPONSE_TOKENS,
                    temperature=0.3,
                ):
                    yield {"type": "content", "content": chunk, "is_final": False}

                # Signal end of content stream
                yield {"type": "content", "content": "", "is_final": True}
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

            # ============================================================
            # DETAILED LOGGING: Show tool results being sent to LLM
            # ============================================================
            self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════════════")
            self.logger.info(f"[{flow_id}] TOOL RESULTS → LLM (Turn {turn_num})")
            self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════════════")
            for i, (tc, result) in enumerate(zip(tool_calls, tool_results)):
                tool_name = tc.name
                status = result.get("status", "unknown")
                formatted_context = result.get("formatted_context")
                data_keys = list(result.get("data", {}).keys()) if result.get("data") else []

                self.logger.info(f"[{flow_id}] [{i+1}] {tool_name} | status={status}")

                if formatted_context:
                    self.logger.info(f"[{flow_id}]     ✅ formatted_context: {len(formatted_context)} chars")
                    # Show first 400 chars to see key indicators
                    self.logger.info(f"[{flow_id}]     Preview: {formatted_context[:400]}...")
                else:
                    self.logger.warning(f"[{flow_id}]     ⚠️ formatted_context: MISSING")
                    self.logger.info(f"[{flow_id}]     data keys: {data_keys[:10]}")

                # For technical indicators, show key data
                if tool_name == "getTechnicalIndicators" and result.get("data"):
                    data = result["data"]
                    outlook = data.get("outlook", {})
                    rec = data.get("trading_recommendation", {})
                    self.logger.info(f"[{flow_id}]     📊 outlook={outlook.get('outlook')} | action={rec.get('overall_action')}")
            self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════════════")

            # Emit results
            yield {
                "type": "tool_results",
                "results": [
                    {"tool": tc.name, "success": r.get("status") in ["success", "200"]}
                    for tc, r in zip(tool_calls, tool_results)
                ],
            }

            # Update messages (preserves thought_signature for Gemini 3+)
            messages.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": [self._build_tool_call_dict(tc) for tc in tool_calls],
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
        MAX_RESPONSE_TOKENS = 32000  # High limit - let model generate full responses

        async for chunk in self.llm_provider.stream_response(
            model_name=effective_model,
            messages=messages,
            provider_type=effective_provider,
            api_key=self.api_key,
            max_tokens=MAX_RESPONSE_TOKENS,
            temperature=0.3,
        ):
            yield {"type": "content", "content": chunk, "is_final": False}

        # Signal end of content stream
        yield {"type": "content", "content": "", "is_final": True}
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
            max_tokens=16000,  # Allow longer responses
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
        system_prompt_override: Optional[str] = None,
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
            system_prompt_override=system_prompt_override,
        )

        async for chunk in self.llm_provider.stream_response(
            model_name=effective_model,
            messages=messages,
            provider_type=effective_provider,
            api_key=self.api_key,
            max_tokens=16000,  # Allow longer responses
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

                # Use per-tool timeout for slow tools
                tool_timeout = self.SLOW_TOOL_TIMEOUTS.get(
                    tool_name, self.DEFAULT_TOOL_TIMEOUT
                )

                self.logger.debug(
                    f"[{flow_id}] Executing {tool_name} with args: {args} (timeout={tool_timeout}s)"
                )

                result = await asyncio.wait_for(
                    self.registry.execute_tool(tool_name=tool_name, params=args),
                    timeout=tool_timeout,
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
                    "error": f"Tool timed out after {tool_timeout}s",
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

        def normalize_tool_name(name: str) -> str:
            """
            Normalize tool name by removing hallucinated prefixes.

            LLMs sometimes add prefixes like:
            - tool_finance.getStockPrice -> getStockPrice
            - functions.getStockPrice -> getStockPrice
            - tools.getStockPrice -> getStockPrice
            """
            # Common hallucinated prefixes to strip
            prefixes_to_strip = [
                "tool_finance.",
                "tool_stock.",
                "tool_crypto.",
                "tool_market.",
                "tool_news.",
                "tool_technical.",
                "tool_fundamental.",
                "tool_risk.",
                "tools.",
                "functions.",
                "function.",
            ]

            original_name = name
            for prefix in prefixes_to_strip:
                if name.lower().startswith(prefix.lower()):
                    name = name[len(prefix):]
                    self.logger.warning(
                        f"[TOOL_NORMALIZE] Stripped prefix: '{original_name}' -> '{name}'"
                    )
                    break

            return name

        async def execute_single(tc: ToolCall) -> Dict[str, Any]:
            try:
                # Normalize tool name (fix LLM hallucinations like "tool_finance.getStockPrice")
                actual_tool_name = normalize_tool_name(tc.name)

                # ============================================================
                # SPECIAL HANDLING: tool_search meta-tool
                # ============================================================
                if actual_tool_name == "tool_search":
                    return await self._execute_tool_search(tc, flow_id)

                # ============================================================
                # REGULAR TOOL EXECUTION
                # ============================================================
                # Use per-tool timeout for slow tools
                tool_timeout = self.SLOW_TOOL_TIMEOUTS.get(
                    actual_tool_name, self.DEFAULT_TOOL_TIMEOUT
                )

                result = await asyncio.wait_for(
                    self.registry.execute_tool(tool_name=actual_tool_name, params=tc.arguments),
                    timeout=tool_timeout,
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
                    "tool_name": actual_tool_name,
                    "status": "timeout",
                    "error": f"Timeout after {tool_timeout}s",
                }
            except Exception as e:
                return {
                    "tool_name": actual_tool_name,
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
                # Normalize tool name for consistency
                normalized_name = normalize_tool_name(tool_calls[i].name)
                processed.append({
                    "tool_name": normalized_name,
                    "status": "error",
                    "error": str(result),
                })
            else:
                processed.append(result)

        # ============================================================
        # DEBUG: Log tool results before sending to LLM
        # ============================================================
        self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════════════")
        self.logger.info(f"[{flow_id}] TOOL RESULTS DETAIL (before sending to LLM)")
        self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════════════")

        for i, res in enumerate(processed):
            tool_name = res.get("tool_name", "unknown")
            status = res.get("status", "unknown")
            formatted_context = res.get("formatted_context")
            data = res.get("data", {})
            data_keys = list(data.keys()) if isinstance(data, dict) else []

            self.logger.info(f"[{flow_id}] [{i+1}] 📊 {tool_name} | status={status}")

            # Log formatted_context status
            if formatted_context:
                self.logger.info(f"[{flow_id}]     ✅ formatted_context: {len(formatted_context)} chars")
                # Show preview (first 500 chars for technical indicators)
                preview = formatted_context[:500].replace('\n', ' | ')
                self.logger.info(f"[{flow_id}]     Preview: {preview}...")
            else:
                self.logger.warning(f"[{flow_id}]     ⚠️ formatted_context: MISSING")
                self.logger.info(f"[{flow_id}]     data keys: {data_keys[:10]}")

            # Log key data from specific tools
            if status == "success" and isinstance(data, dict):
                # Technical Indicators
                if tool_name == "getTechnicalIndicators":
                    outlook = data.get("outlook", {})
                    rec = data.get("trading_recommendation", {})
                    self.logger.info(
                        f"[{flow_id}]     📊 outlook={outlook.get('outlook')} | "
                        f"action={rec.get('overall_action')} | "
                        f"signal_agreement={rec.get('signal_agreement_pct')}%"
                    )

                # Income Statement specific logging
                if "revenue" in data:
                    rev = data.get('revenue')
                    net = data.get('net_income')
                    eps = data.get('eps')
                    rev_str = f"{rev:,.0f}" if isinstance(rev, (int, float)) else str(rev)
                    net_str = f"{net:,.0f}" if isinstance(net, (int, float)) else str(net)
                    self.logger.info(
                        f"[{flow_id}]     └─ revenue={rev_str} | "
                        f"net_income={net_str} | eps={eps}"
                    )

                # Log first statement if available
                if "statements" in data and data["statements"]:
                    stmt = data["statements"][0]
                    stmt_rev = stmt.get('revenue')
                    rev_str = f"{stmt_rev:,.0f}" if isinstance(stmt_rev, (int, float)) else "N/A"
                    self.logger.info(
                        f"[{flow_id}]     └─ Latest: date={stmt.get('date')} | "
                        f"period={stmt.get('period')} | revenue={rev_str}"
                    )

        self.logger.info(f"[{flow_id}] ═══════════════════════════════════════════════════════")

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
                max_tokens=16000,  # High limit for tool calls with thinking models
                temperature=0.1,
            )

            return response if isinstance(response, dict) else {"content": str(response)}

        except Exception as e:
            self.logger.error(f"[{flow_id}] LLM call failed: {e}")
            raise

    def _parse_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """
        Parse tool calls from LLM response.

        For Gemini 3+ models, also extracts thought_signature and _part_proto_bytes
        which must be preserved and sent back in subsequent turns.
        See: https://ai.google.dev/gemini-api/docs/thought-signatures
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

                    # Extract thought_signature for Gemini 3+ models
                    # This MUST be preserved and sent back for function calling to work
                    thought_signature = call.get("thought_signature")

                    # Extract _part_proto_bytes (original Part proto serialized)
                    # This is CRITICAL for Gemini 3+ - preserves thought_signature
                    # even when SDK doesn't expose the field
                    part_proto_bytes = call.get("_part_proto_bytes")

                    if name:
                        tool_calls.append(ToolCall(
                            id=call_id,
                            name=name,
                            arguments=arguments,
                            thought_signature=thought_signature,
                            _part_proto_bytes=part_proto_bytes,
                        ))

            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Failed to parse tool call: {e}")
                continue

        return tool_calls

    def _build_tool_call_dict(self, tc: ToolCall) -> Dict[str, Any]:
        """
        Build OpenAI-compatible tool_call dict from ToolCall object.

        For Gemini 3+ models, includes thought_signature and _part_proto_bytes if present.
        These MUST be preserved for function calling to work correctly.
        See: https://ai.google.dev/gemini-api/docs/thought-signatures
        """
        tool_call_dict = {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments,
            },
        }

        # Include thought_signature for Gemini 3+ models
        # This is CRITICAL - missing signature causes 400 error
        if tc.thought_signature:
            tool_call_dict["thought_signature"] = tc.thought_signature

        # Include _part_proto_bytes - preserves thought_signature even when SDK
        # doesn't expose the field (critical for Gemini 3+ function calling)
        if tc._part_proto_bytes:
            tool_call_dict["_part_proto_bytes"] = tc._part_proto_bytes

        return tool_call_dict

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
## Current Context
- Date: {current_date}
- Language: {system_language.upper()}
{f'- {symbols_hint}' if symbols_hint else ''}
{f'- {categories_hint}' if categories_hint else ''}
{user_context}

{execution_guidelines}

## Data Integrity
- Only use actual numbers from tool results (never fabricate)
- Cite data sources when presenting financial data
- Acknowledge missing data clearly
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
        market_type = "stock"
        if classification:
            market_type = getattr(classification, "market_type", "stock") or "stock"

        skill = self.skill_registry.select_skill(market_type)
        analysis_framework = skill.get_analysis_framework()

        # Check if web search results are present
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

        system_prompt = f"""You are a {skill.config.description}.

{analysis_framework}

---
## Current Context
- Date: {current_date}

## Tool Results Data (CRITICAL - USE ALL DATA)

{results_text}

---
## Your Task

Analyze the data above and provide a **comprehensive, data-driven response** to the user's query.

### Data Integrity Requirements (MANDATORY)
1. **USE ALL DATA**: Every piece of data from tools MUST be included in your analysis
2. **CITE SPECIFIC NUMBERS**: Always quote exact values (prices, ratios, percentages) from tool results
3. **NO FABRICATION**: Only use numbers that appear in the data above - never make up figures
4. **SOURCE ATTRIBUTION**: Reference which tool provided each data point when relevant

### Analysis Quality Requirements
1. **EXPLAIN SIGNIFICANCE**: For every metric, explain what it means for investment decisions
   - Example: "RSI at 72 indicates overbought conditions, suggesting potential short-term pullback"
2. **PROVIDE CONTEXT**: Compare metrics to benchmarks, historical ranges, or sector averages
3. **IDENTIFY PATTERNS**: Connect data points to reveal trends, divergences, or confirmations
4. **BALANCED VIEW**: Present both bullish and bearish signals objectively

### Actionable Output Requirements
1. **SPECIFIC RECOMMENDATIONS**: Provide clear action items based on data
   - Entry/exit levels, support/resistance zones, key price targets
2. **RISK ASSESSMENT**: Identify risks and suggest risk management strategies
3. **STRATEGIC INSIGHTS**: Offer short-term and long-term perspectives
4. **DECISION FRAMEWORK**: Help user understand when to act and what to monitor

### Response Style
- **Language**: Match the user's language naturally
- **Depth**: Be comprehensive - longer is better if it adds value
- **Clarity**: Structure with headers, use bullet points for key data
- **Engagement**: End with 2-3 follow-up questions to explore deeper"""

        if has_web_search:
            system_prompt += """

**Source Citations:**
Include web sources at the end: [Title](URL)"""

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
        system_prompt_override: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build messages for no-tools response."""
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        user_context = ""
        if core_memory:
            user_context += f"\n\n<USER_PROFILE>\n{core_memory}\n</USER_PROFILE>"
        if conversation_summary:
            user_context += f"\n\n<CONVERSATION_SUMMARY>\n{conversation_summary}\n</CONVERSATION_SUMMARY>"

        # Use character persona override if provided, otherwise default
        if system_prompt_override:
            base_prompt = system_prompt_override
        else:
            base_prompt = "You are a friendly financial assistant."

        system_prompt = f"""{base_prompt}

Current Date: {current_date}
Response Language: {system_language.upper()}
{user_context}

Respond naturally and helpfully while staying in character."""

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
        system_prompt_override: Optional[str] = None,
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
            system_prompt_override: If provided, replace the default skill-based
                system prompt with this custom prompt. Used for character agents
                to inject investment personas while keeping tool guidelines.

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
                    system_prompt_override=system_prompt_override,  # Pass character persona
                ):
                    yield {"type": "content", "content": chunk, "is_final": False}

                # Signal end of content stream
                yield {"type": "content", "content": "", "is_final": True}
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
                system_prompt_override=system_prompt_override,  # Character persona override
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

                # Emit thinking content if present (Gemini 3+ chain-of-thought)
                # This shows the model's reasoning process (like "Hiện tiến trình tư duy")
                thinking_content = response.get("thinking_content", "")
                if thinking_content and enable_reasoning:
                    yield {
                        "type": "thinking",
                        "phase": f"turn_{turn_num}",
                        "content": thinking_content,
                    }
                    self.logger.info(f"[{flow_id}] 💭 Thinking: {thinking_content[:100]}...")

                # No tool calls - stream final response
                if not tool_calls:
                    if enable_reasoning:
                        yield {
                            "type": "reasoning",
                            "phase": "synthesis",
                            "action": "start",
                            "content": "Agent decided no more tools needed - generating final response",
                        }

                    # If we already have content from the first LLM call, yield it directly
                    # This avoids making a second streaming call which can lose content (especially with Gemini)
                    if assistant_content and len(assistant_content) > 100:
                        self.logger.info(f"[{flow_id}] Using existing response content ({len(assistant_content)} chars)")

                        # Simulate streaming by yielding content in chunks
                        # Split by sentences/paragraphs for natural streaming feel
                        import re
                        # Split on sentence boundaries while keeping the delimiter
                        chunks = [c for c in re.split(r'(?<=[.!?。\n])\s*', assistant_content) if c.strip()]
                        total_chunks = len(chunks)

                        for i, chunk in enumerate(chunks):
                            is_last = (i == total_chunks - 1)
                            yield {
                                "type": "content",
                                "content": chunk + (" " if not is_last else ""),
                                "is_final": is_last
                            }

                        self.logger.info(f"[{flow_id}] Yielded existing content: {total_chunks} chunks")
                    else:
                        # No content from first call - need to make streaming call
                        messages.append({"role": "assistant", "content": assistant_content})

                        # High limit - let model generate full responses
                        MAX_RESPONSE_TOKENS = 32000

                        self.logger.info(f"[{flow_id}] Streaming final response (max_tokens={MAX_RESPONSE_TOKENS})")
                        content_chunks = []
                        async for chunk in self.llm_provider.stream_response(
                            model_name=effective_model,
                            messages=messages,
                            provider_type=effective_provider,
                            api_key=self.api_key,
                            max_tokens=MAX_RESPONSE_TOKENS,
                            temperature=0.3,
                        ):
                            content_chunks.append(chunk)
                            # Emit chunks as they come, mark all as not final initially
                            yield {"type": "content", "content": chunk, "is_final": False}

                        # Emit final empty chunk to signal completion if we had content
                        if content_chunks:
                            yield {"type": "content", "content": "", "is_final": True}

                        self.logger.info(f"[{flow_id}] Final response streamed: {len(content_chunks)} chunks")

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

                # Update messages (preserves thought_signature for Gemini 3+)
                messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": [self._build_tool_call_dict(tc) for tc in tool_calls],
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
            MAX_RESPONSE_TOKENS = 32000  # High limit - let model generate full responses

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
                yield {"type": "content", "content": chunk, "is_final": False}

            # Signal end of content stream
            yield {"type": "content", "content": "", "is_final": True}
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
        system_prompt_override: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build messages for agent with ALL tools.

        Uses skill-based prompts for natural, conversational responses.
        """
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        # Get market type and symbols from intent result
        market_type = getattr(intent_result, 'market_type', 'stock')
        if hasattr(market_type, 'value'):
            market_type = market_type.value

        # Use merged_symbols if provided, else fallback to intent_result
        validated_symbols = merged_symbols if merged_symbols is not None else getattr(intent_result, 'validated_symbols', [])
        intent_summary = getattr(intent_result, 'intent_summary', '')

        # Select base prompt: use override for character agents, otherwise skill-based
        if system_prompt_override:
            base_prompt = system_prompt_override
        else:
            skill = self.skill_registry.select_skill(market_type)
            base_prompt = skill.get_full_prompt()

        # Build context section
        context_parts = [f"Current Date: {current_date}"]

        if validated_symbols:
            symbols_str = ', '.join(validated_symbols)
            context_parts.append(f"Symbols to analyze: {symbols_str}")

        if intent_summary:
            context_parts.append(f"User intent: {intent_summary}")

        # User context (core_memory, conversation_summary)
        user_context = ""
        if core_memory:
            user_context += f"\n\n<USER_PROFILE>\n{core_memory}\n</USER_PROFILE>"
        if conversation_summary:
            user_context += f"\n\n<CONVERSATION_SUMMARY>\n{conversation_summary}\n</CONVERSATION_SUMMARY>"

        # Build system prompt - simplified and natural
        system_prompt = f"""{base_prompt}

---
## Current Context
{chr(10).join('- ' + p for p in context_parts)}
{user_context}

## Tool Usage

You have access to various data tools. Use them to gather real data before responding.

**Key Principles:**
1. **Call tools first** - Get actual data before making claims
2. **Think before responding** - Analyze data thoroughly before writing your response
3. **Be accurate** - Never fabricate numbers, only use tool results
4. **Respond naturally** - Match the user's language, be conversational

**CRITICAL - Response Style:**
- **NEVER mention internal tool names** in your response (e.g., DON'T say "getStockPrice returned..." or "Using getTechnicalIndicators...")
- Present data naturally as if you already know it (e.g., "AAPL is currently trading at $259.96" NOT "getStockPrice shows AAPL at $259.96")
- Reference data sources generically: "Real-time data shows..." or "Market data indicates..."
- The user should NOT know which specific tools you used - just deliver the insights

**Language (IMPORTANT):**
- **Match the user's language** - If they write in Vietnamese, respond in Vietnamese. If English, respond in English.
- **Never switch languages mid-conversation** unless the user does first
- Be consistent throughout your entire response"""

        # Add think tool instruction if enabled (works for both GPT and Gemini)
        if enable_think_tool:
            system_prompt += """

**Thinking Process (RECOMMENDED):**
Use the `think` tool to organize your reasoning. This helps you:
- Plan which data to gather before calling tools
- Analyze and synthesize results after receiving tool data
- Catch potential errors or contradictions in your analysis
Pattern: think(plan) → call tools → think(analyze) → respond"""

        # Add web search instruction based on mode
        if enable_web_search:
            # FORCED mode: always use web search
            system_prompt += """

**Web Search (REQUIRED - ENABLED):**
You MUST use `webSearch` to get latest news and real-time information.
When presenting web search results:
- Integrate findings naturally into your response
- Include relevant source URLs inline or in a "Sources:" section at the end
- Format: "According to [Source Name](URL), ..." or list sources at the end
- Example Sources section:
  **Sources:**
  - [Bloomberg: AAPL hits new high](https://bloomberg.com/...)
  - [Reuters: Fed rate decision](https://reuters.com/...)"""
        else:
            # AUTO mode: LLM decides when internal tools are insufficient
            system_prompt += """

**Web Search (AVAILABLE - USE WHEN NEEDED):**
Use `webSearch` when your internal data tools don't have sufficient information to answer accurately:
- Latest breaking news or recent events (last 24-48 hours)
- Information not covered by financial data APIs
- User asks about topics beyond market data
When presenting web search results:
- Integrate findings naturally into your response
- Include relevant source URLs in a "Sources:" section
- Format sources as: [Title](URL)"""

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

    async def run_with_all_tools(
        self,
        query: str,
        intent_result: Any,
        session_id: Optional[str] = None,
        user_id: Optional[int] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_language: str = "en",
        core_memory: Optional[str] = None,
        conversation_summary: Optional[str] = None,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None,
        max_turns: int = 6,
        working_memory_symbols: Optional[List[str]] = None,
        system_prompt_override: Optional[str] = None,
    ) -> "AgentRunResult":
        """
        Non-streaming version of run_stream_with_all_tools.

        Collects all stream events and returns a consolidated result.
        Useful for non-streaming API endpoints.

        Args:
            query: User query
            intent_result: IntentResult from IntentClassifier
            system_prompt_override: Optional character persona prompt override

        Returns:
            AgentRunResult with response, total_turns, total_tool_calls
        """
        full_response = []
        total_turns = 0
        total_tool_calls = 0

        async for event in self.run_stream_with_all_tools(
            query=query,
            intent_result=intent_result,
            session_id=session_id,
            user_id=user_id,
            conversation_history=conversation_history,
            system_language=system_language,
            core_memory=core_memory,
            conversation_summary=conversation_summary,
            model_name=model_name,
            provider_type=provider_type,
            max_turns=max_turns,
            working_memory_symbols=working_memory_symbols,
            system_prompt_override=system_prompt_override,
        ):
            event_type = event.get("type", "")

            if event_type == "content":
                content = event.get("content", "")
                full_response.append(content)
            elif event_type == "content_delta":
                content = event.get("content", "")
                full_response.append(content)
            elif event_type == "done":
                total_turns = event.get("total_turns", total_turns)
                total_tool_calls = event.get("total_tool_calls", total_tool_calls)

        return AgentRunResult(
            response="".join(full_response),
            total_turns=total_turns,
            total_tool_calls=total_tool_calls,
        )


@dataclass
class AgentRunResult:
    """Result from non-streaming agent run."""
    response: str
    total_turns: int
    total_tool_calls: int


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
