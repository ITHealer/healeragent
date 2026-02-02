"""
The Orchestrator - State Machine Controller for the invest_agent pipeline.

Why: This is the central brain of the invest_agent module. It implements a
state machine (Setup -> ModeResolution -> ExecutionLoop -> Response) that
adapts its behavior based on the resolved mode (Instant vs Thinking).

How: The orchestrator is an async generator that yields SSEEvent objects.
The router iterates over these events and serializes them to the SSE wire
format. Each state transition emits relevant events so the frontend can
display progress in real-time.

Key design decisions:
- State machine over ad-hoc branching: makes the control flow explicit and testable
- Async generator over callbacks: natural fit for SSE streaming
- Never crashes: every external call is wrapped in try/catch with fallback
"""

import json
import logging
import time
import uuid
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.invest_agent.core.config import AgentMode, ModeConfig, INSTANT_MODE_CONFIG
from src.invest_agent.core.events import (
    SSEEvent,
    event_mode_selecting,
    event_mode_selected,
    event_mode_escalated,
    event_classified,
    event_turn_start,
    event_tool_calls,
    event_tool_results,
    event_thinking_step,
    event_evaluation,
    event_content,
    event_thinking_summary,
    event_done,
    event_error,
    event_artifact_saved,
    SSEEventType,
)
from src.invest_agent.core.exceptions import InvestAgentError
from src.invest_agent.orchestration.intent_wrapper import IntentWrapper, ClassificationResult
from src.invest_agent.orchestration.mode_resolver import ModeResolver, ModeDecision
from src.invest_agent.execution.validator import ToolCallValidator, ValidatedToolCall
from src.invest_agent.execution.tool_executor import ToolExecutor, ToolExecutionResult
from src.invest_agent.execution.evaluator import DataEvaluator, EvaluationResult
from src.invest_agent.storage.artifact_manager import ArtifactManager
from src.invest_agent.memory.context_manager import ContextManager

logger = logging.getLogger(__name__)


class OrchestratorState(str, Enum):
    """States in the orchestrator's state machine."""
    SETUP = "setup"
    MODE_RESOLUTION = "mode_resolution"
    CLASSIFICATION = "classification"
    EXECUTION_LOOP = "execution_loop"
    EVALUATION = "evaluation"
    SYNTHESIS = "synthesis"
    ESCALATION = "escalation"
    DONE = "done"


class Orchestrator:
    """State machine controller for the invest_agent pipeline.

    Why: The orchestrator manages the full lifecycle of a chat request:
    1. Setup: Load context, initialize managers
    2. Mode Resolution: Decide Instant vs Thinking
    3. Classification: Understand user intent
    4. Execution Loop: Call tools iteratively
    5. Evaluation: Check data sufficiency (Thinking mode only)
    6. Synthesis: Generate final response via LLM streaming
    7. Done: Emit final metrics

    How it works:
    - `run()` is an async generator yielding SSEEvent objects
    - Each state transition emits events for frontend display
    - Tool execution results are managed by ArtifactManager (offloading)
    - Context is managed by ContextManager (token budget)
    - Errors are caught and converted to error events, never raised
    """

    def __init__(
        self,
        intent_wrapper: IntentWrapper,
        mode_resolver: ModeResolver,
        tool_validator: ToolCallValidator,
        tool_executor: ToolExecutor,
        evaluator: DataEvaluator,
        artifact_manager: ArtifactManager,
        system_prompt: str = "",
    ):
        self._intent_wrapper = intent_wrapper
        self._mode_resolver = mode_resolver
        self._tool_validator = tool_validator
        self._tool_executor = tool_executor
        self._evaluator = evaluator
        self._artifact_manager = artifact_manager
        self._system_prompt = system_prompt

    async def run(
        self,
        query: str,
        session_id: str,
        response_mode: str = "auto",
        enable_thinking: bool = True,
        model_name_override: Optional[str] = None,
        provider_type: str = "openai",
        ui_context: Optional[Any] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[int] = None,
    ) -> AsyncGenerator[SSEEvent, None]:
        """Main entry point: run the full orchestration pipeline.

        Yields SSEEvent objects that the router serializes to SSE format.
        """
        start_time = time.time()
        total_tool_calls = 0
        total_turns = 0

        # ---- State 1: SETUP ----
        context_mgr = ContextManager(max_history_messages=10)
        context_mgr.set_system_prompt(self._system_prompt)

        if conversation_history:
            context_mgr.add_conversation_history(conversation_history)
        context_mgr.add_user_message(query)

        # ---- State 2: MODE RESOLUTION ----
        try:
            if response_mode == "auto":
                yield event_mode_selecting(method="auto")

            mode_decision = self._mode_resolver.resolve(
                response_mode=response_mode,
                enable_thinking=enable_thinking,
                query=query,
            )

            # Apply model override if specified
            mode_config = mode_decision.config
            if model_name_override:
                mode_config = mode_config.model_copy(update={"model_name": model_name_override})

            # Update context manager with mode-specific history depth
            context_mgr._max_history = mode_config.max_history_messages

            yield event_mode_selected(
                mode=mode_decision.mode.value,
                reason=mode_decision.reason,
                model=mode_config.model_name,
                confidence=mode_decision.confidence,
            )
        except Exception as e:
            logger.error(f"[Orchestrator] Mode resolution failed: {e}")
            mode_config = INSTANT_MODE_CONFIG
            mode_decision = ModeDecision(
                mode=AgentMode.INSTANT,
                config=mode_config,
                reason="fallback_on_error",
            )
            yield event_mode_selected(
                mode="instant",
                reason="fallback_on_error",
                model=mode_config.model_name,
            )

        # ---- State 3: CLASSIFICATION ----
        yield SSEEvent(event=SSEEventType.CLASSIFYING)

        if mode_config.enable_thinking_display:
            yield event_thinking_step(phase="classification", action="Analyzing query...")

        classification = await self._intent_wrapper.classify(
            query=query,
            conversation_history=conversation_history,
            ui_context=ui_context.model_dump() if ui_context else None,
        )

        # Refine mode with classification data (for auto mode)
        if mode_decision.was_auto_resolved:
            refined = self._mode_resolver.resolve(
                response_mode=response_mode,
                enable_thinking=enable_thinking,
                query=query,
                classification=classification,
            )
            if refined.mode != mode_decision.mode:
                mode_config = refined.config
                mode_decision = refined
                yield event_mode_selected(
                    mode=refined.mode.value,
                    reason=f"refined: {refined.reason}",
                    model=mode_config.model_name,
                    confidence=refined.confidence,
                )

        yield event_classified(
            query_type=classification.query_type,
            symbols=classification.symbols,
            complexity=classification.complexity,
            requires_tools=classification.requires_tools,
        )

        # ---- State 4: EXECUTION LOOP ----
        all_tool_results: List[Dict[str, Any]] = []
        gathered_data: Dict[str, Any] = {}

        if classification.requires_tools and mode_config.use_tools:
            async for event in self._execution_loop(
                query=query,
                session_id=session_id,
                classification=classification,
                mode_config=mode_config,
                context_mgr=context_mgr,
                all_tool_results=all_tool_results,
                gathered_data=gathered_data,
                provider_type=provider_type,
            ):
                if isinstance(event, SSEEvent):
                    if event.event == SSEEventType.TURN_START:
                        total_turns += 1
                    elif event.event == SSEEventType.TOOL_RESULTS:
                        total_tool_calls += len(event.data.get("results", []))
                    yield event

            # ---- ESCALATION CHECK (Instant mode only) ----
            if mode_decision.mode == AgentMode.INSTANT:
                escalation = self._mode_resolver.check_escalation(
                    tool_results=all_tool_results,
                    current_mode=mode_decision.mode,
                )
                if escalation.should_escalate:
                    yield event_mode_escalated(
                        from_mode="instant",
                        to_mode="thinking",
                        reason=escalation.reason,
                    )
                    # Re-run execution loop with Thinking config
                    from src.invest_agent.core.config import THINKING_MODE_CONFIG
                    mode_config = THINKING_MODE_CONFIG
                    if model_name_override:
                        mode_config = mode_config.model_copy(update={"model_name": model_name_override})

                    if mode_config.enable_thinking_display:
                        yield event_thinking_step(
                            phase="escalation",
                            action="Upgrading to deeper analysis...",
                        )

                    async for event in self._execution_loop(
                        query=query,
                        session_id=session_id,
                        classification=classification,
                        mode_config=mode_config,
                        context_mgr=context_mgr,
                        all_tool_results=all_tool_results,
                        gathered_data=gathered_data,
                        provider_type=provider_type,
                    ):
                        if isinstance(event, SSEEvent):
                            if event.event == SSEEventType.TURN_START:
                                total_turns += 1
                            elif event.event == SSEEventType.TOOL_RESULTS:
                                total_tool_calls += len(event.data.get("results", []))
                            yield event

        # ---- State 5: SYNTHESIS ----
        if mode_config.enable_thinking_display:
            yield event_thinking_step(phase="synthesis", action="Generating response...")

        async for event in self._synthesize(
            query=query,
            context_mgr=context_mgr,
            classification=classification,
            mode_config=mode_config,
            gathered_data=gathered_data,
            provider_type=provider_type,
        ):
            yield event

        # ---- State 6: DONE ----
        total_time_ms = int((time.time() - start_time) * 1000)

        if mode_config.enable_thinking_display:
            yield event_thinking_summary(
                total_duration_ms=total_time_ms,
                steps=[
                    {"phase": "classification", "action": f"Classified: {classification.complexity}"},
                    {"phase": "execution", "action": f"{total_tool_calls} tool calls in {total_turns} turns"},
                    {"phase": "synthesis", "action": "Response generated"},
                ],
            )

        yield event_done(
            total_turns=total_turns,
            total_tool_calls=total_tool_calls,
            total_time_ms=total_time_ms,
            mode=mode_decision.mode.value,
        )

    async def _execution_loop(
        self,
        query: str,
        session_id: str,
        classification: ClassificationResult,
        mode_config: ModeConfig,
        context_mgr: ContextManager,
        all_tool_results: List[Dict[str, Any]],
        gathered_data: Dict[str, Any],
        provider_type: str,
    ) -> AsyncGenerator[SSEEvent, None]:
        """Execute tools iteratively based on LLM decisions.

        For Instant mode: 1 turn, no evaluation.
        For Thinking mode: multi-turn with evaluation after each turn.
        """
        for turn in range(1, mode_config.max_turns + 1):
            yield event_turn_start(turn=turn, max_turns=mode_config.max_turns)

            if mode_config.enable_thinking_display:
                yield event_thinking_step(
                    phase="tool_selection",
                    action=f"Turn {turn}: Deciding which tools to call...",
                )

            # Ask LLM what tools to call
            tool_calls_raw = await self._get_llm_tool_calls(
                context_mgr=context_mgr,
                mode_config=mode_config,
                provider_type=provider_type,
            )

            if not tool_calls_raw:
                # LLM decided no more tools needed
                break

            # Validate tool calls
            valid_calls, skipped = self._tool_validator.validate_tool_calls(tool_calls_raw)

            if skipped:
                for s in skipped:
                    logger.warning(f"[Orchestrator] Skipped tool call: {s}")

            if not valid_calls:
                break

            # Emit tool_calls event
            yield event_tool_calls([
                {"name": c.resolved_name, "arguments": c.arguments}
                for c in valid_calls
            ])

            # Add assistant message with tool calls to context
            context_mgr.add_tool_calls_message([
                {
                    "id": c.id or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": c.resolved_name,
                        "arguments": json.dumps(c.arguments),
                    },
                }
                for c in valid_calls
            ])

            # Execute tools
            results = await self._tool_executor.execute_batch(
                calls=valid_calls,
                session_id=session_id,
                parallel=True,
            )

            # Process results
            results_for_event = []
            for result in results:
                result_dict = {
                    "tool_name": result.tool_name,
                    "success": result.success,
                    "data": result.data,
                    "formatted_context": result.formatted_context,
                    "execution_time_ms": result.execution_time_ms,
                }
                all_tool_results.append(result_dict)
                results_for_event.append({
                    "name": result.tool_name,
                    "success": result.success,
                    "execution_time_ms": result.execution_time_ms,
                })

                # Add to context
                call_id = result.call_id or f"call_{uuid.uuid4().hex[:8]}"
                content = result.formatted_context or str(result.error or "No data")
                context_mgr.add_tool_result(
                    tool_name=result.tool_name,
                    tool_call_id=call_id,
                    result_content=content,
                    is_summary=result.was_offloaded,
                )

                if result.success and result.data:
                    gathered_data[result.tool_name] = result.data

                if result.artifact_ref:
                    yield event_artifact_saved(
                        artifact_id=result.artifact_ref.artifact_id,
                        tool_name=result.tool_name,
                        summary=result.artifact_ref.summary,
                    )

            yield event_tool_results(results_for_event)

            # Evaluation (Thinking mode only)
            if mode_config.enable_evaluation and turn < mode_config.max_turns:
                eval_result = self._evaluator.evaluate_heuristic(
                    query=query,
                    tool_results=all_tool_results,
                    iteration=turn,
                )

                yield event_evaluation(
                    iteration=turn,
                    sufficient=eval_result.is_sufficient,
                    missing=eval_result.missing_data,
                )

                if eval_result.is_sufficient:
                    if mode_config.enable_thinking_display:
                        yield event_thinking_step(
                            phase="evaluation",
                            action="Data sufficient, moving to synthesis",
                        )
                    break
                else:
                    if mode_config.enable_thinking_display:
                        yield event_thinking_step(
                            phase="evaluation",
                            action=f"Need more data: {', '.join(eval_result.missing_data[:3])}",
                        )
            elif not mode_config.enable_evaluation:
                # Instant mode: single turn, break after execution
                break

    async def _get_llm_tool_calls(
        self,
        context_mgr: ContextManager,
        mode_config: ModeConfig,
        provider_type: str,
    ) -> List[Dict[str, Any]]:
        """Ask the LLM which tools to call based on current context.

        Returns a list of raw tool call dicts, or empty list if LLM decides
        no tools are needed (generates text instead).
        """
        try:
            from src.agents.tools import get_registry

            registry = get_registry()
            tool_schemas = registry.get_all_tool_schemas()

            messages = context_mgr.get_messages()

            # Use the provider factory to make the LLM call
            from src.helpers.llm_helper import LLMGeneratorProvider

            llm = LLMGeneratorProvider()
            response = await llm.generate_response(
                model_name=mode_config.model_name,
                messages=messages,
                provider_type=provider_type,
                tools=tool_schemas,
                tool_choice="auto",
                max_tokens=mode_config.max_tokens,
                temperature=mode_config.temperature,
            )

            # Extract tool calls from response
            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                # LLM generated text content instead of calling tools
                content = response.get("content", "")
                if content:
                    context_mgr.add_assistant_message(content)
                return []

            # Normalize tool calls format
            normalized = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    normalized.append({
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", tc.get("name", "")),
                        "arguments": tc.get("function", {}).get("arguments", tc.get("arguments", {})),
                    })
                else:
                    # Handle OpenAI ChatCompletionMessageToolCall objects
                    normalized.append({
                        "id": getattr(tc, "id", ""),
                        "name": getattr(tc.function, "name", "") if hasattr(tc, "function") else "",
                        "arguments": getattr(tc.function, "arguments", "{}") if hasattr(tc, "function") else "{}",
                    })

            return normalized

        except Exception as e:
            logger.error(f"[Orchestrator] Failed to get LLM tool calls: {e}")
            return []

    async def _synthesize(
        self,
        query: str,
        context_mgr: ContextManager,
        classification: ClassificationResult,
        mode_config: ModeConfig,
        gathered_data: Dict[str, Any],
        provider_type: str,
    ) -> AsyncGenerator[SSEEvent, None]:
        """Generate the final response via LLM streaming.

        The context manager already contains all tool results (or summaries).
        We add a synthesis instruction and stream the LLM response as
        content events.
        """
        try:
            # Add synthesis instruction
            synthesis_instruction = (
                "Based on all the data gathered above, provide a comprehensive and accurate "
                f"response to the user's question. Language: {classification.response_language}. "
                "Be specific with numbers and data. If some data was unavailable, acknowledge it.\n\n"
                "IMPORTANT — Financial Report Rules:\n"
                "- When tool results include HISTORICAL data tables (multiple periods/years/quarters), "
                "you MUST present the full comparison table in your response so the user can compare across periods.\n"
                "- NEVER summarize multi-period financial data into a single number. Show ALL periods returned by the tools.\n"
                "- Use markdown tables to display year-over-year or quarter-over-quarter comparisons.\n"
                "- Include raw numbers from the tool results — do not round excessively or omit data points."
            )
            if gathered_data:
                context_mgr.add_user_message(synthesis_instruction)
            else:
                # No tools were called, just answer directly
                pass

            messages = context_mgr.get_messages()

            from src.helpers.llm_helper import LLMGeneratorProvider

            llm = LLMGeneratorProvider()
            async for chunk in llm.stream_response(
                model_name=mode_config.model_name,
                messages=messages,
                provider_type=provider_type,
                max_tokens=mode_config.max_tokens,
                temperature=mode_config.temperature,
            ):
                if chunk:
                    yield event_content(chunk)

        except Exception as e:
            logger.error(f"[Orchestrator] Synthesis failed: {e}")
            yield event_error(
                message=f"Failed to generate response: {str(e)[:200]}",
                code="synthesis_error",
            )
