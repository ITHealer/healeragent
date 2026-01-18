"""
Deep Research Base Agent - OpenManus-Inspired Pattern

Implements the core agent pattern with:
- State machine for lifecycle management
- think()/act() separation for clarity
- Stuck detection to avoid infinite loops
- Memory for tracking progress

This base class is extended by:
- DeepResearchOrchestrator (Lead Agent)
- BaseWorker (Worker Agents)

Design inspired by OpenManus BaseAgent → ReActAgent → ToolCallAgent hierarchy.
"""

import asyncio
import uuid
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple

from src.agents.deep_research.models import (
    AgentState,
    Artifact,
    ArtifactType,
    DeepResearchConfig,
)
from src.utils.logger.custom_logging import LoggerMixin


# ============================================================================
# BASE AGENT (OpenManus Pattern)
# ============================================================================

class BaseDeepResearchAgent(ABC, LoggerMixin):
    """
    Base class for all Deep Research agents.

    Implements the think()/act() pattern from OpenManus:
    1. THINK: Analyze current state, decide next action
    2. ACT: Execute the decided action
    3. OBSERVE: Process results (implicit in act return)
    4. LOOP: Continue until done or stuck

    Attributes:
        agent_id: Unique identifier for this agent instance
        state: Current agent state (from AgentState enum)
        config: Configuration settings
        memory: List of (thought, action, result) tuples
        max_iterations: Maximum think/act cycles
        current_iteration: Current iteration count
        stuck_threshold: Number of no-progress iterations before stuck
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[DeepResearchConfig] = None,
    ):
        super().__init__()
        self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        self.config = config or DeepResearchConfig()
        self.state = AgentState.IDLE

        # Execution tracking
        self.max_iterations = 10
        self.current_iteration = 0
        self.stuck_threshold = 3

        # Memory - tracks (thought, action, result) for each iteration
        self.memory: List[Dict[str, Any]] = []

        # Stuck detection
        self._last_actions: List[str] = []
        self._no_progress_count = 0

        # Timing
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

        self.logger.info(f"[{self.agent_id}] Agent initialized")

    # ========================================================================
    # ABSTRACT METHODS (must be implemented by subclasses)
    # ========================================================================

    @abstractmethod
    async def think(self) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        THINK phase: Analyze situation and decide next action.

        This is where the agent:
        1. Analyzes current state and memory
        2. Decides what to do next
        3. Returns whether to continue and what action to take

        Returns:
            Tuple of:
            - should_continue (bool): True if more work needed, False if done
            - thought (str): Description of what the agent is thinking
            - action_plan (dict or None): Details of the next action, or None if done

        Example implementation:
            async def think(self) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
                if self.all_sections_complete():
                    return False, "All sections complete, ready to synthesize", None

                next_section = self.get_next_section()
                return True, f"Need to research: {next_section.name}", {
                    "action": "spawn_worker",
                    "section": next_section,
                }
        """
        raise NotImplementedError

    @abstractmethod
    async def act(self, action_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        ACT phase: Execute the planned action.

        This is where the agent:
        1. Executes tools, spawns workers, etc.
        2. Returns the result of the action

        Args:
            action_plan: The action plan from think() phase

        Returns:
            Dict containing action results

        Example implementation:
            async def act(self, action_plan: Dict[str, Any]) -> Dict[str, Any]:
                action = action_plan.get("action")

                if action == "spawn_worker":
                    worker = self.create_worker(action_plan["section"])
                    result = await worker.run()
                    return {"success": True, "worker_result": result}

                return {"success": False, "error": "Unknown action"}
        """
        raise NotImplementedError

    # ========================================================================
    # STUCK DETECTION (OpenManus Pattern)
    # ========================================================================

    def is_stuck(self) -> bool:
        """
        Detect if agent is stuck in a loop.

        Checks:
        1. Exceeded max iterations
        2. No progress after threshold iterations
        3. Repeating same action

        Returns:
            True if agent appears stuck, False otherwise
        """
        # Check max iterations
        if self.current_iteration >= self.max_iterations:
            self.logger.warning(
                f"[{self.agent_id}] Max iterations ({self.max_iterations}) reached"
            )
            return True

        # Check no progress
        if self._no_progress_count >= self.stuck_threshold:
            self.logger.warning(
                f"[{self.agent_id}] No progress for {self._no_progress_count} iterations"
            )
            return True

        # Check repeating actions (last 3 actions same)
        if len(self._last_actions) >= 3:
            if len(set(self._last_actions[-3:])) == 1:
                self.logger.warning(
                    f"[{self.agent_id}] Repeating same action: {self._last_actions[-1]}"
                )
                return True

        return False

    def _record_action(self, action: str, made_progress: bool):
        """Record action for stuck detection."""
        self._last_actions.append(action)
        if len(self._last_actions) > 5:
            self._last_actions.pop(0)

        if made_progress:
            self._no_progress_count = 0
        else:
            self._no_progress_count += 1

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main execution loop: THINK → ACT → OBSERVE → LOOP.

        Yields SSE events for each phase of execution.

        Yields:
            Dict events for streaming to frontend:
            - {"type": "thinking", "thought": "..."}
            - {"type": "acting", "action": "..."}
            - {"type": "result", "data": {...}}
            - {"type": "stuck", "reason": "..."}
            - {"type": "completed", "result": {...}}
            - {"type": "error", "error": "..."}
        """
        self._start_time = datetime.utcnow()
        self.state = AgentState.EXECUTING
        self.logger.info(f"[{self.agent_id}] Starting execution")

        try:
            while self.current_iteration < self.max_iterations:
                self.current_iteration += 1

                # Check stuck before each iteration
                if self.is_stuck():
                    self.state = AgentState.STUCK
                    yield {
                        "type": "stuck",
                        "agent_id": self.agent_id,
                        "iteration": self.current_iteration,
                        "reason": "Agent detected stuck condition",
                    }
                    break

                # THINK Phase
                self.logger.debug(
                    f"[{self.agent_id}] Iteration {self.current_iteration}: THINK"
                )
                try:
                    should_continue, thought, action_plan = await self.think()
                except Exception as e:
                    self.logger.error(f"[{self.agent_id}] Think error: {e}")
                    yield {"type": "error", "phase": "think", "error": str(e)}
                    self.state = AgentState.FAILED
                    break

                yield {
                    "type": "thinking",
                    "agent_id": self.agent_id,
                    "iteration": self.current_iteration,
                    "thought": thought,
                    "should_continue": should_continue,
                }

                # Check if done
                if not should_continue:
                    self.state = AgentState.COMPLETED
                    self.logger.info(
                        f"[{self.agent_id}] Completed after {self.current_iteration} iterations"
                    )
                    break

                if action_plan is None:
                    self.logger.warning(
                        f"[{self.agent_id}] Think returned continue=True but no action_plan"
                    )
                    self._record_action("no_action", made_progress=False)
                    continue

                # ACT Phase
                self.logger.debug(
                    f"[{self.agent_id}] Iteration {self.current_iteration}: ACT"
                )
                action_name = action_plan.get("action", "unknown")

                yield {
                    "type": "acting",
                    "agent_id": self.agent_id,
                    "iteration": self.current_iteration,
                    "action": action_name,
                    "action_plan": action_plan,
                }

                try:
                    result = await self.act(action_plan)
                except Exception as e:
                    self.logger.error(f"[{self.agent_id}] Act error: {e}")
                    result = {"success": False, "error": str(e)}
                    yield {"type": "error", "phase": "act", "error": str(e)}

                # OBSERVE Phase (implicit - record to memory)
                made_progress = result.get("success", False)
                self._record_action(action_name, made_progress)

                self.memory.append({
                    "iteration": self.current_iteration,
                    "thought": thought,
                    "action": action_plan,
                    "result": result,
                    "timestamp": datetime.utcnow().isoformat(),
                })

                yield {
                    "type": "result",
                    "agent_id": self.agent_id,
                    "iteration": self.current_iteration,
                    "success": made_progress,
                    "data": result,
                }

            # Final state update
            self._end_time = datetime.utcnow()

            if self.state == AgentState.EXECUTING:
                # Max iterations reached without explicit completion
                self.state = AgentState.COMPLETED
                self.logger.info(
                    f"[{self.agent_id}] Max iterations reached, marking as completed"
                )

            yield {
                "type": "completed" if self.state == AgentState.COMPLETED else "finished",
                "agent_id": self.agent_id,
                "state": self.state.value,
                "iterations": self.current_iteration,
                "duration_ms": self.get_duration_ms(),
            }

        except Exception as e:
            self.state = AgentState.FAILED
            self._end_time = datetime.utcnow()
            self.logger.error(f"[{self.agent_id}] Fatal error: {e}", exc_info=True)
            yield {
                "type": "error",
                "agent_id": self.agent_id,
                "error": str(e),
                "state": self.state.value,
            }

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_duration_ms(self) -> int:
        """Get execution duration in milliseconds."""
        if not self._start_time:
            return 0
        end = self._end_time or datetime.utcnow()
        return int((end - self._start_time).total_seconds() * 1000)

    def get_memory_summary(self) -> str:
        """Get summary of agent memory for context."""
        if not self.memory:
            return "No actions taken yet."

        lines = []
        for entry in self.memory[-5:]:  # Last 5 entries
            action = entry.get("action", {}).get("action", "unknown")
            success = "SUCCESS" if entry.get("result", {}).get("success") else "FAILED"
            lines.append(f"- {action}: {success}")

        return "\n".join(lines)

    def reset(self):
        """Reset agent state for reuse."""
        self.state = AgentState.IDLE
        self.current_iteration = 0
        self.memory = []
        self._last_actions = []
        self._no_progress_count = 0
        self._start_time = None
        self._end_time = None


# ============================================================================
# REACT AGENT (Extended Base with Tool Calling)
# ============================================================================

class ReActAgent(BaseDeepResearchAgent):
    """
    ReAct-style agent that combines reasoning and acting.

    Extends BaseDeepResearchAgent with:
    - Tool registry integration
    - Tool execution helpers
    - Structured output parsing

    This is the base for worker agents that need to call tools.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[DeepResearchConfig] = None,
        tools: Optional[List[str]] = None,
    ):
        super().__init__(agent_id, config)
        self.tools = tools or []
        self._tool_results: Dict[str, Any] = {}

    async def call_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Call a tool and return results.

        Args:
            tool_name: Name of the tool to call
            params: Parameters to pass to the tool

        Returns:
            Tool execution result
        """
        # Import here to avoid circular imports
        from src.agents.tools.tool_loader import get_registry

        try:
            registry = get_registry()

            # Use registry's execute_tool method which handles circuit breaker
            result = await registry.execute_tool(tool_name, params)

            # Store result for later reference
            self._tool_results[tool_name] = result

            return {
                "success": result.is_success(),
                "data": result.data,
                "error": result.error,
                "execution_time_ms": result.execution_time_ms,
                "formatted_context": result.formatted_context,
            }

        except Exception as e:
            self.logger.error(f"[{self.agent_id}] Tool call error: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def call_tools_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Call multiple tools in parallel.

        Args:
            tool_calls: List of {"name": "...", "params": {...}}

        Returns:
            List of results in same order
        """
        tasks = [
            self.call_tool(tc["name"], tc.get("params", {}))
            for tc in tool_calls
        ]
        return await asyncio.gather(*tasks)

    def get_tool_results_summary(self) -> str:
        """Get summary of all tool results for LLM context."""
        if not self._tool_results:
            return "No tools called yet."

        lines = []
        for tool_name, result in self._tool_results.items():
            if result.is_success():
                lines.append(f"- {tool_name}: SUCCESS")
            else:
                lines.append(f"- {tool_name}: FAILED ({result.error})")

        return "\n".join(lines)


# ============================================================================
# TOOL CALL AGENT (Further Extended for LLM-Driven Tool Selection)
# ============================================================================

class ToolCallAgent(ReActAgent):
    """
    Agent that uses LLM to decide which tools to call.

    Extends ReActAgent with:
    - LLM integration for tool selection
    - Automatic tool call parsing
    - Response streaming

    This pattern matches OpenManus ToolCallAgent.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[DeepResearchConfig] = None,
        tools: Optional[List[str]] = None,
        model_name: Optional[str] = None,
    ):
        super().__init__(agent_id, config, tools)
        self.model_name = model_name or config.worker_model if config else "gpt-4o-mini"
        self._llm_provider = None

    @property
    def llm_provider(self):
        """Lazy load LLM provider."""
        if self._llm_provider is None:
            from src.helpers.llm_helper import LLMGeneratorProvider
            self._llm_provider = LLMGeneratorProvider()
        return self._llm_provider

    async def get_llm_response(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Get response from LLM, optionally with tool calling.

        Args:
            messages: Conversation messages
            tools: Tool definitions for function calling
            temperature: Sampling temperature

        Returns:
            LLM response with content and/or tool_calls
        """
        try:
            # Use the LLM provider to get response
            response = await self.llm_provider.generate_with_tools(
                model_name=self.model_name,
                messages=messages,
                tools=tools,
                temperature=temperature,
            )
            return response
        except Exception as e:
            self.logger.error(f"[{self.agent_id}] LLM error: {e}")
            return {"content": None, "tool_calls": [], "error": str(e)}

    def parse_tool_calls(
        self,
        response: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Parse tool calls from LLM response.

        Args:
            response: LLM response dict

        Returns:
            List of {"name": "...", "params": {...}}
        """
        tool_calls = response.get("tool_calls", [])
        parsed = []

        for tc in tool_calls:
            if isinstance(tc, dict):
                # OpenAI format
                func = tc.get("function", {})
                parsed.append({
                    "id": tc.get("id"),
                    "name": func.get("name"),
                    "params": func.get("arguments", {}),
                })
            else:
                # Direct format
                parsed.append({
                    "name": getattr(tc, "name", None),
                    "params": getattr(tc, "arguments", {}),
                })

        return parsed
