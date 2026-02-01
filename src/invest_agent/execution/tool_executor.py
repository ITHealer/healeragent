"""
Robust Tool Executor with retry, circuit breaker integration, and safe catch.

Why: The agent loop MUST NOT crash because a single tool call fails. This executor
wraps every tool invocation in defensive error handling with classified retry logic:
- Transient errors (timeout, rate limit) -> auto retry up to 2 times
- Data errors (symbol not found) -> no retry, return formatted error
- Auth/config errors -> no retry, skip tool

How: Integrates with the existing ToolRegistry for actual execution, the
ToolCallValidator for pre-validation, and the ArtifactManager for context
offloading of large results.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.agents.tools import ToolRegistry, ToolOutput, get_registry
from src.invest_agent.execution.validator import ToolCallValidator, ValidatedToolCall
from src.invest_agent.storage.artifact_manager import ArtifactManager, ArtifactRef
from src.invest_agent.core.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)

# Errors that indicate transient issues worth retrying
TRANSIENT_ERROR_KEYWORDS = (
    "timeout", "timed out", "rate limit", "rate_limit",
    "429", "503", "502", "504",
    "connection", "network", "temporarily unavailable",
    "circuit breaker",
)

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = [1.0, 2.0]  # Exponential backoff per retry


class ToolExecutionResult(BaseModel):
    """Result of a single tool execution, enriched with metadata."""
    tool_name: str
    call_id: str = ""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    formatted_context: Optional[str] = None
    execution_time_ms: int = 0
    retries_used: int = 0
    was_offloaded: bool = False
    artifact_ref: Optional[ArtifactRef] = None

    class Config:
        arbitrary_types_allowed = True


class ToolExecutor:
    """Executes validated tool calls with retry and artifact offloading.

    Why: The orchestrator needs a single call to execute tools and get back
    clean results ready for LLM context injection. This class handles all
    the messy details: retries, error classification, offloading, timing.

    How it fits in the pipeline:
    1. Orchestrator receives raw tool calls from LLM
    2. ToolCallValidator validates them
    3. ToolExecutor.execute_batch() runs them (parallel or sequential)
    4. Results include formatted_context ready for LLM messages
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        artifact_manager: Optional[ArtifactManager] = None,
        max_retries: int = MAX_RETRIES,
    ):
        self._registry = registry or get_registry()
        self._artifact_manager = artifact_manager
        self._max_retries = max_retries
        # Simple TTL cache: key -> (result, expire_time)
        self._result_cache: Dict[str, tuple[ToolOutput, float]] = {}
        self._cache_ttl_seconds = 300  # 5 minutes

    async def execute_single(
        self,
        call: ValidatedToolCall,
        session_id: Optional[str] = None,
    ) -> ToolExecutionResult:
        """Execute a single validated tool call with retry and offloading.

        This method NEVER raises exceptions. Errors are captured and returned
        as formatted strings in the result, keeping the agent loop alive.
        """
        start_time = time.time()
        retries = 0
        last_error = ""

        # Check cache first
        cache_key = f"{call.resolved_name}:{_stable_hash(call.arguments)}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.info(f"[ToolExecutor] Cache hit for {call.resolved_name}")
            return self._build_result(
                call=call,
                output=cached,
                start_time=start_time,
                retries=0,
                session_id=session_id,
            )

        # Execute with retry
        for attempt in range(1 + self._max_retries):
            try:
                output: ToolOutput = await self._registry.execute_tool(
                    tool_name=call.resolved_name,
                    params=call.arguments,
                )

                if output.status == "success" or output.status == "partial":
                    # Cache successful results
                    self._set_cached(cache_key, output)
                    return self._build_result(
                        call=call,
                        output=output,
                        start_time=start_time,
                        retries=retries,
                        session_id=session_id,
                    )

                # Tool returned an error status
                last_error = output.error or "Unknown tool error"

                if self._is_transient_error(last_error) and attempt < self._max_retries:
                    retries += 1
                    wait = RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
                    logger.warning(
                        f"[ToolExecutor] Transient error in {call.resolved_name}: {last_error}. "
                        f"Retry {retries}/{self._max_retries} after {wait}s"
                    )
                    await asyncio.sleep(wait)
                    continue

                # Non-transient error or max retries reached
                break

            except Exception as exc:
                last_error = str(exc)
                if self._is_transient_error(last_error) and attempt < self._max_retries:
                    retries += 1
                    wait = RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
                    logger.warning(
                        f"[ToolExecutor] Exception in {call.resolved_name}: {exc}. "
                        f"Retry {retries}/{self._max_retries} after {wait}s"
                    )
                    await asyncio.sleep(wait)
                    continue
                break

        # All retries exhausted or non-transient error
        elapsed_ms = int((time.time() - start_time) * 1000)
        formatted_error = (
            f"Tool '{call.resolved_name}' failed after {retries} retries: {last_error}"
        )
        logger.error(f"[ToolExecutor] {formatted_error}")

        return ToolExecutionResult(
            tool_name=call.resolved_name,
            call_id=call.id,
            success=False,
            error=last_error,
            formatted_context=formatted_error,
            execution_time_ms=elapsed_ms,
            retries_used=retries,
        )

    async def execute_batch(
        self,
        calls: List[ValidatedToolCall],
        session_id: Optional[str] = None,
        parallel: bool = True,
    ) -> List[ToolExecutionResult]:
        """Execute multiple tool calls, optionally in parallel.

        Why parallel: Independent tools (e.g., getStockPrice + getTechnicalIndicators
        for the same symbol) can run concurrently, reducing total latency.
        """
        if not calls:
            return []

        if parallel and len(calls) > 1:
            tasks = [
                self.execute_single(call, session_id=session_id)
                for call in calls
            ]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for call in calls:
                result = await self.execute_single(call, session_id=session_id)
                results.append(result)
            return results

    def _build_result(
        self,
        call: ValidatedToolCall,
        output: ToolOutput,
        start_time: float,
        retries: int,
        session_id: Optional[str] = None,
    ) -> ToolExecutionResult:
        """Convert a ToolOutput into a ToolExecutionResult, handling offloading."""
        elapsed_ms = int((time.time() - start_time) * 1000)

        # Determine what goes into context
        context_data = output.formatted_context or output.data or ""
        was_offloaded = False
        artifact_ref = None

        # Try artifact offloading if manager is available and session exists
        if self._artifact_manager and session_id and output.data:
            was_offloaded, context_data, artifact_ref = self._artifact_manager.maybe_offload(
                session_id=session_id,
                tool_name=call.resolved_name,
                result_data=output.data,
                custom_summary=output.formatted_context,
            )

        return ToolExecutionResult(
            tool_name=call.resolved_name,
            call_id=call.id,
            success=(output.status in ("success", "partial")),
            data=output.data,
            error=output.error,
            formatted_context=str(context_data) if context_data else None,
            execution_time_ms=elapsed_ms,
            retries_used=retries,
            was_offloaded=was_offloaded,
            artifact_ref=artifact_ref,
        )

    @staticmethod
    def _is_transient_error(error_msg: str) -> bool:
        """Classify an error as transient (retryable) or permanent."""
        lower = error_msg.lower()
        return any(kw in lower for kw in TRANSIENT_ERROR_KEYWORDS)

    def _get_cached(self, key: str) -> Optional[ToolOutput]:
        """Get a cached tool output if not expired."""
        if key in self._result_cache:
            output, expire_time = self._result_cache[key]
            if time.time() < expire_time:
                return output
            del self._result_cache[key]
        return None

    def _set_cached(self, key: str, output: ToolOutput) -> None:
        """Cache a tool output with TTL."""
        self._result_cache[key] = (output, time.time() + self._cache_ttl_seconds)

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._result_cache.clear()


def _stable_hash(d: Dict[str, Any]) -> str:
    """Create a stable string hash of a dict for cache keying."""
    import hashlib
    import json
    serialized = json.dumps(d, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()[:12]
