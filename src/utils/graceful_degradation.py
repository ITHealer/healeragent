"""
Graceful Degradation Utilities

Provides patterns for handling partial failures in parallel operations,
allowing the system to continue with available data when some operations fail.

Features:
- Configurable minimum success threshold
- Fallback response strategies
- Partial success messaging
- Integration with circuit breaker

Usage:
    from src.utils.graceful_degradation import (
        execute_with_degradation,
        DegradationConfig,
        DegradationResult,
    )

    async def fetch_data():
        results = await execute_with_degradation(
            tasks=[fetch_stock_a(), fetch_stock_b(), fetch_news()],
            config=DegradationConfig(min_required=2),
        )

        if results.should_proceed:
            return synthesize_partial_response(results.successful_results)
        else:
            return results.fallback_response
"""

import asyncio
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)
from enum import Enum
import logging

from src.utils.circuit_breaker import get_circuit_breaker


logger = logging.getLogger(__name__)

T = TypeVar("T")


class DegradationStrategy(str, Enum):
    """Strategies for handling partial failures"""

    # Continue if at least min_required succeed
    THRESHOLD = "threshold"

    # Continue if any succeed
    ANY_SUCCESS = "any_success"

    # Continue if critical operations succeed
    CRITICAL_ONLY = "critical_only"

    # Always continue, even with all failures
    BEST_EFFORT = "best_effort"


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation"""

    # Strategy for handling failures
    strategy: DegradationStrategy = DegradationStrategy.THRESHOLD

    # Minimum number of successful results required (for THRESHOLD strategy)
    min_required: int = 1

    # Minimum success ratio (alternative to absolute count)
    # e.g., 0.5 means at least 50% must succeed
    min_success_ratio: Optional[float] = None

    # Indices of critical tasks (for CRITICAL_ONLY strategy)
    critical_indices: List[int] = field(default_factory=list)

    # Custom fallback response when degradation is not possible
    fallback_response: Optional[Any] = None

    # Fallback message template
    fallback_message_template: str = (
        "Some data sources are temporarily unavailable. "
        "Showing partial results based on {success_count}/{total_count} sources."
    )

    # Whether to log failures
    log_failures: bool = True

    # Timeout for individual tasks (seconds)
    task_timeout: Optional[float] = 30.0

    # Whether to record failures to circuit breaker
    use_circuit_breaker: bool = False
    circuit_breaker_name: Optional[str] = None


@dataclass
class TaskResult(Generic[T]):
    """Result of a single task in degradation context"""

    index: int
    success: bool
    result: Optional[T] = None
    error: Optional[Exception] = None
    task_name: Optional[str] = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "success": self.success,
            "result": self.result if self.success else None,
            "error": str(self.error) if self.error else None,
            "task_name": self.task_name,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class DegradationResult(Generic[T]):
    """Result of graceful degradation execution"""

    # Whether to proceed with partial results
    should_proceed: bool

    # All results (both successful and failed)
    all_results: List[TaskResult[T]]

    # Only successful results
    successful_results: List[T]

    # Only failed results
    failed_results: List[TaskResult[T]]

    # Summary statistics
    total_count: int
    success_count: int
    failure_count: int

    # Fallback response (if should_proceed is False)
    fallback_response: Optional[Any] = None

    # Message about degradation status
    degradation_message: Optional[str] = None

    # Whether results are partial
    is_partial: bool = False

    @property
    def success_ratio(self) -> float:
        """Calculate success ratio"""
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count

    def get_partial_response_header(self) -> str:
        """Generate header message for partial responses"""
        if not self.is_partial:
            return ""

        return (
            f"⚠️ Showing partial results ({self.success_count}/{self.total_count} sources available). "
            f"Some data may be incomplete."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_proceed": self.should_proceed,
            "total_count": self.total_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_ratio": self.success_ratio,
            "is_partial": self.is_partial,
            "degradation_message": self.degradation_message,
            "successful_results": self.successful_results,
            "failed_tasks": [r.to_dict() for r in self.failed_results],
        }


async def execute_with_degradation(
    tasks: List[Union[Coroutine[Any, Any, T], Callable[[], Coroutine[Any, Any, T]]]],
    config: Optional[DegradationConfig] = None,
    task_names: Optional[List[str]] = None,
) -> DegradationResult[T]:
    """
    Execute multiple async tasks with graceful degradation.

    If some tasks fail, continue with partial results based on configuration.

    Args:
        tasks: List of coroutines or async callables to execute
        config: Degradation configuration
        task_names: Optional names for tasks (for logging)

    Returns:
        DegradationResult with successful results and degradation status

    Example:
        results = await execute_with_degradation(
            tasks=[fetch_a(), fetch_b(), fetch_c()],
            config=DegradationConfig(min_required=2),
        )

        if results.should_proceed:
            process(results.successful_results)
        else:
            return results.fallback_response
    """
    import time

    config = config or DegradationConfig()
    task_names = task_names or [f"task_{i}" for i in range(len(tasks))]

    # Ensure task_names matches tasks length
    while len(task_names) < len(tasks):
        task_names.append(f"task_{len(task_names)}")

    # Execute tasks with timeout and exception handling
    async def execute_single(
        index: int,
        task: Union[Coroutine, Callable],
        name: str,
    ) -> TaskResult[T]:
        start_time = time.time()

        try:
            # Handle callable vs coroutine
            coro = task() if callable(task) and not asyncio.iscoroutine(task) else task

            # Apply timeout if configured
            if config.task_timeout:
                result = await asyncio.wait_for(coro, timeout=config.task_timeout)
            else:
                result = await coro

            elapsed_ms = (time.time() - start_time) * 1000

            # Record success to circuit breaker if configured
            if config.use_circuit_breaker and config.circuit_breaker_name:
                get_circuit_breaker().record_success(config.circuit_breaker_name)

            return TaskResult(
                index=index,
                success=True,
                result=result,
                task_name=name,
                elapsed_ms=elapsed_ms,
            )

        except asyncio.TimeoutError as e:
            elapsed_ms = (time.time() - start_time) * 1000

            if config.log_failures:
                logger.warning(f"[DEGRADATION] Task '{name}' timed out after {config.task_timeout}s")

            if config.use_circuit_breaker and config.circuit_breaker_name:
                get_circuit_breaker().record_failure(config.circuit_breaker_name, e)

            return TaskResult(
                index=index,
                success=False,
                error=e,
                task_name=name,
                elapsed_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000

            if config.log_failures:
                logger.warning(f"[DEGRADATION] Task '{name}' failed: {e}")

            if config.use_circuit_breaker and config.circuit_breaker_name:
                get_circuit_breaker().record_failure(config.circuit_breaker_name, e)

            return TaskResult(
                index=index,
                success=False,
                error=e,
                task_name=name,
                elapsed_ms=elapsed_ms,
            )

    # Execute all tasks in parallel
    task_results: List[TaskResult[T]] = await asyncio.gather(
        *[
            execute_single(i, task, task_names[i])
            for i, task in enumerate(tasks)
        ],
        return_exceptions=False,  # We handle exceptions in execute_single
    )

    # Separate successful and failed results
    successful_results = []
    failed_results = []

    for tr in task_results:
        if tr.success:
            successful_results.append(tr.result)
        else:
            failed_results.append(tr)

    total_count = len(task_results)
    success_count = len(successful_results)
    failure_count = len(failed_results)

    # Determine if we should proceed based on strategy
    should_proceed = _evaluate_degradation_strategy(
        config=config,
        task_results=task_results,
        success_count=success_count,
        total_count=total_count,
    )

    # Build degradation message
    is_partial = failure_count > 0 and should_proceed
    degradation_message = None

    if is_partial:
        degradation_message = config.fallback_message_template.format(
            success_count=success_count,
            total_count=total_count,
            failure_count=failure_count,
        )
    elif not should_proceed:
        degradation_message = (
            f"Unable to proceed: only {success_count}/{total_count} sources available, "
            f"minimum required: {config.min_required}"
        )

    # Log summary
    logger.info(
        f"[DEGRADATION] Completed: {success_count}/{total_count} succeeded, "
        f"proceed={should_proceed}, partial={is_partial}"
    )

    return DegradationResult(
        should_proceed=should_proceed,
        all_results=task_results,
        successful_results=successful_results,
        failed_results=failed_results,
        total_count=total_count,
        success_count=success_count,
        failure_count=failure_count,
        fallback_response=config.fallback_response if not should_proceed else None,
        degradation_message=degradation_message,
        is_partial=is_partial,
    )


def _evaluate_degradation_strategy(
    config: DegradationConfig,
    task_results: List[TaskResult],
    success_count: int,
    total_count: int,
) -> bool:
    """Evaluate whether to proceed based on degradation strategy"""

    if config.strategy == DegradationStrategy.BEST_EFFORT:
        return True

    if config.strategy == DegradationStrategy.ANY_SUCCESS:
        return success_count > 0

    if config.strategy == DegradationStrategy.CRITICAL_ONLY:
        # All critical tasks must succeed
        for idx in config.critical_indices:
            if idx < len(task_results) and not task_results[idx].success:
                return False
        return True

    # Default: THRESHOLD strategy
    # Check minimum count
    if success_count < config.min_required:
        return False

    # Check minimum ratio if configured
    if config.min_success_ratio is not None:
        actual_ratio = success_count / total_count if total_count > 0 else 0
        if actual_ratio < config.min_success_ratio:
            return False

    return True


def synthesize_partial_response(
    successful_results: List[Any],
    failed_task_names: List[str],
    template: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a response from partial results with appropriate messaging.

    Args:
        successful_results: Results from successful operations
        failed_task_names: Names of failed operations
        template: Optional message template

    Returns:
        Dictionary with partial results and metadata
    """
    default_template = (
        "Some data sources were unavailable: {failed_sources}. "
        "Results are based on available data."
    )

    message = (template or default_template).format(
        failed_sources=", ".join(failed_task_names) if failed_task_names else "unknown",
        success_count=len(successful_results),
        failure_count=len(failed_task_names),
    )

    return {
        "results": successful_results,
        "is_partial": True,
        "partial_message": message,
        "unavailable_sources": failed_task_names,
        "available_count": len(successful_results),
    }


def create_fallback_response(
    error_type: str = "service_unavailable",
    message: Optional[str] = None,
    retry_after: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create a fallback response when degradation is not possible.

    Args:
        error_type: Type of fallback (service_unavailable, rate_limited, etc.)
        message: Custom error message
        retry_after: Seconds until retry is suggested

    Returns:
        Fallback response dictionary
    """
    default_messages = {
        "service_unavailable": "Our data services are temporarily unavailable. Please try again later.",
        "rate_limited": "Too many requests. Please wait a moment before trying again.",
        "timeout": "The request took too long to process. Please try a simpler query.",
        "partial_failure": "Some data sources could not be reached. Please try again.",
    }

    return {
        "error": True,
        "error_type": error_type,
        "message": message or default_messages.get(error_type, "An error occurred."),
        "retry_after": retry_after,
        "fallback": True,
    }


# ============================================================================
# DECORATOR FOR EASY INTEGRATION
# ============================================================================

def with_graceful_degradation(
    min_required: int = 1,
    strategy: DegradationStrategy = DegradationStrategy.THRESHOLD,
    fallback_response: Optional[Any] = None,
):
    """
    Decorator to wrap async functions that return lists with graceful degradation.

    Usage:
        @with_graceful_degradation(min_required=2)
        async def fetch_all_prices(symbols: List[str]) -> List[PriceData]:
            return await asyncio.gather(*[fetch_price(s) for s in symbols])
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)

                # If result is a DegradationResult, return as-is
                if isinstance(result, DegradationResult):
                    return result

                # Otherwise, wrap in a successful DegradationResult
                if isinstance(result, list):
                    return DegradationResult(
                        should_proceed=True,
                        all_results=[
                            TaskResult(index=i, success=True, result=r)
                            for i, r in enumerate(result)
                        ],
                        successful_results=result,
                        failed_results=[],
                        total_count=len(result),
                        success_count=len(result),
                        failure_count=0,
                        is_partial=False,
                    )

                return result

            except Exception as e:
                logger.error(f"[DEGRADATION] Function {func.__name__} failed: {e}")

                return DegradationResult(
                    should_proceed=False,
                    all_results=[],
                    successful_results=[],
                    failed_results=[TaskResult(index=0, success=False, error=e)],
                    total_count=1,
                    success_count=0,
                    failure_count=1,
                    fallback_response=fallback_response,
                    degradation_message=str(e),
                )

        return wrapper
    return decorator
