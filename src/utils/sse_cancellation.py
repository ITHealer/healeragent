"""
SSE Cancellation Utilities

Provides helpers for handling SSE client disconnection in FastAPI endpoints.
Enables graceful cleanup when clients disconnect during streaming.

Features:
- Client disconnection detection via Request.is_disconnected()
- Resource cleanup on cancellation
- Cancelled event emission
- Background cleanup tasks

Usage:
    from src.utils.sse_cancellation import (
        SSECancellationHandler,
        with_cancellation,
    )

    @router.post("/stream")
    async def stream_chat(request: Request, data: ChatRequest):
        async def _generate():
            yield "data: start\n\n"
            # ... streaming logic
            yield "data: done\n\n"

        return StreamingResponse(
            with_cancellation(request, _generate(), cleanup_fn=cleanup_resources),
            media_type="text/event-stream"
        )
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    TypeVar,
)
from datetime import datetime

from fastapi import Request, BackgroundTasks


logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CancellationState:
    """Tracks cancellation state for an SSE stream"""

    is_cancelled: bool = False
    cancelled_at: Optional[datetime] = None
    reason: str = "client_disconnected"
    cleanup_completed: bool = False
    cleanup_errors: List[str] = field(default_factory=list)

    def mark_cancelled(self, reason: str = "client_disconnected") -> None:
        """Mark the stream as cancelled"""
        self.is_cancelled = True
        self.cancelled_at = datetime.utcnow()
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_cancelled": self.is_cancelled,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "reason": self.reason,
            "cleanup_completed": self.cleanup_completed,
            "cleanup_errors": self.cleanup_errors,
        }


class SSECancellationHandler:
    """
    Handles SSE stream cancellation with resource cleanup.

    Usage:
        handler = SSECancellationHandler(request)

        async for event in generator:
            if await handler.check_cancelled():
                yield handler.get_cancelled_event()
                break
            yield event

        await handler.cleanup(cleanup_fn)
    """

    def __init__(
        self,
        request: Request,
        check_interval: float = 0.1,
        emit_cancelled_event: bool = True,
    ):
        """
        Initialize cancellation handler.

        Args:
            request: FastAPI request object
            check_interval: How often to check for disconnection (seconds)
            emit_cancelled_event: Whether to emit a cancelled SSE event
        """
        self.request = request
        self.check_interval = check_interval
        self.emit_cancelled_event = emit_cancelled_event
        self.state = CancellationState()
        self._cleanup_functions: List[Callable[[], Coroutine]] = []
        self._last_check_time = 0.0

    async def check_cancelled(self, force: bool = False) -> bool:
        """
        Check if client has disconnected.

        Args:
            force: Force check even if interval hasn't passed

        Returns:
            True if cancelled/disconnected
        """
        if self.state.is_cancelled:
            return True

        import time
        current_time = time.time()

        # Rate limit disconnection checks
        if not force and (current_time - self._last_check_time) < self.check_interval:
            return False

        self._last_check_time = current_time

        # Check if client is disconnected
        try:
            if await self.request.is_disconnected():
                self.state.mark_cancelled("client_disconnected")
                logger.info("[SSE_CANCEL] Client disconnected")
                return True
        except Exception as e:
            # If we can't check, assume still connected
            logger.debug(f"[SSE_CANCEL] Disconnect check error: {e}")

        return False

    def mark_cancelled(self, reason: str = "manual") -> None:
        """Manually mark the stream as cancelled"""
        self.state.mark_cancelled(reason)

    def get_cancelled_event(self) -> str:
        """Get SSE event to send when cancelled"""
        if not self.emit_cancelled_event:
            return ""

        import json
        event_data = {
            "type": "cancelled",
            "timestamp": datetime.utcnow().isoformat(),
            "reason": self.state.reason,
        }
        return f"event: cancelled\ndata: {json.dumps(event_data)}\n\n"

    def register_cleanup(
        self,
        cleanup_fn: Callable[[], Coroutine],
    ) -> None:
        """Register a cleanup function to run on cancellation"""
        self._cleanup_functions.append(cleanup_fn)

    async def cleanup(
        self,
        additional_cleanup: Optional[Callable[[], Coroutine]] = None,
    ) -> None:
        """
        Run all registered cleanup functions.

        Args:
            additional_cleanup: Optional additional cleanup function
        """
        if additional_cleanup:
            self._cleanup_functions.append(additional_cleanup)

        for cleanup_fn in self._cleanup_functions:
            try:
                await cleanup_fn()
            except Exception as e:
                error_msg = f"Cleanup error: {e}"
                self.state.cleanup_errors.append(error_msg)
                logger.warning(f"[SSE_CANCEL] {error_msg}")

        self.state.cleanup_completed = True
        logger.debug(
            f"[SSE_CANCEL] Cleanup completed with {len(self.state.cleanup_errors)} errors"
        )


async def with_cancellation(
    request: Request,
    generator: AsyncGenerator[str, None],
    cleanup_fn: Optional[Callable[[], Coroutine]] = None,
    check_interval: float = 0.5,
    emit_cancelled_event: bool = True,
) -> AsyncGenerator[str, None]:
    """
    Wrap an SSE generator with cancellation detection.

    Checks for client disconnection periodically and runs cleanup
    when the client disconnects or the stream ends.

    Args:
        request: FastAPI request object
        generator: The SSE generator to wrap
        cleanup_fn: Optional cleanup function to run on completion/cancellation
        check_interval: How often to check for disconnection (seconds)
        emit_cancelled_event: Whether to emit a cancelled SSE event

    Yields:
        SSE events from the generator

    Example:
        async def _generate():
            for i in range(100):
                yield f"data: {i}\n\n"
                await asyncio.sleep(0.1)

        return StreamingResponse(
            with_cancellation(request, _generate(), cleanup_fn=my_cleanup),
            media_type="text/event-stream"
        )
    """
    handler = SSECancellationHandler(
        request=request,
        check_interval=check_interval,
        emit_cancelled_event=emit_cancelled_event,
    )

    if cleanup_fn:
        handler.register_cleanup(cleanup_fn)

    try:
        async for event in generator:
            # Check for cancellation
            if await handler.check_cancelled():
                logger.info("[SSE_CANCEL] Stream cancelled - stopping iteration")
                if emit_cancelled_event:
                    yield handler.get_cancelled_event()
                break

            yield event

            # Small yield to allow cancellation checks between events
            await asyncio.sleep(0)

    except asyncio.CancelledError:
        logger.info("[SSE_CANCEL] Generator cancelled")
        handler.mark_cancelled("asyncio_cancelled")
        if emit_cancelled_event:
            yield handler.get_cancelled_event()

    except Exception as e:
        logger.error(f"[SSE_CANCEL] Generator error: {e}")
        raise

    finally:
        # Always run cleanup
        await handler.cleanup()


def create_cancellation_background_task(
    background_tasks: BackgroundTasks,
    cleanup_fn: Callable[[], Coroutine],
    delay_seconds: float = 0,
) -> None:
    """
    Add cleanup function as a background task.

    Use this when you can't await cleanup in the generator.

    Args:
        background_tasks: FastAPI BackgroundTasks
        cleanup_fn: Async cleanup function
        delay_seconds: Optional delay before cleanup
    """
    async def _delayed_cleanup():
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        await cleanup_fn()

    background_tasks.add_task(_delayed_cleanup)


# ============================================================================
# INTEGRATION WITH EXISTING CANCELLATION TOKEN
# ============================================================================

class CancellationTokenWithRequest:
    """
    Enhanced CancellationToken that also checks Request.is_disconnected().

    Combines explicit cancellation with automatic disconnection detection.
    """

    def __init__(self, request: Optional[Request] = None):
        self._cancelled = False
        self._cancel_event = asyncio.Event()
        self._request = request

    def cancel(self):
        """Mark request as cancelled."""
        self._cancelled = True
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        """Check if request is cancelled."""
        return self._cancelled

    async def check_cancelled(self) -> bool:
        """
        Check if request is cancelled or client disconnected.

        This is the async method that should be called in long-running operations.
        """
        if self._cancelled:
            return True

        if self._request:
            try:
                if await self._request.is_disconnected():
                    self.cancel()
                    return True
            except Exception:
                pass

        return False

    async def wait_for_cancel(self, timeout: float = None) -> bool:
        """Wait for cancellation with optional timeout. Returns True if cancelled."""
        try:
            await asyncio.wait_for(self._cancel_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


def create_cancellation_token(request: Optional[Request] = None) -> CancellationTokenWithRequest:
    """
    Create a cancellation token, optionally linked to a request.

    Args:
        request: Optional FastAPI request for automatic disconnect detection

    Returns:
        CancellationToken that can detect both manual cancellation and client disconnect
    """
    return CancellationTokenWithRequest(request)
