"""
Request Context Management
=========================

Provides request_id tracing across all log messages using contextvars.
This allows tracking a single request through the entire processing pipeline.

Usage:
------
```python
from src.core.logging.context import RequestContext, get_request_id

# In middleware or request handler
with RequestContext(request_id="req-abc-123"):
    # All logs within this context will include [req-abc-123]
    logger.info("Processing started")
    await some_async_function()  # Context propagates to async calls
    logger.info("Processing completed")

# Or manually
set_request_id("req-abc-123")
try:
    process_request()
finally:
    clear_request_id()
```
"""

import uuid
from contextvars import ContextVar
from typing import Optional

# Context variable for request ID - thread-safe and async-safe
_request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return _request_id_var.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set request ID in context.

    Args:
        request_id: Custom request ID. If None, generates a new UUID.

    Returns:
        The request ID that was set.
    """
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]  # Short UUID for readability
    _request_id_var.set(request_id)
    return request_id


def clear_request_id() -> None:
    """Clear the request ID from context."""
    _request_id_var.set(None)


class RequestContext:
    """
    Context manager for request ID scoping.

    Automatically sets and clears request ID, with proper cleanup
    even if exceptions occur.

    Usage:
        with RequestContext(request_id="abc-123"):
            # All operations here will have request_id in logs
            await process_request()
    """

    def __init__(self, request_id: Optional[str] = None):
        """
        Initialize request context.

        Args:
            request_id: Custom request ID. If None, generates a new UUID.
        """
        self.request_id = request_id
        self._token = None

    def __enter__(self):
        self._token = _request_id_var.set(
            self.request_id if self.request_id else str(uuid.uuid4())[:8]
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _request_id_var.reset(self._token)
        return False  # Don't suppress exceptions

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


def generate_flow_id(prefix: str = "flow") -> str:
    """
    Generate a short flow ID for tracking sub-operations.

    Args:
        prefix: Prefix for the flow ID (e.g., "agent", "tool")

    Returns:
        Flow ID like "agent-a1b2c3d4"
    """
    return f"{prefix}-{uuid.uuid4().hex[:8]}"
