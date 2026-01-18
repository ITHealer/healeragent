"""
Logging Middleware for FastAPI
=============================

Provides middleware for:
- Request ID injection and propagation
- API request/response logging
- Performance timing

Usage:
------
```python
from fastapi import FastAPI
from src.core.logging import LoggingMiddleware

app = FastAPI()
app.add_middleware(LoggingMiddleware)
```

All requests will automatically:
- Get a unique request_id
- Log request details to api/ logs
- Log response with duration
- Propagate request_id to all downstream logs
"""

import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.core.logging.context import set_request_id, clear_request_id
from src.core.logging.config import get_logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for request logging and tracing.

    Features:
    - Generates unique request_id for each request
    - Logs request start with method, path, client info
    - Logs response with status code and duration
    - Injects request_id into context for all downstream logs
    - Adds X-Request-ID header to response
    """

    def __init__(self, app, exclude_paths: list[str] = None):
        """
        Initialize logging middleware.

        Args:
            app: FastAPI application
            exclude_paths: Paths to exclude from logging (e.g., /health, /metrics)
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]
        self.logger = get_logger("api.middleware", category="api")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if path should be excluded
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())[:12]

        # Set request ID in context
        set_request_id(request_id)

        # Record start time
        start_time = time.perf_counter()

        # Extract request info
        client_host = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query = str(request.query_params) if request.query_params else ""

        # Log request start
        self.logger.info(
            f"→ {method} {path}"
            + (f"?{query}" if query else "")
            + f" | client={client_host}"
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log response
            status_emoji = "✓" if response.status_code < 400 else "✗"
            self.logger.info(
                f"{status_emoji} {method} {path} | "
                f"status={response.status_code} | "
                f"duration={duration_ms:.1f}ms"
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log error
            self.logger.error(
                f"✗ {method} {path} | "
                f"error={type(e).__name__}: {str(e)[:100]} | "
                f"duration={duration_ms:.1f}ms",
                exc_info=True,
            )
            raise

        finally:
            # Clear request ID from context
            clear_request_id()


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    Middleware for detailed performance logging.

    Logs to performance/ category with timing breakdowns.
    Use together with LoggingMiddleware for complete logging.
    """

    def __init__(self, app, slow_request_threshold_ms: float = 1000):
        """
        Initialize performance middleware.

        Args:
            app: FastAPI application
            slow_request_threshold_ms: Threshold for warning about slow requests
        """
        super().__init__(app)
        self.slow_threshold = slow_request_threshold_ms
        self.logger = get_logger("performance.api", category="performance")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Log performance metric
        log_data = {
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round(duration_ms, 2),
        }

        if duration_ms > self.slow_threshold:
            self.logger.warning(f"SLOW REQUEST: {log_data}")
        else:
            self.logger.debug(f"Request timing: {log_data}")

        return response


def setup_request_logging(app) -> None:
    """
    Setup all logging middleware for a FastAPI app.

    Args:
        app: FastAPI application instance

    Usage:
        from src.core.logging.middleware import setup_request_logging

        app = FastAPI()
        setup_request_logging(app)
    """
    from starlette.middleware import Middleware

    # Add middlewares in order (last added = first executed)
    app.add_middleware(PerformanceMiddleware)
    app.add_middleware(LoggingMiddleware)
