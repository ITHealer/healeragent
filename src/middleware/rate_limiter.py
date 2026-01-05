"""
Rate Limiting Middleware for Production

Provides Redis-backed rate limiting with:
- Per-user rate limiting (by user_id)
- Per-IP rate limiting (fallback for unauthenticated)
- Configurable limits per endpoint
- Sliding window algorithm for accurate limiting

Usage in main.py:
    from src.middleware.rate_limiter import RateLimitMiddleware, RateLimitConfig

    app.add_middleware(
        RateLimitMiddleware,
        config=RateLimitConfig(
            default_limit=100,
            default_window=60,
        )
    )
"""

import time
import asyncio
import hashlib
from typing import Optional, Dict, Callable
from dataclasses import dataclass, field

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.utils.logger.custom_logging import LoggerMixin


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""

    # Default limits
    default_limit: int = 100  # requests per window
    default_window: int = 60  # seconds

    # Chat endpoint limits (more restrictive)
    chat_limit: int = 30  # requests per window
    chat_window: int = 60  # seconds

    # Streaming endpoint limits
    stream_limit: int = 20  # requests per window
    stream_window: int = 60  # seconds

    # Burst protection
    burst_limit: int = 10  # max requests in burst window
    burst_window: int = 5  # seconds

    # Skip rate limiting for these paths
    skip_paths: list = field(default_factory=lambda: [
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v2/health",
    ])

    # Endpoint-specific limits: path_pattern -> (limit, window)
    endpoint_limits: Dict[str, tuple] = field(default_factory=lambda: {
        "/api/v2/chat-assistant/chat": (30, 60),
        "/api/v2/assistant/stream": (20, 60),
        "/api/v2/llm_conversation": (50, 60),
    })

    # Enable/disable
    enabled: bool = True

    # Use Redis (True) or in-memory (False)
    use_redis: bool = True


class InMemoryRateLimiter:
    """Simple in-memory rate limiter (fallback when Redis unavailable)"""

    def __init__(self):
        self._requests: Dict[str, list] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window: int
    ) -> tuple[bool, int, int]:
        """
        Check if request is allowed.

        Returns:
            (allowed, remaining, reset_time)
        """
        now = time.time()
        window_start = now - window

        async with self._lock:
            # Get or create request list
            if key not in self._requests:
                self._requests[key] = []

            # Clean old requests
            self._requests[key] = [
                ts for ts in self._requests[key]
                if ts > window_start
            ]

            current_count = len(self._requests[key])
            remaining = max(0, limit - current_count - 1)
            reset_time = int(window_start + window)

            if current_count >= limit:
                return False, 0, reset_time

            # Add current request
            self._requests[key].append(now)
            return True, remaining, reset_time

    async def cleanup(self):
        """Remove expired entries"""
        now = time.time()
        max_window = 3600  # 1 hour max

        async with self._lock:
            keys_to_delete = []
            for key, timestamps in self._requests.items():
                # Remove old timestamps
                self._requests[key] = [
                    ts for ts in timestamps
                    if ts > now - max_window
                ]
                if not self._requests[key]:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del self._requests[key]


class RedisRateLimiter:
    """Redis-backed rate limiter using sliding window"""

    def __init__(self):
        self._redis = None
        self._fallback = InMemoryRateLimiter()

    async def _get_redis(self):
        """Get Redis client lazily"""
        if self._redis is None:
            try:
                from src.helpers.redis_cache import get_redis_client_llm
                self._redis = await get_redis_client_llm()
            except Exception:
                self._redis = None
        return self._redis

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window: int
    ) -> tuple[bool, int, int]:
        """
        Check if request is allowed using Redis sliding window.

        Returns:
            (allowed, remaining, reset_time)
        """
        redis = await self._get_redis()

        if redis is None:
            # Fallback to in-memory
            return await self._fallback.is_allowed(key, limit, window)

        try:
            now = time.time()
            window_start = now - window
            redis_key = f"ratelimit:{key}"

            pipe = redis.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(redis_key, 0, window_start)

            # Count current entries
            pipe.zcard(redis_key)

            # Add current request
            pipe.zadd(redis_key, {str(now): now})

            # Set expiry
            pipe.expire(redis_key, window + 1)

            results = await pipe.execute()
            current_count = results[1]  # zcard result

            remaining = max(0, limit - current_count - 1)
            reset_time = int(now + window)

            if current_count >= limit:
                # Remove the request we just added
                await redis.zrem(redis_key, str(now))
                return False, 0, reset_time

            return True, remaining, reset_time

        except Exception:
            # Fallback to in-memory on Redis error
            return await self._fallback.is_allowed(key, limit, window)


class RateLimitMiddleware(BaseHTTPMiddleware, LoggerMixin):
    """
    Production-ready rate limiting middleware.

    Features:
    - Redis-backed with in-memory fallback
    - Per-user and per-IP limiting
    - Endpoint-specific limits
    - Burst protection
    - Standard rate limit headers
    """

    def __init__(self, app, config: Optional[RateLimitConfig] = None):
        super().__init__(app)
        self.config = config or RateLimitConfig()
        self.limiter = RedisRateLimiter() if self.config.use_redis else InMemoryRateLimiter()

    def _get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier (user_id or IP)"""
        # Try to get user_id from request state (set by auth middleware)
        user_id = getattr(request.state, 'user_id', None)
        if user_id:
            return f"user:{user_id}"

        # Fallback to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return f"ip:{ip}"

    def _get_limit_for_path(self, path: str) -> tuple[int, int]:
        """Get rate limit for specific path"""
        # Check endpoint-specific limits
        for pattern, (limit, window) in self.config.endpoint_limits.items():
            if path.startswith(pattern):
                return limit, window

        # Check if it's a chat/stream endpoint
        if "/chat" in path or "/stream" in path:
            return self.config.chat_limit, self.config.chat_window

        return self.config.default_limit, self.config.default_window

    def _should_skip(self, path: str) -> bool:
        """Check if path should skip rate limiting"""
        return any(path.startswith(skip) for skip in self.config.skip_paths)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting"""

        # Skip if disabled
        if not self.config.enabled:
            return await call_next(request)

        # Skip certain paths
        if self._should_skip(request.url.path):
            return await call_next(request)

        # Get client identifier
        client_id = self._get_client_identifier(request)
        path = request.url.path

        # Create rate limit key
        # Hash the path to avoid overly long keys
        path_hash = hashlib.md5(path.encode()).hexdigest()[:8]
        rate_key = f"{client_id}:{path_hash}"

        # Get limits for this endpoint
        limit, window = self._get_limit_for_path(path)

        # Check rate limit
        allowed, remaining, reset_time = await self.limiter.is_allowed(
            rate_key, limit, window
        )

        # Also check burst limit
        burst_key = f"{client_id}:burst"
        burst_allowed, _, _ = await self.limiter.is_allowed(
            burst_key,
            self.config.burst_limit,
            self.config.burst_window
        )

        if not allowed or not burst_allowed:
            self.logger.warning(
                f"[RATE_LIMIT] Blocked: {client_id} on {path} "
                f"(limit={limit}/{window}s)"
            )

            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please slow down.",
                    "retry_after": reset_time - int(time.time()),
                },
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                    "Retry-After": str(reset_time - int(time.time())),
                }
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)

        return response


# Convenience function to create middleware with default config
def create_rate_limit_middleware(
    default_limit: int = 100,
    chat_limit: int = 30,
    enabled: bool = True
) -> tuple:
    """
    Create rate limit middleware with custom config.

    Usage:
        middleware_class, middleware_kwargs = create_rate_limit_middleware(
            default_limit=100,
            chat_limit=30
        )
        app.add_middleware(middleware_class, **middleware_kwargs)
    """
    config = RateLimitConfig(
        default_limit=default_limit,
        chat_limit=chat_limit,
        enabled=enabled,
    )
    return RateLimitMiddleware, {"config": config}
