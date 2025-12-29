"""
Global HTTP Client Pool - Singleton pattern to prevent connection leaks

This module provides a centralized HTTP client pool that should be used
across the entire application instead of creating new clients per request.
"""
import asyncio
import aiohttp
import httpx
from typing import Optional
from contextlib import asynccontextmanager

from src.utils.config import settings
from src.utils.logger.custom_logging import LoggerMixin


# =============================================================================
# CONFIGURATION
# =============================================================================

# Connection pool limits
MAX_CONNECTIONS_AIOHTTP = 100
MAX_CONNECTIONS_PER_HOST = 20

# Timeouts
DEFAULT_TIMEOUT = 30  # seconds
CONNECT_TIMEOUT = 10  # seconds

# Semaphore for rate limiting external API calls
MAX_CONCURRENT_EXTERNAL_CALLS = 20


# =============================================================================
# SINGLETON HTTP CLIENT MANAGER
# =============================================================================

class HTTPClientManager(LoggerMixin):
    """
    Singleton HTTP client manager for centralized connection pooling.

    Usage:
        client_manager = HTTPClientManager.get_instance()

        # For aiohttp
        async with client_manager.get_aiohttp_session() as session:
            async with session.get(url) as response:
                data = await response.json()

        # For httpx
        async with client_manager.get_httpx_client() as client:
            response = await client.get(url)
    """

    _instance: Optional['HTTPClientManager'] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self):
        super().__init__()

        # aiohttp session (lazy init)
        self._aiohttp_session: Optional[aiohttp.ClientSession] = None
        self._aiohttp_connector: Optional[aiohttp.TCPConnector] = None

        # httpx client (lazy init)
        self._httpx_client: Optional[httpx.AsyncClient] = None

        # Semaphore for rate limiting
        self._external_api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXTERNAL_CALLS)

        # Initialization lock
        self._init_lock = asyncio.Lock()

        self.logger.info("[HTTP_POOL] HTTPClientManager initialized")

    @classmethod
    async def get_instance(cls) -> 'HTTPClientManager':
        """Get or create singleton instance"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def _ensure_aiohttp_session(self) -> aiohttp.ClientSession:
        """Lazily initialize aiohttp session with connection pooling"""
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            async with self._init_lock:
                if self._aiohttp_session is None or self._aiohttp_session.closed:
                    # Create connector with connection limits
                    self._aiohttp_connector = aiohttp.TCPConnector(
                        limit=MAX_CONNECTIONS_AIOHTTP,
                        limit_per_host=MAX_CONNECTIONS_PER_HOST,
                        ttl_dns_cache=300,  # Cache DNS for 5 minutes
                        enable_cleanup_closed=True,
                    )

                    # Create timeout configuration
                    timeout = aiohttp.ClientTimeout(
                        total=DEFAULT_TIMEOUT,
                        connect=CONNECT_TIMEOUT,
                        sock_read=DEFAULT_TIMEOUT,
                    )

                    # Create session
                    self._aiohttp_session = aiohttp.ClientSession(
                        connector=self._aiohttp_connector,
                        timeout=timeout,
                        headers={
                            'User-Agent': 'HealerAgent/1.0'
                        }
                    )

                    self.logger.info("[HTTP_POOL] Created aiohttp session with pooling")

        return self._aiohttp_session

    async def _ensure_httpx_client(self) -> httpx.AsyncClient:
        """Lazily initialize httpx client with connection pooling"""
        if self._httpx_client is None or self._httpx_client.is_closed:
            async with self._init_lock:
                if self._httpx_client is None or self._httpx_client.is_closed:
                    # Create client with limits
                    limits = httpx.Limits(
                        max_connections=MAX_CONNECTIONS_AIOHTTP,
                        max_keepalive_connections=MAX_CONNECTIONS_PER_HOST,
                        keepalive_expiry=30.0,
                    )

                    timeout = httpx.Timeout(
                        timeout=DEFAULT_TIMEOUT,
                        connect=CONNECT_TIMEOUT,
                    )

                    self._httpx_client = httpx.AsyncClient(
                        limits=limits,
                        timeout=timeout,
                        http2=True,  # Enable HTTP/2 for better performance
                    )

                    self.logger.info("[HTTP_POOL] Created httpx client with pooling")

        return self._httpx_client

    @asynccontextmanager
    async def get_aiohttp_session(self):
        """Get the shared aiohttp session"""
        session = await self._ensure_aiohttp_session()
        yield session

    @asynccontextmanager
    async def get_httpx_client(self):
        """Get the shared httpx client"""
        client = await self._ensure_httpx_client()
        yield client

    @asynccontextmanager
    async def rate_limited_request(self):
        """
        Context manager for rate-limited external API calls.

        Usage:
            async with client_manager.rate_limited_request():
                # Make external API call
                response = await session.get(url)
        """
        async with self._external_api_semaphore:
            yield

    async def close(self):
        """Close all connections gracefully"""
        try:
            if self._aiohttp_session and not self._aiohttp_session.closed:
                await self._aiohttp_session.close()
                self.logger.info("[HTTP_POOL] Closed aiohttp session")

            if self._httpx_client and not self._httpx_client.is_closed:
                await self._httpx_client.aclose()
                self.logger.info("[HTTP_POOL] Closed httpx client")

        except Exception as e:
            self.logger.error(f"[HTTP_POOL] Error closing clients: {e}")

    async def health_check(self) -> dict:
        """Check health of HTTP clients"""
        status = {
            "aiohttp_active": False,
            "httpx_active": False,
            "semaphore_available": self._external_api_semaphore._value,
        }

        try:
            if self._aiohttp_session and not self._aiohttp_session.closed:
                status["aiohttp_active"] = True

            if self._httpx_client and not self._httpx_client.is_closed:
                status["httpx_active"] = True

        except Exception as e:
            self.logger.error(f"[HTTP_POOL] Health check error: {e}")

        return status


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def get_http_client_manager() -> HTTPClientManager:
    """Get the global HTTP client manager instance"""
    return await HTTPClientManager.get_instance()


async def fetch_with_retry(
    url: str,
    method: str = "GET",
    max_retries: int = 3,
    retry_delay: float = 1.0,
    **kwargs
) -> dict:
    """
    Fetch URL with automatic retry and rate limiting.

    Args:
        url: URL to fetch
        method: HTTP method (GET, POST, etc.)
        max_retries: Maximum retry attempts
        retry_delay: Base delay between retries (exponential backoff)
        **kwargs: Additional arguments for the request

    Returns:
        Response data as dict
    """
    client_manager = await get_http_client_manager()

    last_error = None
    for attempt in range(max_retries):
        try:
            async with client_manager.rate_limited_request():
                async with client_manager.get_aiohttp_session() as session:
                    async with session.request(method, url, **kwargs) as response:
                        response.raise_for_status()
                        return await response.json()

        except aiohttp.ClientResponseError as e:
            if e.status in [429, 503]:  # Rate limit or service unavailable
                delay = retry_delay * (2 ** attempt)
                await asyncio.sleep(delay)
                last_error = e
                continue
            raise

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            delay = retry_delay * (2 ** attempt)
            await asyncio.sleep(delay)
            last_error = e
            continue

    raise last_error or Exception(f"Failed to fetch {url} after {max_retries} retries")
