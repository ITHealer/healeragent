"""
Base class for Crypto Tools using Internal API

All crypto market tools inherit from this class which provides:
- Internal API configuration
- HTTP client management
- Redis caching
- Common error handling
"""

import json
from typing import Dict, Any, Optional
from abc import ABC

import httpx

from src.agents.tools.base import BaseTool
from src.helpers.redis_cache import get_redis_client_llm
from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings


class BaseCryptoTool(BaseTool, LoggerMixin, ABC):
    """
    Base class for crypto tools using internal API

    Internal API: http://10.10.0.2:20073
    """

    # API Configuration
    DEFAULT_BASE_URL = "http://10.10.0.2:20073"
    API_PREFIX = "/api/v1/market/crypto"

    # Default cache TTL (5 minutes for market data)
    DEFAULT_CACHE_TTL = 300

    # HTTP client settings
    REQUEST_TIMEOUT = 30.0

    def __init__(self):
        """Initialize base crypto tool"""
        super().__init__()

        # Get base URL from settings or use default
        self.base_url = getattr(settings, 'CRYPTO_API_BASE_URL', self.DEFAULT_BASE_URL)
        self.api_url = f"{self.base_url}{self.API_PREFIX}"

        self.logger.debug(f"[{self.__class__.__name__}] API URL: {self.api_url}")

    async def _fetch_api(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET"
    ) -> Dict[str, Any]:
        """
        Fetch data from internal API

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            method: HTTP method

        Returns:
            API response as dict

        Raises:
            httpx.HTTPError: On HTTP errors
        """
        url = f"{self.api_url}{endpoint}"

        try:
            async with httpx.AsyncClient(timeout=self.REQUEST_TIMEOUT) as client:
                if method.upper() == "GET":
                    response = await client.get(url, params=params)
                else:
                    response = await client.post(url, json=params)

                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            self.logger.error(
                f"[{self.__class__.__name__}] HTTP {e.response.status_code} "
                f"for {url}: {e.response.text[:200]}"
            )
            raise
        except httpx.RequestError as e:
            self.logger.error(f"[{self.__class__.__name__}] Request error for {url}: {e}")
            raise

    async def _get_cached_result(
        self,
        cache_key: str,
        ttl: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result from Redis

        Args:
            cache_key: Redis key
            ttl: Cache TTL (unused for get, but for consistency)

        Returns:
            Cached data or None
        """
        try:
            redis_client = await get_redis_client_llm()
            if redis_client:
                cached_data = await redis_client.get(cache_key)
                if cached_data:
                    # Handle both bytes and str
                    if isinstance(cached_data, bytes):
                        cached_data = cached_data.decode('utf-8')
                    return json.loads(cached_data)
        except Exception as e:
            self.logger.warning(f"[CACHE] Read error: {e}")
        return None

    async def _set_cached_result(
        self,
        cache_key: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache result to Redis

        Args:
            cache_key: Redis key
            data: Data to cache
            ttl: Cache TTL in seconds
        """
        try:
            redis_client = await get_redis_client_llm()
            if redis_client:
                await redis_client.set(
                    cache_key,
                    json.dumps(data, ensure_ascii=False),
                    ex=ttl or self.DEFAULT_CACHE_TTL
                )
                self.logger.debug(f"[CACHE] SET {cache_key} TTL={ttl or self.DEFAULT_CACHE_TTL}s")
        except Exception as e:
            self.logger.warning(f"[CACHE] Write error: {e}")

    def _format_price(self, price: float) -> str:
        """Format price with appropriate precision"""
        if price >= 1000:
            return f"${price:,.2f}"
        elif price >= 1:
            return f"${price:.4f}"
        else:
            return f"${price:.8f}"

    def _format_large_number(self, num: float) -> str:
        """Format large numbers (volume, market cap)"""
        if num >= 1_000_000_000:
            return f"${num / 1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"${num / 1_000_000:.2f}M"
        elif num >= 1_000:
            return f"${num / 1_000:.2f}K"
        else:
            return f"${num:.2f}"

    def _format_percent(self, pct: float) -> str:
        """Format percentage with sign"""
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.2f}%"
