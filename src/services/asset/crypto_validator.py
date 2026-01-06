import asyncio
import aiohttp
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings


@dataclass
class CryptoSearchResult:
    """Result from crypto symbol search"""
    symbol: str
    name: str
    last_price: float
    price_change_percent: float
    exchange: str
    image_url: Optional[str] = None
    is_exact_match: bool = False


@dataclass
class CryptoValidationResult:
    """Result of crypto symbol validation"""
    symbol: str
    is_valid: bool
    exact_match: Optional[CryptoSearchResult] = None
    similar_symbols: List[CryptoSearchResult] = field(default_factory=list)
    source: str = "external_api"
    error: Optional[str] = None


class CryptoSymbolValidator(LoggerMixin):
    """
    Validates crypto symbols using external market API.

    API configured via settings:
    - CRYPTO_INTERNAL_API_URL (default: http://10.10.0.2:20073)
    - CRYPTO_INTERNAL_API_PREFIX (default: /api/v1/market/crypto)

    Features:
    - Validates if symbol is supported
    - Returns similar symbols for disambiguation
    - Caches results in Redis for performance
    - Handles API errors gracefully
    """

    # Cache settings
    CACHE_TTL_SECONDS = 3600  # 1 hour
    CACHE_KEY_PREFIX = "crypto_validate:"

    # Request settings
    REQUEST_TIMEOUT = 10  # seconds
    MAX_RESULTS = 10

    _instance: Optional['CryptoSymbolValidator'] = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        base_url: Optional[str] = None,
        redis_client=None,
        timeout: int = REQUEST_TIMEOUT
    ):
        if hasattr(self, '_initialized') and self._initialized:
            return

        super().__init__()

        # Get API config from settings
        self.base_url = base_url or settings.CRYPTO_INTERNAL_API_URL
        self.api_prefix = settings.CRYPTO_INTERNAL_API_PREFIX
        self.search_endpoint = f"{self.api_prefix}/search"
        self.redis = redis_client
        self.timeout = timeout

        # In-memory cache for quick lookups
        self._cache: Dict[str, tuple] = {}  # symbol -> (result, timestamp)
        self._cache_max_size = 1000

        self._initialized = True
        self.logger.info(f"[CRYPTO_VALIDATOR] Initialized with base_url={self.base_url}")

    async def validate(self, symbol: str) -> CryptoValidationResult:
        """
        Validate a crypto symbol against the external API.

        Args:
            symbol: Crypto symbol to validate (e.g., "BTC", "ETH")

        Returns:
            CryptoValidationResult with validation details
        """
        symbol = symbol.upper().strip()

        # Check memory cache first
        cached = self._get_from_cache(symbol)
        if cached:
            self.logger.debug(f"[CRYPTO_VALIDATOR] Cache hit for {symbol}")
            return cached

        # Check Redis cache
        if self.redis:
            redis_cached = await self._get_from_redis(symbol)
            if redis_cached:
                self._add_to_cache(symbol, redis_cached)
                return redis_cached

        # Call external API
        try:
            result = await self._search_symbol(symbol)

            # Cache the result
            self._add_to_cache(symbol, result)
            if self.redis:
                await self._save_to_redis(symbol, result)

            return result

        except Exception as e:
            self.logger.error(f"[CRYPTO_VALIDATOR] Error validating {symbol}: {e}")
            return CryptoValidationResult(
                symbol=symbol,
                is_valid=False,
                error=str(e),
                source="error"
            )

    async def search(
        self,
        query: str,
        max_results: int = MAX_RESULTS
    ) -> List[CryptoSearchResult]:
        """
        Search for crypto symbols matching a query.

        Args:
            query: Search query (e.g., "BTC", "Bitcoin")
            max_results: Maximum number of results

        Returns:
            List of matching CryptoSearchResult
        """
        try:
            url = f"{self.base_url}{self.search_endpoint}"
            params = {
                "SearchQuery": query,
                "PageNumber": max_results
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        self.logger.warning(
                            f"[CRYPTO_VALIDATOR] API returned {response.status} for query={query}"
                        )
                        return []

                    data = await response.json()
                    return self._parse_search_results(data, query)

        except asyncio.TimeoutError:
            self.logger.warning(f"[CRYPTO_VALIDATOR] Timeout searching for {query}")
            return []
        except Exception as e:
            self.logger.error(f"[CRYPTO_VALIDATOR] Error searching {query}: {e}")
            return []

    async def get_symbol_info(self, symbol: str) -> Optional[CryptoSearchResult]:
        """
        Get detailed info for a specific crypto symbol.

        Args:
            symbol: Exact symbol to look up (e.g., "BTC")

        Returns:
            CryptoSearchResult if found, None otherwise
        """
        result = await self.validate(symbol)
        return result.exact_match

    async def batch_validate(
        self,
        symbols: List[str]
    ) -> Dict[str, CryptoValidationResult]:
        """
        Validate multiple symbols in parallel.

        Args:
            symbols: List of symbols to validate

        Returns:
            Dict mapping symbol to validation result
        """
        tasks = [self.validate(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            symbols[i]: (
                result if isinstance(result, CryptoValidationResult)
                else CryptoValidationResult(
                    symbol=symbols[i],
                    is_valid=False,
                    error=str(result),
                    source="error"
                )
            )
            for i, result in enumerate(results)
        }

    # ========================================
    # PRIVATE METHODS
    # ========================================

    async def _search_symbol(self, symbol: str) -> CryptoValidationResult:
        """Search for symbol via external API"""
        results = await self.search(symbol, max_results=self.MAX_RESULTS)

        if not results:
            return CryptoValidationResult(
                symbol=symbol,
                is_valid=False,
                source="external_api"
            )

        # Find exact match
        exact_match = None
        similar = []

        for result in results:
            if result.symbol.upper() == symbol:
                exact_match = result
                result.is_exact_match = True
            else:
                similar.append(result)

        return CryptoValidationResult(
            symbol=symbol,
            is_valid=exact_match is not None,
            exact_match=exact_match,
            similar_symbols=similar[:5],  # Limit similar results
            source="external_api"
        )

    def _parse_search_results(
        self,
        data: Dict[str, Any],
        query: str
    ) -> List[CryptoSearchResult]:
        """Parse API response into CryptoSearchResult list"""
        results = []

        try:
            # Navigate to data array
            items = data.get("data", {}).get("data", [])

            if not isinstance(items, list):
                return results

            for item in items:
                symbol = item.get("symbol", "").upper()
                result = CryptoSearchResult(
                    symbol=symbol,
                    name=item.get("name", symbol),
                    last_price=float(item.get("lastPrice", 0) or 0),
                    price_change_percent=float(item.get("priceChangePercent", 0) or 0),
                    exchange=item.get("exchange", ""),
                    image_url=item.get("imageUrl"),
                    is_exact_match=(symbol == query.upper())
                )
                results.append(result)

        except Exception as e:
            self.logger.warning(f"[CRYPTO_VALIDATOR] Error parsing results: {e}")

        return results

    def _get_from_cache(self, symbol: str) -> Optional[CryptoValidationResult]:
        """Get from in-memory cache"""
        if symbol not in self._cache:
            return None

        result, timestamp = self._cache[symbol]
        age = (datetime.now() - timestamp).total_seconds()

        if age > self.CACHE_TTL_SECONDS:
            del self._cache[symbol]
            return None

        return result

    def _add_to_cache(self, symbol: str, result: CryptoValidationResult) -> None:
        """Add to in-memory cache with LRU eviction"""
        # Evict oldest if at capacity
        while len(self._cache) >= self._cache_max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        self._cache[symbol] = (result, datetime.now())

    async def _get_from_redis(self, symbol: str) -> Optional[CryptoValidationResult]:
        """Get from Redis cache"""
        try:
            import json
            key = f"{self.CACHE_KEY_PREFIX}{symbol}"
            data = await self.redis.get(key)

            if data:
                parsed = json.loads(data)
                return self._dict_to_result(parsed)

        except Exception as e:
            self.logger.debug(f"[CRYPTO_VALIDATOR] Redis get error: {e}")

        return None

    async def _save_to_redis(self, symbol: str, result: CryptoValidationResult) -> None:
        """Save to Redis cache"""
        try:
            import json
            key = f"{self.CACHE_KEY_PREFIX}{symbol}"
            data = self._result_to_dict(result)
            await self.redis.set(key, json.dumps(data), ex=self.CACHE_TTL_SECONDS)

        except Exception as e:
            self.logger.debug(f"[CRYPTO_VALIDATOR] Redis save error: {e}")

    def _result_to_dict(self, result: CryptoValidationResult) -> Dict[str, Any]:
        """Convert result to dict for serialization"""
        return {
            "symbol": result.symbol,
            "is_valid": result.is_valid,
            "exact_match": {
                "symbol": result.exact_match.symbol,
                "name": result.exact_match.name,
                "last_price": result.exact_match.last_price,
                "price_change_percent": result.exact_match.price_change_percent,
                "exchange": result.exact_match.exchange,
                "image_url": result.exact_match.image_url,
            } if result.exact_match else None,
            "similar_symbols": [
                {
                    "symbol": s.symbol,
                    "name": s.name,
                    "last_price": s.last_price,
                    "price_change_percent": s.price_change_percent,
                    "exchange": s.exchange,
                    "image_url": s.image_url,
                }
                for s in result.similar_symbols
            ],
            "source": result.source,
            "error": result.error,
        }

    def _dict_to_result(self, data: Dict[str, Any]) -> CryptoValidationResult:
        """Convert dict back to result"""
        exact_match = None
        if data.get("exact_match"):
            em = data["exact_match"]
            exact_match = CryptoSearchResult(
                symbol=em["symbol"],
                name=em["name"],
                last_price=em["last_price"],
                price_change_percent=em["price_change_percent"],
                exchange=em["exchange"],
                image_url=em.get("image_url"),
                is_exact_match=True,
            )

        similar = [
            CryptoSearchResult(
                symbol=s["symbol"],
                name=s["name"],
                last_price=s["last_price"],
                price_change_percent=s["price_change_percent"],
                exchange=s["exchange"],
                image_url=s.get("image_url"),
            )
            for s in data.get("similar_symbols", [])
        ]

        return CryptoValidationResult(
            symbol=data["symbol"],
            is_valid=data["is_valid"],
            exact_match=exact_match,
            similar_symbols=similar,
            source=data.get("source", "redis_cache"),
            error=data.get("error"),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get validator statistics"""
        return {
            "cache_size": len(self._cache),
            "cache_max_size": self._cache_max_size,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }

    def clear_cache(self) -> int:
        """Clear in-memory cache"""
        count = len(self._cache)
        self._cache.clear()
        return count


# ========================================
# SINGLETON ACCESSOR
# ========================================

_crypto_validator: Optional[CryptoSymbolValidator] = None


def get_crypto_validator(
    base_url: Optional[str] = None,
    redis_client=None
) -> CryptoSymbolValidator:
    """Get singleton CryptoSymbolValidator instance"""
    global _crypto_validator

    if _crypto_validator is None:
        _crypto_validator = CryptoSymbolValidator(
            base_url=base_url,
            redis_client=redis_client
        )

    return _crypto_validator