# src/services/asset/symbol_cache.py
"""
Simplified Symbol Cache Service

Simple O(1) symbol lookup with ambiguity detection.
No quote currency mapping - BE team handles that.

Key Features:
1. O(1) lookup for crypto/stock symbols
2. Ambiguity detection (symbol exists in both crypto and stock)
3. Redis cache for distributed systems
4. Async DB loading at startup
"""

from typing import Optional, Set, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from src.utils.logger.custom_logging import LoggerMixin


class AssetClass(str, Enum):
    """Asset class enumeration"""
    CRYPTO = "crypto"
    STOCK = "stock"
    UNKNOWN = "unknown"


@dataclass
class SymbolInfo:
    """Symbol information for disambiguation"""
    symbol: str
    name: str
    asset_class: AssetClass
    exchange: Optional[str] = None
    description: Optional[str] = None


class SymbolCacheService(LoggerMixin):
    """
    Simple symbol cache for O(1) ambiguity detection.

    Usage:
        cache = get_symbol_cache()
        await cache.initialize()  # Call once at startup

        if cache.is_ambiguous("SOL"):
            options = cache.get_disambiguation_options("SOL")
            # Ask user to choose
    """

    # Redis key for caching ambiguous symbols
    REDIS_KEY = "symbol_cache:ambiguous"
    REDIS_TTL = 3600 * 24  # 24 hours

    _instance: Optional['SymbolCacheService'] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, redis_client=None):
        if self._initialized:
            return

        super().__init__()
        self.redis = redis_client

        # In-memory cache (O(1) lookup)
        self._crypto_symbols: Set[str] = set()
        self._stock_symbols: Set[str] = set()
        self._ambiguous_symbols: Set[str] = set()

        # Symbol info for disambiguation UI
        self._crypto_info: Dict[str, SymbolInfo] = {}
        self._stock_info: Dict[str, SymbolInfo] = {}

        # Cache metadata
        self._last_refresh: Optional[datetime] = None
        self._cache_ttl: timedelta = timedelta(hours=24)

        # Thread pool for DB operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="symbol_cache")

        self._initialized = True
        self.logger.info("[SYMBOL_CACHE] Initialized")

    async def initialize(self) -> bool:
        """
        Load symbols at startup.
        Call once during app initialization.

        Returns:
            True if loaded successfully
        """
        self.logger.info("[SYMBOL_CACHE] Starting initialization...")

        try:
            # Try Redis cache first (fast)
            if self.redis:
                loaded = await self._load_from_redis()
                if loaded:
                    self.logger.info("[SYMBOL_CACHE] Loaded from Redis cache")
                    return True

            # Load from database (slower)
            await self._load_from_database()
            self._compute_ambiguous_symbols()
            self._last_refresh = datetime.utcnow()

            # Save to Redis for next time
            if self.redis:
                await self._save_to_redis()

            self.logger.info(
                f"[SYMBOL_CACHE] Loaded: "
                f"crypto={len(self._crypto_symbols)}, "
                f"stock={len(self._stock_symbols)}, "
                f"ambiguous={len(self._ambiguous_symbols)}"
            )
            return True

        except Exception as e:
            self.logger.error(f"[SYMBOL_CACHE] Initialization failed: {e}")
            self._load_default_symbols()
            self._compute_ambiguous_symbols()
            return False

    def is_ambiguous(self, symbol: str) -> bool:
        """
        Check if symbol exists in both crypto and stock.
        O(1) lookup.
        """
        return symbol.upper().strip() in self._ambiguous_symbols

    def get_disambiguation_options(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get all options for an ambiguous symbol.

        Returns:
            List of options with asset_class, name, description, exchange
        """
        symbol = symbol.upper().strip()
        options = []

        # Add crypto option
        if symbol in self._crypto_symbols:
            info = self._crypto_info.get(symbol)
            options.append({
                "asset_class": "crypto",
                "symbol": symbol,
                "name": info.name if info else symbol,
                "description": info.description if info else "Cryptocurrency",
                "exchange": "Crypto",
            })

        # Add stock option
        if symbol in self._stock_symbols:
            info = self._stock_info.get(symbol)
            options.append({
                "asset_class": "stock",
                "symbol": symbol,
                "name": info.name if info else symbol,
                "description": info.description if info else "Stock",
                "exchange": info.exchange if info else "US",
            })

        return options

    def exists(self, symbol: str, asset_class: Optional[AssetClass] = None) -> bool:
        """Check if symbol exists in cache"""
        symbol = symbol.upper().strip()

        if asset_class is None:
            return symbol in self._crypto_symbols or symbol in self._stock_symbols
        elif asset_class == AssetClass.CRYPTO:
            return symbol in self._crypto_symbols
        elif asset_class == AssetClass.STOCK:
            return symbol in self._stock_symbols
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "crypto_count": len(self._crypto_symbols),
            "stock_count": len(self._stock_symbols),
            "ambiguous_count": len(self._ambiguous_symbols),
            "ambiguous_symbols": list(self._ambiguous_symbols)[:20],
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
        }

    # ========================================
    # PRIVATE METHODS
    # ========================================

    async def _load_from_database(self) -> None:
        """Load symbols from PostgreSQL (runs in thread pool)"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._load_from_db_sync)

    def _load_from_db_sync(self) -> None:
        """Synchronous DB loading (runs in thread pool)"""
        try:
            from src.database.models.symbol_directory import Stock, Cryptocurrency
            from sqlalchemy import select
            from src.database import get_postgres_db

            db = get_postgres_db().get_session()
            try:
                # Load stocks
                stmt = select(Stock.symbol, Stock.name, Stock.exchange)
                result = db.execute(stmt)
                for row in result:
                    symbol = row.symbol.upper().strip()
                    self._stock_symbols.add(symbol)
                    self._stock_info[symbol] = SymbolInfo(
                        symbol=symbol,
                        name=row.name or symbol,
                        asset_class=AssetClass.STOCK,
                        exchange=row.exchange,
                        description=f"Stock - {row.exchange or 'US'}",
                    )

                # Load cryptos
                stmt = select(Cryptocurrency.symbol, Cryptocurrency.name)
                result = db.execute(stmt)
                for row in result:
                    # Normalize: BTCUSD -> BTC
                    full_symbol = row.symbol.upper().strip()
                    base_symbol = self._normalize_crypto_symbol(full_symbol)

                    self._crypto_symbols.add(base_symbol)
                    if base_symbol not in self._crypto_info:
                        self._crypto_info[base_symbol] = SymbolInfo(
                            symbol=base_symbol,
                            name=row.name or base_symbol,
                            asset_class=AssetClass.CRYPTO,
                            description="Cryptocurrency",
                        )

            finally:
                db.close()

        except Exception as e:
            self.logger.error(f"[SYMBOL_CACHE] DB load failed: {e}")
            raise

    def _normalize_crypto_symbol(self, symbol: str) -> str:
        """Remove quote currency suffixes: BTCUSD -> BTC"""
        for suffix in ["USDT", "USD", "BUSD", "USDC"]:
            if symbol.endswith(suffix) and len(symbol) > len(suffix):
                return symbol[:-len(suffix)]
        return symbol

    def _compute_ambiguous_symbols(self) -> None:
        """Find symbols that exist in both crypto and stock"""
        self._ambiguous_symbols = self._crypto_symbols & self._stock_symbols

        if self._ambiguous_symbols:
            self.logger.info(
                f"[SYMBOL_CACHE] Ambiguous symbols: {list(self._ambiguous_symbols)}"
            )

    def _load_default_symbols(self) -> None:
        """Fallback: load common symbols"""
        # Common cryptos
        crypto_defaults = {
            "BTC": "Bitcoin",
            "ETH": "Ethereum",
            "SOL": "Solana",
            "BNB": "Binance Coin",
            "XRP": "Ripple",
            "ADA": "Cardano",
            "DOGE": "Dogecoin",
            "LINK": "Chainlink",
            "UNI": "Uniswap",
            "AVAX": "Avalanche",
        }

        for symbol, name in crypto_defaults.items():
            self._crypto_symbols.add(symbol)
            self._crypto_info[symbol] = SymbolInfo(
                symbol=symbol,
                name=name,
                asset_class=AssetClass.CRYPTO,
                description="Cryptocurrency",
            )

        # Common stocks (including ambiguous ones)
        stock_defaults = {
            "AAPL": ("Apple Inc", "NASDAQ"),
            "TSLA": ("Tesla Inc", "NASDAQ"),
            "NVDA": ("NVIDIA Corp", "NASDAQ"),
            "MSFT": ("Microsoft Corp", "NASDAQ"),
            "SOL": ("Renesola Ltd", "NYSE"),      # Ambiguous with Solana
            "LINK": ("Interlink Electronics", "NASDAQ"),  # Ambiguous with Chainlink
        }

        for symbol, (name, exchange) in stock_defaults.items():
            self._stock_symbols.add(symbol)
            self._stock_info[symbol] = SymbolInfo(
                symbol=symbol,
                name=name,
                asset_class=AssetClass.STOCK,
                exchange=exchange,
                description=f"Stock - {exchange}",
            )

    async def _load_from_redis(self) -> bool:
        """Load cached ambiguous symbols from Redis"""
        try:
            import json
            data = await self.redis.get(self.REDIS_KEY)
            if not data:
                return False

            cached = json.loads(data)
            self._crypto_symbols = set(cached.get("crypto", []))
            self._stock_symbols = set(cached.get("stock", []))
            self._ambiguous_symbols = set(cached.get("ambiguous", []))

            # Rebuild info from cached data
            for symbol in self._crypto_symbols:
                self._crypto_info[symbol] = SymbolInfo(
                    symbol=symbol,
                    name=symbol,
                    asset_class=AssetClass.CRYPTO,
                    description="Cryptocurrency",
                )
            for symbol in self._stock_symbols:
                self._stock_info[symbol] = SymbolInfo(
                    symbol=symbol,
                    name=symbol,
                    asset_class=AssetClass.STOCK,
                    description="Stock",
                )

            self._last_refresh = datetime.fromisoformat(cached.get("timestamp", ""))
            return True

        except Exception as e:
            self.logger.debug(f"[SYMBOL_CACHE] Redis load failed: {e}")
            return False

    async def _save_to_redis(self) -> None:
        """Save symbol sets to Redis"""
        try:
            import json
            data = {
                "crypto": list(self._crypto_symbols),
                "stock": list(self._stock_symbols),
                "ambiguous": list(self._ambiguous_symbols),
                "timestamp": datetime.utcnow().isoformat(),
            }
            await self.redis.set(
                self.REDIS_KEY,
                json.dumps(data),
                ex=self.REDIS_TTL
            )
        except Exception as e:
            self.logger.debug(f"[SYMBOL_CACHE] Redis save failed: {e}")


# ========================================
# SINGLETON ACCESSOR
# ========================================

_symbol_cache: Optional[SymbolCacheService] = None


def get_symbol_cache(redis_client=None) -> SymbolCacheService:
    """Get singleton SymbolCacheService instance"""
    global _symbol_cache
    if _symbol_cache is None:
        _symbol_cache = SymbolCacheService(redis_client=redis_client)
    return _symbol_cache


async def initialize_symbol_cache(redis_client=None) -> SymbolCacheService:
    """Initialize and return symbol cache (call at startup)"""
    cache = get_symbol_cache(redis_client=redis_client)
    await cache.initialize()
    return cache
