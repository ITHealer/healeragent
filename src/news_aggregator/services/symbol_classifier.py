# src/news_aggregator/services/symbol_classifier.py
"""
Symbol Classifier Service - Production-Ready Implementation

High-performance symbol classification (crypto vs stock) with:
- In-memory caching for O(1) lookup
- Async-safe operations with Lock
- TTL-based background refresh
- Graceful fallback if DB unavailable
- No blocking of other APIs

Architecture:
- Preload crypto symbols from PostgreSQL into memory (Set)
- Use Set membership check for O(1) classification
- Background refresh every CACHE_TTL_SECONDS
- Fallback to common crypto list if DB unavailable

Usage:
    classifier = await get_symbol_classifier()
    asset_type = classifier.classify("BTC")  # Returns "crypto"
    asset_type = classifier.classify("TSLA")  # Returns "stock"

    # Batch classification
    crypto_symbols, stock_symbols = classifier.classify_batch(["BTC", "TSLA", "ETH"])
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SymbolClassifier:
    """
    Production-ready symbol classifier with in-memory caching.

    Features:
    - O(1) symbol classification
    - Async-safe with Lock
    - TTL-based cache refresh
    - Graceful degradation
    """

    # Cache settings
    CACHE_TTL_SECONDS = 3600  # 1 hour
    REFRESH_RETRY_DELAY = 60  # 1 minute retry on failure

    # Common crypto symbols as fallback (used if DB unavailable)
    FALLBACK_CRYPTO_SYMBOLS: FrozenSet[str] = frozenset({
        # Major cryptocurrencies
        "BTC", "ETH", "USDT", "BNB", "XRP", "USDC", "SOL", "ADA", "DOGE", "TRX",
        "TON", "DOT", "MATIC", "LTC", "SHIB", "AVAX", "LINK", "ATOM", "XLM", "XMR",
        "ETC", "BCH", "FIL", "HBAR", "APT", "ICP", "VET", "NEAR", "OP", "ARB",
        "MKR", "AAVE", "GRT", "QNT", "ALGO", "EGLD", "FTM", "SAND", "MANA", "AXS",
        "THETA", "EOS", "XTZ", "CHZ", "CRV", "LDO", "SNX", "COMP", "ZEC", "DASH",
        # With USD suffix (FMP format)
        "BTCUSD", "ETHUSD", "SOLUSD", "ADAUSD", "DOGEUSD", "XRPUSD", "DOTUSD",
        "LINKUSD", "AVAXUSD", "MATICUSD", "ATOMUSD", "LTCUSD", "BCHUSD", "XLMUSD",
    })

    def __init__(self):
        """Initialize classifier with empty cache."""
        self._crypto_symbols: FrozenSet[str] = frozenset()
        self._last_refresh: Optional[float] = None
        self._refresh_lock = asyncio.Lock()
        self._initialized = False
        self._db_available = True

    @property
    def is_initialized(self) -> bool:
        """Check if cache has been initialized."""
        return self._initialized

    @property
    def cache_age_seconds(self) -> Optional[float]:
        """Get cache age in seconds."""
        if self._last_refresh is None:
            return None
        return time.time() - self._last_refresh

    @property
    def needs_refresh(self) -> bool:
        """Check if cache needs refresh."""
        if not self._initialized:
            return True
        age = self.cache_age_seconds
        return age is None or age > self.CACHE_TTL_SECONDS

    async def _load_crypto_symbols_from_db(self) -> Set[str]:
        """
        Load crypto symbols from PostgreSQL.

        Returns:
            Set of crypto symbols (uppercase)
        """
        try:
            # Import here to avoid circular imports
            from src.database import get_postgres_db
            from src.database.models.symbol_directory import Cryptocurrency
            from sqlalchemy import select

            db = get_postgres_db()
            session = db.get_session()

            try:
                # Query all crypto symbols
                stmt = select(Cryptocurrency.symbol)
                result = session.execute(stmt).scalars().all()

                # Normalize to uppercase set
                symbols = set(s.upper() for s in result if s)

                logger.info(f"[SymbolClassifier] Loaded {len(symbols)} crypto symbols from DB")
                return symbols

            finally:
                session.close()

        except Exception as e:
            logger.error(f"[SymbolClassifier] DB load failed: {e}")
            raise

    async def refresh_cache(self, force: bool = False) -> bool:
        """
        Refresh crypto symbols cache from DB.

        Args:
            force: Force refresh even if cache is still valid

        Returns:
            True if refresh successful
        """
        # Check if refresh needed
        if not force and not self.needs_refresh:
            return True

        # Use lock to prevent concurrent refreshes
        async with self._refresh_lock:
            # Double-check after acquiring lock
            if not force and not self.needs_refresh:
                return True

            try:
                logger.info("[SymbolClassifier] Refreshing crypto symbols cache...")
                start_time = time.time()

                # Load from DB
                symbols = await self._load_crypto_symbols_from_db()

                # Update cache atomically
                self._crypto_symbols = frozenset(symbols)
                self._last_refresh = time.time()
                self._initialized = True
                self._db_available = True

                elapsed_ms = int((time.time() - start_time) * 1000)
                logger.info(
                    f"[SymbolClassifier] Cache refreshed: {len(symbols)} symbols in {elapsed_ms}ms"
                )
                return True

            except Exception as e:
                logger.error(f"[SymbolClassifier] Cache refresh failed: {e}")
                self._db_available = False

                # Use fallback if not initialized
                if not self._initialized:
                    logger.warning(
                        f"[SymbolClassifier] Using fallback list: {len(self.FALLBACK_CRYPTO_SYMBOLS)} symbols"
                    )
                    self._crypto_symbols = self.FALLBACK_CRYPTO_SYMBOLS
                    self._initialized = True
                    self._last_refresh = time.time()

                return False

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol for classification.

        Handles various formats:
        - BTC -> BTC
        - BTC-USD -> BTC
        - BTCUSD -> BTCUSD (keep as is, check directly)
        """
        symbol = symbol.upper().strip()

        # Remove common suffixes for checking
        if symbol.endswith("-USD"):
            return symbol[:-4]  # BTC-USD -> BTC

        return symbol

    def _is_crypto_symbol(self, symbol: str) -> bool:
        """
        Check if symbol is crypto.

        Args:
            symbol: Normalized symbol

        Returns:
            True if crypto
        """
        normalized = self._normalize_symbol(symbol)

        # Check exact match
        if normalized in self._crypto_symbols:
            return True

        # Check with USD suffix (FMP format: BTCUSD)
        if f"{normalized}USD" in self._crypto_symbols:
            return True

        # Check if symbol itself ends with USD (already in format BTCUSD)
        if normalized.endswith("USD") and normalized in self._crypto_symbols:
            return True

        return False

    def classify(self, symbol: str) -> str:
        """
        Classify single symbol as 'crypto' or 'stock'.

        Args:
            symbol: Symbol to classify

        Returns:
            'crypto' or 'stock'
        """
        if self._is_crypto_symbol(symbol):
            return "crypto"
        return "stock"

    def classify_batch(
        self,
        symbols: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Classify multiple symbols efficiently.

        Args:
            symbols: List of symbols to classify

        Returns:
            Tuple of (crypto_symbols, stock_symbols)
        """
        crypto_symbols = []
        stock_symbols = []

        for symbol in symbols:
            symbol_upper = symbol.upper().strip()
            if self._is_crypto_symbol(symbol_upper):
                crypto_symbols.append(symbol_upper)
            else:
                stock_symbols.append(symbol_upper)

        return crypto_symbols, stock_symbols

    def get_stats(self) -> Dict:
        """Get classifier statistics."""
        return {
            "initialized": self._initialized,
            "crypto_count": len(self._crypto_symbols),
            "cache_age_seconds": self.cache_age_seconds,
            "db_available": self._db_available,
            "using_fallback": not self._db_available,
            "last_refresh": datetime.fromtimestamp(self._last_refresh).isoformat()
                if self._last_refresh else None,
        }


# =============================================================================
# Singleton Instance
# =============================================================================

_classifier: Optional[SymbolClassifier] = None
_init_lock = asyncio.Lock()


async def get_symbol_classifier() -> SymbolClassifier:
    """
    Get or create singleton SymbolClassifier instance.

    Returns:
        Initialized SymbolClassifier
    """
    global _classifier

    if _classifier is None:
        async with _init_lock:
            if _classifier is None:
                _classifier = SymbolClassifier()
                await _classifier.refresh_cache()

    # Background refresh if needed (non-blocking)
    elif _classifier.needs_refresh:
        # Schedule background refresh without waiting
        asyncio.create_task(_background_refresh())

    return _classifier


async def _background_refresh():
    """Background refresh task (fire-and-forget)."""
    global _classifier
    if _classifier:
        try:
            await _classifier.refresh_cache()
        except Exception as e:
            logger.warning(f"[SymbolClassifier] Background refresh failed: {e}")


def get_symbol_classifier_sync() -> Optional[SymbolClassifier]:
    """
    Get classifier synchronously (if already initialized).

    Returns:
        SymbolClassifier if initialized, None otherwise
    """
    return _classifier
