import threading
import time
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum

from src.utils.logger.custom_logging import LoggerMixin


# ============================================================================
# CONSTANTS
# ============================================================================

class AssetClass(str, Enum):
    """Asset class types"""
    STOCK = "stock"
    CRYPTO = "crypto"
    UNKNOWN = "unknown"


# Common false positives from regex matching
FALSE_POSITIVE_SYMBOLS = frozenset({
    # Common English words
    "I", "A", "OK", "US", "UK", "EU", "UN", "TV", "AM", "PM", "NO", "IT",
    "AI", "AR", "VR", "CEO", "CFO", "CTO", "COO", "VP", "HR", "PR", "QA",
    "USA", "USD", "EUR", "GBP", "JPY", "CNY", "VND", "THB", "SGD", "HKD",
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "HIS", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HOW",
    "MAN", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "BOY", "DID", "ITS",
    "SAY", "SHE", "TWO", "USE", "HER", "MAY", "DOG", "CAT", "RED", "BIG",
    # Vietnamese common words that might be uppercase
    "GIA", "BAN", "MUA", "CON", "NAM", "TEN", "MOT", "HAI", "BA", "BON",
    # Common tech/finance abbreviations
    "API", "URL", "SQL", "ETF", "IPO", "ATH", "YTD", "QTD", "MTD", "EPS",
    "ROI", "ROE", "ROA", "P/E", "P/B", "DCF", "NPV", "IRR", "WACC", "EBITDA",
    # Common crypto terms that aren't symbols
    "NFT", "DEX", "CEX", "DAO", "TVL", "APY", "APR", "HODL", "FUD", "FOMO",
})


@dataclass
class CacheEntry:
    """Single cache entry for a symbol"""
    symbol: str
    is_valid: bool
    asset_class: AssetClass
    name: Optional[str] = None
    exchange: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    hits: int = 0
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry is expired"""
        return datetime.now() > self.created_at + timedelta(seconds=ttl_seconds)


@dataclass
class ValidationResult:
    """Result of symbol validation"""
    symbol: str
    is_valid: bool
    asset_class: AssetClass
    source: str  # "cache" or "database"
    name: Optional[str] = None
    exchange: Optional[str] = None


# ============================================================================
# SYMBOL VALIDATION CACHE
# ============================================================================

class SymbolValidationCache(LoggerMixin):
    """
    Production-ready Symbol Validation Cache
    
    Features:
    - In-memory cache with LRU eviction
    - TTL-based expiration
    - Batch validation support
    - Thread-safe operations
    - Pre-loaded common symbols
    
    Memory Usage:
    - ~50 bytes per entry
    - 10,000 symbols ≈ 500KB
    - Max 50,000 entries ≈ 2.5MB
    """
    
    # Configuration
    DEFAULT_TTL_SECONDS = 1800  # 30 minutes
    MAX_CACHE_SIZE = 50000  # Max entries
    PRELOAD_TOP_SYMBOLS = 500  # Preload top N symbols on init
    
    # Singleton instance
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for global cache"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_size: int = MAX_CACHE_SIZE,
        auto_preload: bool = True
    ):
        """
        Initialize cache
        
        Args:
            ttl_seconds: Time-to-live for cache entries
            max_size: Maximum number of entries
            auto_preload: Whether to preload common symbols
        """
        # Skip if already initialized (singleton)
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        super().__init__()
        
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        
        # LRU cache using OrderedDict
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Thread lock for cache operations
        self._cache_lock = threading.RLock()
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "db_lookups": 0,
            "evictions": 0,
            "preloaded": 0
        }
        
        # DB service reference (lazy loaded)
        self._db_service = None
        
        # Mark as initialized
        self._initialized = True
        
        # Preload common symbols (optional)
        if auto_preload:
            self._preload_common_symbols()
        
        self.logger.info(
            f"✅ SymbolValidationCache initialized: "
            f"ttl={ttl_seconds}s, max_size={max_size}"
        )
    
    # ========================================================================
    # PRELOADING
    # ========================================================================
    
    def _preload_common_symbols(self) -> None:
        """
        Preload common/popular symbols into cache
        
        This runs once on startup to warm the cache with frequently used symbols.
        """
        # Popular US stocks
        common_stocks = [
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
            "BRK.A", "BRK.B", "V", "JNJ", "WMT", "JPM", "MA", "PG", "UNH", "HD",
            "DIS", "PYPL", "BAC", "INTC", "VZ", "CMCSA", "NFLX", "KO", "PEP",
            "T", "MRK", "ABT", "TMO", "COST", "NKE", "ADBE", "CRM", "AVGO", "LLY",
            "ORCL", "CSCO", "ACN", "DHR", "TXN", "QCOM", "AMD", "NEE", "BMY", "PM"
        ]
        
        # Popular crypto
        common_crypto = [
            "BTCUSD", "ETHUSD", "BNBUSD", "XRPUSD", "SOLUSD", "ADAUSD", "DOGEUSD",
            "DOTUSD", "MATICUSD", "LINKUSD", "LTCUSD", "SHIBUSD", "AVAXUSD",
            "UNIUSD", "ATOMUSD", "XLMUSD", "NEARUSD", "ALGOUSD", "VETUSD", "ICPUSD"
        ]
        
        # Pre-populate cache with known valid symbols
        for symbol in common_stocks:
            self._add_to_cache(
                symbol=symbol,
                is_valid=True,
                asset_class=AssetClass.STOCK,
                source="preload"
            )
        
        for symbol in common_crypto:
            self._add_to_cache(
                symbol=symbol,
                is_valid=True,
                asset_class=AssetClass.CRYPTO,
                source="preload"
            )
        
        # Pre-populate invalid symbols (false positives)
        for symbol in list(FALSE_POSITIVE_SYMBOLS)[:100]:
            self._add_to_cache(
                symbol=symbol,
                is_valid=False,
                asset_class=AssetClass.UNKNOWN,
                source="preload"
            )
        
        self._stats["preloaded"] = len(common_stocks) + len(common_crypto)
        self.logger.info(f"✅ Preloaded {self._stats['preloaded']} common symbols")
    
    # ========================================================================
    # CACHE OPERATIONS
    # ========================================================================
    
    def _add_to_cache(
        self,
        symbol: str,
        is_valid: bool,
        asset_class: AssetClass,
        name: Optional[str] = None,
        exchange: Optional[str] = None,
        source: str = "database"
    ) -> None:
        """Add entry to cache with LRU eviction"""
        symbol = symbol.upper()
        
        with self._cache_lock:
            # If already exists, move to end (most recently used)
            if symbol in self._cache:
                self._cache.move_to_end(symbol)
                return
            
            # Evict oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                self._stats["evictions"] += 1
            
            # Add new entry
            self._cache[symbol] = CacheEntry(
                symbol=symbol,
                is_valid=is_valid,
                asset_class=asset_class,
                name=name,
                exchange=exchange,
            )
    
    def _get_from_cache(self, symbol: str) -> Optional[CacheEntry]:
        """Get entry from cache if exists and not expired"""
        symbol = symbol.upper()
        
        with self._cache_lock:
            if symbol not in self._cache:
                return None
            
            entry = self._cache[symbol]
            
            # Check expiration
            if entry.is_expired(self.ttl_seconds):
                del self._cache[symbol]
                return None
            
            # Move to end (LRU update)
            self._cache.move_to_end(symbol)
            entry.hits += 1
            
            return entry
    
    # ========================================================================
    # VALIDATION METHODS
    # ========================================================================
    
    def validate(
        self,
        symbol: str,
        skip_false_positives: bool = True
    ) -> ValidationResult:
        """
        Validate a single symbol
        
        Args:
            symbol: Symbol to validate
            skip_false_positives: Skip known false positives
            
        Returns:
            ValidationResult with is_valid, asset_class, source
        """
        symbol = symbol.upper().strip()
        
        # Quick check: known false positives
        if skip_false_positives and symbol in FALSE_POSITIVE_SYMBOLS:
            self._stats["hits"] += 1
            return ValidationResult(
                symbol=symbol,
                is_valid=False,
                asset_class=AssetClass.UNKNOWN,
                source="false_positive_list"
            )
        
        # Check cache first
        cached = self._get_from_cache(symbol)
        if cached:
            self._stats["hits"] += 1
            return ValidationResult(
                symbol=cached.symbol,
                is_valid=cached.is_valid,
                asset_class=cached.asset_class,
                name=cached.name,
                exchange=cached.exchange,
                source="cache"
            )
        
        # Cache miss - lookup in database
        self._stats["misses"] += 1
        return self._lookup_in_database(symbol)
    
    def validate_batch(
        self,
        symbols: List[str],
        skip_false_positives: bool = True
    ) -> Dict[str, ValidationResult]:
        """
        Validate multiple symbols efficiently
        
        Args:
            symbols: List of symbols to validate
            skip_false_positives: Skip known false positives
            
        Returns:
            Dict mapping symbol to ValidationResult
        """
        if not symbols:
            return {}
        
        results = {}
        symbols_to_lookup = []
        
        # Check cache first for all symbols
        for symbol in symbols:
            symbol = symbol.upper().strip()
            
            # Skip false positives
            if skip_false_positives and symbol in FALSE_POSITIVE_SYMBOLS:
                results[symbol] = ValidationResult(
                    symbol=symbol,
                    is_valid=False,
                    asset_class=AssetClass.UNKNOWN,
                    source="false_positive_list"
                )
                continue
            
            # Check cache
            cached = self._get_from_cache(symbol)
            if cached:
                self._stats["hits"] += 1
                results[symbol] = ValidationResult(
                    symbol=cached.symbol,
                    is_valid=cached.is_valid,
                    asset_class=cached.asset_class,
                    name=cached.name,
                    exchange=cached.exchange,
                    source="cache"
                )
            else:
                symbols_to_lookup.append(symbol)
        
        # Batch lookup remaining symbols in database
        if symbols_to_lookup:
            db_results = self._batch_lookup_in_database(symbols_to_lookup)
            results.update(db_results)
        
        return results
    
    def validate_and_filter(
        self,
        symbols: List[str],
        asset_class: Optional[AssetClass] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Validate symbols and return valid/invalid lists
        
        Args:
            symbols: List of symbols to validate
            asset_class: Optional filter by asset class
            
        Returns:
            Tuple of (valid_symbols, invalid_symbols)
        """
        results = self.validate_batch(symbols)
        
        valid = []
        invalid = []
        
        for symbol, result in results.items():
            if result.is_valid:
                if asset_class is None or result.asset_class == asset_class:
                    valid.append(symbol)
                else:
                    invalid.append(symbol)
            else:
                invalid.append(symbol)
        
        return valid, invalid
    
    # ========================================================================
    # DATABASE LOOKUP
    # ========================================================================
    
    def _get_db_service(self):
        """Lazy load database service"""
        if self._db_service is None:
            try:
                from src.services.data_providers.fmp.symbol_directory_service import (
                    get_symbol_directory_service
                )
                self._db_service = get_symbol_directory_service()
            except Exception as e:
                self.logger.warning(f"Failed to load DB service: {e}")
        return self._db_service
    
    def _lookup_in_database(self, symbol: str) -> ValidationResult:
        """
        Lookup single symbol in database
        
        This is called on cache miss.
        """
        self._stats["db_lookups"] += 1
        
        try:
            service = self._get_db_service()
            if service is None:
                # Fallback: assume valid if can't check
                return ValidationResult(
                    symbol=symbol,
                    is_valid=True,  # Fail open
                    asset_class=AssetClass.UNKNOWN,
                    source="fallback"
                )
            
            result = service.validate_symbol(symbol)
            
            # Map service result to our types
            is_valid = result.is_valid
            asset_class = AssetClass.UNKNOWN
            if result.asset_class:
                asset_class = AssetClass(result.asset_class.value)
            
            # Add to cache
            self._add_to_cache(
                symbol=symbol,
                is_valid=is_valid,
                asset_class=asset_class,
                source="database"
            )
            
            return ValidationResult(
                symbol=symbol,
                is_valid=is_valid,
                asset_class=asset_class,
                source="database"
            )
            
        except Exception as e:
            self.logger.warning(f"DB lookup failed for {symbol}: {e}")
            # Fail open - assume valid
            return ValidationResult(
                symbol=symbol,
                is_valid=True,
                asset_class=AssetClass.UNKNOWN,
                source="error_fallback"
            )
    
    def _batch_lookup_in_database(
        self,
        symbols: List[str]
    ) -> Dict[str, ValidationResult]:
        """
        Batch lookup symbols in database
        
        More efficient than individual lookups.
        """
        self._stats["db_lookups"] += len(symbols)
        results = {}
        
        try:
            service = self._get_db_service()
            if service is None:
                # Fallback: assume all valid
                for symbol in symbols:
                    results[symbol] = ValidationResult(
                        symbol=symbol,
                        is_valid=True,
                        asset_class=AssetClass.UNKNOWN,
                        source="fallback"
                    )
                return results
            
            # Use repository directly for batch operation
            for symbol in symbols:
                db_result = service.repo.validate_symbol(symbol)
                
                is_valid = db_result["is_valid"]
                asset_class = AssetClass.UNKNOWN
                if db_result.get("asset_class"):
                    asset_class = AssetClass(db_result["asset_class"])
                
                # Add to cache
                self._add_to_cache(
                    symbol=symbol,
                    is_valid=is_valid,
                    asset_class=asset_class,
                    source="database"
                )
                
                results[symbol] = ValidationResult(
                    symbol=symbol,
                    is_valid=is_valid,
                    asset_class=asset_class,
                    source="database"
                )
            
        except Exception as e:
            self.logger.warning(f"Batch DB lookup failed: {e}")
            # Fail open for remaining symbols
            for symbol in symbols:
                if symbol not in results:
                    results[symbol] = ValidationResult(
                        symbol=symbol,
                        is_valid=True,
                        asset_class=AssetClass.UNKNOWN,
                        source="error_fallback"
                    )
        
        return results
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def is_likely_symbol(self, text: str) -> bool:
        """
        Quick check if text looks like a symbol
        
        Used for pre-filtering before validation.
        """
        text = text.upper().strip()
        
        # Length check: 1-10 chars
        if not 1 <= len(text) <= 10:
            return False
        
        # Must be uppercase alphanumeric (with optional . for BRK.A style)
        if not all(c.isalnum() or c in './-' for c in text):
            return False
        
        # Known false positives
        if text in FALSE_POSITIVE_SYMBOLS:
            return False
        
        return True
    
    def extract_potential_symbols(self, text: str) -> List[str]:
        """
        Extract potential symbols from text for validation
        
        More conservative than regex - uses pattern matching + filtering.
        """
        import re
        
        # Pattern: 1-5 uppercase letters, optionally followed by USD
        pattern = r'\b([A-Z]{1,5}(?:\.?[A-Z])?(?:USD)?)\b'
        
        matches = re.findall(pattern, text.upper())
        
        # Filter out false positives
        potential = [
            m for m in matches
            if m not in FALSE_POSITIVE_SYMBOLS and len(m) >= 2
        ]
        
        return list(set(potential))[:20]  # Limit to 20
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._cache_lock:
            hit_rate = 0
            total = self._stats["hits"] + self._stats["misses"]
            if total > 0:
                hit_rate = self._stats["hits"] / total * 100
            
            return {
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate_percent": round(hit_rate, 2),
                "db_lookups": self._stats["db_lookups"],
                "evictions": self._stats["evictions"],
                "preloaded": self._stats["preloaded"],
            }
    
    def clear_cache(self) -> int:
        """Clear entire cache"""
        with self._cache_lock:
            count = len(self._cache)
            self._cache.clear()
            self.logger.info(f"Cleared {count} entries from cache")
            return count
    
    def invalidate(self, symbol: str) -> bool:
        """Invalidate specific symbol in cache"""
        symbol = symbol.upper()
        with self._cache_lock:
            if symbol in self._cache:
                del self._cache[symbol]
                return True
            return False


# ============================================================================
# SINGLETON ACCESSOR
# ============================================================================

_validator_instance: Optional[SymbolValidationCache] = None
_validator_lock = threading.Lock()


def get_symbol_validator(
    ttl_seconds: int = SymbolValidationCache.DEFAULT_TTL_SECONDS,
    auto_preload: bool = True
) -> SymbolValidationCache:
    """
    Get singleton SymbolValidationCache instance
    
    Usage:
        validator = get_symbol_validator()
        result = validator.validate("AAPL")
    """
    global _validator_instance
    
    if _validator_instance is None:
        with _validator_lock:
            if _validator_instance is None:
                _validator_instance = SymbolValidationCache(
                    ttl_seconds=ttl_seconds,
                    auto_preload=auto_preload
                )
    
    return _validator_instance


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_symbols_from_planning(
    symbols: List[str],
    logger: Optional[Any] = None
) -> Tuple[List[str], List[str]]:
    """
    Validate symbols extracted by Planning Agent
    
    This is the main function to call after Planning Agent extracts symbols.
    
    Args:
        symbols: Symbols from Planning Agent
        logger: Optional logger
        
    Returns:
        Tuple of (valid_symbols, invalid_symbols)
    """
    if not symbols:
        return [], []
    
    validator = get_symbol_validator()
    valid, invalid = validator.validate_and_filter(symbols)
    
    if logger and invalid:
        logger.warning(f"[SYMBOL_VALIDATION] Invalid symbols filtered: {invalid}")
    
    return valid, invalid


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Demo usage
    validator = get_symbol_validator()
    
    print("=" * 60)
    print("SYMBOL VALIDATION CACHE DEMO")
    print("=" * 60)
    
    # Single validation
    result = validator.validate("AAPL")
    print(f"\nSingle: AAPL -> valid={result.is_valid}, class={result.asset_class}, source={result.source}")
    
    # Batch validation
    symbols = ["AAPL", "NVDA", "BTCUSD", "AI", "CEO", "INVALID123"]
    results = validator.validate_batch(symbols)
    
    print(f"\nBatch validation:")
    for sym, res in results.items():
        print(f"  {sym}: valid={res.is_valid}, class={res.asset_class}, source={res.source}")
    
    # Filter
    valid, invalid = validator.validate_and_filter(symbols)
    print(f"\nFiltered: valid={valid}, invalid={invalid}")
    
    # Stats
    stats = validator.get_stats()
    print(f"\nCache stats: {stats}")