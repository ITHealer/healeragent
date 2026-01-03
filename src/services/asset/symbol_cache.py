from typing import Optional, Set, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timedelta

from src.utils.logger.custom_logging import LoggerMixin


class AssetClass(str, Enum):
    """Asset class enumeration"""
    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"
    COMMODITY = "commodity"
    UNKNOWN = "unknown"


class AssetSubClass(str, Enum):
    """Asset sub-class for more specific categorization"""
    # Crypto
    CRYPTO_LAYER1 = "crypto_layer1"          # BTC, ETH, SOL
    CRYPTO_LAYER2 = "crypto_layer2"          # ARB, OP
    CRYPTO_DEFI = "crypto_defi"              # UNI, AAVE
    CRYPTO_MEME = "crypto_meme"              # DOGE, SHIB
    CRYPTO_STABLECOIN = "crypto_stablecoin"  # USDT, USDC
    CRYPTO_OTHER = "crypto_other"            # Default crypto

    # Stock
    STOCK_US = "stock_us"
    STOCK_VN = "stock_vn"
    STOCK_HK = "stock_hk"
    STOCK_OTHER = "stock_other"

    # Forex
    FOREX_MAJOR = "forex_major"
    FOREX_MINOR = "forex_minor"

    # Commodity
    COMMODITY_PRECIOUS = "commodity_precious"
    COMMODITY_ENERGY = "commodity_energy"

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class SymbolInfo:
    """Detailed symbol information"""
    symbol: str
    name: str
    asset_class: AssetClass
    sub_class: AssetSubClass = AssetSubClass.UNKNOWN

    # For crypto: quote currencies available
    quote_currencies: List[str] = field(default_factory=lambda: ["USDT", "USD"])
    default_quote: str = "USDT"

    # For stocks: exchange info
    exchange: Optional[str] = None
    currency: str = "USD"

    # Data sources
    data_sources: List[str] = field(default_factory=list)

    # Is this symbol ambiguous (exists in multiple asset classes)?
    is_ambiguous: bool = False
    ambiguous_alternatives: List[str] = field(default_factory=list)


@dataclass
class QuoteCurrencyConfig:
    """Configuration for quote currency handling"""

    # Default quote currencies by region/preference
    default_crypto_quote: str = "USDT"
    fallback_crypto_quote: str = "USD"

    # Stablecoin peg thresholds
    usdt_peg_min: float = 0.99
    usdt_peg_max: float = 1.01

    # Trading pair formats by exchange
    exchange_formats: Dict[str, str] = field(default_factory=lambda: {
        "binance": "{base}{quote}",      # BTCUSDT
        "coinbase": "{base}-{quote}",    # BTC-USD
        "kraken": "{base}/{quote}",      # BTC/USD
        "coingecko": "{base}",           # Just base, quote in params
        "coinmarketcap": "{base}",       # Just base
    })


class SymbolCacheService(LoggerMixin):
    """
    High-performance symbol lookup with multi-layer caching.

    Design Goals:
    1. O(1) lookup - no DB calls during normal operation
    2. Memory efficient - ~5MB for 60,000 symbols
    3. Auto-refresh - background job every 24h
    4. Ambiguity detection - pre-computed set

    Cache Layers:
    1. In-Memory (fastest) - Python dict/set
    2. Redis (shared) - for multi-instance deployment
    3. PostgreSQL (source) - only on cache miss/refresh
    """

    _instance: Optional['SymbolCacheService'] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        """Singleton pattern - one cache instance per process"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        redis_client=None,
        db_session_factory=None,
        quote_config: Optional[QuoteCurrencyConfig] = None
    ):
        if self._initialized:
            return

        super().__init__()

        self.redis = redis_client
        self.db_factory = db_session_factory
        self.quote_config = quote_config or QuoteCurrencyConfig()

        # ========================================
        # IN-MEMORY CACHE (Layer 1)
        # ========================================
        # Fast O(1) sets for existence check
        self._crypto_symbols: Set[str] = set()
        self._stock_symbols: Set[str] = set()
        self._forex_symbols: Set[str] = set()
        self._commodity_symbols: Set[str] = set()

        # Pre-computed ambiguous symbols (exist in multiple classes)
        self._ambiguous_symbols: Set[str] = set()

        # Detailed info cache
        self._symbol_info: Dict[str, SymbolInfo] = {}

        # Crypto symbol to base mapping (BTCUSD -> BTC)
        self._crypto_base_map: Dict[str, str] = {}

        # Cache metadata
        self._last_refresh: Optional[datetime] = None
        self._cache_ttl: timedelta = timedelta(hours=24)

        self._initialized = True
        self.logger.info("SymbolCacheService initialized (singleton)")

    # ========================================
    # PUBLIC API
    # ========================================

    async def initialize(self) -> None:
        """
        Load all symbols into memory cache.
        Call this once on application startup.
        """
        self.logger.info("[SYMBOL_CACHE] Initializing from database...")

        try:
            await self._load_from_database()
            self._compute_ambiguous_symbols()
            self._last_refresh = datetime.utcnow()

            self.logger.info(
                f"[SYMBOL_CACHE] Initialized: "
                f"crypto={len(self._crypto_symbols)}, "
                f"stock={len(self._stock_symbols)}, "
                f"ambiguous={len(self._ambiguous_symbols)}"
            )
        except Exception as e:
            self.logger.error(f"[SYMBOL_CACHE] Failed to initialize: {e}")
            # Load defaults for basic functionality
            self._load_default_symbols()
            self._compute_ambiguous_symbols()
            self._last_refresh = datetime.utcnow()

    def lookup(self, symbol: str) -> Tuple[Optional[SymbolInfo], bool]:
        """
        Fast O(1) symbol lookup.

        Args:
            symbol: Symbol to lookup (e.g., "BTC", "AAPL", "SOL")

        Returns:
            Tuple of (SymbolInfo or None, is_ambiguous)

        Performance: ~0.001ms (in-memory dict lookup)
        """
        symbol_upper = symbol.upper().strip()

        # Remove common suffixes for lookup (BTCUSD -> BTC, BTCUSDT -> BTC)
        base_symbol = self._normalize_symbol(symbol_upper)

        # Check if ambiguous first
        is_ambiguous = base_symbol in self._ambiguous_symbols

        # Get detailed info
        info = self._symbol_info.get(base_symbol)

        # If not found directly, try base from crypto mapping
        if info is None and symbol_upper in self._crypto_base_map:
            base = self._crypto_base_map[symbol_upper]
            info = self._symbol_info.get(base)

        return info, is_ambiguous

    def exists(self, symbol: str, asset_class: Optional[AssetClass] = None) -> bool:
        """
        Fast O(1) existence check.

        Args:
            symbol: Symbol to check
            asset_class: Optional filter by asset class

        Returns:
            True if symbol exists

        Performance: ~0.0005ms (set membership)
        """
        symbol_upper = self._normalize_symbol(symbol.upper().strip())

        if asset_class is None:
            # Check all sets
            return (
                symbol_upper in self._crypto_symbols or
                symbol_upper in self._stock_symbols or
                symbol_upper in self._forex_symbols or
                symbol_upper in self._commodity_symbols
            )

        # Check specific set
        symbol_sets = {
            AssetClass.CRYPTO: self._crypto_symbols,
            AssetClass.STOCK: self._stock_symbols,
            AssetClass.FOREX: self._forex_symbols,
            AssetClass.COMMODITY: self._commodity_symbols,
        }

        return symbol_upper in symbol_sets.get(asset_class, set())

    def is_ambiguous(self, symbol: str) -> bool:
        """Check if symbol exists in multiple asset classes"""
        return self._normalize_symbol(symbol.upper().strip()) in self._ambiguous_symbols

    def get_ambiguous_options(self, symbol: str) -> List[SymbolInfo]:
        """
        Get all possible interpretations of an ambiguous symbol.

        Args:
            symbol: Ambiguous symbol (e.g., "SOL")

        Returns:
            List of SymbolInfo for each possible interpretation
        """
        symbol_upper = self._normalize_symbol(symbol.upper().strip())

        if symbol_upper not in self._ambiguous_symbols:
            return []

        options = []

        # Check each asset class
        if symbol_upper in self._crypto_symbols:
            crypto_info = self._get_crypto_info(symbol_upper)
            if crypto_info:
                options.append(crypto_info)

        if symbol_upper in self._stock_symbols:
            stock_info = self._get_stock_info(symbol_upper)
            if stock_info:
                options.append(stock_info)

        return options

    def get_trading_pair(
        self,
        base_symbol: str,
        exchange: str = "binance",
        quote: Optional[str] = None
    ) -> str:
        """
        Get formatted trading pair for API calls.

        Args:
            base_symbol: Base asset (e.g., "BTC")
            exchange: Target exchange
            quote: Quote currency (default from config)

        Returns:
            Formatted trading pair (e.g., "BTCUSDT" for Binance)

        Example:
            get_trading_pair("BTC", "binance") → "BTCUSDT"
            get_trading_pair("BTC", "coinbase") → "BTC-USD"
        """
        quote = quote or self.quote_config.default_crypto_quote

        format_template = self.quote_config.exchange_formats.get(
            exchange.lower(),
            "{base}{quote}"
        )

        return format_template.format(
            base=base_symbol.upper(),
            quote=quote.upper()
        )

    def get_all_crypto_symbols(self) -> Set[str]:
        """Get all crypto symbols"""
        return self._crypto_symbols.copy()

    def get_all_stock_symbols(self) -> Set[str]:
        """Get all stock symbols"""
        return self._stock_symbols.copy()

    # ========================================
    # CACHE MANAGEMENT
    # ========================================

    async def refresh_if_needed(self) -> bool:
        """
        Check if cache needs refresh and refresh if needed.
        Called periodically by background task.

        Returns:
            True if cache was refreshed
        """
        if self._last_refresh is None:
            await self.initialize()
            return True

        age = datetime.utcnow() - self._last_refresh
        if age > self._cache_ttl:
            self.logger.info(f"[SYMBOL_CACHE] Cache expired (age={age}), refreshing...")
            await self.initialize()
            return True

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "crypto_count": len(self._crypto_symbols),
            "stock_count": len(self._stock_symbols),
            "forex_count": len(self._forex_symbols),
            "commodity_count": len(self._commodity_symbols),
            "ambiguous_count": len(self._ambiguous_symbols),
            "ambiguous_symbols": list(self._ambiguous_symbols)[:20],
            "total_detailed_info": len(self._symbol_info),
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "cache_ttl_hours": self._cache_ttl.total_seconds() / 3600,
        }

    def add_symbol(
        self,
        symbol: str,
        name: str,
        asset_class: AssetClass,
        sub_class: Optional[AssetSubClass] = None,
        exchange: Optional[str] = None,
    ) -> None:
        """Manually add a symbol to cache"""
        symbol_upper = symbol.upper().strip()

        # Add to appropriate set
        if asset_class == AssetClass.CRYPTO:
            self._crypto_symbols.add(symbol_upper)
        elif asset_class == AssetClass.STOCK:
            self._stock_symbols.add(symbol_upper)
        elif asset_class == AssetClass.FOREX:
            self._forex_symbols.add(symbol_upper)
        elif asset_class == AssetClass.COMMODITY:
            self._commodity_symbols.add(symbol_upper)

        # Add detailed info
        self._symbol_info[symbol_upper] = SymbolInfo(
            symbol=symbol_upper,
            name=name,
            asset_class=asset_class,
            sub_class=sub_class or self._detect_sub_class(symbol_upper, asset_class),
            exchange=exchange,
        )

        # Recompute ambiguous
        self._compute_ambiguous_symbols()

    # ========================================
    # PRIVATE METHODS
    # ========================================

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol by removing common suffixes"""
        # Remove USD/USDT suffix for crypto
        for suffix in ["USDT", "USD", "BUSD", "USDC"]:
            if symbol.endswith(suffix) and len(symbol) > len(suffix):
                return symbol[:-len(suffix)]
        return symbol

    async def _load_from_database(self) -> None:
        """Load all symbols from PostgreSQL into memory"""

        try:
            from src.database.models.symbol_directory import Stock, Cryptocurrency
            from sqlalchemy import select
            from src.database import get_postgres_db

            db = get_postgres_db().get_session()
            try:
                # Load all stocks
                stmt = select(Stock.symbol, Stock.name, Stock.exchange)
                result = db.execute(stmt)
                stock_count = 0
                for row in result:
                    base_symbol = row.symbol.upper()
                    self._stock_symbols.add(base_symbol)
                    self._symbol_info[base_symbol] = SymbolInfo(
                        symbol=base_symbol,
                        name=row.name or base_symbol,
                        asset_class=AssetClass.STOCK,
                        sub_class=self._detect_stock_subclass(row.exchange),
                        exchange=row.exchange,
                        data_sources=["fmp"]
                    )
                    stock_count += 1

                # Load all cryptos
                stmt = select(Cryptocurrency.symbol, Cryptocurrency.name)
                result = db.execute(stmt)
                crypto_count = 0
                for row in result:
                    full_symbol = row.symbol.upper()  # e.g., BTCUSD
                    base_symbol = self._normalize_symbol(full_symbol)  # e.g., BTC

                    self._crypto_symbols.add(base_symbol)
                    self._crypto_base_map[full_symbol] = base_symbol

                    # Only add if not already present (avoid duplicates)
                    if base_symbol not in self._symbol_info:
                        self._symbol_info[base_symbol] = SymbolInfo(
                            symbol=base_symbol,
                            name=row.name or base_symbol,
                            asset_class=AssetClass.CRYPTO,
                            sub_class=self._detect_crypto_subclass(base_symbol),
                            quote_currencies=["USDT", "USD", "USDC"],
                            default_quote="USDT",
                            data_sources=["binance", "coingecko"]
                        )
                    crypto_count += 1

                self.logger.info(
                    f"[SYMBOL_CACHE] Loaded from DB: {stock_count} stocks, {crypto_count} cryptos"
                )
            finally:
                db.close()

        except Exception as e:
            self.logger.error(f"[SYMBOL_CACHE] DB load failed: {e}")
            self._load_default_symbols()

    def _load_default_symbols(self) -> None:
        """Load default symbols for testing/fallback"""

        # Popular cryptos
        crypto_defaults = {
            "BTC": ("Bitcoin", AssetSubClass.CRYPTO_LAYER1),
            "ETH": ("Ethereum", AssetSubClass.CRYPTO_LAYER1),
            "SOL": ("Solana", AssetSubClass.CRYPTO_LAYER1),
            "BNB": ("Binance Coin", AssetSubClass.CRYPTO_LAYER1),
            "XRP": ("Ripple", AssetSubClass.CRYPTO_LAYER1),
            "ADA": ("Cardano", AssetSubClass.CRYPTO_LAYER1),
            "AVAX": ("Avalanche", AssetSubClass.CRYPTO_LAYER1),
            "DOT": ("Polkadot", AssetSubClass.CRYPTO_LAYER1),
            "MATIC": ("Polygon", AssetSubClass.CRYPTO_LAYER2),
            "ARB": ("Arbitrum", AssetSubClass.CRYPTO_LAYER2),
            "OP": ("Optimism", AssetSubClass.CRYPTO_LAYER2),
            "DOGE": ("Dogecoin", AssetSubClass.CRYPTO_MEME),
            "SHIB": ("Shiba Inu", AssetSubClass.CRYPTO_MEME),
            "PEPE": ("Pepe", AssetSubClass.CRYPTO_MEME),
            "UNI": ("Uniswap", AssetSubClass.CRYPTO_DEFI),
            "AAVE": ("Aave", AssetSubClass.CRYPTO_DEFI),
            "LINK": ("Chainlink", AssetSubClass.CRYPTO_DEFI),
            "USDT": ("Tether", AssetSubClass.CRYPTO_STABLECOIN),
            "USDC": ("USD Coin", AssetSubClass.CRYPTO_STABLECOIN),
        }

        for symbol, (name, sub_class) in crypto_defaults.items():
            self._crypto_symbols.add(symbol)
            self._symbol_info[symbol] = SymbolInfo(
                symbol=symbol,
                name=name,
                asset_class=AssetClass.CRYPTO,
                sub_class=sub_class,
                data_sources=["binance", "coingecko"]
            )

        # Popular stocks (including ambiguous ones)
        stock_defaults = {
            "AAPL": ("Apple Inc", "NASDAQ"),
            "TSLA": ("Tesla Inc", "NASDAQ"),
            "NVDA": ("NVIDIA Corp", "NASDAQ"),
            "MSFT": ("Microsoft Corp", "NASDAQ"),
            "GOOGL": ("Alphabet Inc", "NASDAQ"),
            "AMZN": ("Amazon.com Inc", "NASDAQ"),
            "META": ("Meta Platforms Inc", "NASDAQ"),
            "AMD": ("Advanced Micro Devices", "NASDAQ"),
            "SOL": ("Renesola Ltd", "NYSE"),       # AMBIGUOUS with Solana!
            "COMP": ("Compass Inc", "NYSE"),        # AMBIGUOUS with Compound!
            "LINK": ("Interlink Electronics", "NASDAQ"),  # AMBIGUOUS with Chainlink!
        }

        for symbol, (name, exchange) in stock_defaults.items():
            self._stock_symbols.add(symbol)
            # Don't overwrite crypto info for ambiguous symbols
            if symbol not in self._symbol_info:
                self._symbol_info[symbol] = SymbolInfo(
                    symbol=symbol,
                    name=name,
                    asset_class=AssetClass.STOCK,
                    sub_class=AssetSubClass.STOCK_US,
                    exchange=exchange,
                    data_sources=["fmp"]
                )

    def _compute_ambiguous_symbols(self) -> None:
        """Find symbols that exist in multiple asset classes"""

        # Find overlaps between sets
        crypto_stock_overlap = self._crypto_symbols & self._stock_symbols
        crypto_forex_overlap = self._crypto_symbols & self._forex_symbols
        stock_forex_overlap = self._stock_symbols & self._forex_symbols

        self._ambiguous_symbols = (
            crypto_stock_overlap |
            crypto_forex_overlap |
            stock_forex_overlap
        )

        # Update symbol info with ambiguity flag
        for symbol in self._ambiguous_symbols:
            if symbol in self._symbol_info:
                self._symbol_info[symbol].is_ambiguous = True

        if self._ambiguous_symbols:
            self.logger.info(
                f"[SYMBOL_CACHE] Found {len(self._ambiguous_symbols)} ambiguous symbols: "
                f"{list(self._ambiguous_symbols)[:10]}..."
            )

    def _detect_sub_class(self, symbol: str, asset_class: AssetClass) -> AssetSubClass:
        """Detect sub-class based on symbol and asset class"""
        if asset_class == AssetClass.CRYPTO:
            return self._detect_crypto_subclass(symbol)
        elif asset_class == AssetClass.STOCK:
            return AssetSubClass.STOCK_US
        return AssetSubClass.UNKNOWN

    def _detect_crypto_subclass(self, symbol: str) -> AssetSubClass:
        """Detect crypto sub-class from symbol"""

        layer1 = {"BTC", "ETH", "SOL", "AVAX", "ADA", "DOT", "ATOM", "NEAR", "SUI", "APT"}
        layer2 = {"ARB", "OP", "MATIC", "IMX", "STRK", "ZK"}
        defi = {"UNI", "AAVE", "CRV", "COMP", "MKR", "SNX", "LINK", "GRT"}
        meme = {"DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF"}
        stablecoin = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "FRAX"}

        symbol_upper = symbol.upper()

        if symbol_upper in layer1:
            return AssetSubClass.CRYPTO_LAYER1
        elif symbol_upper in layer2:
            return AssetSubClass.CRYPTO_LAYER2
        elif symbol_upper in defi:
            return AssetSubClass.CRYPTO_DEFI
        elif symbol_upper in meme:
            return AssetSubClass.CRYPTO_MEME
        elif symbol_upper in stablecoin:
            return AssetSubClass.CRYPTO_STABLECOIN
        else:
            return AssetSubClass.CRYPTO_OTHER

    def _detect_stock_subclass(self, exchange: Optional[str]) -> AssetSubClass:
        """Detect stock sub-class from exchange"""
        if not exchange:
            return AssetSubClass.STOCK_US

        exchange_lower = exchange.lower()
        if any(x in exchange_lower for x in ["nyse", "nasdaq", "amex", "us"]):
            return AssetSubClass.STOCK_US
        elif any(x in exchange_lower for x in ["hose", "hnx", "upcom", "vn"]):
            return AssetSubClass.STOCK_VN
        elif any(x in exchange_lower for x in ["hkex", "hk"]):
            return AssetSubClass.STOCK_HK
        else:
            return AssetSubClass.STOCK_OTHER

    def _get_crypto_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get crypto-specific info for ambiguous symbol"""
        if symbol not in self._crypto_symbols:
            return None

        # Check if we have detailed info
        info = self._symbol_info.get(symbol)
        if info and info.asset_class == AssetClass.CRYPTO:
            return SymbolInfo(
                symbol=info.symbol,
                name=info.name,
                asset_class=AssetClass.CRYPTO,
                sub_class=info.sub_class,
                is_ambiguous=True,
                quote_currencies=["USDT", "USD"],
                default_quote="USDT",
                data_sources=["binance", "coingecko"]
            )

        # Create crypto-specific info
        return SymbolInfo(
            symbol=symbol,
            name=f"{symbol} (Crypto)",
            asset_class=AssetClass.CRYPTO,
            sub_class=self._detect_crypto_subclass(symbol),
            is_ambiguous=True,
            data_sources=["binance", "coingecko"]
        )

    def _get_stock_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get stock-specific info for ambiguous symbol"""
        if symbol not in self._stock_symbols:
            return None

        # Check if we have detailed info
        info = self._symbol_info.get(symbol)
        if info and info.asset_class == AssetClass.STOCK:
            return SymbolInfo(
                symbol=info.symbol,
                name=info.name,
                asset_class=AssetClass.STOCK,
                sub_class=info.sub_class,
                exchange=info.exchange,
                is_ambiguous=True,
                data_sources=["fmp"]
            )

        # Create stock-specific info
        return SymbolInfo(
            symbol=symbol,
            name=f"{symbol} (Stock)",
            asset_class=AssetClass.STOCK,
            sub_class=AssetSubClass.STOCK_US,
            is_ambiguous=True,
            data_sources=["fmp"]
        )


# ========================================
# SINGLETON ACCESSOR
# ========================================

_symbol_cache: Optional[SymbolCacheService] = None


def get_symbol_cache() -> SymbolCacheService:
    """Get the singleton symbol cache instance"""
    global _symbol_cache
    if _symbol_cache is None:
        _symbol_cache = SymbolCacheService()
    return _symbol_cache


async def initialize_symbol_cache(
    redis_client=None,
    db_session_factory=None
) -> SymbolCacheService:
    """Initialize and return the symbol cache"""
    global _symbol_cache
    _symbol_cache = SymbolCacheService(
        redis_client=redis_client,
        db_session_factory=db_session_factory
    )
    await _symbol_cache.initialize()
    return _symbol_cache