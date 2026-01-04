"""
Symbol Resolution Models

Models for the Multi-Asset Classification System with Soft Context Inheritance.

Principle: "Assume smartly, Confirm explicitly, Correct gracefully"

These models support:
1. Pattern-based detection (regex for exchange suffixes)
2. UI context inheritance (active tab context)
3. LLM semantic understanding (multilingual company names)
4. Resolution info with alternatives
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


class ActiveTab(str, Enum):
    """UI tab context from frontend"""
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    NONE = "none"  # No tab context (general chat)


class Exchange(str, Enum):
    """Supported exchanges"""
    # US
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    AMEX = "AMEX"

    # Asia
    HKEX = "HKEX"       # Hong Kong (.HK)
    SSE = "SSE"         # Shanghai (.SS)
    SZSE = "SZSE"       # Shenzhen (.SZ)
    TSE = "TSE"         # Tokyo (.T)

    # Europe
    LSE = "LSE"         # London (.L)
    XETRA = "XETRA"     # Germany (.DE)
    EURONEXT = "EURONEXT"  # Paris (.PA)

    # Vietnam
    HOSE = "HOSE"
    HNX = "HNX"
    UPCOM = "UPCOM"

    # Crypto
    BINANCE = "BINANCE"
    COINBASE = "COINBASE"

    # Default
    UNKNOWN = "UNKNOWN"


class ResolutionMethod(str, Enum):
    """How the symbol was resolved"""
    PATTERN = "pattern"       # Regex pattern matched (.HK, .SS, etc.)
    CACHE = "cache"           # Found in symbol cache
    LLM_SEMANTIC = "llm"      # LLM understood company name
    UI_CONTEXT = "ui_context" # Inferred from UI tab
    DEFAULT = "default"       # Default assumption (US for stock)
    USER_CONFIRMED = "user"   # User explicitly confirmed


class ConfidenceLevel(str, Enum):
    """Resolution confidence level"""
    HIGH = "high"         # 0.9+ : Pattern match or explicit mention
    MEDIUM = "medium"     # 0.7-0.9 : UI context + semantic match
    LOW = "low"           # <0.7 : Needs confirmation
    AMBIGUOUS = "ambiguous"  # Multiple valid interpretations


@dataclass
class UIContext:
    """
    Context passed from frontend UI.

    Enables "Soft Context Inheritance" - chat inherits context from current tab.
    """
    active_tab: ActiveTab = ActiveTab.NONE

    # Optional: recent symbols viewed in UI
    recent_symbols: List[str] = field(default_factory=list)

    # Optional: user's preferred exchange/market
    preferred_exchange: Optional[Exchange] = None
    preferred_quote_currency: str = "USD"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_tab": self.active_tab.value,
            "recent_symbols": self.recent_symbols,
            "preferred_exchange": self.preferred_exchange.value if self.preferred_exchange else None,
            "preferred_quote_currency": self.preferred_quote_currency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UIContext":
        if not data:
            return cls()

        active_tab = ActiveTab.NONE
        try:
            active_tab = ActiveTab(data.get("active_tab", "none"))
        except ValueError:
            pass

        preferred_exchange = None
        if data.get("preferred_exchange"):
            try:
                preferred_exchange = Exchange(data["preferred_exchange"])
            except ValueError:
                pass

        return cls(
            active_tab=active_tab,
            recent_symbols=data.get("recent_symbols", []),
            preferred_exchange=preferred_exchange,
            preferred_quote_currency=data.get("preferred_quote_currency", "USD"),
        )


@dataclass
class ResolutionInfo:
    """
    Information about how a symbol was resolved.

    Provides transparency for debugging and user feedback.
    """
    method: ResolutionMethod
    confidence: float  # 0.0 - 1.0
    confidence_level: ConfidenceLevel

    # What triggered this resolution
    pattern_matched: Optional[str] = None      # e.g., ".HK suffix"
    context_used: Optional[str] = None         # e.g., "Stock tab active"
    semantic_match: Optional[str] = None       # e.g., "腾讯 -> Tencent"

    # For debugging
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "pattern_matched": self.pattern_matched,
            "context_used": self.context_used,
            "semantic_match": self.semantic_match,
            "reasoning": self.reasoning,
        }

    @classmethod
    def high_confidence(
        cls,
        method: ResolutionMethod,
        reasoning: str = "",
        **kwargs
    ) -> "ResolutionInfo":
        """Create high confidence resolution"""
        return cls(
            method=method,
            confidence=0.95,
            confidence_level=ConfidenceLevel.HIGH,
            reasoning=reasoning,
            **kwargs
        )

    @classmethod
    def medium_confidence(
        cls,
        method: ResolutionMethod,
        reasoning: str = "",
        **kwargs
    ) -> "ResolutionInfo":
        """Create medium confidence resolution"""
        return cls(
            method=method,
            confidence=0.8,
            confidence_level=ConfidenceLevel.MEDIUM,
            reasoning=reasoning,
            **kwargs
        )

    @classmethod
    def low_confidence(
        cls,
        method: ResolutionMethod,
        reasoning: str = "",
        **kwargs
    ) -> "ResolutionInfo":
        """Create low confidence resolution (needs confirmation)"""
        return cls(
            method=method,
            confidence=0.6,
            confidence_level=ConfidenceLevel.LOW,
            reasoning=reasoning,
            **kwargs
        )


@dataclass
class AlternativeSymbol:
    """
    Alternative interpretation of a symbol.

    Shown when symbol is ambiguous (e.g., SOL = Solana or Renesola).
    """
    symbol: str
    name: str
    asset_type: str           # "crypto", "stock"
    exchange: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "asset_type": self.asset_type,
            "exchange": self.exchange,
            "description": self.description,
        }


@dataclass
class ResolvedSymbol:
    """
    Fully resolved symbol with all context.

    This is the output of the symbol resolution process.
    """
    # Core identity
    symbol: str               # Normalized symbol (e.g., "AAPL", "BTC")
    name: str                 # Full name (e.g., "Apple Inc", "Bitcoin")
    asset_type: str           # "stock", "crypto", "forex", "commodity"

    # Exchange/Market info
    exchange: Optional[Exchange] = None
    trading_pair: Optional[str] = None    # For crypto: "BTCUSDT"
    quote_currency: str = "USD"

    # Resolution details
    resolution_info: Optional[ResolutionInfo] = None

    # Ambiguity handling
    clarification_needed: bool = False
    alternatives: List[AlternativeSymbol] = field(default_factory=list)
    clarification_message: Optional[str] = None

    # Original text from query
    original_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "asset_type": self.asset_type,
            "exchange": self.exchange.value if self.exchange else None,
            "trading_pair": self.trading_pair,
            "quote_currency": self.quote_currency,
            "resolution_info": self.resolution_info.to_dict() if self.resolution_info else None,
            "clarification_needed": self.clarification_needed,
            "alternatives": [alt.to_dict() for alt in self.alternatives],
            "clarification_message": self.clarification_message,
            "original_text": self.original_text,
        }

    @classmethod
    def from_cache_lookup(
        cls,
        symbol: str,
        name: str,
        asset_type: str,
        exchange: Optional[str] = None,
    ) -> "ResolvedSymbol":
        """Create from symbol cache lookup result"""
        exchange_enum = None
        if exchange:
            try:
                exchange_enum = Exchange(exchange)
            except ValueError:
                pass

        return cls(
            symbol=symbol,
            name=name,
            asset_type=asset_type,
            exchange=exchange_enum,
            resolution_info=ResolutionInfo.high_confidence(
                method=ResolutionMethod.CACHE,
                reasoning=f"Found in symbol cache: {symbol}"
            ),
        )

    @classmethod
    def ambiguous(
        cls,
        symbol: str,
        alternatives: List[AlternativeSymbol],
        original_text: str = "",
    ) -> "ResolvedSymbol":
        """Create ambiguous result needing confirmation"""
        alt_names = ", ".join([f"{a.name} ({a.asset_type})" for a in alternatives[:3]])
        message = f"'{symbol}' có thể là: {alt_names}. Bạn muốn phân tích loại nào?"

        return cls(
            symbol=symbol,
            name=f"{symbol} (ambiguous)",
            asset_type="unknown",
            clarification_needed=True,
            alternatives=alternatives,
            clarification_message=message,
            original_text=original_text,
            resolution_info=ResolutionInfo(
                method=ResolutionMethod.DEFAULT,
                confidence=0.5,
                confidence_level=ConfidenceLevel.AMBIGUOUS,
                reasoning=f"Multiple interpretations: {alt_names}"
            ),
        )


@dataclass
class SymbolResolutionResult:
    """
    Complete result from symbol resolution.

    Contains all resolved symbols and any unresolved entities.
    """
    # Successfully resolved symbols
    resolved_symbols: List[ResolvedSymbol] = field(default_factory=list)

    # Symbols that need clarification
    needs_clarification: List[ResolvedSymbol] = field(default_factory=list)

    # Entities that couldn't be resolved
    unresolved: List[str] = field(default_factory=list)

    # Overall confidence
    overall_confidence: float = 1.0

    # Any clarification messages
    clarification_messages: List[str] = field(default_factory=list)

    def has_ambiguity(self) -> bool:
        """Check if any symbols need clarification"""
        return len(self.needs_clarification) > 0

    def get_all_symbols(self) -> List[str]:
        """Get all resolved symbols (excluding ambiguous)"""
        return [rs.symbol for rs in self.resolved_symbols]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolved_symbols": [rs.to_dict() for rs in self.resolved_symbols],
            "needs_clarification": [rs.to_dict() for rs in self.needs_clarification],
            "unresolved": self.unresolved,
            "overall_confidence": self.overall_confidence,
            "clarification_messages": self.clarification_messages,
            "has_ambiguity": self.has_ambiguity(),
        }


# ============================================================================
# Pattern Detection Constants
# ============================================================================

# Exchange suffix patterns (for pattern-based detection)
EXCHANGE_SUFFIX_PATTERNS = {
    # Hong Kong
    r"\.HK$": Exchange.HKEX,
    r"-HK$": Exchange.HKEX,

    # China
    r"\.SS$": Exchange.SSE,      # Shanghai
    r"\.SH$": Exchange.SSE,      # Shanghai alt
    r"\.SZ$": Exchange.SZSE,     # Shenzhen

    # Japan
    r"\.T$": Exchange.TSE,
    r"\.JP$": Exchange.TSE,

    # UK
    r"\.L$": Exchange.LSE,
    r"\.LON$": Exchange.LSE,

    # Germany
    r"\.DE$": Exchange.XETRA,
    r"\.GER$": Exchange.XETRA,

    # France
    r"\.PA$": Exchange.EURONEXT,
    r"\.PAR$": Exchange.EURONEXT,
}

# Crypto trading pair patterns
CRYPTO_PAIR_PATTERNS = [
    r"^([A-Z]{2,10})[-/]?(USDT?)$",   # BTC-USD, BTCUSDT, BTC/USD
    r"^([A-Z]{2,10})[-/]?(USDC)$",    # BTC-USDC, BTCUSDC
    r"^([A-Z]{2,10})[-/]?(EUR)$",     # BTC-EUR
    r"^([A-Z]{2,10})[-/]?(BTC)$",     # ETH-BTC, ETHBTC
]

# Default exchange by region
DEFAULT_EXCHANGES = {
    "US": Exchange.NYSE,
    "HK": Exchange.HKEX,
    "CN": Exchange.SSE,
    "JP": Exchange.TSE,
    "UK": Exchange.LSE,
    "DE": Exchange.XETRA,
    "VN": Exchange.HOSE,
}

# ============================================================================
# Bitcoin and Crypto ETF Mappings
# ============================================================================

# Bitcoin ETF symbols - map "cổ phiếu Bitcoin" / "Bitcoin stock" to these
BITCOIN_ETF_SYMBOLS = [
    "IBIT",   # iShares Bitcoin Trust (BlackRock) - largest
    "GBTC",   # Grayscale Bitcoin Trust
    "FBTC",   # Fidelity Wise Origin Bitcoin Fund
    "ARKB",   # ARK 21Shares Bitcoin ETF
    "BTCO",   # Invesco Galaxy Bitcoin ETF
    "BITO",   # ProShares Bitcoin Strategy ETF (futures-based)
    "HODL",   # VanEck Bitcoin Trust
    "BITB",   # Bitwise Bitcoin ETF
]

# Ethereum ETF symbols - map "cổ phiếu Ethereum" / "Ethereum stock" to these
ETHEREUM_ETF_SYMBOLS = [
    "ETHE",   # Grayscale Ethereum Trust
    "ETHA",   # iShares Ethereum Trust (BlackRock)
]

# Primary symbol for each crypto type (when comparing "crypto stock" vs "real crypto")
CRYPTO_ETF_PRIMARY = {
    "BTC": "IBIT",    # Most popular Bitcoin ETF
    "BITCOIN": "IBIT",
    "ETH": "ETHA",    # Most popular Ethereum ETF
    "ETHEREUM": "ETHA",
}

# Keywords that indicate "Bitcoin/crypto as stock/ETF" (not actual crypto)
CRYPTO_STOCK_KEYWORDS = [
    # Vietnamese
    "cổ phiếu bitcoin", "cổ phiếu btc", "cổ phiếu ethereum", "cổ phiếu eth",
    "quỹ bitcoin", "etf bitcoin", "etf ethereum",
    # English
    "bitcoin stock", "bitcoin etf", "bitcoin trust", "bitcoin fund",
    "ethereum stock", "ethereum etf", "ethereum trust", "ethereum fund",
    "btc etf", "eth etf", "gbtc", "ibit", "fbtc",
]

# Keywords that indicate "actual crypto" (not stock/ETF)
REAL_CRYPTO_KEYWORDS = [
    # Vietnamese
    "bitcoin thật", "btc thật", "bitcoin thực", "tiền điện tử",
    "ethereum thật", "eth thật", "coin", "token",
    # English
    "real bitcoin", "actual bitcoin", "bitcoin crypto", "btc crypto",
    "real ethereum", "actual ethereum", "cryptocurrency",
]

# ============================================================================
# Ambiguous Symbol Mappings
# ============================================================================

# Symbols that can be both stock and crypto
AMBIGUOUS_SYMBOLS = {
    "BTC": {
        "crypto": {"symbol": "BTC", "name": "Bitcoin", "trading_pair": "BTCUSD"},
        "stock": {"symbol": "BTC", "name": "Grayscale Bitcoin Mini Trust", "exchange": "NYSE"},
    },
    "SOL": {
        "crypto": {"symbol": "SOL", "name": "Solana", "trading_pair": "SOLUSD"},
        "stock": {"symbol": "SOL", "name": "Renesola Ltd", "exchange": "NYSE"},
    },
    "COMP": {
        "crypto": {"symbol": "COMP", "name": "Compound", "trading_pair": "COMPUSD"},
        "stock": {"symbol": "COMP", "name": "Compass Inc", "exchange": "NYSE"},
    },
    "COIN": {
        "crypto": None,  # Not a crypto symbol
        "stock": {"symbol": "COIN", "name": "Coinbase Global Inc", "exchange": "NASDAQ"},
    },
    "MARA": {
        "crypto": None,
        "stock": {"symbol": "MARA", "name": "Marathon Digital Holdings", "exchange": "NASDAQ"},
    },
    "RIOT": {
        "crypto": None,
        "stock": {"symbol": "RIOT", "name": "Riot Platforms Inc", "exchange": "NASDAQ"},
    },
}
