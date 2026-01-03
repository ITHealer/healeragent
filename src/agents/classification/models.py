from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================================
# UI CONTEXT - Soft Context Inheritance from Frontend
# ============================================================================

class AssetContext(str, Enum):
    """
    Asset context from UI tab selection.

    Frontend sends this based on which tab user is currently viewing.
    Used for soft disambiguation of ambiguous symbols.
    """
    CRYPTO = "crypto"   # User is on Crypto tab
    STOCK = "stock"     # User is on Stock tab
    AUTO = "auto"       # No preference, use query analysis


@dataclass
class UIContext:
    """
    Context inherited from frontend UI state.

    This enables "Soft Context Inheritance" where:
    1. Chat inherits context from current tab (crypto/stock)
    2. Ambiguous symbols are resolved using this context
    3. User can override with explicit keywords
    4. Response shows what context was used

    Example:
        User on Crypto tab asks "giá BTC"
        → BTC resolved as Bitcoin (crypto) not BTC Digital (stock)
        → Response shows: "Đang xem: BTC (Crypto)" + alternative link
    """

    # Primary context from UI tab
    current_tab: AssetContext = AssetContext.AUTO

    # Recent symbols user viewed (for context reinforcement)
    recent_symbols: List[str] = field(default_factory=list)

    # Watchlist type currently displayed
    watchlist_type: Optional[str] = None

    # User's language preference from UI
    language: str = "vi"

    def __post_init__(self):
        """Normalize after initialization"""
        # Ensure current_tab is AssetContext enum
        if isinstance(self.current_tab, str):
            try:
                self.current_tab = AssetContext(self.current_tab.lower())
            except ValueError:
                self.current_tab = AssetContext.AUTO

        # Uppercase recent symbols
        self.recent_symbols = [s.upper() for s in self.recent_symbols]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "current_tab": self.current_tab.value,
            "recent_symbols": self.recent_symbols,
            "watchlist_type": self.watchlist_type,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "UIContext":
        """Create from dictionary (API request)"""
        if not data:
            return cls()

        return cls(
            current_tab=data.get("current_tab", "auto"),
            recent_symbols=data.get("recent_symbols", []),
            watchlist_type=data.get("watchlist_type"),
            language=data.get("language", "vi"),
        )

    def get_preferred_asset_class(self) -> Optional[str]:
        """Get preferred asset class from context"""
        if self.current_tab == AssetContext.CRYPTO:
            return "crypto"
        elif self.current_tab == AssetContext.STOCK:
            return "stock"
        return None


# ============================================================================
# CONTEXT RESOLUTION SOURCE - Track how symbol was resolved
# ============================================================================

class ContextSource(str, Enum):
    """How a symbol's asset type was determined"""
    EXPLICIT = "explicit"       # User said "stock BTC" or "crypto SOL"
    UI_TAB = "ui_tab"          # Inherited from current UI tab
    QUERY_HINT = "query_hint"   # Keywords in query (e.g., "cổ phiếu", "coin")
    UNAMBIGUOUS = "unambiguous" # Symbol only exists in one asset class
    DEFAULT = "default"         # No context, used default (crypto)


@dataclass
class SymbolResolution:
    """Resolution result for a single symbol"""
    symbol: str
    resolved_type: str              # "crypto" or "stock"
    context_source: ContextSource   # How it was resolved
    confidence: float = 1.0
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "resolved_type": self.resolved_type,
            "context_source": self.context_source.value,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
        }


# ============================================================================
# QUERY TYPE ENUM
# ============================================================================

class QueryType(str, Enum):
    """Types of user queries"""
    STOCK_SPECIFIC = "stock_specific"      # Query about specific stock(s)
    CRYPTO_SPECIFIC = "crypto_specific"    # Query about specific crypto
    SCREENER = "screener"                  # Finding stocks by criteria
    MARKET_LEVEL = "market_level"          # Market overview, indices
    CONVERSATIONAL = "conversational"      # Greetings, thanks, bye
    MEMORY_RECALL = "memory_recall"        # Past conversation questions
    GENERAL_KNOWLEDGE = "general_knowledge"  # Financial concepts, no tools needed
    REAL_TIME_INFO = "real_time_info"      # Current events, latest news, needs web search


class MarketType(str, Enum):
    """Market types for symbol classification"""
    STOCK = "stock"
    CRYPTO = "crypto"
    BOTH = "both"


# Valid tool categories
VALID_CATEGORIES = [
    "price",        # Stock prices, quotes, performance
    "technical",    # Technical indicators, chart patterns
    "fundamentals", # P/E, ROE, financial ratios
    "news",         # News, events, announcements
    "market",       # Market overview, indices, trending
    "risk",         # Risk assessment, volatility
    "crypto",       # Cryptocurrency
    "discovery",    # Stock screening
    "memory",       # Cross-session memory search
    "web",          # Web search for additional information
]


@dataclass
class UnifiedClassificationResult:
    """
    Result of unified classification.

    Combines what was previously:
    - Stage 1: Query Classification
    - Thinking: Tool Necessity Validation

    Into a single output from 1 LLM call.
    """

    # Classification (from Stage 1)
    query_type: QueryType
    symbols: List[str] = field(default_factory=list)
    tool_categories: List[str] = field(default_factory=list)
    market_type: Optional[MarketType] = None
    response_language: str = "en"

    # Validation (from Thinking layer - now merged)
    requires_tools: bool = True
    confidence: float = 0.9

    # Intent description
    intent_summary: str = ""

    # Debug/Logging
    reasoning: str = ""
    classification_method: str = "llm"  # llm | fallback

    # Metadata
    classified_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate and normalize after initialization"""
        # Validate categories
        self.tool_categories = [
            cat for cat in self.tool_categories
            if cat in VALID_CATEGORIES
        ]

        # Ensure symbols are uppercase
        self.symbols = [s.upper() for s in self.symbols]

        # Auto-add categories based on query type
        if self.query_type == QueryType.MEMORY_RECALL:
            if "memory" not in self.tool_categories:
                self.tool_categories.append("memory")

        # Conversational never needs tools
        if self.query_type == QueryType.CONVERSATIONAL:
            self.requires_tools = False
            self.tool_categories = []

        # General knowledge doesn't need tools
        if self.query_type == QueryType.GENERAL_KNOWLEDGE:
            self.requires_tools = False
            self.tool_categories = []

        # Real-time info always needs web search
        if self.query_type == QueryType.REAL_TIME_INFO:
            self.requires_tools = True
            if "web" not in self.tool_categories:
                self.tool_categories.append("web")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization"""
        return {
            "query_type": self.query_type.value if isinstance(self.query_type, QueryType) else self.query_type,
            "symbols": self.symbols,
            "tool_categories": self.tool_categories,
            "market_type": self.market_type.value if self.market_type else None,
            "response_language": self.response_language,
            "requires_tools": self.requires_tools,
            "confidence": self.confidence,
            "intent_summary": self.intent_summary,
            "reasoning": self.reasoning,
            "classification_method": self.classification_method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedClassificationResult":
        """Create from dictionary (LLM response)"""
        # Handle query_type
        query_type_str = data.get("query_type", "stock_specific")
        try:
            query_type = QueryType(query_type_str)
        except ValueError:
            query_type = QueryType.STOCK_SPECIFIC

        # Handle market_type
        market_type_str = data.get("market_type")
        market_type = None
        if market_type_str:
            try:
                market_type = MarketType(market_type_str)
            except ValueError:
                pass

        return cls(
            query_type=query_type,
            symbols=data.get("symbols", []),
            tool_categories=data.get("tool_categories", data.get("categories", [])),
            market_type=market_type,
            response_language=data.get("response_language", "en"),
            requires_tools=data.get("requires_tools", True),
            confidence=data.get("confidence", 0.9),
            intent_summary=data.get("intent_summary", data.get("final_intent", "")),
            reasoning=data.get("reasoning", ""),
        )

    @classmethod
    def fallback(cls, reason: str = "Classification failed") -> "UnifiedClassificationResult":
        """Create fallback result when classification fails"""
        return cls(
            query_type=QueryType.GENERAL_KNOWLEDGE,
            requires_tools=False,
            confidence=0.3,
            reasoning=f"Fallback: {reason}",
            classification_method="fallback",
        )


@dataclass
class ClassifierContext:
    """
    Context provided to the classifier.

    Assembled from various memory sources before classification.
    Now includes UI context for soft disambiguation.
    """

    # User query
    query: str

    # Conversation history (last N turns)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    # Memory context
    core_memory_summary: str = ""      # User profile, preferences
    working_memory_summary: str = ""   # Current session state, recent symbols

    # UI Context - Soft Context Inheritance
    ui_context: Optional[UIContext] = None

    # Token limits
    max_history_tokens: int = 2000

    def get_ui_context(self) -> UIContext:
        """Get UI context, creating default if not set"""
        return self.ui_context or UIContext()

    def format_ui_context(self) -> str:
        """Format UI context for classifier prompt"""
        ui = self.get_ui_context()

        lines = []
        if ui.current_tab != AssetContext.AUTO:
            lines.append(f"Current UI Tab: {ui.current_tab.value.upper()}")

        if ui.recent_symbols:
            lines.append(f"Recently Viewed: {', '.join(ui.recent_symbols[:5])}")

        if ui.watchlist_type:
            lines.append(f"Watchlist Type: {ui.watchlist_type}")

        return "\n".join(lines) if lines else ""

    def format_history(self, max_turns: int = 5) -> str:
        """Format conversation history for prompt"""
        if not self.conversation_history:
            return "No previous conversation."

        recent = self.conversation_history[-max_turns:]
        lines = []
        for i, msg in enumerate(recent):
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")[:300]  # Truncate long messages
            lines.append(f"[{role}]: {content}")

        return "\n".join(lines)

    def format_memory_context(self) -> str:
        """Format memory context for prompt"""
        parts = []

        if self.working_memory_summary:
            parts.append(f"<working_memory>\n{self.working_memory_summary}\n</working_memory>")

        if self.core_memory_summary:
            parts.append(f"<user_profile>\n{self.core_memory_summary}\n</user_profile>")

        return "\n\n".join(parts) if parts else ""