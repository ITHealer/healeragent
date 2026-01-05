from enum import Enum
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime

if TYPE_CHECKING:
    from src.services.asset.symbol_resolution_models import (
        UIContext,
        ResolvedSymbol,
        SymbolResolutionResult,
    )


class QueryType(str, Enum):
    """Types of user queries"""
    STOCK_SPECIFIC = "stock_specific"      # Query about specific stock(s)
    CRYPTO_SPECIFIC = "crypto_specific"    # Query about specific crypto
    COMPARISON = "comparison"              # Comparing multiple assets (stock vs crypto, etc.)
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

    Now also includes symbol resolution information for Soft Context Inheritance.
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

    # Symbol Resolution (Soft Context Inheritance)
    # These are populated after classification when resolve_symbols=True
    resolved_symbols: Optional[List[Dict[str, Any]]] = None
    clarification_needed: bool = False
    clarification_messages: List[str] = field(default_factory=list)

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

        # Comparison queries always need tools from multiple categories
        if self.query_type == QueryType.COMPARISON:
            self.requires_tools = True
            # Ensure price category is present for comparisons
            if "price" not in self.tool_categories:
                self.tool_categories.append("price")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization"""
        result = {
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

        # Add symbol resolution info if present
        if self.resolved_symbols is not None:
            result["resolved_symbols"] = self.resolved_symbols
            result["clarification_needed"] = self.clarification_needed
            result["clarification_messages"] = self.clarification_messages

        return result

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
            resolved_symbols=data.get("resolved_symbols"),
            clarification_needed=data.get("clarification_needed", False),
            clarification_messages=data.get("clarification_messages", []),
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

    Now includes UI context for Soft Context Inheritance.
    Now supports multimodal input (images) for better intent classification.
    """

    # User query
    query: str

    # Conversation history (last N turns)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    # Memory context
    core_memory_summary: str = ""      # User profile, preferences
    working_memory_summary: str = ""   # Current session state, recent symbols

    # Token limits
    max_history_tokens: int = 2000

    # UI Context (Soft Context Inheritance)
    # Passed from frontend to enable context-aware symbol resolution
    ui_context: Optional[Dict[str, Any]] = None

    # Multimodal Input (Images)
    # List of ProcessedImage for vision-based classification
    images: Optional[List[Any]] = None

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

    def format_ui_context(self) -> str:
        """Format UI context for prompt (Soft Context Inheritance)"""
        if not self.ui_context:
            return ""

        active_tab = self.ui_context.get("active_tab", "none")
        if active_tab == "none":
            return ""

        parts = [f"<ui_context>"]
        parts.append(f"Active Tab: {active_tab}")

        recent_symbols = self.ui_context.get("recent_symbols", [])
        if recent_symbols:
            parts.append(f"Recently Viewed: {', '.join(recent_symbols[:5])}")

        parts.append("</ui_context>")
        return "\n".join(parts)

    def get_active_tab(self) -> str:
        """Get the active UI tab (for symbol resolution)"""
        if self.ui_context:
            return self.ui_context.get("active_tab", "none")
        return "none"

    def has_images(self) -> bool:
        """Check if context has images for multimodal classification."""
        return bool(self.images and len(self.images) > 0)

    def get_image_count(self) -> int:
        """Get number of images in context."""
        return len(self.images) if self.images else 0

    def format_image_context(self) -> str:
        """Format image context description for text-only prompts."""
        if not self.has_images():
            return ""

        parts = ["<image_context>"]
        parts.append(f"User has attached {self.get_image_count()} image(s).")
        parts.append("The image(s) may contain:")
        parts.append("- Financial charts or graphs")
        parts.append("- Stock/crypto screenshots")
        parts.append("- Documents or reports")
        parts.append("- Other visual content related to their query")
        parts.append("</image_context>")
        return "\n".join(parts)