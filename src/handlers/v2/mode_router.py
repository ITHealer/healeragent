"""
Mode Router - Decides between NORMAL and DEEP_RESEARCH modes.

================================================================================
LEGACY NOTICE: This module is ONLY used by the legacy /chat endpoint.
               The new /chat/v2 endpoint does NOT use ModeRouter.

For /chat/v2:
- IntentClassifier determines complexity (direct vs agent_loop)
- No separate DEEP_RESEARCH routing - agent loop handles all queries
- Deep research functionality can be triggered via query keywords directly

See: docs/ARCHITECTURE_CHAT_V2.md for the new architecture.
================================================================================
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from src.utils.logger.custom_logging import LoggerMixin
from src.agents.classification.models import UnifiedClassificationResult, QueryType


class QueryMode(str, Enum):
    """Query processing modes."""
    NORMAL = "normal"
    DEEP_RESEARCH = "deep_research"
    AUTO = "auto"


@dataclass
class ModeDecision:
    """Mode routing decision result."""
    mode: QueryMode
    reason: str
    confidence: float
    detection_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# Keywords triggering Deep Research mode
DEEP_RESEARCH_KEYWORDS = {
    # English
    "comprehensive", "in-depth", "detailed analysis", "full analysis",
    "compare", "comparison", "versus", "vs",
    "portfolio", "portfolio analysis", "holdings",
    "research report", "investment thesis", "due diligence",
    "sector analysis", "industry comparison",
    "valuation", "dcf analysis", "intrinsic value",
    "risk assessment", "risk analysis",
    # Vietnamese
    "phân tích toàn diện", "phân tích chi tiết", "phân tích sâu",
    "so sánh", "đối chiếu",
    "danh mục", "danh mục đầu tư",
    "báo cáo nghiên cứu", "luận điểm đầu tư",
    "phân tích ngành", "so sánh ngành",
    "định giá", "giá trị nội tại",
    "đánh giá rủi ro", "phân tích rủi ro",
}

# Query types requiring Deep Research
DEEP_RESEARCH_QUERY_TYPES = {QueryType.SCREENER}

# Symbol count threshold for Deep Research
MULTI_SYMBOL_THRESHOLD = 3


class ModeRouter(LoggerMixin):
    """Routes queries to appropriate processing mode."""

    def __init__(
        self,
        multi_symbol_threshold: int = MULTI_SYMBOL_THRESHOLD,
        enable_keyword_detection: bool = True,
    ):
        super().__init__()
        self.multi_symbol_threshold = multi_symbol_threshold
        self.enable_keyword_detection = enable_keyword_detection
        self.logger.debug(
            f"[ROUTER:INIT] threshold={multi_symbol_threshold} | keywords={enable_keyword_detection}"
        )

    def determine_mode(
        self,
        query: str,
        explicit_mode: Optional[str] = None,
        classification: Optional[UnifiedClassificationResult] = None,
        symbols: Optional[List[str]] = None,
    ) -> ModeDecision:
        """Determine processing mode for query based on priority rules."""
        self.logger.debug(f"[ROUTER:START] query={query[:60]}...")

        # Priority 1: Explicit mode from request
        if explicit_mode and explicit_mode != QueryMode.AUTO.value:
            try:
                mode = QueryMode(explicit_mode.lower())
                return self._make_decision(
                    mode=mode,
                    reason="Explicitly selected by user",
                    confidence=1.0,
                    method="explicit",
                )
            except ValueError:
                self.logger.warning(f"[ROUTER:WARN] Invalid mode: {explicit_mode}")

        # Extract symbols from classification if not provided
        if symbols is None and classification:
            symbols = classification.symbols or []

        # Priority 2: Multi-symbol triggers Deep Research
        if symbols and len(symbols) >= self.multi_symbol_threshold:
            return self._make_decision(
                mode=QueryMode.DEEP_RESEARCH,
                reason=f"Multi-symbol ({len(symbols)} >= {self.multi_symbol_threshold})",
                confidence=0.9,
                method="symbol_count",
                metadata={"symbol_count": len(symbols), "symbols": symbols},
            )

        # Priority 3: Keyword matching
        if self.enable_keyword_detection:
            keyword = self._find_deep_research_keyword(query)
            if keyword:
                return self._make_decision(
                    mode=QueryMode.DEEP_RESEARCH,
                    reason=f"Keyword matched: '{keyword}'",
                    confidence=0.85,
                    method="keyword_match",
                    metadata={"matched_keyword": keyword},
                )

        # Priority 4: Classification-based detection
        # if classification:
        #     if classification.query_type in DEEP_RESEARCH_QUERY_TYPES:
        #         if self._is_screener_with_analysis(query):
        #             return self._make_decision(
        #                 mode=QueryMode.DEEP_RESEARCH,
        #                 reason="Screener with analysis intent",
        #                 confidence=0.75,
        #                 method="classification",
        #                 metadata={"query_type": classification.query_type.value},
        #             )

        #     if len(classification.tool_categories or []) >= 4:
        #         return self._make_decision(
        #             mode=QueryMode.DEEP_RESEARCH,
        #             reason=f"Complex query ({len(classification.tool_categories)} categories)",
        #             confidence=0.7,
        #             method="classification",
        #             metadata={"categories": classification.tool_categories},
        #         )

        # Default: Normal mode
        return self._make_decision(
            mode=QueryMode.NORMAL,
            reason="Default for simple queries",
            confidence=0.95,
            method="default",
        )

    def _make_decision(
        self,
        mode: QueryMode,
        reason: str,
        confidence: float,
        method: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModeDecision:
        """Create decision and log it."""
        decision = ModeDecision(
            mode=mode,
            reason=reason,
            confidence=confidence,
            detection_method=method,
            metadata=metadata or {},
        )
        self.logger.info(
            f"[ROUTER:DECISION] mode={mode.value} | method={method} | "
            f"conf={confidence:.2f} | reason={reason}"
        )
        return decision

    def _find_deep_research_keyword(self, query: str) -> Optional[str]:
        """Find matching deep research keyword in query."""
        query_lower = query.lower()
        for keyword in DEEP_RESEARCH_KEYWORDS:
            if keyword in query_lower:
                return keyword
        return None

    def _is_screener_with_analysis(self, query: str) -> bool:
        """Check if screener query has analysis intent."""
        query_lower = query.lower()
        analysis_keywords = ["analyze", "phân tích", "review", "evaluate", "đánh giá"]
        return any(kw in query_lower for kw in analysis_keywords)


# Singleton instance
_router_instance: Optional[ModeRouter] = None


def get_mode_router() -> ModeRouter:
    """Get or create singleton ModeRouter."""
    global _router_instance
    if _router_instance is None:
        _router_instance = ModeRouter()
    return _router_instance


def reset_mode_router() -> None:
    """Reset singleton for testing."""
    global _router_instance
    _router_instance = None


def should_use_deep_research(
    query: str,
    explicit_mode: Optional[str] = None,
    classification: Optional[UnifiedClassificationResult] = None,
    symbols: Optional[List[str]] = None,
) -> bool:
    """Check if query requires Deep Research mode."""
    decision = get_mode_router().determine_mode(
        query=query,
        explicit_mode=explicit_mode,
        classification=classification,
        symbols=symbols,
    )
    return decision.mode == QueryMode.DEEP_RESEARCH


def get_mode_from_request(
    query: str,
    mode_param: Optional[str] = None,
    classification: Optional[UnifiedClassificationResult] = None,
) -> QueryMode:
    """Resolve mode from API request parameters."""
    decision = get_mode_router().determine_mode(
        query=query,
        explicit_mode=mode_param,
        classification=classification,
    )
    return decision.mode