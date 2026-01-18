from enum import Enum
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field

from src.utils.logger.custom_logging import LoggerMixin


class ChartType(str, Enum):
    """Supported chart types by frontend"""
    STOCK_PRICE = "showStockPrice"
    STOCK_FINANCIALS = "showStockFinancials"
    STOCK_CHART = "showStockChart"  # Technical chart
    STOCK_NEWS = "showStockNews"
    MARKET_OVERVIEW = "showMarketOverview"
    TRENDING_STOCKS = "showTrendingStocks"
    STOCK_HEATMAP = "showStockHeatmap"
    CRYPTO_CHART = "cryptoChart"


# Priority order for charts (lower index = higher priority)
CHART_PRIORITY: List[ChartType] = [
    ChartType.STOCK_CHART,      # Technical analysis - main chart
    ChartType.STOCK_PRICE,      # Basic price info
    ChartType.CRYPTO_CHART,     # Crypto queries
    ChartType.STOCK_FINANCIALS, # Financial reports
    ChartType.STOCK_NEWS,       # News
    ChartType.MARKET_OVERVIEW,  # Market overview
    ChartType.TRENDING_STOCKS,  # Discovery
    ChartType.STOCK_HEATMAP,    # Heatmap
]


# =============================================================================
# DIRECT TOOL NAME → CHART TYPE MAPPING (Production-Ready)
# =============================================================================
# This mapping is based on exact tool names from ToolSchema.name
# NO keyword matching - deterministic and language-agnostic
# =============================================================================

TOOL_TO_CHART: Dict[str, ChartType] = {
    # -------------------------------------------------------------------------
    # TECHNICAL TOOLS → STOCK_CHART (Technical Analysis Chart)
    # -------------------------------------------------------------------------
    "getTechnicalIndicators": ChartType.STOCK_CHART,
    "detectChartPatterns": ChartType.STOCK_CHART,
    "getRelativeStrength": ChartType.STOCK_CHART,
    "getSupportResistance": ChartType.STOCK_CHART,

    # -------------------------------------------------------------------------
    # PRICE TOOLS → STOCK_PRICE
    # -------------------------------------------------------------------------
    "getStockPrice": ChartType.STOCK_PRICE,
    "getStockPerformance": ChartType.STOCK_PRICE,
    "getPriceTargets": ChartType.STOCK_PRICE,

    # -------------------------------------------------------------------------
    # FUNDAMENTALS TOOLS → STOCK_FINANCIALS
    # -------------------------------------------------------------------------
    "getIncomeStatement": ChartType.STOCK_FINANCIALS,
    "getBalanceSheet": ChartType.STOCK_FINANCIALS,
    "getCashFlow": ChartType.STOCK_FINANCIALS,
    "getFinancialRatios": ChartType.STOCK_FINANCIALS,
    "getGrowthMetrics": ChartType.STOCK_FINANCIALS,

    # -------------------------------------------------------------------------
    # NEWS TOOLS → STOCK_NEWS
    # -------------------------------------------------------------------------
    "getStockNews": ChartType.STOCK_NEWS,
    "getEarningsCalendar": ChartType.STOCK_NEWS,
    "getCompanyEvents": ChartType.STOCK_NEWS,

    # -------------------------------------------------------------------------
    # MARKET TOOLS → MARKET_OVERVIEW (or STOCK_HEATMAP)
    # -------------------------------------------------------------------------
    "getMarketIndices": ChartType.MARKET_OVERVIEW,
    "getSectorPerformance": ChartType.MARKET_OVERVIEW,
    "getMarketMovers": ChartType.MARKET_OVERVIEW,
    "getMarketBreadth": ChartType.MARKET_OVERVIEW,
    "getMarketNews": ChartType.MARKET_OVERVIEW,
    "getTopGainers": ChartType.TRENDING_STOCKS,
    "getTopLosers": ChartType.TRENDING_STOCKS,
    "getMostActives": ChartType.TRENDING_STOCKS,
    "getStockHeatmap": ChartType.STOCK_HEATMAP,

    # -------------------------------------------------------------------------
    # CRYPTO TOOLS → CRYPTO_CHART
    # -------------------------------------------------------------------------
    "getCryptoPrice": ChartType.CRYPTO_CHART,
    "getCryptoTechnicals": ChartType.CRYPTO_CHART,

    # -------------------------------------------------------------------------
    # DISCOVERY TOOLS → TRENDING_STOCKS
    # -------------------------------------------------------------------------
    "stockScreener": ChartType.TRENDING_STOCKS,

    # -------------------------------------------------------------------------
    # RISK TOOLS → STOCK_CHART (show with technical data)
    # -------------------------------------------------------------------------
    "assessRisk": ChartType.STOCK_CHART,
    "getVolumeProfile": ChartType.STOCK_CHART,
    "getSentiment": ChartType.STOCK_PRICE,
    "suggestStopLoss": ChartType.STOCK_CHART,

    # -------------------------------------------------------------------------
    # NO CHART MAPPING (memory, reasoning, web)
    # -------------------------------------------------------------------------
    # "searchConversationHistory": None,
    # "getRecentConversations": None,
    # "memoryUserEdits": None,
    # "think": None,
    # "webSearch": None,
}


# Category fallback mapping (used when tool name not in TOOL_TO_CHART)
CATEGORY_TO_CHART: Dict[str, ChartType] = {
    "price": ChartType.STOCK_PRICE,
    "technical": ChartType.STOCK_CHART,
    "fundamentals": ChartType.STOCK_FINANCIALS,
    "news": ChartType.STOCK_NEWS,
    "market": ChartType.MARKET_OVERVIEW,
    "discovery": ChartType.TRENDING_STOCKS,
    "crypto": ChartType.CRYPTO_CHART,
    "risk": ChartType.STOCK_CHART,
    # "memory" - no chart mapping
    # "reasoning" - no chart mapping
    # "web" - no chart mapping
}


# Categories that should NOT produce charts
NO_CHART_CATEGORIES: Set[str] = {"memory", "reasoning", "web"}


@dataclass
class ChartInfo:
    """
    Chart information to be returned to frontend.

    Attributes:
        type: Chart type (one of ChartType values)
        symbols: List of symbols to display (e.g., ["NVDA", "AAPL"])
        primary: Whether this is the primary chart (first chart)
        metadata: Optional additional data for the chart
    """
    type: str
    symbols: List[str] = field(default_factory=list)
    primary: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "type": self.type,
            "symbols": self.symbols,
            "primary": self.primary,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class SimpleChartResolver(LoggerMixin):
    """
    Resolves chart types based on executed tools.

    Production-ready mapping logic:
    - Direct tool name → chart type (deterministic)
    - Category fallback for unknown tools
    - No keyword matching (language-agnostic)
    """

    def __init__(self):
        super().__init__()

    def resolve_from_tool_results(
        self,
        tool_results: List[Dict[str, Any]],
        symbols: List[str],
        query: str = "",
        max_charts: int = 3,
    ) -> List[ChartInfo]:
        """
        Resolve chart types from actual tool execution results.

        Uses direct tool name mapping - no keyword patterns.

        Args:
            tool_results: List of tool results with tool names
            symbols: List of symbols
            query: Original query (unused, kept for API compatibility)
            max_charts: Maximum charts to return

        Returns:
            List of ChartInfo
        """
        if not tool_results:
            return []

        charts: List[ChartInfo] = []
        seen_types: Set[str] = set()

        for result in tool_results:
            # Extract tool name from result (support multiple key formats)
            tool_name = self._extract_tool_name(result)

            if not tool_name:
                continue

            # Get chart type for this tool
            chart_type = self._get_chart_type_for_tool(tool_name, result)

            if chart_type and chart_type.value not in seen_types:
                seen_types.add(chart_type.value)
                charts.append(ChartInfo(
                    type=chart_type.value,
                    symbols=symbols.copy(),
                    primary=False,
                    metadata={"source_tool": tool_name}
                ))

        # Sort by priority
        charts = self._sort_by_priority(charts)

        # Limit number of charts
        charts = charts[:max_charts]

        # Mark first chart as primary
        if charts:
            charts[0].primary = True

        self.logger.info(
            f"[CHART_RESOLVER] Resolved {len(charts)} charts from "
            f"{len(tool_results)} tool results, symbols={symbols}"
        )

        return charts

    def resolve_from_categories(
        self,
        categories: List[str],
        symbols: List[str],
        query: str = "",
        max_charts: int = 3,
    ) -> List[ChartInfo]:
        """
        Resolve chart types from tool categories (fallback method).

        Args:
            categories: List of tool categories (e.g., ["price", "technical"])
            symbols: List of symbols (e.g., ["NVDA", "AAPL"])
            query: Original query (unused)
            max_charts: Maximum number of charts to return

        Returns:
            List of ChartInfo sorted by priority
        """
        if not categories:
            return []

        charts: List[ChartInfo] = []
        seen_types: Set[str] = set()

        for category in categories:
            if category in NO_CHART_CATEGORIES:
                continue

            chart_type = CATEGORY_TO_CHART.get(category)

            if chart_type and chart_type.value not in seen_types:
                seen_types.add(chart_type.value)
                charts.append(ChartInfo(
                    type=chart_type.value,
                    symbols=symbols.copy(),
                    primary=False,
                    metadata={"source_category": category}
                ))

        # Sort by priority
        charts = self._sort_by_priority(charts)

        # Limit number of charts
        charts = charts[:max_charts]

        # Mark first chart as primary
        if charts:
            charts[0].primary = True

        self.logger.debug(
            f"[CHART_RESOLVER] Resolved {len(charts)} charts from "
            f"categories={categories}, symbols={symbols}"
        )

        return charts

    def _extract_tool_name(self, result: Dict[str, Any]) -> Optional[str]:
        """
        Extract tool name from result dict.

        Supports multiple key formats used by different API versions.
        """
        # Try different key names used across the codebase
        for key in ["tool", "name", "tool_name", "toolName"]:
            if key in result and result[key]:
                return str(result[key])
        return None

    def _get_chart_type_for_tool(
        self,
        tool_name: str,
        result: Dict[str, Any]
    ) -> Optional[ChartType]:
        """
        Get chart type for a specific tool.

        Priority:
        1. Direct tool name mapping (TOOL_TO_CHART)
        2. Category-based fallback (CATEGORY_TO_CHART)
        """
        # 1. Direct mapping (exact match)
        if tool_name in TOOL_TO_CHART:
            return TOOL_TO_CHART[tool_name]

        # 2. Try category from result metadata
        category = result.get("category", result.get("tool_category", ""))
        if category:
            if category in NO_CHART_CATEGORIES:
                return None
            return CATEGORY_TO_CHART.get(category)

        # 3. No mapping found
        self.logger.debug(
            f"[CHART_RESOLVER] No chart mapping for tool: {tool_name}"
        )
        return None

    def _sort_by_priority(self, charts: List[ChartInfo]) -> List[ChartInfo]:
        """Sort charts by priority order"""
        priority_map = {ct.value: i for i, ct in enumerate(CHART_PRIORITY)}

        return sorted(
            charts,
            key=lambda c: priority_map.get(c.type, len(CHART_PRIORITY))
        )


# =============================================================================
# SINGLETON & HELPER FUNCTIONS
# =============================================================================

_chart_resolver: Optional[SimpleChartResolver] = None


def get_chart_resolver() -> SimpleChartResolver:
    """Get singleton chart resolver instance"""
    global _chart_resolver
    if _chart_resolver is None:
        _chart_resolver = SimpleChartResolver()
    return _chart_resolver


def resolve_charts_from_categories(
    categories: List[str],
    symbols: List[str],
    query: str = "",
    max_charts: int = 3,
) -> List[ChartInfo]:
    """
    Convenience function to resolve charts from categories.

    Args:
        categories: Tool categories
        symbols: Stock/crypto symbols
        query: Original query
        max_charts: Max charts to return

    Returns:
        List of ChartInfo
    """
    resolver = get_chart_resolver()
    return resolver.resolve_from_categories(
        categories=categories,
        symbols=symbols,
        query=query,
        max_charts=max_charts,
    )


def resolve_charts_from_classification(
    classification: Any,
    query: str = "",
    max_charts: int = 3,
) -> List[ChartInfo]:
    """
    Resolve charts from UnifiedClassificationResult.

    Args:
        classification: UnifiedClassificationResult object
        query: Original query (fallback if not in classification)
        max_charts: Max charts to return

    Returns:
        List of ChartInfo
    """
    # Extract from classification object
    categories = getattr(classification, "tool_categories", [])
    symbols = getattr(classification, "symbols", [])

    # Use query from classification if available
    intent = getattr(classification, "intent_summary", "")
    if not query and intent:
        query = intent

    return resolve_charts_from_categories(
        categories=categories,
        symbols=symbols,
        query=query,
        max_charts=max_charts,
    )


def charts_to_dict_list(charts: List[ChartInfo]) -> List[Dict[str, Any]]:
    """Convert list of ChartInfo to list of dicts"""
    return [chart.to_dict() for chart in charts]
