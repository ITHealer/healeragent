from enum import Enum
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field, asdict

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


# Category to chart type mapping
CATEGORY_TO_CHART: Dict[str, ChartType] = {
    "price": ChartType.STOCK_PRICE,
    "technical": ChartType.STOCK_CHART,
    "fundamentals": ChartType.STOCK_FINANCIALS,
    "news": ChartType.STOCK_NEWS,
    "market": ChartType.MARKET_OVERVIEW,
    "discovery": ChartType.TRENDING_STOCKS,  # Default, may become HEATMAP
    "crypto": ChartType.CRYPTO_CHART,
    "risk": ChartType.STOCK_PRICE,  # Fallback to price chart
    # "memory" - no chart mapping
}


# Keywords that trigger heatmap instead of trending
HEATMAP_KEYWORDS: Set[str] = {
    "heatmap", "heat map", "heat-map",
    "sector map", "sector performance",
    "market map", "bản đồ nhiệt"
}


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
    Resolves chart types based on tool categories.

    Simple mapping logic - no LLM calls required.
    Frontend will fetch chart data based on type and symbols.
    """

    def __init__(self):
        super().__init__()

    def resolve_from_categories(
        self,
        categories: List[str],
        symbols: List[str],
        query: str = "",
        max_charts: int = 3,
    ) -> List[ChartInfo]:
        """
        Resolve chart types from tool categories.

        Args:
            categories: List of tool categories (e.g., ["price", "technical"])
            symbols: List of symbols (e.g., ["NVDA", "AAPL"])
            query: Original query (used for heatmap detection)
            max_charts: Maximum number of charts to return

        Returns:
            List of ChartInfo sorted by priority
        """
        if not categories:
            return []

        charts: List[ChartInfo] = []
        seen_types: Set[str] = set()
        query_lower = query.lower() if query else ""

        # Check for heatmap keywords
        use_heatmap = any(kw in query_lower for kw in HEATMAP_KEYWORDS)

        # Map categories to chart types
        for category in categories:
            chart_type = self._get_chart_type(category, use_heatmap)

            if chart_type and chart_type.value not in seen_types:
                seen_types.add(chart_type.value)
                charts.append(ChartInfo(
                    type=chart_type.value,
                    symbols=symbols.copy(),
                    primary=False,  # Will set primary later
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

    def resolve_from_tool_results(
        self,
        tool_results: List[Dict[str, Any]],
        symbols: List[str],
        query: str = "",
        max_charts: int = 3,
    ) -> List[ChartInfo]:
        """
        Resolve chart types from actual tool execution results.

        Args:
            tool_results: List of tool results with tool names
            symbols: List of symbols
            query: Original query
            max_charts: Maximum charts to return

        Returns:
            List of ChartInfo
        """
        if not tool_results:
            return []

        # Extract categories from tool names
        categories = self._extract_categories_from_tools(tool_results)

        return self.resolve_from_categories(
            categories=categories,
            symbols=symbols,
            query=query,
            max_charts=max_charts,
        )

    def _get_chart_type(
        self,
        category: str,
        use_heatmap: bool = False
    ) -> Optional[ChartType]:
        """Get chart type for a category"""
        if category == "memory":
            return None  # No chart for memory

        if category == "discovery" and use_heatmap:
            return ChartType.STOCK_HEATMAP

        return CATEGORY_TO_CHART.get(category)

    def _sort_by_priority(self, charts: List[ChartInfo]) -> List[ChartInfo]:
        """Sort charts by priority order"""
        priority_map = {ct.value: i for i, ct in enumerate(CHART_PRIORITY)}

        return sorted(
            charts,
            key=lambda c: priority_map.get(c.type, len(CHART_PRIORITY))
        )

    def _extract_categories_from_tools(
        self,
        tool_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract unique categories from tool results"""
        categories: Set[str] = set()

        # Tool name patterns to category mapping
        # Pattern matching is case-insensitive (tool_name.lower())
        tool_patterns = {
            "price": [
                "get_stock_price", "get_quote", "price", "getstockprice",
                "getquote", "stockprice"
            ],
            "technical": [
                "get_technical", "indicators", "rsi", "macd", "sma",
                "gettechnical", "detectchartpatterns", "supportresistance",
                "chart_pattern", "technical_indicator"
            ],
            "fundamentals": [
                "get_financials", "get_fundamentals", "earnings",
                "getincomestatement", "getbalancesheet", "getcashflow",
                "getfinancialratios", "getgrowthmetrics", "income_statement",
                "balance_sheet", "cash_flow", "financial_ratio", "growth"
            ],
            "news": ["get_news", "news", "getnews", "stocknews"],
            "market": [
                "get_market", "market_overview", "indices", "getmarket",
                "marketoverview", "sector"
            ],
            "discovery": [
                "screen", "trending", "gainers", "losers", "screener",
                "gettrending", "gettopgainers", "gettoplosers"
            ],
            "crypto": ["crypto", "bitcoin", "ethereum", "getcrypto"],
            "risk": ["risk", "volatility", "getrisk"],
        }

        for result in tool_results:
            # Support multiple key formats: "tool", "name", "tool_name"
            tool_name = result.get("tool", result.get("name", result.get("tool_name", ""))).lower()

            for category, patterns in tool_patterns.items():
                if any(p in tool_name for p in patterns):
                    categories.add(category)
                    break

        return list(categories)


# Singleton instance
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
        query: Original query (for heatmap detection)
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