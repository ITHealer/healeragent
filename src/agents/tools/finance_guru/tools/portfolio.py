"""
Finance Guru - Portfolio Analysis Tools (Phase 4)

Layer 3 of the 3-layer architecture: Agent-callable tool interfaces.

These tools provide the agent-facing interface for portfolio analysis:
- OptimizePortfolioTool: Portfolio optimization (Max Sharpe, Min Variance, etc.)
- GetCorrelationMatrixTool: Asset correlation analysis
- GetEfficientFrontierTool: Efficient frontier generation
- AnalyzePortfolioDiversificationTool: Comprehensive diversification analysis
- SuggestRebalancingTool: Portfolio rebalancing suggestions

WHAT: Agent-callable tools for portfolio analysis
WHY: Provides standardized interface for LLM agents to optimize portfolios
ARCHITECTURE: Layer 3 of 3-layer type-safe architecture

Author: HealerAgent Development Team
"""

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from src.agents.tools.base import (
    BaseTool,
    ToolOutput,
    ToolParameter,
    ToolSchema,
)
from src.agents.tools.finance_guru.calculators.portfolio import (
    PortfolioOptimizer,
    CorrelationEngine,
    RebalancingCalculator,
)
from src.agents.tools.finance_guru.models.portfolio import (
    # Enums
    OptimizationMethod,
    CorrelationMethod,
    RiskModel,
    # Input
    AssetData,
    PortfolioDataInput,
    PortfolioPriceData,
    OptimizationConfig,
    CorrelationConfig,
    EfficientFrontierConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_success_output(
    tool_name: str,
    data: Dict[str, Any],
    formatted_context: Optional[str] = None,
    execution_time_ms: int = 0,
) -> ToolOutput:
    """Create a successful ToolOutput."""
    return ToolOutput(
        tool_name=tool_name,
        status="success",
        data=data,
        formatted_context=formatted_context,
        execution_time_ms=execution_time_ms,
    )


def create_error_output(
    tool_name: str,
    error: str,
    execution_time_ms: int = 0,
) -> ToolOutput:
    """Create an error ToolOutput."""
    return ToolOutput(
        tool_name=tool_name,
        status="error",
        error=error,
        execution_time_ms=execution_time_ms,
    )


async def fetch_portfolio_prices(
    symbols: List[str],
    days: int = 252,
    fmp_api_key: Optional[str] = None,
) -> PortfolioPriceData:
    """
    Fetch price data for multiple assets from FMP API.

    Args:
        symbols: List of stock symbols
        days: Number of days of data
        fmp_api_key: FMP API key

    Returns:
        PortfolioPriceData with price matrix
    """
    try:
        from src.services.fmp_service import FMPService

        fmp = FMPService(api_key=fmp_api_key)
        price_matrix = []
        common_dates = None

        for symbol in symbols:
            historical = await fmp.get_historical_price(symbol, days=days)

            if not historical or len(historical) < 30:
                raise ValueError(f"Insufficient data for {symbol}")

            # FMP returns newest first, reverse for chronological order
            historical = list(reversed(historical))

            dates = [d["date"] for d in historical]
            prices = [float(d["close"]) for d in historical]

            if common_dates is None:
                common_dates = dates
            else:
                # Align dates (simple approach - use min length)
                min_len = min(len(common_dates), len(dates))
                common_dates = common_dates[-min_len:]
                prices = prices[-min_len:]

            price_matrix.append(prices[-len(common_dates):])

        # Ensure all rows have same length
        min_len = min(len(row) for row in price_matrix)
        price_matrix = [row[-min_len:] for row in price_matrix]
        common_dates = common_dates[-min_len:] if common_dates else None

        return PortfolioPriceData(
            symbols=symbols,
            price_matrix=price_matrix,
            dates=common_dates,
        )

    except ImportError:
        logger.warning("FMPService not available, using mock data")
        return _create_mock_portfolio_data(symbols, days)


def _create_mock_portfolio_data(
    symbols: List[str],
    days: int = 252,
) -> PortfolioPriceData:
    """Create mock portfolio price data for testing."""
    import random

    price_matrix = []
    dates = []
    today = date.today()

    # Generate dates
    for i in range(days):
        d = today - timedelta(days=days - i - 1)
        if d.weekday() < 5:  # Skip weekends
            dates.append(d.isoformat())

    # Generate correlated price data for each symbol
    base_prices = {"AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 350.0, "AMZN": 3400.0, "TSLA": 250.0}

    for symbol in symbols:
        base = base_prices.get(symbol, 100.0)
        prices = [base]
        for _ in range(len(dates) - 1):
            change = random.gauss(0.0005, 0.015)  # Small drift + volatility
            prices.append(prices[-1] * (1 + change))
        price_matrix.append(prices)

    return PortfolioPriceData(
        symbols=symbols,
        price_matrix=price_matrix,
        dates=dates,
    )


async def fetch_portfolio_data(
    symbols: List[str],
    weights: Optional[List[float]] = None,
    days: int = 252,
    risk_free_rate: float = 0.02,
    fmp_api_key: Optional[str] = None,
) -> PortfolioDataInput:
    """
    Fetch portfolio data with prices for optimization.

    Args:
        symbols: List of stock symbols
        weights: Current portfolio weights (optional)
        days: Number of days of data
        risk_free_rate: Risk-free rate
        fmp_api_key: FMP API key

    Returns:
        PortfolioDataInput for optimization
    """
    price_data = await fetch_portfolio_prices(symbols, days, fmp_api_key)

    if weights is None:
        weights = [1.0 / len(symbols)] * len(symbols)

    assets = [
        AssetData(
            symbol=symbols[i],
            prices=price_data.price_matrix[i],
            weight=weights[i],
        )
        for i in range(len(symbols))
    ]

    return PortfolioDataInput(
        assets=assets,
        risk_free_rate=risk_free_rate,
    )


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================


class OptimizePortfolioTool(BaseTool):
    """Tool for portfolio optimization.

    Finds optimal asset allocation using various methods:
    - MAX_SHARPE: Maximize risk-adjusted returns
    - MIN_VARIANCE: Minimize portfolio volatility
    - MEAN_VARIANCE: Target specific return with minimum risk
    - RISK_PARITY: Equal risk contribution from each asset

    Usage by agent:
        optimizePortfolio(symbols=["AAPL", "GOOGL", "MSFT"], method="max_sharpe")
    """

    def __init__(self):
        super().__init__()
        self.optimizer = PortfolioOptimizer()

        self.schema = ToolSchema(
            name="optimizePortfolio",
            category="portfolio",
            description=(
                "Optimize portfolio allocation to find the best asset weights. "
                "Methods: max_sharpe (maximize Sharpe ratio), min_variance (minimize risk), "
                "mean_variance (target return), risk_parity (equal risk contribution)."
            ),
            parameters=[
                ToolParameter(
                    name="symbols",
                    type="array",
                    description="List of stock symbols (e.g., ['AAPL', 'GOOGL', 'MSFT'])",
                    required=True,
                ),
                ToolParameter(
                    name="method",
                    type="string",
                    description="Optimization method: max_sharpe, min_variance, mean_variance, risk_parity",
                    required=False,
                    default="max_sharpe",
                ),
                ToolParameter(
                    name="target_return",
                    type="number",
                    description="Target annual return for mean_variance method (e.g., 0.15 for 15%)",
                    required=False,
                ),
                ToolParameter(
                    name="min_weight",
                    type="number",
                    description="Minimum weight per asset (0.0 = no short selling)",
                    required=False,
                    default=0.0,
                ),
                ToolParameter(
                    name="max_weight",
                    type="number",
                    description="Maximum weight per asset (prevents concentration)",
                    required=False,
                    default=1.0,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolOutput:
        """Execute portfolio optimization."""
        try:
            symbols = kwargs.get("symbols", [])
            method_str = kwargs.get("method", "max_sharpe")
            target_return = kwargs.get("target_return")
            min_weight = kwargs.get("min_weight", 0.0)
            max_weight = kwargs.get("max_weight", 1.0)

            if not symbols or len(symbols) < 2:
                return create_error_output(
                    self.schema.name,
                    "Need at least 2 symbols for portfolio optimization"
                )

            # Map method string to enum
            method_map = {
                "max_sharpe": OptimizationMethod.MAX_SHARPE,
                "min_variance": OptimizationMethod.MIN_VARIANCE,
                "mean_variance": OptimizationMethod.MEAN_VARIANCE,
                "risk_parity": OptimizationMethod.RISK_PARITY,
            }
            method = method_map.get(method_str, OptimizationMethod.MAX_SHARPE)

            # Fetch data
            data = await fetch_portfolio_data(symbols)

            # Configure optimization
            config = OptimizationConfig(
                method=method,
                target_return=target_return,
                min_weight=min_weight,
                max_weight=max_weight,
            )

            # Run optimization
            result = self.optimizer.optimize(data, config)

            # Format output
            allocations_summary = []
            for alloc in result.allocations:
                allocations_summary.append(
                    f"  {alloc.symbol}: {alloc.weight:.1%} (expected return: {alloc.expected_return:.1%})"
                )

            formatted = (
                f"Portfolio Optimization Results ({method_str}):\n"
                f"Allocations:\n{''.join(allocations_summary)}\n"
                f"Expected Return: {result.expected_return:.2%}\n"
                f"Expected Volatility: {result.expected_volatility:.2%}\n"
                f"Sharpe Ratio: {result.sharpe_ratio:.3f}"
            )

            return create_success_output(
                self.schema.name,
                data={
                    "allocations": [
                        {"symbol": a.symbol, "weight": a.weight, "expected_return": a.expected_return}
                        for a in result.allocations
                    ],
                    "expected_return": result.expected_return,
                    "expected_volatility": result.expected_volatility,
                    "sharpe_ratio": result.sharpe_ratio,
                    "method": method_str,
                },
                formatted_context=formatted,
                execution_time_ms=result.calculation_time_ms or 0,
            )

        except Exception as e:
            logger.exception(f"Portfolio optimization failed: {e}")
            return create_error_output(self.schema.name, str(e))


class GetCorrelationMatrixTool(BaseTool):
    """Tool for calculating asset correlation matrix.

    Calculates correlation between assets to assess diversification.
    Correlation ranges from -1 (perfect negative) to +1 (perfect positive).

    Usage by agent:
        getCorrelationMatrix(symbols=["AAPL", "GOOGL", "MSFT"])
    """

    def __init__(self):
        super().__init__()
        self.engine = CorrelationEngine()

        self.schema = ToolSchema(
            name="getCorrelationMatrix",
            category="portfolio",
            description=(
                "Calculate correlation matrix between assets. "
                "Correlation ranges from -1 (move opposite) to +1 (move together). "
                "Lower correlations = better diversification."
            ),
            parameters=[
                ToolParameter(
                    name="symbols",
                    type="array",
                    description="List of stock symbols (e.g., ['AAPL', 'GOOGL', 'MSFT'])",
                    required=True,
                ),
                ToolParameter(
                    name="method",
                    type="string",
                    description="Correlation method: pearson, spearman, or kendall",
                    required=False,
                    default="pearson",
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolOutput:
        """Execute correlation matrix calculation."""
        try:
            symbols = kwargs.get("symbols", [])
            method_str = kwargs.get("method", "pearson")

            if not symbols or len(symbols) < 2:
                return create_error_output(
                    self.schema.name,
                    "Need at least 2 symbols for correlation analysis"
                )

            # Map method string to enum
            method_map = {
                "pearson": CorrelationMethod.PEARSON,
                "spearman": CorrelationMethod.SPEARMAN,
                "kendall": CorrelationMethod.KENDALL,
            }
            method = method_map.get(method_str, CorrelationMethod.PEARSON)

            # Fetch data
            price_data = await fetch_portfolio_prices(symbols)

            # Configure and calculate
            config = CorrelationConfig(method=method)
            result = self.engine.calculate_correlation_matrix(price_data, config)

            # Format matrix for display
            header = "       " + "  ".join(f"{s:>7}" for s in symbols)
            rows = []
            for s1 in symbols:
                row_values = [f"{result.matrix[s1][s2]:>7.3f}" for s2 in symbols]
                rows.append(f"{s1:>7}" + "  ".join(row_values))

            formatted = (
                f"Correlation Matrix ({method_str}):\n{header}\n" +
                "\n".join(rows) +
                f"\n\nAverage Correlation: {result.average_correlation:.3f}\n"
                f"Most Correlated: {result.max_correlation[0]}-{result.max_correlation[1]}: {result.max_correlation[2]:.3f}\n"
                f"Least Correlated: {result.min_correlation[0]}-{result.min_correlation[1]}: {result.min_correlation[2]:.3f}"
            )

            return create_success_output(
                self.schema.name,
                data={
                    "matrix": result.matrix,
                    "symbols": result.symbols,
                    "method": method_str,
                    "average_correlation": result.average_correlation,
                    "max_correlation": result.max_correlation,
                    "min_correlation": result.min_correlation,
                },
                formatted_context=formatted,
                execution_time_ms=result.calculation_time_ms or 0,
            )

        except Exception as e:
            logger.exception(f"Correlation calculation failed: {e}")
            return create_error_output(self.schema.name, str(e))


class GetEfficientFrontierTool(BaseTool):
    """Tool for generating efficient frontier.

    Creates the efficient frontier showing optimal risk-return tradeoffs.
    Identifies Max Sharpe and Min Variance portfolios.

    Usage by agent:
        getEfficientFrontier(symbols=["AAPL", "GOOGL", "MSFT"])
    """

    def __init__(self):
        super().__init__()
        self.optimizer = PortfolioOptimizer()

        self.schema = ToolSchema(
            name="getEfficientFrontier",
            category="portfolio",
            description=(
                "Generate efficient frontier showing optimal portfolios at each risk level. "
                "Returns Max Sharpe and Min Variance portfolios with their allocations."
            ),
            parameters=[
                ToolParameter(
                    name="symbols",
                    type="array",
                    description="List of stock symbols",
                    required=True,
                ),
                ToolParameter(
                    name="num_portfolios",
                    type="integer",
                    description="Number of portfolios on frontier (10-100)",
                    required=False,
                    default=50,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolOutput:
        """Execute efficient frontier generation."""
        try:
            symbols = kwargs.get("symbols", [])
            num_portfolios = min(max(kwargs.get("num_portfolios", 50), 10), 100)

            if not symbols or len(symbols) < 2:
                return create_error_output(
                    self.schema.name,
                    "Need at least 2 symbols for efficient frontier"
                )

            # Fetch data
            data = await fetch_portfolio_data(symbols)

            # Configure and generate
            config = EfficientFrontierConfig(
                num_portfolios=num_portfolios,
                include_assets=True,
            )
            result = self.optimizer.generate_efficient_frontier(data, config)

            # Format Max Sharpe portfolio
            max_sharpe_alloc = ", ".join(
                f"{s}: {w:.1%}" for s, w in result.max_sharpe_portfolio.weights.items()
            )

            # Format Min Variance portfolio
            min_var_alloc = ", ".join(
                f"{s}: {w:.1%}" for s, w in result.min_variance_portfolio.weights.items()
            )

            formatted = (
                f"Efficient Frontier Analysis:\n\n"
                f"Max Sharpe Portfolio (Optimal Risk-Adjusted):\n"
                f"  Allocation: {max_sharpe_alloc}\n"
                f"  Expected Return: {result.max_sharpe_portfolio.expected_return:.2%}\n"
                f"  Volatility: {result.max_sharpe_portfolio.expected_volatility:.2%}\n"
                f"  Sharpe Ratio: {result.max_sharpe_portfolio.sharpe_ratio:.3f}\n\n"
                f"Min Variance Portfolio (Lowest Risk):\n"
                f"  Allocation: {min_var_alloc}\n"
                f"  Expected Return: {result.min_variance_portfolio.expected_return:.2%}\n"
                f"  Volatility: {result.min_variance_portfolio.expected_volatility:.2%}\n"
                f"  Sharpe Ratio: {result.min_variance_portfolio.sharpe_ratio:.3f}"
            )

            return create_success_output(
                self.schema.name,
                data={
                    "max_sharpe_portfolio": {
                        "weights": result.max_sharpe_portfolio.weights,
                        "expected_return": result.max_sharpe_portfolio.expected_return,
                        "expected_volatility": result.max_sharpe_portfolio.expected_volatility,
                        "sharpe_ratio": result.max_sharpe_portfolio.sharpe_ratio,
                    },
                    "min_variance_portfolio": {
                        "weights": result.min_variance_portfolio.weights,
                        "expected_return": result.min_variance_portfolio.expected_return,
                        "expected_volatility": result.min_variance_portfolio.expected_volatility,
                        "sharpe_ratio": result.min_variance_portfolio.sharpe_ratio,
                    },
                    "frontier_points_count": len(result.frontier_points),
                    "asset_points": result.asset_points,
                },
                formatted_context=formatted,
                execution_time_ms=result.calculation_time_ms or 0,
            )

        except Exception as e:
            logger.exception(f"Efficient frontier generation failed: {e}")
            return create_error_output(self.schema.name, str(e))


class AnalyzePortfolioDiversificationTool(BaseTool):
    """Tool for comprehensive portfolio diversification analysis.

    Analyzes portfolio diversification through correlation analysis,
    effective N calculation, and provides recommendations.

    Usage by agent:
        analyzePortfolioDiversification(symbols=["AAPL", "GOOGL", "MSFT"])
    """

    def __init__(self):
        super().__init__()
        self.engine = CorrelationEngine()

        self.schema = ToolSchema(
            name="analyzePortfolioDiversification",
            category="portfolio",
            description=(
                "Comprehensive portfolio diversification analysis including correlation matrix, "
                "diversification ratio, effective number of assets, and recommendations."
            ),
            parameters=[
                ToolParameter(
                    name="symbols",
                    type="array",
                    description="List of stock symbols",
                    required=True,
                ),
                ToolParameter(
                    name="weights",
                    type="array",
                    description="Current portfolio weights (optional, defaults to equal)",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolOutput:
        """Execute diversification analysis."""
        try:
            symbols = kwargs.get("symbols", [])
            weights = kwargs.get("weights")

            if not symbols or len(symbols) < 2:
                return create_error_output(
                    self.schema.name,
                    "Need at least 2 symbols for diversification analysis"
                )

            if weights is None:
                weights = [1.0 / len(symbols)] * len(symbols)

            if len(weights) != len(symbols):
                return create_error_output(
                    self.schema.name,
                    "Number of weights must match number of symbols"
                )

            # Fetch data
            price_data = await fetch_portfolio_prices(symbols)

            # Configure and analyze
            config = CorrelationConfig(method=CorrelationMethod.PEARSON)
            result = self.engine.analyze_portfolio_correlation(price_data, weights, config)

            # Format recommendations
            recs = "\n".join(f"  • {r}" for r in result.recommendations) or "  None"

            # Format highest/lowest correlations
            high_corrs = "\n".join(
                f"  {h[0]}-{h[1]}: {h[2]:.3f}" for h in result.highest_correlations
            )
            low_corrs = "\n".join(
                f"  {l[0]}-{l[1]}: {l[2]:.3f}" for l in result.lowest_correlations
            )

            formatted = (
                f"Portfolio Diversification Analysis:\n\n"
                f"Diversification Ratio: {result.diversification_ratio:.2f}\n"
                f"  (Higher = better diversified, >1.0 shows diversification benefit)\n\n"
                f"Effective Number of Assets: {result.effective_n:.1f} / {len(symbols)}\n"
                f"  (How many independent assets the portfolio behaves like)\n\n"
                f"Highest Correlations (concentration risk):\n{high_corrs}\n\n"
                f"Lowest Correlations (diversification opportunities):\n{low_corrs}\n\n"
                f"Recommendations:\n{recs}"
            )

            return create_success_output(
                self.schema.name,
                data={
                    "diversification_ratio": result.diversification_ratio,
                    "effective_n": result.effective_n,
                    "highest_correlations": result.highest_correlations,
                    "lowest_correlations": result.lowest_correlations,
                    "recommendations": result.recommendations,
                    "average_correlation": result.correlation_matrix.average_correlation,
                },
                formatted_context=formatted,
                execution_time_ms=result.calculation_time_ms or 0,
            )

        except Exception as e:
            logger.exception(f"Diversification analysis failed: {e}")
            return create_error_output(self.schema.name, str(e))


class SuggestRebalancingTool(BaseTool):
    """Tool for portfolio rebalancing suggestions.

    Compares current portfolio to optimal and suggests rebalancing actions.

    Usage by agent:
        suggestRebalancing(
            symbols=["AAPL", "GOOGL", "MSFT"],
            current_weights=[0.5, 0.3, 0.2]
        )
    """

    def __init__(self):
        super().__init__()
        self.calculator = RebalancingCalculator()

        self.schema = ToolSchema(
            name="suggestRebalancing",
            category="portfolio",
            description=(
                "Analyze current portfolio and suggest rebalancing to optimal allocation. "
                "Provides buy/sell recommendations with urgency levels."
            ),
            parameters=[
                ToolParameter(
                    name="symbols",
                    type="array",
                    description="List of stock symbols in portfolio",
                    required=True,
                ),
                ToolParameter(
                    name="current_weights",
                    type="array",
                    description="Current portfolio weights (must sum to 1)",
                    required=True,
                ),
                ToolParameter(
                    name="optimization_method",
                    type="string",
                    description="Target optimization method: max_sharpe, min_variance",
                    required=False,
                    default="max_sharpe",
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolOutput:
        """Execute rebalancing analysis."""
        try:
            symbols = kwargs.get("symbols", [])
            current_weights = kwargs.get("current_weights", [])
            method_str = kwargs.get("optimization_method", "max_sharpe")

            if not symbols or len(symbols) < 2:
                return create_error_output(
                    self.schema.name,
                    "Need at least 2 symbols for rebalancing analysis"
                )

            if len(current_weights) != len(symbols):
                return create_error_output(
                    self.schema.name,
                    "Number of weights must match number of symbols"
                )

            # Map method
            method_map = {
                "max_sharpe": OptimizationMethod.MAX_SHARPE,
                "min_variance": OptimizationMethod.MIN_VARIANCE,
            }
            method = method_map.get(method_str, OptimizationMethod.MAX_SHARPE)

            # Fetch data
            data = await fetch_portfolio_data(symbols, current_weights)

            # Configure and analyze
            config = OptimizationConfig(method=method)
            result = self.calculator.suggest_rebalancing(data, config)

            # Format suggestions
            if result.suggestions:
                suggestions_text = "\n".join(
                    f"  [{s.urgency.upper()}] {s.symbol}: {s.action.upper()} "
                    f"({s.current_weight:.1%} → {s.target_weight:.1%})"
                    for s in result.suggestions
                )
            else:
                suggestions_text = "  Portfolio is already well-balanced"

            formatted = (
                f"Portfolio Rebalancing Analysis:\n\n"
                f"Current Sharpe Ratio: {result.current_sharpe:.3f}\n"
                f"Target Sharpe Ratio: {result.target_sharpe:.3f}\n"
                f"Expected Improvement: {result.improvement.get('sharpe_improvement', 0):.3f}\n\n"
                f"Total Turnover: {result.total_turnover:.1%}\n\n"
                f"Rebalancing Suggestions:\n{suggestions_text}"
            )

            return create_success_output(
                self.schema.name,
                data={
                    "suggestions": [
                        {
                            "symbol": s.symbol,
                            "current_weight": s.current_weight,
                            "target_weight": s.target_weight,
                            "action": s.action,
                            "urgency": s.urgency,
                        }
                        for s in result.suggestions
                    ],
                    "current_sharpe": result.current_sharpe,
                    "target_sharpe": result.target_sharpe,
                    "total_turnover": result.total_turnover,
                    "improvement": result.improvement,
                },
                formatted_context=formatted,
                execution_time_ms=result.calculation_time_ms or 0,
            )

        except Exception as e:
            logger.exception(f"Rebalancing analysis failed: {e}")
            return create_error_output(self.schema.name, str(e))
