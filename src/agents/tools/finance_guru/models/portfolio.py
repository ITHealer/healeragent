"""
Finance Guru - Portfolio Analysis Models (Phase 4)

Layer 1: Pydantic models for portfolio optimization and correlation analysis.

This module provides validated data structures for:
- Portfolio optimization (Mean-Variance, Risk Parity, Min Variance, Max Sharpe, Black-Litterman)
- Correlation analysis (Pearson, Spearman, Kendall)
- Covariance matrix calculations
- Efficient frontier generation

EDUCATIONAL NOTES:
- Portfolio optimization: Finding the best asset allocation to maximize returns
  for a given risk level (or minimize risk for a given return)
- Correlation: Measures how assets move together (-1 to +1)
- Covariance: Measures joint variability of assets
- Efficient Frontier: Set of optimal portfolios offering highest return for each risk level

Author: HealerAgent Development Team
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import Field, field_validator

from src.agents.tools.finance_guru.models.base import (
    BaseFinanceModel,
    BaseCalculationResult,
)


# =============================================================================
# ENUMS
# =============================================================================


class OptimizationMethod(str, Enum):
    """Portfolio optimization methods available.

    MEAN_VARIANCE: Classic Markowitz optimization balancing risk and return
    MIN_VARIANCE: Minimize portfolio volatility regardless of return
    MAX_SHARPE: Maximize risk-adjusted returns (Sharpe ratio)
    RISK_PARITY: Equal risk contribution from each asset
    BLACK_LITTERMAN: Combines market equilibrium with investor views
    """
    MEAN_VARIANCE = "mean_variance"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"


class CorrelationMethod(str, Enum):
    """Correlation calculation methods.

    PEARSON: Linear correlation (most common), assumes normal distribution
    SPEARMAN: Rank-based, captures monotonic relationships
    KENDALL: Rank-based, robust to outliers
    """
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class RiskModel(str, Enum):
    """Risk model for covariance estimation.

    SAMPLE: Simple historical sample covariance
    SHRINKAGE: Ledoit-Wolf shrinkage for stability
    EXPONENTIAL: Exponentially weighted (recent data weighted more)
    """
    SAMPLE = "sample"
    SHRINKAGE = "shrinkage"
    EXPONENTIAL = "exponential"


# =============================================================================
# INPUT MODELS
# =============================================================================


class AssetData(BaseFinanceModel):
    """Data for a single asset in the portfolio.

    Attributes:
        symbol: Asset ticker symbol (e.g., "AAPL")
        prices: Historical price series
        returns: Optional pre-calculated returns (if not provided, calculated from prices)
        weight: Current portfolio weight (0.0 to 1.0)
    """
    symbol: str = Field(..., min_length=1, max_length=20, description="Asset ticker symbol")
    prices: list[float] = Field(..., min_length=2, description="Historical price series")
    returns: Optional[list[float]] = Field(None, description="Pre-calculated returns")
    weight: float = Field(0.0, ge=0.0, le=1.0, description="Current portfolio weight")

    @field_validator("prices")
    @classmethod
    def validate_prices(cls, v: list[float]) -> list[float]:
        """Ensure all prices are positive."""
        if any(p <= 0 for p in v):
            raise ValueError("All prices must be positive")
        return v


class PortfolioDataInput(BaseFinanceModel):
    """Input data for portfolio analysis.

    Contains price/return data for multiple assets to be analyzed together.

    Attributes:
        assets: List of asset data (prices, returns, weights)
        risk_free_rate: Annual risk-free rate for calculations (default 0.02 = 2%)
        trading_days_per_year: Number of trading days for annualization

    Example:
        portfolio = PortfolioDataInput(
            assets=[
                AssetData(symbol="AAPL", prices=[150.0, 152.0, 148.0, ...]),
                AssetData(symbol="GOOGL", prices=[2800.0, 2850.0, 2780.0, ...]),
            ],
            risk_free_rate=0.02
        )
    """
    assets: list[AssetData] = Field(..., min_length=2, description="List of assets in portfolio")
    risk_free_rate: float = Field(0.02, ge=0.0, le=0.5, description="Annual risk-free rate")
    trading_days_per_year: int = Field(252, ge=200, le=365, description="Trading days per year")

    @field_validator("assets")
    @classmethod
    def validate_assets_length(cls, v: list[AssetData]) -> list[AssetData]:
        """Ensure all assets have the same number of data points."""
        if len(v) < 2:
            raise ValueError("Portfolio must contain at least 2 assets")

        lengths = [len(asset.prices) for asset in v]
        if len(set(lengths)) > 1:
            raise ValueError(f"All assets must have the same number of price points. Found: {lengths}")

        return v


class PortfolioPriceData(BaseFinanceModel):
    """Simplified price data input for correlation analysis.

    Attributes:
        symbols: List of asset symbols
        price_matrix: 2D matrix where each row is an asset's prices
        dates: Optional date labels for the price series
    """
    symbols: list[str] = Field(..., min_length=2, description="Asset symbols")
    price_matrix: list[list[float]] = Field(..., description="Price matrix [assets x time]")
    dates: Optional[list[str]] = Field(None, description="Date labels")

    @field_validator("price_matrix")
    @classmethod
    def validate_matrix(cls, v: list[list[float]]) -> list[list[float]]:
        """Ensure matrix is rectangular and has valid values."""
        if not v:
            raise ValueError("Price matrix cannot be empty")

        length = len(v[0])
        for row in v:
            if len(row) != length:
                raise ValueError("All rows in price matrix must have the same length")
            if any(p <= 0 for p in row):
                raise ValueError("All prices must be positive")

        return v


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class OptimizationConfig(BaseFinanceModel):
    """Configuration for portfolio optimization.

    Attributes:
        method: Optimization method to use
        target_return: Target annual return (for mean-variance optimization)
        target_volatility: Target annual volatility (alternative constraint)
        min_weight: Minimum weight per asset (0.0 for no short selling)
        max_weight: Maximum weight per asset (prevents concentration)
        risk_aversion: Risk aversion parameter (higher = more conservative)

    EDUCATIONAL NOTE:
    - Risk aversion of 1 = neutral, >1 = risk-averse, <1 = risk-seeking
    - Setting min_weight=0 prevents short selling
    - Max_weight < 1 ensures diversification
    """
    method: OptimizationMethod = Field(
        OptimizationMethod.MAX_SHARPE,
        description="Optimization method"
    )
    target_return: Optional[float] = Field(
        None, ge=0.0, le=2.0,
        description="Target annual return (for mean-variance)"
    )
    target_volatility: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Target annual volatility"
    )
    min_weight: float = Field(
        0.0, ge=-1.0, le=1.0,
        description="Minimum weight per asset"
    )
    max_weight: float = Field(
        1.0, ge=0.0, le=1.0,
        description="Maximum weight per asset"
    )
    risk_aversion: float = Field(
        1.0, ge=0.1, le=10.0,
        description="Risk aversion parameter"
    )

    @field_validator("max_weight")
    @classmethod
    def validate_weights(cls, v: float, info) -> float:
        """Ensure max_weight >= min_weight."""
        min_w = info.data.get("min_weight", 0.0)
        if v < min_w:
            raise ValueError(f"max_weight ({v}) must be >= min_weight ({min_w})")
        return v


class CorrelationConfig(BaseFinanceModel):
    """Configuration for correlation analysis.

    Attributes:
        method: Correlation calculation method
        risk_model: Covariance estimation method
        rolling_window: Window size for rolling correlation (optional)
        min_periods: Minimum periods required for calculation
        annualize: Whether to annualize results
    """
    method: CorrelationMethod = Field(
        CorrelationMethod.PEARSON,
        description="Correlation method"
    )
    risk_model: RiskModel = Field(
        RiskModel.SAMPLE,
        description="Risk/covariance model"
    )
    rolling_window: Optional[int] = Field(
        None, ge=10, le=500,
        description="Rolling window size"
    )
    min_periods: int = Field(
        20, ge=5, le=100,
        description="Minimum periods for calculation"
    )
    annualize: bool = Field(True, description="Annualize covariance matrix")
    trading_days: int = Field(252, ge=200, le=365, description="Trading days per year")


class BlackLittermanConfig(BaseFinanceModel):
    """Configuration for Black-Litterman model.

    Black-Litterman combines market equilibrium returns with investor views
    to generate adjusted expected returns for optimization.

    Attributes:
        market_weights: Market capitalization weights
        views: Dictionary of view portfolios and expected returns
        view_confidences: Confidence levels for each view (0-1)
        tau: Scaling factor for uncertainty in equilibrium returns

    EDUCATIONAL NOTE:
    - Views express investor beliefs about relative or absolute returns
    - Higher confidence = view has more weight in final estimates
    - tau typically small (0.01-0.05) to reflect uncertainty
    """
    market_weights: dict[str, float] = Field(..., description="Market cap weights by symbol")
    views: dict[str, float] = Field(default_factory=dict, description="Return views by symbol")
    view_confidences: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence for each view (0-1)"
    )
    tau: float = Field(0.05, ge=0.001, le=0.5, description="Uncertainty scaling factor")


class EfficientFrontierConfig(BaseFinanceModel):
    """Configuration for efficient frontier generation.

    Attributes:
        num_portfolios: Number of portfolios to generate on frontier
        return_range: Optional (min, max) return range
        include_assets: Include individual asset points
    """
    num_portfolios: int = Field(50, ge=10, le=200, description="Number of frontier portfolios")
    return_range: Optional[tuple[float, float]] = Field(
        None,
        description="(min, max) return range for frontier"
    )
    include_assets: bool = Field(True, description="Include individual asset points")


# =============================================================================
# OUTPUT MODELS
# =============================================================================


class AssetAllocation(BaseFinanceModel):
    """Optimal allocation for a single asset.

    Attributes:
        symbol: Asset ticker
        weight: Optimal weight (0-1 or negative for short)
        expected_return: Expected annual return
        risk_contribution: Contribution to portfolio risk
    """
    symbol: str = Field(..., description="Asset symbol")
    weight: float = Field(..., description="Optimal weight")
    expected_return: float = Field(..., description="Expected annual return")
    risk_contribution: float = Field(0.0, description="Risk contribution to portfolio")


class OptimizationOutput(BaseCalculationResult):
    """Result of portfolio optimization.

    Attributes:
        allocations: List of asset allocations
        expected_return: Portfolio expected annual return
        expected_volatility: Portfolio expected annual volatility
        sharpe_ratio: Portfolio Sharpe ratio
        method_used: Optimization method that was used
        converged: Whether optimization converged successfully
    """
    allocations: list[AssetAllocation] = Field(..., description="Optimal allocations")
    expected_return: float = Field(..., description="Portfolio expected return")
    expected_volatility: float = Field(..., description="Portfolio expected volatility")
    sharpe_ratio: float = Field(..., description="Portfolio Sharpe ratio")
    method_used: OptimizationMethod = Field(..., description="Method used")
    converged: bool = Field(True, description="Optimization convergence status")
    iterations: Optional[int] = Field(None, description="Number of iterations")

    # Risk metrics
    var_95: Optional[float] = Field(None, description="95% Value at Risk")
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown")


class FrontierPoint(BaseFinanceModel):
    """Single point on the efficient frontier.

    Attributes:
        expected_return: Expected annual return at this point
        expected_volatility: Expected annual volatility at this point
        sharpe_ratio: Sharpe ratio at this point
        weights: Asset weights for this portfolio
    """
    expected_return: float = Field(..., description="Expected return")
    expected_volatility: float = Field(..., description="Expected volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    weights: dict[str, float] = Field(..., description="Asset weights")


class EfficientFrontierOutput(BaseCalculationResult):
    """Efficient frontier calculation result.

    The efficient frontier represents the set of optimal portfolios that offer
    the highest expected return for each level of risk.

    Attributes:
        frontier_points: Points along the efficient frontier
        max_sharpe_portfolio: Portfolio with maximum Sharpe ratio
        min_variance_portfolio: Minimum variance portfolio
        asset_points: Risk/return coordinates of individual assets
    """
    frontier_points: list[FrontierPoint] = Field(..., description="Frontier portfolios")
    max_sharpe_portfolio: FrontierPoint = Field(..., description="Max Sharpe portfolio")
    min_variance_portfolio: FrontierPoint = Field(..., description="Min variance portfolio")
    asset_points: Optional[dict[str, tuple[float, float]]] = Field(
        None,
        description="Individual asset (volatility, return) points"
    )
    symbols: list[str] = Field(..., description="Asset symbols in analysis")


class CorrelationMatrixOutput(BaseCalculationResult):
    """Correlation matrix calculation result.

    Correlation measures linear relationship between assets (-1 to +1):
    - +1: Perfect positive correlation (move together)
    - 0: No linear relationship
    - -1: Perfect negative correlation (move opposite)

    DIVERSIFICATION TIP:
    Lower correlations between assets provide better diversification benefits.

    Attributes:
        matrix: Correlation matrix as nested dict
        symbols: Asset symbols (row/column labels)
        method: Correlation method used
        average_correlation: Average pairwise correlation
        min_correlation: Minimum correlation pair
        max_correlation: Maximum correlation pair (excluding diagonal)
    """
    matrix: dict[str, dict[str, float]] = Field(..., description="Correlation matrix")
    symbols: list[str] = Field(..., description="Asset symbols")
    method: CorrelationMethod = Field(..., description="Method used")
    average_correlation: float = Field(..., description="Average pairwise correlation")
    min_correlation: tuple[str, str, float] = Field(
        ...,
        description="Min correlation (symbol1, symbol2, value)"
    )
    max_correlation: tuple[str, str, float] = Field(
        ...,
        description="Max correlation (symbol1, symbol2, value)"
    )


class CovarianceMatrixOutput(BaseCalculationResult):
    """Covariance matrix calculation result.

    Covariance measures joint variability of assets. Unlike correlation,
    covariance is not bounded and depends on asset volatilities.

    Used in portfolio optimization to calculate portfolio variance.

    Attributes:
        matrix: Covariance matrix as nested dict
        symbols: Asset symbols
        risk_model: Risk model used for estimation
        is_annualized: Whether matrix is annualized
        determinant: Matrix determinant (measure of total variance)
    """
    matrix: dict[str, dict[str, float]] = Field(..., description="Covariance matrix")
    symbols: list[str] = Field(..., description="Asset symbols")
    risk_model: RiskModel = Field(..., description="Risk model used")
    is_annualized: bool = Field(..., description="Whether annualized")
    determinant: Optional[float] = Field(None, description="Matrix determinant")
    condition_number: Optional[float] = Field(None, description="Condition number")


class RollingCorrelationOutput(BaseCalculationResult):
    """Rolling correlation calculation result.

    Rolling correlation shows how the relationship between two assets
    changes over time, useful for detecting regime changes.

    Attributes:
        asset_pair: The two assets being compared
        correlations: Time series of rolling correlations
        dates: Date labels for each correlation value
        window_size: Rolling window size used
        average: Average correlation over period
        std: Standard deviation of rolling correlations
        trend: Whether correlation is trending up, down, or stable
    """
    asset_pair: tuple[str, str] = Field(..., description="Asset pair (symbol1, symbol2)")
    correlations: list[float] = Field(..., description="Rolling correlation values")
    dates: Optional[list[str]] = Field(None, description="Date labels")
    window_size: int = Field(..., description="Window size used")
    average: float = Field(..., description="Average correlation")
    std: float = Field(..., description="Correlation volatility")
    trend: str = Field(..., description="Trend direction: up/down/stable")
    current: float = Field(..., description="Most recent correlation")


class PortfolioCorrelationOutput(BaseCalculationResult):
    """Complete portfolio correlation analysis.

    Comprehensive view of portfolio diversification through correlation analysis.

    Attributes:
        correlation_matrix: Full correlation matrix
        covariance_matrix: Full covariance matrix
        diversification_ratio: Ratio of weighted avg vol to portfolio vol
        effective_n: Effective number of independent assets
        highest_correlations: Most correlated pairs (potential concentration risk)
        lowest_correlations: Least correlated pairs (diversification opportunities)
    """
    correlation_matrix: CorrelationMatrixOutput = Field(..., description="Correlation matrix")
    covariance_matrix: CovarianceMatrixOutput = Field(..., description="Covariance matrix")
    diversification_ratio: float = Field(
        ...,
        description="Diversification ratio (higher = better diversified)"
    )
    effective_n: float = Field(
        ...,
        description="Effective number of independent assets"
    )
    highest_correlations: list[tuple[str, str, float]] = Field(
        ...,
        description="Most correlated pairs"
    )
    lowest_correlations: list[tuple[str, str, float]] = Field(
        ...,
        description="Least correlated pairs"
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Diversification recommendations"
    )


class RebalancingSuggestion(BaseFinanceModel):
    """Suggestion for portfolio rebalancing.

    Attributes:
        symbol: Asset to adjust
        current_weight: Current portfolio weight
        target_weight: Optimal/target weight
        action: "buy", "sell", or "hold"
        urgency: "high", "medium", or "low"
        reason: Explanation for the suggestion
    """
    symbol: str = Field(..., description="Asset symbol")
    current_weight: float = Field(..., description="Current weight")
    target_weight: float = Field(..., description="Target weight")
    action: str = Field(..., description="Action: buy/sell/hold")
    urgency: str = Field(..., description="Urgency: high/medium/low")
    reason: str = Field(..., description="Explanation")

    @property
    def weight_change(self) -> float:
        """Calculate the weight change needed."""
        return self.target_weight - self.current_weight


class RebalancingOutput(BaseCalculationResult):
    """Portfolio rebalancing analysis result.

    Attributes:
        suggestions: List of rebalancing suggestions per asset
        total_turnover: Total portfolio turnover (sum of |weight changes|)
        estimated_cost: Estimated transaction cost
        current_sharpe: Current portfolio Sharpe ratio
        target_sharpe: Expected Sharpe ratio after rebalancing
        improvement: Expected improvement metrics
    """
    suggestions: list[RebalancingSuggestion] = Field(..., description="Rebalancing suggestions")
    total_turnover: float = Field(..., description="Total turnover (0-2)")
    estimated_cost: Optional[float] = Field(None, description="Estimated cost ($)")
    current_sharpe: float = Field(..., description="Current Sharpe ratio")
    target_sharpe: float = Field(..., description="Target Sharpe ratio")
    improvement: dict[str, float] = Field(
        default_factory=dict,
        description="Expected improvements"
    )
