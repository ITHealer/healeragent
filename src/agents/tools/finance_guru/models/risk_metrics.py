"""
Finance Guru - Enhanced Risk Metrics Models

Pydantic models for comprehensive risk analysis:
- Value at Risk (VaR) - Historical & Parametric
- Conditional VaR (CVaR / Expected Shortfall)
- Sharpe Ratio - Risk-adjusted return
- Sortino Ratio - Downside risk-adjusted return
- Treynor Ratio - Return per unit of systematic risk
- Information Ratio - Active return per tracking error
- Calmar Ratio - Return per unit of max drawdown
- Maximum Drawdown - Worst peak-to-trough decline
- Omega Ratio - Probability-weighted gains vs losses
- Beta/Alpha - Market sensitivity and excess return

WHAT: Data models for risk metrics inputs, configuration, and outputs
WHY: Type-safe, validated data structures for Finance Guru risk analysis
ARCHITECTURE: Layer 1 of 3-layer type-safe architecture

Used by: Risk Assessment, Portfolio Management, Compliance workflows
Author: HealerAgent Development Team
Created: 2025-01-18
"""

from datetime import date
from typing import List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from src.agents.tools.finance_guru.models.base import (
    BaseCalculationResult,
    BaseFinanceModel,
    FinanceValidationError,
    SignalType,
)


# =============================================================================
# INPUT MODELS
# =============================================================================

class RiskDataInput(BaseFinanceModel):
    """
    WHAT: Historical price/return data for risk calculations
    WHY: Ensures valid data before running risk metrics

    VALIDATES:
      - Prices are positive
      - Dates are chronological
      - Sufficient data points (min 30 for statistical validity)

    EDUCATIONAL NOTE:
    Risk metrics require sufficient historical data for statistical validity:
    - VaR/CVaR: Need enough tail observations (min 100 days recommended)
    - Sharpe/Sortino: Need enough periods to estimate mean/std (min 30 days)
    - Max Drawdown: Needs enough time to capture drawdown cycles
    - Beta/Alpha: Need 60+ days for reliable regression estimates
    """

    ticker: str = Field(
        ...,
        description="Stock ticker symbol (e.g., 'AAPL', 'TSLA')",
        pattern=r"^[A-Z0-9\-\.]{1,20}$",
    )
    dates: List[date] = Field(
        ...,
        min_length=30,
        description="Trading dates (min 30 days for statistical validity)"
    )
    prices: List[float] = Field(
        ...,
        min_length=30,
        description="Daily closing prices"
    )
    volumes: Optional[List[float]] = Field(
        default=None,
        description="Daily trading volumes (optional)"
    )

    @field_validator("prices")
    @classmethod
    def prices_must_be_positive(cls, v: List[float]) -> List[float]:
        """Validate all prices are positive."""
        for i, price in enumerate(v):
            if price <= 0:
                raise FinanceValidationError(
                    f"Price at index {i} must be positive",
                    field="prices",
                    value=price,
                    suggestion="Check for data errors or adjust for stock splits",
                )
        return v

    @field_validator("dates")
    @classmethod
    def dates_must_be_sorted(cls, v: List[date]) -> List[date]:
        """Validate dates are in chronological order."""
        if v != sorted(v):
            raise FinanceValidationError(
                "Dates must be in chronological order",
                field="dates",
                suggestion="Sort data by date ascending before analysis",
            )
        return v

    @model_validator(mode="after")
    def validate_alignment(self) -> "RiskDataInput":
        """Ensure prices array matches dates length."""
        if len(self.prices) != len(self.dates):
            raise FinanceValidationError(
                f"Length mismatch: {len(self.dates)} dates but {len(self.prices)} prices",
                field="prices/dates",
                suggestion="Ensure all arrays have the same length",
            )
        if self.volumes is not None and len(self.volumes) != len(self.dates):
            raise FinanceValidationError(
                f"Length mismatch: {len(self.dates)} dates but {len(self.volumes)} volumes",
                field="volumes",
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "ticker": "AAPL",
                "dates": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "prices": [150.0, 151.5, 152.0],
            }]
        }
    }


class BenchmarkDataInput(BaseFinanceModel):
    """
    WHAT: Benchmark data for relative risk metrics (Beta, Alpha, Information Ratio)
    WHY: Many risk metrics compare performance to a benchmark

    EDUCATIONAL NOTE:
    Common benchmarks:
    - SPY: S&P 500 ETF (US large-cap stocks)
    - QQQ: Nasdaq 100 ETF (US tech stocks)
    - IWM: Russell 2000 ETF (US small-cap)
    - VTI: Total US stock market
    - AGG: US aggregate bonds
    """

    ticker: str = Field(
        ...,
        description="Benchmark ticker symbol (e.g., 'SPY')",
        pattern=r"^[A-Z0-9\-\.]{1,20}$",
    )
    dates: List[date] = Field(
        ...,
        min_length=30,
        description="Trading dates"
    )
    prices: List[float] = Field(
        ...,
        min_length=30,
        description="Daily closing prices"
    )

    @field_validator("prices")
    @classmethod
    def prices_must_be_positive(cls, v: List[float]) -> List[float]:
        """Validate all prices are positive."""
        for i, price in enumerate(v):
            if price <= 0:
                raise FinanceValidationError(
                    f"Benchmark price at index {i} must be positive",
                    field="prices",
                    value=price,
                )
        return v


# =============================================================================
# CONFIGURATION
# =============================================================================

class RiskCalculationConfig(BaseFinanceModel):
    """
    WHAT: Configuration for risk metric calculations
    WHY: Standardizes risk calculations with sensible defaults

    EDUCATIONAL NOTE:
    Key configuration parameters:

    confidence_level (0.95):
    - VaR at 95% means: "95% of days, losses won't exceed this amount"
    - Higher confidence = more conservative estimate

    var_method ("historical"):
    - Historical: Uses actual data percentile (no distribution assumption)
    - Parametric: Assumes normal distribution (faster but less accurate for fat tails)

    risk_free_rate (0.045):
    - Current US Treasury rates are ~4.5% (as of 2025)
    - Used for Sharpe, Sortino, Treynor calculations

    trading_days (252):
    - US markets have ~252 trading days per year
    - Used for annualizing daily returns/volatility
    """

    # VaR Configuration
    confidence_level: float = Field(
        default=0.95,
        ge=0.90,
        le=0.99,
        description="Confidence level for VaR/CVaR (default: 0.95 = 95%)"
    )
    var_method: Literal["historical", "parametric"] = Field(
        default="historical",
        description="VaR calculation method"
    )

    # Return Calculation
    risk_free_rate: float = Field(
        default=0.045,
        ge=0.0,
        le=0.20,
        description="Annual risk-free rate (default: 4.5%)"
    )
    target_return: Optional[float] = Field(
        default=None,
        ge=-0.5,
        le=1.0,
        description="Target return for Omega ratio (if not set, uses risk-free rate)"
    )

    # Annualization
    trading_days: int = Field(
        default=252,
        ge=200,
        le=365,
        description="Trading days per year for annualization (default: 252)"
    )

    # Rolling Window
    rolling_window: Optional[int] = Field(
        default=None,
        ge=20,
        le=252,
        description="Rolling window for metrics (if None, uses all data)"
    )


# =============================================================================
# OUTPUT MODELS - INDIVIDUAL METRICS
# =============================================================================

class VaROutput(BaseFinanceModel):
    """
    WHAT: Value at Risk calculation result
    WHY: Shows maximum expected loss at given confidence level

    EDUCATIONAL NOTE:
    VaR Formula (Historical): percentile(returns, (1 - confidence) × 100)
    VaR Formula (Parametric): mean + z-score × std_dev

    INTERPRETATION:
    VaR of -0.035 at 95% confidence means:
    - "95% of days, daily losses won't exceed 3.5%"
    - "1 in 20 days, losses may exceed 3.5%"

    For a $500K portfolio with -3.5% VaR:
    - Expected max daily loss (95% of time): $17,500
    """

    var_value: float = Field(
        ...,
        description="VaR value (negative number = loss)"
    )
    var_percent: float = Field(
        ...,
        description="VaR as percentage"
    )
    var_dollar: Optional[float] = Field(
        default=None,
        description="VaR in dollar terms (if portfolio value provided)"
    )
    confidence_level: float = Field(
        ...,
        ge=0.90,
        le=0.99,
        description="Confidence level used"
    )
    method: Literal["historical", "parametric"] = Field(
        ...,
        description="Calculation method used"
    )


class CVaROutput(BaseFinanceModel):
    """
    WHAT: Conditional VaR (Expected Shortfall) result
    WHY: Shows expected loss WHEN losses exceed VaR

    EDUCATIONAL NOTE:
    CVaR = average of all returns worse than VaR

    INTERPRETATION:
    If VaR is -3.5% and CVaR is -5.2%:
    - "When losses DO exceed 3.5% (worst 5% of days)"
    - "The average loss on those bad days is 5.2%"

    CVaR is preferred by regulators because:
    1. Captures tail risk (extreme events)
    2. Is a "coherent" risk measure
    3. Better for fat-tailed distributions
    """

    cvar_value: float = Field(
        ...,
        description="CVaR value (more extreme than VaR)"
    )
    cvar_percent: float = Field(
        ...,
        description="CVaR as percentage"
    )
    cvar_dollar: Optional[float] = Field(
        default=None,
        description="CVaR in dollar terms"
    )
    tail_observations: int = Field(
        ...,
        ge=0,
        description="Number of observations in the tail"
    )


class SharpeRatioOutput(BaseFinanceModel):
    """
    WHAT: Sharpe Ratio - risk-adjusted return
    WHY: Answers "Am I being paid enough for the risk I'm taking?"

    EDUCATIONAL NOTE:
    Formula: (Return - Risk-Free Rate) / Volatility

    INTERPRETATION:
    Sharpe of 1.25 means:
    - "For every 1% of volatility, you earn 1.25% excess return"

    Benchmarks:
    - < 0: Losing money relative to risk-free rate
    - 0-1: Poor risk-adjusted return
    - 1-2: Good risk-adjusted return
    - 2-3: Very good
    - > 3: Excellent (rare, often unsustainable)
    """

    sharpe_ratio: float = Field(
        ...,
        description="Annualized Sharpe Ratio"
    )
    excess_return: float = Field(
        ...,
        description="Annualized excess return over risk-free rate"
    )
    volatility: float = Field(
        ...,
        ge=0,
        description="Annualized volatility used in calculation"
    )
    risk_free_rate: float = Field(
        ...,
        description="Risk-free rate used"
    )
    quality: Literal["poor", "acceptable", "good", "excellent"] = Field(
        ...,
        description="Sharpe ratio quality assessment"
    )


class SortinoRatioOutput(BaseFinanceModel):
    """
    WHAT: Sortino Ratio - downside risk-adjusted return
    WHY: Like Sharpe but only penalizes DOWNSIDE volatility

    EDUCATIONAL NOTE:
    Formula: (Return - Risk-Free Rate) / Downside Deviation
    Downside Deviation = std(returns where returns < 0)

    WHY SORTINO > SHARPE:
    - Investors don't mind upside volatility (big gains are good!)
    - We only care about downside volatility (losses hurt)
    - Sortino captures this asymmetry

    INTERPRETATION:
    Sortino of 1.8 means:
    - "For every 1% of DOWNSIDE volatility, you earn 1.8% excess return"
    - Higher than Sharpe suggests positive skew (more upside than downside)
    """

    sortino_ratio: float = Field(
        ...,
        description="Annualized Sortino Ratio"
    )
    downside_deviation: float = Field(
        ...,
        ge=0,
        description="Annualized downside deviation"
    )
    negative_returns_count: int = Field(
        ...,
        ge=0,
        description="Number of negative return days"
    )
    quality: Literal["poor", "acceptable", "good", "excellent"] = Field(
        ...,
        description="Sortino ratio quality assessment"
    )


class TreynorRatioOutput(BaseFinanceModel):
    """
    WHAT: Treynor Ratio - return per unit of systematic (market) risk
    WHY: Measures reward for bearing market risk specifically

    EDUCATIONAL NOTE:
    Formula: (Return - Risk-Free Rate) / Beta

    KEY DIFFERENCE FROM SHARPE:
    - Sharpe uses TOTAL volatility
    - Treynor uses BETA (systematic risk only)

    WHY IT MATTERS:
    - For diversified portfolios, unsystematic risk is diversified away
    - Only systematic (market) risk remains
    - Treynor rewards you for bearing undiversifiable risk

    INTERPRETATION:
    Treynor of 0.08 means:
    - "For every 1 unit of beta, you earn 8% excess return"
    """

    treynor_ratio: float = Field(
        ...,
        description="Annualized Treynor Ratio"
    )
    beta: float = Field(
        ...,
        description="Portfolio beta used in calculation"
    )
    systematic_risk_premium: float = Field(
        ...,
        description="Return earned per unit of beta"
    )


class InformationRatioOutput(BaseFinanceModel):
    """
    WHAT: Information Ratio - active return per unit of tracking error
    WHY: Measures skill in beating a benchmark consistently

    EDUCATIONAL NOTE:
    Formula: (Portfolio Return - Benchmark Return) / Tracking Error
    Tracking Error = std(Portfolio Return - Benchmark Return)

    INTERPRETATION:
    IR of 0.75 means:
    - "For every 1% of deviation from benchmark"
    - "You earn 0.75% excess return"

    Benchmarks:
    - < 0: Underperforming benchmark
    - 0-0.5: Poor active management
    - 0.5-1.0: Good active management
    - > 1.0: Excellent active management (rare)

    USED FOR:
    - Evaluating fund managers
    - Comparing active vs passive strategies
    - Assessing consistency of outperformance
    """

    information_ratio: float = Field(
        ...,
        description="Information Ratio"
    )
    active_return: float = Field(
        ...,
        description="Annualized active return (portfolio - benchmark)"
    )
    tracking_error: float = Field(
        ...,
        ge=0,
        description="Annualized tracking error"
    )
    benchmark_ticker: str = Field(
        ...,
        description="Benchmark used for comparison"
    )
    quality: Literal["underperforming", "poor", "good", "excellent"] = Field(
        ...,
        description="Information ratio quality assessment"
    )


class CalmarRatioOutput(BaseFinanceModel):
    """
    WHAT: Calmar Ratio - return per unit of maximum drawdown
    WHY: Measures return relative to worst-case historical loss

    EDUCATIONAL NOTE:
    Formula: Annualized Return / |Max Drawdown|

    INTERPRETATION:
    Calmar of 0.85 means:
    - "For every 1% of max drawdown risk"
    - "You earned 0.85% annual return"

    WHY IT MATTERS:
    - Focuses on worst-case scenarios
    - Important for risk-averse investors
    - Considers psychological pain of drawdowns
    """

    calmar_ratio: float = Field(
        ...,
        description="Calmar Ratio"
    )
    annual_return: float = Field(
        ...,
        description="Annualized return"
    )
    max_drawdown: float = Field(
        ...,
        le=0,
        description="Maximum drawdown (negative value)"
    )
    max_drawdown_duration_days: Optional[int] = Field(
        default=None,
        ge=0,
        description="Days to recover from max drawdown"
    )


class MaxDrawdownOutput(BaseFinanceModel):
    """
    WHAT: Maximum Drawdown analysis
    WHY: Shows worst peak-to-trough decline in history

    EDUCATIONAL NOTE:
    Formula: (Trough Price - Peak Price) / Peak Price

    INTERPRETATION:
    Max Drawdown of -32% means:
    - "At worst point, down 32% from previous peak"

    Pain Tolerance Guide:
    - -10%: Mild correction (most can handle)
    - -20%: Bear market territory (tests conviction)
    - -30%: Major decline (many capitulate)
    - -50%: Catastrophic (most capitulate)
    """

    max_drawdown: float = Field(
        ...,
        le=0,
        description="Maximum drawdown percentage"
    )
    max_drawdown_dollar: Optional[float] = Field(
        default=None,
        le=0,
        description="Max drawdown in dollar terms"
    )
    peak_date: date = Field(
        ...,
        description="Date of the peak before drawdown"
    )
    trough_date: date = Field(
        ...,
        description="Date of the trough (bottom)"
    )
    peak_price: float = Field(
        ...,
        gt=0,
        description="Price at peak"
    )
    trough_price: float = Field(
        ...,
        gt=0,
        description="Price at trough"
    )
    recovery_date: Optional[date] = Field(
        default=None,
        description="Date when recovered to peak (None if not recovered)"
    )
    drawdown_duration_days: int = Field(
        ...,
        ge=0,
        description="Days from peak to trough"
    )
    recovery_duration_days: Optional[int] = Field(
        default=None,
        ge=0,
        description="Days from trough to recovery"
    )
    current_drawdown: float = Field(
        ...,
        le=0,
        description="Current drawdown from most recent peak"
    )


class OmegaRatioOutput(BaseFinanceModel):
    """
    WHAT: Omega Ratio - probability-weighted gains vs losses
    WHY: Considers entire return distribution, not just mean/variance

    EDUCATIONAL NOTE:
    Formula: Sum(returns above threshold) / Sum(|returns below threshold|)

    INTERPRETATION:
    Omega of 1.35 means:
    - "Probability-weighted gains are 1.35x probability-weighted losses"
    - Omega > 1: More upside than downside
    - Omega < 1: More downside than upside
    - Omega = 1: Balanced (threshold is the median)

    WHY OMEGA MATTERS:
    - Doesn't assume normal distribution
    - Captures skewness and kurtosis
    - Better for asymmetric return distributions
    """

    omega_ratio: float = Field(
        ...,
        ge=0,
        description="Omega Ratio"
    )
    threshold_return: float = Field(
        ...,
        description="Threshold return used (often risk-free rate)"
    )
    gains_above_threshold: float = Field(
        ...,
        ge=0,
        description="Sum of returns above threshold"
    )
    losses_below_threshold: float = Field(
        ...,
        le=0,
        description="Sum of returns below threshold"
    )
    win_rate: float = Field(
        ...,
        ge=0,
        le=1,
        description="Percentage of returns above threshold"
    )


class BetaAlphaOutput(BaseFinanceModel):
    """
    WHAT: Beta (market sensitivity) and Alpha (excess return)
    WHY: Core CAPM metrics for understanding risk/return

    EDUCATIONAL NOTE:
    Beta Formula: Cov(asset, benchmark) / Var(benchmark)
    Alpha Formula: Return - (Rf + Beta × (Benchmark Return - Rf))

    BETA INTERPRETATION:
    - Beta = 1.0: Moves with market
    - Beta = 1.5: 50% more volatile than market
    - Beta = 0.5: 50% less volatile than market
    - Beta < 0: Moves opposite to market (rare)

    ALPHA INTERPRETATION:
    - Alpha > 0: Outperforming risk-adjusted expectations
    - Alpha = 0: Performing as expected for risk level
    - Alpha < 0: Underperforming risk-adjusted expectations
    """

    beta: float = Field(
        ...,
        description="Portfolio beta"
    )
    alpha: float = Field(
        ...,
        description="Annualized alpha (Jensen's alpha)"
    )
    r_squared: float = Field(
        ...,
        ge=0,
        le=1,
        description="R-squared (% of variance explained by benchmark)"
    )
    benchmark_ticker: str = Field(
        ...,
        description="Benchmark used"
    )
    correlation: float = Field(
        ...,
        ge=-1,
        le=1,
        description="Correlation with benchmark"
    )

    @property
    def beta_category(self) -> str:
        """Categorize beta level."""
        if self.beta < 0:
            return "inverse"
        elif self.beta < 0.5:
            return "defensive"
        elif self.beta < 1.0:
            return "low_volatility"
        elif self.beta < 1.5:
            return "moderate"
        else:
            return "aggressive"


class VolatilityOutput(BaseFinanceModel):
    """
    WHAT: Comprehensive volatility metrics
    WHY: Multiple views of price variability

    EDUCATIONAL NOTE:
    Volatility is the square root of variance.
    Annualized by multiplying daily by sqrt(252).

    BENCHMARKS:
    - 10-20%: Low volatility (large-cap, bonds)
    - 20-40%: Medium (typical stocks)
    - 40-80%: High (growth stocks, small-cap)
    - 80%+: Extreme (crypto, penny stocks)
    """

    annual_volatility: float = Field(
        ...,
        ge=0,
        description="Annualized volatility"
    )
    daily_volatility: float = Field(
        ...,
        ge=0,
        description="Daily volatility"
    )
    monthly_volatility: float = Field(
        ...,
        ge=0,
        description="Monthly volatility"
    )
    volatility_regime: Literal["low", "normal", "high", "extreme"] = Field(
        ...,
        description="Current volatility regime"
    )
    volatility_percentile: float = Field(
        ...,
        ge=0,
        le=100,
        description="Current volatility vs historical (percentile)"
    )


# =============================================================================
# COMBINED OUTPUT MODEL
# =============================================================================

class RiskMetricsOutput(BaseCalculationResult):
    """
    WHAT: Complete risk metrics analysis output
    WHY: Provides comprehensive risk profile in single response

    AGENT USE CASES:
    - Risk Assessment: Full risk profile for investment decisions
    - Portfolio Review: Periodic risk monitoring
    - Compliance: Risk limit monitoring
    - Strategy Evaluation: Compare strategies on risk-adjusted basis
    """

    # Core Risk Metrics
    var: VaROutput = Field(
        ...,
        description="Value at Risk analysis"
    )
    cvar: CVaROutput = Field(
        ...,
        description="Conditional VaR (Expected Shortfall)"
    )
    max_drawdown: MaxDrawdownOutput = Field(
        ...,
        description="Maximum drawdown analysis"
    )
    volatility: VolatilityOutput = Field(
        ...,
        description="Volatility metrics"
    )

    # Risk-Adjusted Returns
    sharpe_ratio: SharpeRatioOutput = Field(
        ...,
        description="Sharpe Ratio analysis"
    )
    sortino_ratio: SortinoRatioOutput = Field(
        ...,
        description="Sortino Ratio analysis"
    )
    calmar_ratio: CalmarRatioOutput = Field(
        ...,
        description="Calmar Ratio analysis"
    )
    omega_ratio: OmegaRatioOutput = Field(
        ...,
        description="Omega Ratio analysis"
    )

    # Benchmark-Relative (Optional)
    beta_alpha: Optional[BetaAlphaOutput] = Field(
        default=None,
        description="Beta/Alpha analysis (requires benchmark)"
    )
    treynor_ratio: Optional[TreynorRatioOutput] = Field(
        default=None,
        description="Treynor Ratio (requires benchmark)"
    )
    information_ratio: Optional[InformationRatioOutput] = Field(
        default=None,
        description="Information Ratio (requires benchmark)"
    )

    # Summary
    risk_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall risk score (0=safest, 100=riskiest)"
    )
    risk_level: Literal["conservative", "moderate", "aggressive", "speculative"] = Field(
        ...,
        description="Risk level classification"
    )
    summary: str = Field(
        ...,
        description="Human-readable risk summary"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "ticker": "TSLA",
                "calculation_date": "2025-01-18",
                "risk_score": 72.5,
                "risk_level": "aggressive",
                "summary": "High volatility growth stock with strong returns but significant drawdown risk"
            }]
        }
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Input Models
    "RiskDataInput",
    "BenchmarkDataInput",
    # Configuration
    "RiskCalculationConfig",
    # Individual Metric Outputs
    "VaROutput",
    "CVaROutput",
    "SharpeRatioOutput",
    "SortinoRatioOutput",
    "TreynorRatioOutput",
    "InformationRatioOutput",
    "CalmarRatioOutput",
    "MaxDrawdownOutput",
    "OmegaRatioOutput",
    "BetaAlphaOutput",
    "VolatilityOutput",
    # Combined Output
    "RiskMetricsOutput",
]
