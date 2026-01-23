"""
Finance Guru - Enhanced Risk Metrics Calculator

Implements comprehensive risk calculations:
1. Value at Risk (VaR) - Historical & Parametric methods
2. Conditional VaR (CVaR / Expected Shortfall)
3. Sharpe Ratio - Risk-adjusted return
4. Sortino Ratio - Downside risk-adjusted return
5. Treynor Ratio - Return per unit of systematic risk
6. Information Ratio - Active return per tracking error
7. Calmar Ratio - Return per unit of max drawdown
8. Maximum Drawdown - Worst peak-to-trough decline
9. Omega Ratio - Probability-weighted gains vs losses
10. Beta/Alpha - Market sensitivity and excess return

WHAT: Calculators for comprehensive risk analysis
WHY: Provides validated, type-safe risk metrics for portfolio decisions
ARCHITECTURE: Layer 2 of 3-layer type-safe architecture

Author: HealerAgent Development Team
Created: 2025-01-18
"""

import logging
import math
import warnings
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.agents.tools.finance_guru.calculators.base import (
    BaseCalculator,
    CalculationError,
    InsufficientDataError,
)
from src.agents.tools.finance_guru.models.base import FinanceValidationError
from src.agents.tools.finance_guru.models.risk_metrics import (
    # Inputs
    RiskDataInput,
    BenchmarkDataInput,
    RiskCalculationConfig,
    # Outputs
    VaROutput,
    CVaROutput,
    SharpeRatioOutput,
    SortinoRatioOutput,
    TreynorRatioOutput,
    InformationRatioOutput,
    CalmarRatioOutput,
    MaxDrawdownOutput,
    OmegaRatioOutput,
    BetaAlphaOutput,
    VolatilityOutput,
    RiskMetricsOutput,
)

logger = logging.getLogger(__name__)


# =============================================================================
# RISK METRICS CALCULATOR
# =============================================================================

class RiskMetricsCalculator(BaseCalculator[RiskDataInput, RiskMetricsOutput, RiskCalculationConfig]):
    """
    WHAT: Comprehensive risk metrics calculator
    WHY: Provides validated, type-safe risk analysis for Finance Guru agents
    HOW: Uses Pydantic models for I/O, numpy/pandas/scipy for calculations

    EDUCATIONAL NOTE:
    This calculator implements industry-standard risk metrics used by
    professional portfolio managers and risk analysts:

    1. Market Risk Metrics (VaR, CVaR, Volatility)
    2. Risk-Adjusted Return Metrics (Sharpe, Sortino, Calmar, Omega)
    3. Benchmark-Relative Metrics (Beta, Alpha, Treynor, Information Ratio)
    4. Drawdown Analysis (Max Drawdown, Recovery Time)

    All calculations follow CFA Institute and industry-standard formulas.
    """

    def __init__(self, config: Optional[RiskCalculationConfig] = None):
        """Initialize with configuration."""
        super().__init__(config or RiskCalculationConfig())

    def _get_minimum_data_points(self) -> int:
        """Risk metrics need at least 30 data points."""
        return 30

    def calculate(
        self,
        data: RiskDataInput,
        benchmark: Optional[BenchmarkDataInput] = None,
        portfolio_value: Optional[float] = None,
        **kwargs,
    ) -> RiskMetricsOutput:
        """
        Calculate all risk metrics for given price data.

        Args:
            data: Historical price data (validated by Pydantic)
            benchmark: Optional benchmark for relative metrics
            portfolio_value: Optional portfolio value for dollar metrics

        Returns:
            RiskMetricsOutput with all calculated metrics
        """
        # Convert to DataFrame
        df = pd.DataFrame({
            "date": data.dates,
            "price": data.prices,
        })
        df = df.set_index("date")

        # Calculate daily returns
        returns = df["price"].pct_change().dropna()

        if len(returns) < self._get_minimum_data_points():
            raise InsufficientDataError(
                "Need at least 30 days of returns for risk calculations",
                required=30,
                provided=len(returns),
                calculator=self.name,
            )

        # Calculate each metric group
        var_output = self._calculate_var(returns, portfolio_value)
        cvar_output = self._calculate_cvar(returns, var_output.var_value, portfolio_value)
        volatility_output = self._calculate_volatility(returns)
        max_dd_output = self._calculate_max_drawdown(df["price"])

        sharpe_output = self._calculate_sharpe(returns)
        sortino_output = self._calculate_sortino(returns)
        calmar_output = self._calculate_calmar(returns, max_dd_output.max_drawdown)
        omega_output = self._calculate_omega(returns)

        # Benchmark-relative metrics (if benchmark provided)
        beta_alpha_output = None
        treynor_output = None
        info_ratio_output = None

        if benchmark is not None:
            beta_alpha_output = self._calculate_beta_alpha(returns, benchmark)
            if beta_alpha_output and beta_alpha_output.beta != 0:
                treynor_output = self._calculate_treynor(returns, beta_alpha_output.beta)
            info_ratio_output = self._calculate_information_ratio(returns, benchmark)

        # Calculate risk score and level
        risk_score = self._calculate_risk_score(
            volatility_output.annual_volatility,
            max_dd_output.max_drawdown,
            var_output.var_percent,
        )
        risk_level = self._classify_risk_level(risk_score)

        # Generate summary
        summary = self._generate_summary(
            data.ticker,
            risk_score,
            risk_level,
            sharpe_output,
            max_dd_output,
            volatility_output,
        )

        return RiskMetricsOutput(
            ticker=data.ticker,
            calculation_date=data.dates[-1],
            calculation_method="Enhanced Risk Metrics (Phase 3)",
            data_points_used=len(returns),
            var=var_output,
            cvar=cvar_output,
            max_drawdown=max_dd_output,
            volatility=volatility_output,
            sharpe_ratio=sharpe_output,
            sortino_ratio=sortino_output,
            calmar_ratio=calmar_output,
            omega_ratio=omega_output,
            beta_alpha=beta_alpha_output,
            treynor_ratio=treynor_output,
            information_ratio=info_ratio_output,
            risk_score=risk_score,
            risk_level=risk_level,
            summary=summary,
        )

    # =========================================================================
    # MARKET RISK METRICS
    # =========================================================================

    def _calculate_var(
        self,
        returns: pd.Series,
        portfolio_value: Optional[float] = None,
    ) -> VaROutput:
        """
        Calculate Value at Risk.

        FORMULA (Historical): percentile(returns, (1 - confidence) × 100)
        FORMULA (Parametric): mean + z-score × std_dev
        """
        confidence = self.config.confidence_level

        if self.config.var_method == "historical":
            var_value = float(np.percentile(returns, (1 - confidence) * 100))
        else:
            mean_return = returns.mean()
            std_return = returns.std()
            z_score = stats.norm.ppf(1 - confidence)
            var_value = float(mean_return + (z_score * std_return))

        var_percent = var_value * 100
        var_dollar = var_value * portfolio_value if portfolio_value else None

        return VaROutput(
            var_value=round(var_value, 6),
            var_percent=round(var_percent, 2),
            var_dollar=round(var_dollar, 2) if var_dollar else None,
            confidence_level=confidence,
            method=self.config.var_method,
        )

    def _calculate_cvar(
        self,
        returns: pd.Series,
        var_value: float,
        portfolio_value: Optional[float] = None,
    ) -> CVaROutput:
        """
        Calculate Conditional VaR (Expected Shortfall).

        FORMULA: mean(returns where returns <= VaR)
        """
        tail_returns = returns[returns <= var_value]

        if len(tail_returns) == 0:
            self.add_warning("No tail observations for CVaR, using VaR as fallback")
            cvar_value = var_value
            tail_count = 0
        else:
            cvar_value = float(tail_returns.mean())
            tail_count = len(tail_returns)

        cvar_percent = cvar_value * 100
        cvar_dollar = cvar_value * portfolio_value if portfolio_value else None

        return CVaROutput(
            cvar_value=round(cvar_value, 6),
            cvar_percent=round(cvar_percent, 2),
            cvar_dollar=round(cvar_dollar, 2) if cvar_dollar else None,
            tail_observations=tail_count,
        )

    def _calculate_volatility(self, returns: pd.Series) -> VolatilityOutput:
        """
        Calculate comprehensive volatility metrics.

        FORMULA: Annual Vol = Daily Std × sqrt(252)
        """
        trading_days = self.config.trading_days

        daily_vol = float(returns.std())
        annual_vol = daily_vol * np.sqrt(trading_days)
        monthly_vol = daily_vol * np.sqrt(21)  # ~21 trading days/month

        # Determine volatility regime
        if annual_vol < 0.15:
            regime = "low"
        elif annual_vol < 0.30:
            regime = "normal"
        elif annual_vol < 0.50:
            regime = "high"
        else:
            regime = "extreme"

        # Calculate percentile (where current vol ranks historically)
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(trading_days)
        current_vol = rolling_vol.iloc[-1]
        percentile = float(stats.percentileofscore(rolling_vol.dropna(), current_vol))

        return VolatilityOutput(
            annual_volatility=round(annual_vol, 4),
            daily_volatility=round(daily_vol, 6),
            monthly_volatility=round(monthly_vol, 4),
            volatility_regime=regime,
            volatility_percentile=round(percentile, 1),
        )

    def _calculate_max_drawdown(self, prices: pd.Series) -> MaxDrawdownOutput:
        """
        Calculate Maximum Drawdown with full analysis.

        FORMULA: (Trough - Peak) / Peak
        """
        # Calculate running maximum
        running_max = prices.expanding().max()
        drawdowns = (prices - running_max) / running_max

        # Find max drawdown
        max_dd = float(drawdowns.min())
        max_dd_idx = drawdowns.idxmin()

        # Find peak before max drawdown
        peak_idx = prices.loc[:max_dd_idx].idxmax()
        peak_price = float(prices.loc[peak_idx])
        trough_price = float(prices.loc[max_dd_idx])

        # Find recovery date (if any)
        recovery_date = None
        recovery_duration = None
        prices_after_trough = prices.loc[max_dd_idx:]
        recovered = prices_after_trough[prices_after_trough >= peak_price]

        if len(recovered) > 0:
            recovery_date = recovered.index[0]
            recovery_duration = (recovery_date - max_dd_idx).days

        # Duration from peak to trough
        drawdown_duration = (max_dd_idx - peak_idx).days

        # Current drawdown
        current_max = prices.iloc[-1]
        current_dd = float((prices.iloc[-1] - running_max.iloc[-1]) / running_max.iloc[-1])

        return MaxDrawdownOutput(
            max_drawdown=round(max_dd, 4),
            max_drawdown_dollar=None,
            peak_date=peak_idx,
            trough_date=max_dd_idx,
            peak_price=round(peak_price, 2),
            trough_price=round(trough_price, 2),
            recovery_date=recovery_date,
            drawdown_duration_days=drawdown_duration,
            recovery_duration_days=recovery_duration,
            current_drawdown=round(current_dd, 4),
        )

    # =========================================================================
    # RISK-ADJUSTED RETURN METRICS
    # =========================================================================

    def _calculate_sharpe(self, returns: pd.Series) -> SharpeRatioOutput:
        """
        Calculate Sharpe Ratio.

        FORMULA: (Return - Rf) / Volatility (annualized)
        """
        trading_days = self.config.trading_days
        rf = self.config.risk_free_rate

        daily_rf = rf / trading_days
        excess_returns = returns - daily_rf

        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(trading_days)
        annual_excess = excess_returns.mean() * trading_days
        annual_vol = returns.std() * np.sqrt(trading_days)

        # Quality assessment
        if sharpe < 0:
            quality = "poor"
        elif sharpe < 1.0:
            quality = "acceptable"
        elif sharpe < 2.0:
            quality = "good"
        else:
            quality = "excellent"

        return SharpeRatioOutput(
            sharpe_ratio=round(float(sharpe), 2),
            excess_return=round(float(annual_excess), 4),
            volatility=round(float(annual_vol), 4),
            risk_free_rate=rf,
            quality=quality,
        )

    def _calculate_sortino(self, returns: pd.Series) -> SortinoRatioOutput:
        """
        Calculate Sortino Ratio.

        FORMULA: (Return - Rf) / Downside Deviation
        """
        trading_days = self.config.trading_days
        rf = self.config.risk_free_rate

        daily_rf = rf / trading_days
        excess_returns = returns - daily_rf

        # Downside returns only
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            self.add_warning("No negative returns, using Sharpe as Sortino fallback")
            sharpe = self._calculate_sharpe(returns)
            return SortinoRatioOutput(
                sortino_ratio=sharpe.sharpe_ratio,
                downside_deviation=sharpe.volatility,
                negative_returns_count=0,
                quality=sharpe.quality,
            )

        downside_std = float(downside_returns.std())
        sortino = (excess_returns.mean() / downside_std) * np.sqrt(trading_days)
        annual_downside = downside_std * np.sqrt(trading_days)

        # Quality assessment
        if sortino < 0:
            quality = "poor"
        elif sortino < 1.0:
            quality = "acceptable"
        elif sortino < 2.0:
            quality = "good"
        else:
            quality = "excellent"

        return SortinoRatioOutput(
            sortino_ratio=round(float(sortino), 2),
            downside_deviation=round(annual_downside, 4),
            negative_returns_count=len(downside_returns),
            quality=quality,
        )

    def _calculate_calmar(self, returns: pd.Series, max_drawdown: float) -> CalmarRatioOutput:
        """
        Calculate Calmar Ratio.

        FORMULA: Annualized Return / |Max Drawdown|
        """
        trading_days = self.config.trading_days
        annual_return = float(returns.mean() * trading_days)

        if max_drawdown == 0:
            self.add_warning("Max drawdown is zero, Calmar undefined")
            calmar = float("inf")
        else:
            calmar = annual_return / abs(max_drawdown)

        return CalmarRatioOutput(
            calmar_ratio=round(calmar, 2) if calmar != float("inf") else 999.99,
            annual_return=round(annual_return, 4),
            max_drawdown=round(max_drawdown, 4),
            max_drawdown_duration_days=None,
        )

    def _calculate_omega(self, returns: pd.Series) -> OmegaRatioOutput:
        """
        Calculate Omega Ratio.

        FORMULA: Sum(gains above threshold) / Sum(|losses below threshold|)
        """
        # Use target return or risk-free rate as threshold
        threshold = self.config.target_return or (self.config.risk_free_rate / self.config.trading_days)

        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]

        sum_gains = float(gains.sum())
        sum_losses = float(losses.sum())

        if sum_losses == 0:
            omega = float("inf")
        else:
            omega = sum_gains / sum_losses

        win_rate = len(returns[returns > threshold]) / len(returns)

        return OmegaRatioOutput(
            omega_ratio=round(omega, 2) if omega != float("inf") else 999.99,
            threshold_return=round(threshold, 6),
            gains_above_threshold=round(sum_gains, 4),
            losses_below_threshold=round(-sum_losses, 4),
            win_rate=round(win_rate, 4),
        )

    # =========================================================================
    # BENCHMARK-RELATIVE METRICS
    # =========================================================================

    def _calculate_beta_alpha(
        self,
        returns: pd.Series,
        benchmark: BenchmarkDataInput,
    ) -> Optional[BetaAlphaOutput]:
        """
        Calculate Beta and Alpha vs benchmark.

        FORMULA:
        Beta = Cov(asset, benchmark) / Var(benchmark)
        Alpha = Asset Return - (Rf + Beta × (Benchmark Return - Rf))
        """
        # Convert benchmark to returns
        bench_df = pd.DataFrame({
            "date": benchmark.dates,
            "price": benchmark.prices,
        })
        bench_df = bench_df.set_index("date")
        bench_returns = bench_df["price"].pct_change().dropna()

        # Align dates
        aligned = pd.DataFrame({
            "asset": returns,
            "benchmark": bench_returns,
        }).dropna()

        if len(aligned) < 30:
            self.add_warning(f"Only {len(aligned)} overlapping days for beta/alpha")
            return None

        asset_ret = aligned["asset"]
        bench_ret = aligned["benchmark"]

        # Calculate beta
        covariance = asset_ret.cov(bench_ret)
        bench_variance = bench_ret.var()
        beta = float(covariance / bench_variance)

        # Calculate alpha
        trading_days = self.config.trading_days
        rf = self.config.risk_free_rate
        daily_rf = rf / trading_days

        asset_annual = (asset_ret.mean() - daily_rf) * trading_days
        bench_annual = (bench_ret.mean() - daily_rf) * trading_days

        alpha = asset_annual - (beta * bench_annual)

        # R-squared and correlation
        correlation = float(asset_ret.corr(bench_ret))
        r_squared = correlation ** 2

        return BetaAlphaOutput(
            beta=round(beta, 2),
            alpha=round(float(alpha), 4),
            r_squared=round(r_squared, 4),
            benchmark_ticker=benchmark.ticker,
            correlation=round(correlation, 4),
        )

    def _calculate_treynor(self, returns: pd.Series, beta: float) -> Optional[TreynorRatioOutput]:
        """
        Calculate Treynor Ratio.

        FORMULA: (Return - Rf) / Beta
        """
        if beta == 0:
            self.add_warning("Beta is zero, Treynor undefined")
            return None

        trading_days = self.config.trading_days
        rf = self.config.risk_free_rate

        annual_return = float(returns.mean() * trading_days)
        excess_return = annual_return - rf
        treynor = excess_return / beta

        return TreynorRatioOutput(
            treynor_ratio=round(treynor, 4),
            beta=round(beta, 2),
            systematic_risk_premium=round(excess_return, 4),
        )

    def _calculate_information_ratio(
        self,
        returns: pd.Series,
        benchmark: BenchmarkDataInput,
    ) -> Optional[InformationRatioOutput]:
        """
        Calculate Information Ratio.

        FORMULA: (Portfolio Return - Benchmark Return) / Tracking Error
        """
        # Convert benchmark to returns
        bench_df = pd.DataFrame({
            "date": benchmark.dates,
            "price": benchmark.prices,
        })
        bench_df = bench_df.set_index("date")
        bench_returns = bench_df["price"].pct_change().dropna()

        # Align dates
        aligned = pd.DataFrame({
            "asset": returns,
            "benchmark": bench_returns,
        }).dropna()

        if len(aligned) < 30:
            return None

        # Active returns
        active_returns = aligned["asset"] - aligned["benchmark"]

        trading_days = self.config.trading_days
        active_return_annual = float(active_returns.mean() * trading_days)
        tracking_error = float(active_returns.std() * np.sqrt(trading_days))

        if tracking_error == 0:
            self.add_warning("Tracking error is zero, IR undefined")
            return None

        ir = active_return_annual / tracking_error

        # Quality assessment
        if ir < 0:
            quality = "underperforming"
        elif ir < 0.5:
            quality = "poor"
        elif ir < 1.0:
            quality = "good"
        else:
            quality = "excellent"

        return InformationRatioOutput(
            information_ratio=round(ir, 2),
            active_return=round(active_return_annual, 4),
            tracking_error=round(tracking_error, 4),
            benchmark_ticker=benchmark.ticker,
            quality=quality,
        )

    # =========================================================================
    # SUMMARY AND SCORING
    # =========================================================================

    def _calculate_risk_score(
        self,
        annual_vol: float,
        max_drawdown: float,
        var_percent: float,
    ) -> float:
        """
        Calculate overall risk score (0-100).

        Combines volatility, drawdown, and VaR into single score.
        """
        # Normalize each metric (higher = riskier)
        vol_score = min(annual_vol / 0.50 * 40, 40)  # Vol up to 50% = 40 points
        dd_score = min(abs(max_drawdown) / 0.40 * 35, 35)  # DD up to 40% = 35 points
        var_score = min(abs(var_percent) / 5 * 25, 25)  # VaR up to 5% = 25 points

        total = vol_score + dd_score + var_score
        return round(min(total, 100), 1)

    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify risk level based on score."""
        if risk_score < 25:
            return "conservative"
        elif risk_score < 50:
            return "moderate"
        elif risk_score < 75:
            return "aggressive"
        else:
            return "speculative"

    def _generate_summary(
        self,
        ticker: str,
        risk_score: float,
        risk_level: str,
        sharpe: SharpeRatioOutput,
        max_dd: MaxDrawdownOutput,
        vol: VolatilityOutput,
    ) -> str:
        """Generate human-readable risk summary."""
        parts = [
            f"{ticker} Risk Profile: {risk_level.upper()} (Score: {risk_score}/100)",
            f"Volatility: {vol.annual_volatility:.1%} ({vol.volatility_regime})",
            f"Max Drawdown: {max_dd.max_drawdown:.1%}",
            f"Sharpe Ratio: {sharpe.sharpe_ratio:.2f} ({sharpe.quality})",
        ]

        if max_dd.current_drawdown < -0.05:
            parts.append(f"⚠️ Currently in {abs(max_dd.current_drawdown):.1%} drawdown")

        return " | ".join(parts)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_risk_metrics(
    data: RiskDataInput,
    benchmark: Optional[BenchmarkDataInput] = None,
    config: Optional[RiskCalculationConfig] = None,
    portfolio_value: Optional[float] = None,
) -> RiskMetricsOutput:
    """
    Convenience function to calculate all risk metrics.

    Args:
        data: Price data input
        benchmark: Optional benchmark data
        config: Optional configuration
        portfolio_value: Optional portfolio value for dollar metrics

    Returns:
        RiskMetricsOutput with all metrics
    """
    calculator = RiskMetricsCalculator(config)
    return calculator.safe_calculate(data, benchmark=benchmark, portfolio_value=portfolio_value)


def calculate_var(
    data: RiskDataInput,
    config: Optional[RiskCalculationConfig] = None,
) -> VaROutput:
    """Calculate VaR only."""
    calculator = RiskMetricsCalculator(config)
    df = pd.DataFrame({"price": data.prices})
    returns = df["price"].pct_change().dropna()
    return calculator._calculate_var(returns, None)


def calculate_sharpe(
    data: RiskDataInput,
    config: Optional[RiskCalculationConfig] = None,
) -> SharpeRatioOutput:
    """Calculate Sharpe Ratio only."""
    calculator = RiskMetricsCalculator(config)
    df = pd.DataFrame({"price": data.prices})
    returns = df["price"].pct_change().dropna()
    return calculator._calculate_sharpe(returns)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "RiskMetricsCalculator",
    "calculate_risk_metrics",
    "calculate_var",
    "calculate_sharpe",
]
