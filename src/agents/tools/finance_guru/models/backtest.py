"""
Finance Guru - Backtest Models (Phase 5)

Layer 1: Pydantic models for strategy backtesting.

This module provides validated data structures for:
- Backtest configuration (capital, commissions, slippage)
- Trading signals (buy/sell/hold)
- Trade execution records
- Performance metrics (returns, Sharpe, drawdown)
- Strategy comparison

EDUCATIONAL NOTES:
- Backtesting: Testing strategies on historical data before risking real capital
- Slippage: Price difference between signal and execution (market moves)
- Drawdown: Peak-to-trough decline in portfolio value
- Sharpe Ratio: Risk-adjusted return (excess return per unit of risk)

Author: HealerAgent Development Team
"""

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field, field_validator, model_validator

from src.agents.tools.finance_guru.models.base import (
    BaseFinanceModel,
    BaseCalculationResult,
)


# =============================================================================
# ENUMS
# =============================================================================


class StrategyType(str, Enum):
    """Available backtest strategies.

    SMA_CROSSOVER: Buy when short MA crosses above long MA, sell on reverse
    RSI_MEAN_REVERSION: Buy when RSI oversold, sell when overbought
    MACD_SIGNAL: Buy on MACD/signal line bullish crossover
    BOLLINGER_BANDS: Trade mean reversion within bands
    DUAL_MOMENTUM: Momentum rotation strategy
    BUY_AND_HOLD: Simple buy and hold benchmark
    CUSTOM: User-defined strategy rules
    """
    SMA_CROSSOVER = "sma_crossover"
    RSI_MEAN_REVERSION = "rsi_mean_reversion"
    MACD_SIGNAL = "macd_signal"
    BOLLINGER_BANDS = "bollinger_bands"
    DUAL_MOMENTUM = "dual_momentum"
    BUY_AND_HOLD = "buy_and_hold"
    CUSTOM = "custom"


class TradeAction(str, Enum):
    """Trading actions."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class BacktestRecommendation(str, Enum):
    """Backtest verdict recommendations."""
    DEPLOY = "DEPLOY"       # Strong performance, ready for live trading
    OPTIMIZE = "OPTIMIZE"   # Promising but needs parameter tuning
    REJECT = "REJECT"       # Poor performance or excessive risk


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class BacktestConfig(BaseFinanceModel):
    """Configuration for backtest execution.

    EDUCATIONAL NOTE:
    These settings make backtests realistic by accounting for real-world costs:
    - Initial Capital: How much money you're starting with
    - Commission: Trading fees (e.g., $0 for Robinhood, $1-10 for traditional)
    - Slippage: Price moves between decision and execution (usually 0.05-0.1%)
    - Position Size: How much to allocate per trade

    WITHOUT these costs, backtests look unrealistically good!

    Attributes:
        initial_capital: Starting capital for backtest
        commission_per_trade: Commission per trade in dollars
        slippage_pct: Slippage as percentage (0.001 = 0.1%)
        position_size_pct: Position size as % of capital
        allow_fractional_shares: Allow fractional share purchases
        benchmark_symbol: Benchmark for comparison (default: SPY)
    """
    initial_capital: float = Field(
        100000.0, gt=0.0,
        description="Starting capital for backtest"
    )
    commission_per_trade: float = Field(
        0.0, ge=0.0,
        description="Commission per trade (e.g., 0.0 for commission-free)"
    )
    slippage_pct: float = Field(
        0.001, ge=0.0, le=0.05,
        description="Slippage as percentage (0.001 = 0.1%)"
    )
    position_size_pct: float = Field(
        1.0, gt=0.0, le=1.0,
        description="Position size as % of capital (1.0 = 100%)"
    )
    allow_fractional_shares: bool = Field(
        True,
        description="Allow fractional share purchases"
    )
    benchmark_symbol: str = Field(
        "SPY",
        description="Benchmark symbol for comparison"
    )


class StrategyConfig(BaseFinanceModel):
    """Configuration for a specific strategy.

    Attributes:
        strategy_type: Type of strategy to backtest
        symbols: List of symbols to trade
        start_date: Backtest start date
        end_date: Backtest end date
        parameters: Strategy-specific parameters
    """
    strategy_type: StrategyType = Field(..., description="Strategy type")
    symbols: List[str] = Field(..., min_length=1, description="Symbols to trade")
    start_date: date = Field(..., description="Backtest start date")
    end_date: date = Field(..., description="Backtest end date")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific parameters"
    )

    @model_validator(mode="after")
    def validate_dates(self) -> "StrategyConfig":
        """Ensure end_date is after start_date."""
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self


class SMAStrategyParams(BaseFinanceModel):
    """Parameters for SMA Crossover strategy.

    EDUCATIONAL NOTE:
    SMA Crossover is a trend-following strategy:
    - BUY when short-term MA crosses ABOVE long-term MA (golden cross)
    - SELL when short-term MA crosses BELOW long-term MA (death cross)

    Common periods:
    - Fast: 10-20 days (react quickly)
    - Slow: 50-200 days (filter noise)

    Attributes:
        fast_period: Short-term moving average period
        slow_period: Long-term moving average period
    """
    fast_period: int = Field(20, ge=5, le=50, description="Fast MA period")
    slow_period: int = Field(50, ge=20, le=200, description="Slow MA period")

    @model_validator(mode="after")
    def validate_periods(self) -> "SMAStrategyParams":
        """Ensure fast_period < slow_period."""
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be less than slow_period")
        return self


class RSIStrategyParams(BaseFinanceModel):
    """Parameters for RSI Mean Reversion strategy.

    EDUCATIONAL NOTE:
    RSI (Relative Strength Index) measures momentum on a 0-100 scale:
    - Below 30: Oversold (potential buy signal)
    - Above 70: Overbought (potential sell signal)

    This strategy assumes prices revert to the mean after extreme readings.

    Attributes:
        period: RSI calculation period
        oversold_threshold: RSI level considered oversold
        overbought_threshold: RSI level considered overbought
    """
    period: int = Field(14, ge=5, le=30, description="RSI period")
    oversold_threshold: float = Field(30.0, ge=10, le=40, description="Oversold level")
    overbought_threshold: float = Field(70.0, ge=60, le=90, description="Overbought level")


class MACDStrategyParams(BaseFinanceModel):
    """Parameters for MACD Signal strategy.

    EDUCATIONAL NOTE:
    MACD (Moving Average Convergence Divergence):
    - MACD Line = Fast EMA - Slow EMA
    - Signal Line = EMA of MACD Line
    - BUY when MACD crosses above Signal (bullish)
    - SELL when MACD crosses below Signal (bearish)

    Attributes:
        fast_period: Fast EMA period (typically 12)
        slow_period: Slow EMA period (typically 26)
        signal_period: Signal line period (typically 9)
    """
    fast_period: int = Field(12, ge=5, le=20, description="Fast EMA period")
    slow_period: int = Field(26, ge=20, le=50, description="Slow EMA period")
    signal_period: int = Field(9, ge=5, le=15, description="Signal line period")


class BollingerStrategyParams(BaseFinanceModel):
    """Parameters for Bollinger Bands strategy.

    EDUCATIONAL NOTE:
    Bollinger Bands create a price envelope:
    - Middle Band = SMA
    - Upper Band = SMA + (std_dev × num_std)
    - Lower Band = SMA - (std_dev × num_std)

    Strategy: Buy at lower band (oversold), sell at upper band (overbought)

    Attributes:
        period: SMA period for middle band
        num_std: Number of standard deviations
    """
    period: int = Field(20, ge=10, le=50, description="SMA period")
    num_std: float = Field(2.0, ge=1.0, le=3.0, description="Standard deviations")


# =============================================================================
# SIGNAL AND TRADE MODELS
# =============================================================================


class TradeSignal(BaseFinanceModel):
    """A single buy or sell signal from a strategy.

    EDUCATIONAL NOTE:
    A trading strategy generates "signals" - instructions to buy or sell.
    Example signals:
    - "Buy TSLA when RSI < 30" (oversold)
    - "Sell TSLA when price crosses above upper Bollinger Band"

    Attributes:
        signal_date: Date of signal
        symbol: Asset ticker
        action: Trade action (BUY/SELL/HOLD)
        price: Price at signal
        signal_strength: Optional signal strength (0-1)
        reason: Optional reason for signal
    """
    signal_date: date = Field(..., description="Date of signal")
    symbol: str = Field(..., description="Asset ticker")
    action: TradeAction = Field(..., description="Trade action")
    price: float = Field(..., gt=0.0, description="Price at signal")
    signal_strength: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Optional signal strength (0-1)"
    )
    reason: Optional[str] = Field(None, description="Reason for signal")


class TradeExecution(BaseFinanceModel):
    """A completed trade with actual execution details.

    EDUCATIONAL NOTE:
    The difference between SIGNAL and EXECUTION is critical:
    - Signal: "I want to buy TSLA at $250"
    - Execution: "I bought 10 shares at $250.50 with $5 commission"

    Slippage and fees mean execution differs from signal!

    Attributes:
        entry_date: Trade entry date
        exit_date: Trade exit date (None if still open)
        symbol: Asset ticker
        entry_price: Actual entry price (after slippage)
        shares: Number of shares traded
        entry_commission: Commission paid on entry
        exit_price: Actual exit price (after slippage)
        exit_commission: Commission paid on exit
        pnl: Profit/Loss in dollars
        pnl_pct: Profit/Loss as percentage
        signal_reason: Original signal reason
    """
    entry_date: date = Field(..., description="Trade entry date")
    exit_date: Optional[date] = Field(None, description="Trade exit date")
    symbol: str = Field(..., description="Asset ticker")
    entry_price: float = Field(..., gt=0.0, description="Entry price after slippage")
    shares: float = Field(..., gt=0.0, description="Number of shares")
    entry_commission: float = Field(..., ge=0.0, description="Entry commission")
    exit_price: Optional[float] = Field(None, description="Exit price after slippage")
    exit_commission: Optional[float] = Field(None, description="Exit commission")
    pnl: Optional[float] = Field(None, description="Profit/Loss in dollars")
    pnl_pct: Optional[float] = Field(None, description="Profit/Loss percentage")
    signal_reason: Optional[str] = Field(None, description="Signal reason")

    @property
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_date is None

    @property
    def is_winning(self) -> bool:
        """Check if trade is profitable."""
        return self.pnl is not None and self.pnl > 0


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================


class BacktestPerformanceMetrics(BaseFinanceModel):
    """Performance statistics for the backtest.

    EDUCATIONAL NOTE:
    These metrics answer key questions about your strategy:
    1. Total Return: How much did I make/lose?
    2. Sharpe Ratio: Was return worth the risk?
    3. Max Drawdown: Worst losing streak (can you stomach it?)
    4. Win Rate: What % of trades were profitable?
    5. Profit Factor: How much winners made vs losers lost

    A good strategy has:
    - Positive returns (obviously!)
    - Sharpe > 1.0 (good risk-adjusted returns)
    - Max drawdown < 25% (tolerable losses)
    - Win rate > 50% OR profit factor > 2.0

    Attributes:
        initial_capital: Starting capital
        final_capital: Ending capital
        total_return: Total return in dollars
        total_return_pct: Total return as percentage
        annualized_return_pct: Annualized return percentage
        sharpe_ratio: Sharpe ratio (risk-adjusted return)
        sortino_ratio: Sortino ratio (downside risk-adjusted)
        max_drawdown: Maximum peak-to-trough decline
        max_drawdown_pct: Max drawdown as percentage
        total_trades: Total number of trades
        winning_trades: Number of profitable trades
        losing_trades: Number of losing trades
        win_rate: Win rate (winning/total)
        avg_win: Average profit on winning trades
        avg_loss: Average loss on losing trades
        profit_factor: Gross profits / gross losses
        avg_trade_duration: Average holding period in days
        total_commissions: Total commissions paid
        total_slippage: Total slippage cost
    """
    # Capital metrics
    initial_capital: float = Field(..., description="Starting capital")
    final_capital: float = Field(..., description="Ending capital")
    total_return: float = Field(..., description="Total return in dollars")
    total_return_pct: float = Field(..., description="Total return percentage")
    annualized_return_pct: Optional[float] = Field(None, description="Annualized return")

    # Risk metrics
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    sortino_ratio: Optional[float] = Field(None, description="Sortino ratio")
    max_drawdown: float = Field(..., description="Max drawdown in dollars")
    max_drawdown_pct: float = Field(..., description="Max drawdown percentage")

    # Trade statistics
    total_trades: int = Field(..., ge=0, description="Total trades")
    winning_trades: int = Field(..., ge=0, description="Winning trades")
    losing_trades: int = Field(..., ge=0, description="Losing trades")
    win_rate: float = Field(..., ge=0.0, le=1.0, description="Win rate")

    # Profit metrics
    avg_win: Optional[float] = Field(None, description="Average winning trade")
    avg_loss: Optional[float] = Field(None, description="Average losing trade")
    profit_factor: Optional[float] = Field(None, description="Profit factor")
    avg_trade_duration: Optional[float] = Field(None, description="Avg trade duration days")

    # Cost analysis
    total_commissions: float = Field(..., ge=0.0, description="Total commissions")
    total_slippage: float = Field(..., ge=0.0, description="Total slippage")


class BenchmarkComparison(BaseFinanceModel):
    """Comparison with benchmark (e.g., SPY).

    Attributes:
        benchmark_symbol: Benchmark symbol
        benchmark_return_pct: Benchmark total return
        alpha: Strategy excess return vs benchmark
        beta: Strategy volatility relative to benchmark
        correlation: Correlation with benchmark
        outperformed: Whether strategy beat benchmark
    """
    benchmark_symbol: str = Field(..., description="Benchmark symbol")
    benchmark_return_pct: float = Field(..., description="Benchmark return %")
    alpha: float = Field(..., description="Excess return vs benchmark")
    beta: Optional[float] = Field(None, description="Beta to benchmark")
    correlation: Optional[float] = Field(None, description="Correlation")
    outperformed: bool = Field(..., description="Beat benchmark?")


# =============================================================================
# OUTPUT MODELS
# =============================================================================


class EquityCurvePoint(BaseFinanceModel):
    """Single point on equity curve.

    Attributes:
        date: Date
        equity: Portfolio value
        drawdown_pct: Current drawdown from peak
    """
    date: date = Field(..., description="Date")
    equity: float = Field(..., description="Portfolio value")
    drawdown_pct: float = Field(0.0, description="Drawdown from peak")


class BacktestResult(BaseCalculationResult):
    """Complete backtest results with all trades and metrics.

    Attributes:
        symbol: Primary asset tested
        strategy_type: Strategy used
        strategy_name: Descriptive strategy name
        start_date: Backtest start
        end_date: Backtest end
        config: Backtest configuration
        performance: Performance metrics
        benchmark: Benchmark comparison
        trades: All executed trades
        equity_curve: Portfolio value over time
        recommendation: Deployment recommendation
        reasoning: Recommendation reasoning
    """
    # Metadata
    symbol: str = Field(..., description="Primary symbol tested")
    strategy_type: StrategyType = Field(..., description="Strategy type")
    strategy_name: str = Field(..., description="Strategy description")
    start_date: date = Field(..., description="Start date")
    end_date: date = Field(..., description="End date")

    # Configuration
    config: BacktestConfig = Field(..., description="Backtest config")

    # Results
    performance: BacktestPerformanceMetrics = Field(..., description="Performance metrics")
    benchmark: Optional[BenchmarkComparison] = Field(None, description="Benchmark comparison")

    # Trade history
    trades: List[TradeExecution] = Field(..., description="All trades")

    # Equity curve
    equity_curve: List[EquityCurvePoint] = Field(..., description="Equity curve")

    # Verdict
    recommendation: BacktestRecommendation = Field(..., description="Recommendation")
    reasoning: str = Field(..., description="Recommendation reasoning")


class StrategyComparisonResult(BaseCalculationResult):
    """Comparison of multiple strategies.

    Attributes:
        strategies: Results for each strategy
        winner: Best performing strategy
        comparison_metrics: Side-by-side metrics
        ranking: Strategies ranked by performance
        recommendation: Overall recommendation
    """
    strategies: List[BacktestResult] = Field(..., description="All strategy results")
    winner: str = Field(..., description="Best strategy name")
    comparison_metrics: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Metrics by strategy"
    )
    ranking: List[Tuple[str, float]] = Field(
        ...,
        description="Strategies ranked by score"
    )
    recommendation: str = Field(..., description="Overall recommendation")
