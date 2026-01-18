"""
Finance Guru - Backtest Engine Calculator (Phase 5)

Layer 2: Pure computation functions for strategy backtesting.

This module implements:
- Historical strategy simulation
- Transaction cost modeling (commissions + slippage)
- Performance metrics calculation
- Equity curve generation
- Benchmark comparison
- Strategy comparison

EDUCATIONAL NOTES:
- Backtesting validates strategies on historical data
- Transaction costs (commissions, slippage) are critical for realism
- Max drawdown is the psychological test of a strategy
- Sharpe ratio measures risk-adjusted performance

Author: HealerAgent Development Team
"""

from datetime import date, timedelta
from typing import Optional

import numpy as np

from src.agents.tools.finance_guru.calculators.base import (
    BaseCalculator,
    CalculationContext,
    CalculationError,
    InsufficientDataError,
)
from src.agents.tools.finance_guru.models.backtest import (
    # Enums
    StrategyType,
    TradeAction,
    BacktestRecommendation,
    # Config
    BacktestConfig,
    StrategyConfig,
    # Signals/Trades
    TradeSignal,
    TradeExecution,
    # Output
    BacktestPerformanceMetrics,
    BenchmarkComparison,
    EquityCurvePoint,
    BacktestResult,
    StrategyComparisonResult,
)
from src.agents.tools.finance_guru.strategies.base_strategy import (
    BaseStrategy,
    StrategyContext,
)
from src.agents.tools.finance_guru.strategies import (
    SMAcrossoverStrategy,
    RSIMeanReversionStrategy,
    MACDSignalStrategy,
    BollingerBandsStrategy,
    BuyAndHoldStrategy,
)


class BacktestEngine(BaseCalculator):
    """Backtest engine for strategy validation.

    Executes trading strategies on historical data with realistic
    cost modeling and generates comprehensive performance analysis.

    FEATURES:
    - Strategy execution simulation
    - Transaction cost modeling (commissions + slippage)
    - Performance metrics calculation
    - Equity curve tracking
    - Benchmark comparison
    - Multi-strategy comparison
    """

    # Strategy registry
    STRATEGIES: dict[StrategyType, type[BaseStrategy]] = {
        StrategyType.SMA_CROSSOVER: SMAcrossoverStrategy,
        StrategyType.RSI_MEAN_REVERSION: RSIMeanReversionStrategy,
        StrategyType.MACD_SIGNAL: MACDSignalStrategy,
        StrategyType.BOLLINGER_BANDS: BollingerBandsStrategy,
        StrategyType.BUY_AND_HOLD: BuyAndHoldStrategy,
    }

    def __init__(self):
        super().__init__()
        self.context = CalculationContext(calculator_name="BacktestEngine")

    def calculate(self, data, **kwargs):
        """Implement abstract method - routes to run_backtest()."""
        strategy_config = kwargs.get("strategy_config")
        backtest_config = kwargs.get("backtest_config", BacktestConfig())
        price_data = kwargs.get("price_data", data)
        benchmark_data = kwargs.get("benchmark_data")
        return self.run_backtest(strategy_config, backtest_config, price_data, benchmark_data)

    def run_backtest(
        self,
        strategy_config: StrategyConfig,
        backtest_config: BacktestConfig,
        price_data: dict[str, dict],
        benchmark_data: Optional[dict] = None,
    ) -> BacktestResult:
        """Run a complete backtest for a strategy.

        Args:
            strategy_config: Strategy configuration
            backtest_config: Backtest settings (capital, costs)
            price_data: Price data dict by symbol {symbol: {dates, opens, highs, lows, closes, volumes}}
            benchmark_data: Optional benchmark price data

        Returns:
            BacktestResult with complete analysis
        """
        self.context.start()

        try:
            # Get primary symbol
            symbol = strategy_config.symbols[0]

            if symbol not in price_data:
                raise ValueError(f"Price data not found for {symbol}")

            data = price_data[symbol]

            # Create strategy context
            context = StrategyContext(
                symbol=symbol,
                dates=data["dates"],
                opens=data["opens"],
                highs=data["highs"],
                lows=data["lows"],
                closes=data["closes"],
                volumes=data.get("volumes"),
                parameters=strategy_config.parameters,
            )

            # Get strategy class
            if strategy_config.strategy_type not in self.STRATEGIES:
                raise ValueError(f"Unknown strategy type: {strategy_config.strategy_type}")

            strategy_class = self.STRATEGIES[strategy_config.strategy_type]
            strategy = strategy_class()

            # Generate signals
            signals = strategy.generate_signals(context)

            # Execute trades
            trades, equity_curve = self._execute_trades(
                signals=signals,
                dates=data["dates"],
                prices=data["closes"],
                config=backtest_config,
            )

            # Calculate performance metrics
            performance = self._calculate_performance(
                trades=trades,
                equity_curve=equity_curve,
                config=backtest_config,
                dates=data["dates"],
            )

            # Compare with benchmark
            benchmark = None
            if benchmark_data:
                benchmark = self._compare_benchmark(
                    equity_curve=equity_curve,
                    benchmark_data=benchmark_data,
                    config=backtest_config,
                )

            # Generate recommendation
            recommendation, reasoning = self._generate_recommendation(performance, benchmark)

            self.context.complete()

            return BacktestResult(
                symbol=symbol,
                strategy_type=strategy_config.strategy_type,
                strategy_name=strategy.name,
                start_date=strategy_config.start_date,
                end_date=strategy_config.end_date,
                config=backtest_config,
                performance=performance,
                benchmark=benchmark,
                trades=trades,
                equity_curve=equity_curve,
                recommendation=recommendation,
                reasoning=reasoning,
                calculation_time_ms=self.context.elapsed_ms,
            )

        except Exception as e:
            self.context.fail(str(e))
            raise CalculationError(f"Backtest failed: {e}")

    def _execute_trades(
        self,
        signals: list[TradeSignal],
        dates: list[date],
        prices: list[float],
        config: BacktestConfig,
    ) -> tuple[list[TradeExecution], list[EquityCurvePoint]]:
        """Execute trades based on signals.

        Args:
            signals: List of trade signals
            dates: Price dates
            prices: Close prices
            config: Backtest configuration

        Returns:
            Tuple of (trades, equity_curve)
        """
        capital = config.initial_capital
        trades: list[TradeExecution] = []
        equity_curve: list[EquityCurvePoint] = []
        current_position: Optional[TradeExecution] = None
        peak_equity = capital

        # Create date-to-index mapping
        date_to_idx = {d: i for i, d in enumerate(dates)}

        # Track equity for each date
        signal_dates = {s.signal_date: s for s in signals}

        for i, (d, price) in enumerate(zip(dates, prices)):
            # Check for signal on this date
            if d in signal_dates:
                signal = signal_dates[d]

                if signal.action == TradeAction.BUY and current_position is None:
                    # Open position
                    current_position = self._open_position(
                        signal=signal,
                        capital=capital,
                        config=config,
                    )
                    if current_position:
                        capital -= (
                            current_position.shares * current_position.entry_price
                            + current_position.entry_commission
                        )

                elif signal.action == TradeAction.SELL and current_position is not None:
                    # Close position
                    closed_trade = self._close_position(
                        position=current_position,
                        exit_date=d,
                        exit_price=price,
                        config=config,
                    )
                    trades.append(closed_trade)
                    capital += (
                        closed_trade.shares * closed_trade.exit_price
                        - closed_trade.exit_commission
                    )
                    current_position = None

            # Calculate current equity
            current_equity = capital
            if current_position is not None:
                position_value = current_position.shares * price
                current_equity += position_value

            # Track drawdown
            if current_equity > peak_equity:
                peak_equity = current_equity
            drawdown_pct = (peak_equity - current_equity) / peak_equity * 100 if peak_equity > 0 else 0

            equity_curve.append(EquityCurvePoint(
                date=d,
                equity=current_equity,
                drawdown_pct=drawdown_pct,
            ))

        # Close any open position at end
        if current_position is not None:
            closed_trade = self._close_position(
                position=current_position,
                exit_date=dates[-1],
                exit_price=prices[-1],
                config=config,
            )
            trades.append(closed_trade)

        return trades, equity_curve

    def _open_position(
        self,
        signal: TradeSignal,
        capital: float,
        config: BacktestConfig,
    ) -> Optional[TradeExecution]:
        """Open a new trading position.

        Args:
            signal: BUY signal
            capital: Available capital
            config: Backtest configuration

        Returns:
            TradeExecution or None if insufficient capital
        """
        # Calculate position size
        position_capital = capital * config.position_size_pct

        # Apply slippage (price moves UP when buying)
        entry_price = signal.price * (1 + config.slippage_pct)

        # Calculate shares
        shares = position_capital / entry_price
        if not config.allow_fractional_shares:
            shares = float(int(shares))

        if shares <= 0:
            return None

        # Check capital requirement
        total_cost = shares * entry_price + config.commission_per_trade
        if total_cost > capital:
            return None

        return TradeExecution(
            entry_date=signal.signal_date,
            exit_date=None,
            symbol=signal.symbol,
            entry_price=entry_price,
            shares=shares,
            entry_commission=config.commission_per_trade,
            exit_price=None,
            exit_commission=None,
            pnl=None,
            pnl_pct=None,
            signal_reason=signal.reason,
        )

    def _close_position(
        self,
        position: TradeExecution,
        exit_date: date,
        exit_price: float,
        config: BacktestConfig,
    ) -> TradeExecution:
        """Close an existing position.

        Args:
            position: Open position to close
            exit_date: Exit date
            exit_price: Exit price (before slippage)
            config: Backtest configuration

        Returns:
            Completed TradeExecution
        """
        # Apply slippage (price moves DOWN when selling)
        actual_exit_price = exit_price * (1 - config.slippage_pct)

        # Calculate proceeds
        proceeds = position.shares * actual_exit_price
        exit_commission = config.commission_per_trade

        # Calculate P&L
        entry_cost = position.shares * position.entry_price + position.entry_commission
        net_proceeds = proceeds - exit_commission
        pnl = net_proceeds - entry_cost
        pnl_pct = (pnl / entry_cost) * 100 if entry_cost > 0 else 0

        return TradeExecution(
            entry_date=position.entry_date,
            exit_date=exit_date,
            symbol=position.symbol,
            entry_price=position.entry_price,
            shares=position.shares,
            entry_commission=position.entry_commission,
            exit_price=actual_exit_price,
            exit_commission=exit_commission,
            pnl=pnl,
            pnl_pct=pnl_pct,
            signal_reason=position.signal_reason,
        )

    def _calculate_performance(
        self,
        trades: list[TradeExecution],
        equity_curve: list[EquityCurvePoint],
        config: BacktestConfig,
        dates: list[date],
    ) -> BacktestPerformanceMetrics:
        """Calculate comprehensive performance metrics.

        Args:
            trades: List of executed trades
            equity_curve: Equity curve data
            config: Backtest configuration
            dates: Full date range

        Returns:
            BacktestPerformanceMetrics
        """
        initial = config.initial_capital
        final = equity_curve[-1].equity if equity_curve else initial

        total_return = final - initial
        total_return_pct = (total_return / initial) * 100 if initial > 0 else 0

        # Annualized return
        if len(dates) >= 2:
            days = (dates[-1] - dates[0]).days
            years = days / 365.25
            if years > 0 and final > 0 and initial > 0:
                annualized = ((final / initial) ** (1 / years) - 1) * 100
            else:
                annualized = None
        else:
            annualized = None

        # Calculate returns series for Sharpe/Sortino
        equity_values = [e.equity for e in equity_curve]
        if len(equity_values) > 1:
            returns = np.diff(equity_values) / equity_values[:-1]
            returns = returns[~np.isnan(returns)]

            if len(returns) > 0 and np.std(returns) > 0:
                sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

                # Sortino (downside deviation)
                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0:
                    downside_std = np.std(negative_returns)
                    sortino = float(np.mean(returns) / downside_std * np.sqrt(252)) if downside_std > 0 else None
                else:
                    sortino = None
            else:
                sharpe = None
                sortino = None
        else:
            sharpe = None
            sortino = None

        # Max drawdown
        max_dd = max(e.drawdown_pct for e in equity_curve) if equity_curve else 0
        max_dd_dollars = initial * (max_dd / 100)

        # Trade statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl and t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl and t.pnl < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Profit metrics
        wins = [t.pnl for t in trades if t.pnl and t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl and t.pnl < 0]

        avg_win = float(np.mean(wins)) if wins else None
        avg_loss = float(np.mean(losses)) if losses else None

        gross_profits = sum(wins) if wins else 0
        gross_losses = abs(sum(losses)) if losses else 0
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else None

        # Average trade duration
        durations = []
        for t in trades:
            if t.exit_date:
                duration = (t.exit_date - t.entry_date).days
                durations.append(duration)
        avg_duration = float(np.mean(durations)) if durations else None

        # Cost analysis
        total_commissions = sum(
            t.entry_commission + (t.exit_commission or 0) for t in trades
        )
        total_slippage = sum(
            t.shares * t.entry_price * config.slippage_pct +
            (t.shares * t.exit_price * config.slippage_pct if t.exit_price else 0)
            for t in trades
        )

        return BacktestPerformanceMetrics(
            initial_capital=initial,
            final_capital=final,
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd_dollars,
            max_drawdown_pct=max_dd,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_trade_duration=avg_duration,
            total_commissions=total_commissions,
            total_slippage=total_slippage,
        )

    def _compare_benchmark(
        self,
        equity_curve: list[EquityCurvePoint],
        benchmark_data: dict,
        config: BacktestConfig,
    ) -> BenchmarkComparison:
        """Compare strategy performance with benchmark.

        Args:
            equity_curve: Strategy equity curve
            benchmark_data: Benchmark price data
            config: Backtest configuration

        Returns:
            BenchmarkComparison
        """
        # Calculate benchmark return (buy and hold)
        benchmark_prices = benchmark_data.get("closes", [])
        if len(benchmark_prices) >= 2:
            benchmark_return = (benchmark_prices[-1] - benchmark_prices[0]) / benchmark_prices[0] * 100
        else:
            benchmark_return = 0

        # Strategy return
        if equity_curve:
            strategy_return = (equity_curve[-1].equity - config.initial_capital) / config.initial_capital * 100
        else:
            strategy_return = 0

        # Alpha (excess return)
        alpha = strategy_return - benchmark_return

        # Beta and correlation (simplified)
        if len(equity_curve) > 10 and len(benchmark_prices) > 10:
            strategy_values = np.array([e.equity for e in equity_curve])
            benchmark_values = np.array(benchmark_prices[:len(strategy_values)])

            if len(strategy_values) == len(benchmark_values):
                strategy_returns = np.diff(strategy_values) / strategy_values[:-1]
                benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]

                # Remove NaN/inf
                valid = ~(np.isnan(strategy_returns) | np.isnan(benchmark_returns) |
                         np.isinf(strategy_returns) | np.isinf(benchmark_returns))
                strategy_returns = strategy_returns[valid]
                benchmark_returns = benchmark_returns[valid]

                if len(strategy_returns) > 5:
                    cov_matrix = np.cov(strategy_returns, benchmark_returns)
                    if cov_matrix[1, 1] > 0:
                        beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])
                    else:
                        beta = None

                    corr = np.corrcoef(strategy_returns, benchmark_returns)[0, 1]
                    correlation = float(corr) if not np.isnan(corr) else None
                else:
                    beta = None
                    correlation = None
            else:
                beta = None
                correlation = None
        else:
            beta = None
            correlation = None

        return BenchmarkComparison(
            benchmark_symbol=benchmark_data.get("symbol", "SPY"),
            benchmark_return_pct=benchmark_return,
            alpha=alpha,
            beta=beta,
            correlation=correlation,
            outperformed=strategy_return > benchmark_return,
        )

    def _generate_recommendation(
        self,
        performance: BacktestPerformanceMetrics,
        benchmark: Optional[BenchmarkComparison],
    ) -> tuple[BacktestRecommendation, str]:
        """Generate deployment recommendation.

        CRITERIA:
        - DEPLOY: Strong performance, acceptable risk
        - OPTIMIZE: Promising but needs improvement
        - REJECT: Poor performance or excessive risk

        Args:
            performance: Performance metrics
            benchmark: Benchmark comparison

        Returns:
            Tuple of (recommendation, reasoning)
        """
        reasons = []
        score = 0

        # Check return
        if performance.total_return_pct > 15:
            reasons.append(f"Strong return ({performance.total_return_pct:.1f}%)")
            score += 2
        elif performance.total_return_pct > 5:
            reasons.append(f"Moderate return ({performance.total_return_pct:.1f}%)")
            score += 1
        else:
            reasons.append(f"Weak return ({performance.total_return_pct:.1f}%)")
            score -= 1

        # Check Sharpe ratio
        if performance.sharpe_ratio is not None:
            if performance.sharpe_ratio > 1.5:
                reasons.append(f"Excellent Sharpe ({performance.sharpe_ratio:.2f})")
                score += 2
            elif performance.sharpe_ratio > 1.0:
                reasons.append(f"Good Sharpe ({performance.sharpe_ratio:.2f})")
                score += 1
            elif performance.sharpe_ratio > 0.5:
                reasons.append(f"Acceptable Sharpe ({performance.sharpe_ratio:.2f})")
            else:
                reasons.append(f"Poor Sharpe ({performance.sharpe_ratio:.2f})")
                score -= 1

        # Check max drawdown
        if performance.max_drawdown_pct < 15:
            reasons.append(f"Low drawdown ({performance.max_drawdown_pct:.1f}%)")
            score += 1
        elif performance.max_drawdown_pct < 25:
            reasons.append(f"Acceptable drawdown ({performance.max_drawdown_pct:.1f}%)")
        else:
            reasons.append(f"High drawdown ({performance.max_drawdown_pct:.1f}%)")
            score -= 2

        # Check win rate
        if performance.win_rate > 0.6:
            reasons.append(f"High win rate ({performance.win_rate * 100:.0f}%)")
            score += 1
        elif performance.win_rate > 0.4:
            reasons.append(f"Acceptable win rate ({performance.win_rate * 100:.0f}%)")
        else:
            reasons.append(f"Low win rate ({performance.win_rate * 100:.0f}%)")
            score -= 1

        # Check benchmark comparison
        if benchmark:
            if benchmark.outperformed:
                reasons.append(f"Beat benchmark by {benchmark.alpha:.1f}%")
                score += 1
            else:
                reasons.append(f"Underperformed benchmark by {-benchmark.alpha:.1f}%")
                score -= 1

        # Generate recommendation
        if score >= 4:
            recommendation = BacktestRecommendation.DEPLOY
        elif score >= 1:
            recommendation = BacktestRecommendation.OPTIMIZE
        else:
            recommendation = BacktestRecommendation.REJECT

        reasoning = "; ".join(reasons)

        return recommendation, reasoning

    def compare_strategies(
        self,
        strategy_configs: list[StrategyConfig],
        backtest_config: BacktestConfig,
        price_data: dict[str, dict],
        benchmark_data: Optional[dict] = None,
    ) -> StrategyComparisonResult:
        """Compare multiple strategies.

        Args:
            strategy_configs: List of strategy configurations
            backtest_config: Common backtest settings
            price_data: Price data by symbol
            benchmark_data: Optional benchmark data

        Returns:
            StrategyComparisonResult with rankings
        """
        self.context.start()

        try:
            results = []
            comparison_metrics: dict[str, dict[str, float]] = {}

            for config in strategy_configs:
                result = self.run_backtest(
                    strategy_config=config,
                    backtest_config=backtest_config,
                    price_data=price_data,
                    benchmark_data=benchmark_data,
                )
                results.append(result)

                # Extract key metrics for comparison
                comparison_metrics[result.strategy_name] = {
                    "total_return_pct": result.performance.total_return_pct,
                    "sharpe_ratio": result.performance.sharpe_ratio or 0,
                    "max_drawdown_pct": result.performance.max_drawdown_pct,
                    "win_rate": result.performance.win_rate * 100,
                    "total_trades": result.performance.total_trades,
                }

            # Rank strategies by a composite score
            rankings = []
            for result in results:
                perf = result.performance
                # Composite score: return + sharpe - drawdown
                score = (
                    perf.total_return_pct * 0.4 +
                    (perf.sharpe_ratio or 0) * 20 +
                    (100 - perf.max_drawdown_pct) * 0.2 +
                    perf.win_rate * 20
                )
                rankings.append((result.strategy_name, score))

            rankings.sort(key=lambda x: x[1], reverse=True)
            winner = rankings[0][0]

            # Generate recommendation
            winner_result = next(r for r in results if r.strategy_name == winner)
            recommendation = (
                f"Best strategy: {winner} with {winner_result.performance.total_return_pct:.1f}% return, "
                f"Sharpe {winner_result.performance.sharpe_ratio or 0:.2f}, "
                f"Max DD {winner_result.performance.max_drawdown_pct:.1f}%"
            )

            self.context.complete()

            return StrategyComparisonResult(
                strategies=results,
                winner=winner,
                comparison_metrics=comparison_metrics,
                ranking=rankings,
                recommendation=recommendation,
                calculation_time_ms=self.context.elapsed_ms,
            )

        except Exception as e:
            self.context.fail(str(e))
            raise CalculationError(f"Strategy comparison failed: {e}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def run_backtest(
    symbol: str,
    strategy_type: StrategyType,
    price_data: dict,
    start_date: date,
    end_date: date,
    initial_capital: float = 100000.0,
    parameters: Optional[dict] = None,
) -> BacktestResult:
    """Convenience function for running a backtest.

    Args:
        symbol: Asset symbol
        strategy_type: Strategy type to test
        price_data: Price data dict {dates, opens, highs, lows, closes}
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        parameters: Strategy-specific parameters

    Returns:
        BacktestResult
    """
    strategy_config = StrategyConfig(
        strategy_type=strategy_type,
        symbols=[symbol],
        start_date=start_date,
        end_date=end_date,
        parameters=parameters or {},
    )

    backtest_config = BacktestConfig(
        initial_capital=initial_capital,
    )

    engine = BacktestEngine()
    return engine.run_backtest(
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        price_data={symbol: price_data},
    )


def compare_strategies(
    symbol: str,
    strategy_types: list[StrategyType],
    price_data: dict,
    start_date: date,
    end_date: date,
    initial_capital: float = 100000.0,
) -> StrategyComparisonResult:
    """Convenience function for comparing strategies.

    Args:
        symbol: Asset symbol
        strategy_types: List of strategy types to compare
        price_data: Price data dict
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital

    Returns:
        StrategyComparisonResult
    """
    configs = [
        StrategyConfig(
            strategy_type=st,
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
        )
        for st in strategy_types
    ]

    backtest_config = BacktestConfig(initial_capital=initial_capital)

    engine = BacktestEngine()
    return engine.compare_strategies(
        strategy_configs=configs,
        backtest_config=backtest_config,
        price_data={symbol: price_data},
    )
