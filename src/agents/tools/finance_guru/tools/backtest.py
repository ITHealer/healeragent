"""
Finance Guru - Backtest Tools (Phase 5)

Layer 3 of the 3-layer architecture: Agent-callable tool interfaces.

These tools provide the agent-facing interface for backtesting:
- RunBacktestTool: Execute a single strategy backtest
- CompareStrategiesTool: Compare multiple strategies

WHAT: Agent-callable tools for strategy backtesting
WHY: Enables agents to validate investment strategies on historical data
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
from src.agents.tools.finance_guru.calculators.backtest import (
    BacktestEngine,
)
from src.agents.tools.finance_guru.models.backtest import (
    StrategyType,
    StrategyConfig,
    BacktestConfig,
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


async def fetch_historical_prices(
    symbol: str,
    days: int = 365,
    fmp_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch historical OHLCV data from FMP API.

    Args:
        symbol: Stock symbol
        days: Number of days of history
        fmp_api_key: FMP API key

    Returns:
        Dict with dates, opens, highs, lows, closes, volumes
    """
    try:
        from src.services.fmp_service import FMPService

        fmp = FMPService(api_key=fmp_api_key)
        historical = await fmp.get_historical_price(symbol, days=days)

        if not historical or len(historical) < 50:
            raise ValueError(f"Insufficient data for {symbol}")

        # FMP returns newest first, reverse for chronological order
        historical = list(reversed(historical))

        return {
            "symbol": symbol,
            "dates": [date.fromisoformat(d["date"]) for d in historical],
            "opens": [float(d["open"]) for d in historical],
            "highs": [float(d["high"]) for d in historical],
            "lows": [float(d["low"]) for d in historical],
            "closes": [float(d["close"]) for d in historical],
            "volumes": [float(d.get("volume", 0)) for d in historical],
        }

    except ImportError:
        logger.warning("FMPService not available, using mock data")
        return _create_mock_historical_data(symbol, days)


def _create_mock_historical_data(symbol: str, days: int = 365) -> Dict[str, Any]:
    """Create mock historical data for testing."""
    import random

    base_prices = {"AAPL": 150.0, "GOOGL": 140.0, "MSFT": 350.0, "TSLA": 250.0, "SPY": 450.0}
    base_price = base_prices.get(symbol.upper(), 100.0)

    today = date.today()
    dates = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    price = base_price

    for i in range(days):
        d = today - timedelta(days=days - i - 1)
        if d.weekday() < 5:  # Skip weekends
            dates.append(d)

            # Random walk with drift
            change = random.gauss(0.0003, 0.015)
            open_price = price
            close_price = price * (1 + change)
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
            volume = random.randint(10000000, 100000000)

            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)

            price = close_price

    return {
        "symbol": symbol,
        "dates": dates,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
    }


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================


class RunBacktestTool(BaseTool):
    """Tool for running a strategy backtest.

    Tests a trading strategy on historical data with realistic
    transaction costs and generates performance metrics.

    Usage by agent:
        runBacktest(
            symbol="AAPL",
            strategy="sma_crossover",
            days=365,
            initial_capital=100000
        )
    """

    def __init__(self):
        super().__init__()
        self.engine = BacktestEngine()

        self.schema = ToolSchema(
            name="runBacktest",
            category="backtest",
            description=(
                "Run a backtest to test a trading strategy on historical data. "
                "Strategies: sma_crossover (moving average), rsi_mean_reversion (RSI oversold/overbought), "
                "macd_signal (MACD crossover), bollinger_bands (band reversion), buy_and_hold (benchmark). "
                "Returns performance metrics including return, Sharpe ratio, max drawdown, and win rate."
            ),
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock symbol to backtest (e.g., 'AAPL')",
                    required=True,
                ),
                ToolParameter(
                    name="strategy",
                    type="string",
                    description="Strategy type: sma_crossover, rsi_mean_reversion, macd_signal, bollinger_bands, buy_and_hold",
                    required=True,
                ),
                ToolParameter(
                    name="days",
                    type="integer",
                    description="Number of days of historical data (default: 365)",
                    required=False,
                    default=365,
                ),
                ToolParameter(
                    name="initial_capital",
                    type="number",
                    description="Starting capital in dollars (default: 100000)",
                    required=False,
                    default=100000,
                ),
                ToolParameter(
                    name="parameters",
                    type="object",
                    description="Strategy-specific parameters (e.g., {fast_period: 20, slow_period: 50})",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolOutput:
        """Execute the backtest."""
        try:
            symbol = kwargs.get("symbol", "").upper()
            strategy_str = kwargs.get("strategy", "sma_crossover")
            days = min(max(kwargs.get("days", 365), 60), 1000)
            initial_capital = kwargs.get("initial_capital", 100000)
            parameters = kwargs.get("parameters", {})

            if not symbol:
                return create_error_output(self.schema.name, "Symbol is required")

            # Map strategy string to enum
            strategy_map = {
                "sma_crossover": StrategyType.SMA_CROSSOVER,
                "rsi_mean_reversion": StrategyType.RSI_MEAN_REVERSION,
                "macd_signal": StrategyType.MACD_SIGNAL,
                "bollinger_bands": StrategyType.BOLLINGER_BANDS,
                "buy_and_hold": StrategyType.BUY_AND_HOLD,
            }

            strategy_type = strategy_map.get(strategy_str.lower())
            if not strategy_type:
                return create_error_output(
                    self.schema.name,
                    f"Unknown strategy: {strategy_str}. "
                    f"Available: {', '.join(strategy_map.keys())}"
                )

            # Fetch historical data
            price_data = await fetch_historical_prices(symbol, days)

            if len(price_data["dates"]) < 50:
                return create_error_output(
                    self.schema.name,
                    f"Insufficient data for {symbol}: need 50+ days"
                )

            # Fetch benchmark data (SPY)
            benchmark_data = await fetch_historical_prices("SPY", days)

            # Configure backtest
            strategy_config = StrategyConfig(
                strategy_type=strategy_type,
                symbols=[symbol],
                start_date=price_data["dates"][0],
                end_date=price_data["dates"][-1],
                parameters=parameters,
            )

            backtest_config = BacktestConfig(
                initial_capital=initial_capital,
                commission_per_trade=0.0,
                slippage_pct=0.001,
            )

            # Run backtest
            result = self.engine.run_backtest(
                strategy_config=strategy_config,
                backtest_config=backtest_config,
                price_data={symbol: price_data},
                benchmark_data=benchmark_data,
            )

            # Format output
            perf = result.performance
            benchmark = result.benchmark

            formatted = (
                f"Backtest Results: {result.strategy_name} on {symbol}\n"
                f"Period: {result.start_date} to {result.end_date} ({len(price_data['dates'])} days)\n\n"
                f"Performance Summary:\n"
                f"  Initial Capital: ${perf.initial_capital:,.0f}\n"
                f"  Final Capital: ${perf.final_capital:,.0f}\n"
                f"  Total Return: {perf.total_return_pct:+.1f}%\n"
                f"  Annualized Return: {perf.annualized_return_pct:+.1f}%\n" if perf.annualized_return_pct else ""
                f"\nRisk Metrics:\n"
                f"  Sharpe Ratio: {perf.sharpe_ratio:.2f}\n" if perf.sharpe_ratio else ""
                f"  Max Drawdown: {perf.max_drawdown_pct:.1f}%\n"
                f"\nTrade Statistics:\n"
                f"  Total Trades: {perf.total_trades}\n"
                f"  Win Rate: {perf.win_rate * 100:.1f}%\n"
                f"  Profit Factor: {perf.profit_factor:.2f}\n" if perf.profit_factor else ""
            )

            if benchmark:
                formatted += (
                    f"\nBenchmark Comparison ({benchmark.benchmark_symbol}):\n"
                    f"  Benchmark Return: {benchmark.benchmark_return_pct:+.1f}%\n"
                    f"  Alpha: {benchmark.alpha:+.1f}%\n"
                    f"  {'Outperformed' if benchmark.outperformed else 'Underperformed'} benchmark\n"
                )

            formatted += (
                f"\nVerdict: {result.recommendation.value.upper()}\n"
                f"Reasoning: {result.reasoning}"
            )

            return create_success_output(
                self.schema.name,
                data={
                    "symbol": symbol,
                    "strategy": result.strategy_name,
                    "strategy_type": strategy_str,
                    "period": {
                        "start": str(result.start_date),
                        "end": str(result.end_date),
                        "days": len(price_data["dates"]),
                    },
                    "performance": {
                        "initial_capital": perf.initial_capital,
                        "final_capital": perf.final_capital,
                        "total_return_pct": perf.total_return_pct,
                        "annualized_return_pct": perf.annualized_return_pct,
                        "sharpe_ratio": perf.sharpe_ratio,
                        "max_drawdown_pct": perf.max_drawdown_pct,
                        "total_trades": perf.total_trades,
                        "win_rate": perf.win_rate,
                        "profit_factor": perf.profit_factor,
                    },
                    "benchmark": {
                        "symbol": benchmark.benchmark_symbol,
                        "return_pct": benchmark.benchmark_return_pct,
                        "alpha": benchmark.alpha,
                        "outperformed": benchmark.outperformed,
                    } if benchmark else None,
                    "recommendation": result.recommendation.value,
                    "reasoning": result.reasoning,
                },
                formatted_context=formatted,
                execution_time_ms=result.calculation_time_ms or 0,
            )

        except Exception as e:
            logger.exception(f"Backtest failed: {e}")
            return create_error_output(self.schema.name, str(e))


class CompareStrategiesTool(BaseTool):
    """Tool for comparing multiple trading strategies.

    Runs backtests on multiple strategies and provides
    a comparison with rankings.

    Usage by agent:
        compareStrategies(
            symbol="AAPL",
            strategies=["sma_crossover", "rsi_mean_reversion", "macd_signal"]
        )
    """

    def __init__(self):
        super().__init__()
        self.engine = BacktestEngine()

        self.schema = ToolSchema(
            name="compareStrategies",
            category="backtest",
            description=(
                "Compare multiple trading strategies on the same stock. "
                "Runs backtests on each strategy and ranks them by performance. "
                "Identifies the best strategy and provides recommendations."
            ),
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock symbol to backtest",
                    required=True,
                ),
                ToolParameter(
                    name="strategies",
                    type="array",
                    description="List of strategies to compare (e.g., ['sma_crossover', 'rsi_mean_reversion'])",
                    required=False,
                ),
                ToolParameter(
                    name="days",
                    type="integer",
                    description="Number of days of historical data",
                    required=False,
                    default=365,
                ),
                ToolParameter(
                    name="initial_capital",
                    type="number",
                    description="Starting capital",
                    required=False,
                    default=100000,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolOutput:
        """Execute strategy comparison."""
        try:
            symbol = kwargs.get("symbol", "").upper()
            strategies = kwargs.get("strategies", [
                "sma_crossover", "rsi_mean_reversion", "macd_signal", "buy_and_hold"
            ])
            days = min(max(kwargs.get("days", 365), 60), 1000)
            initial_capital = kwargs.get("initial_capital", 100000)

            if not symbol:
                return create_error_output(self.schema.name, "Symbol is required")

            # Map strategy strings to enums
            strategy_map = {
                "sma_crossover": StrategyType.SMA_CROSSOVER,
                "rsi_mean_reversion": StrategyType.RSI_MEAN_REVERSION,
                "macd_signal": StrategyType.MACD_SIGNAL,
                "bollinger_bands": StrategyType.BOLLINGER_BANDS,
                "buy_and_hold": StrategyType.BUY_AND_HOLD,
            }

            strategy_types = []
            for s in strategies:
                st = strategy_map.get(s.lower())
                if st:
                    strategy_types.append(st)

            if len(strategy_types) < 2:
                return create_error_output(
                    self.schema.name,
                    "Need at least 2 valid strategies to compare"
                )

            # Fetch historical data
            price_data = await fetch_historical_prices(symbol, days)

            if len(price_data["dates"]) < 50:
                return create_error_output(
                    self.schema.name,
                    f"Insufficient data for {symbol}"
                )

            # Fetch benchmark
            benchmark_data = await fetch_historical_prices("SPY", days)

            # Create strategy configs
            configs = [
                StrategyConfig(
                    strategy_type=st,
                    symbols=[symbol],
                    start_date=price_data["dates"][0],
                    end_date=price_data["dates"][-1],
                )
                for st in strategy_types
            ]

            backtest_config = BacktestConfig(
                initial_capital=initial_capital,
                commission_per_trade=0.0,
                slippage_pct=0.001,
            )

            # Run comparison
            result = self.engine.compare_strategies(
                strategy_configs=configs,
                backtest_config=backtest_config,
                price_data={symbol: price_data},
                benchmark_data=benchmark_data,
            )

            # Format output
            formatted = f"Strategy Comparison for {symbol}\n"
            formatted += f"Period: {price_data['dates'][0]} to {price_data['dates'][-1]}\n\n"

            formatted += "Rankings (by composite score):\n"
            for i, (name, score) in enumerate(result.ranking, 1):
                metrics = result.comparison_metrics.get(name, {})
                formatted += (
                    f"  {i}. {name}: {metrics.get('total_return_pct', 0):+.1f}% return, "
                    f"Sharpe {metrics.get('sharpe_ratio', 0):.2f}, "
                    f"DD {metrics.get('max_drawdown_pct', 0):.1f}%\n"
                )

            formatted += f"\nWinner: {result.winner}\n"
            formatted += f"Recommendation: {result.recommendation}"

            return create_success_output(
                self.schema.name,
                data={
                    "symbol": symbol,
                    "winner": result.winner,
                    "ranking": result.ranking,
                    "comparison_metrics": result.comparison_metrics,
                    "recommendation": result.recommendation,
                },
                formatted_context=formatted,
                execution_time_ms=result.calculation_time_ms or 0,
            )

        except Exception as e:
            logger.exception(f"Strategy comparison failed: {e}")
            return create_error_output(self.schema.name, str(e))
