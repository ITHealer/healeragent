"""
Finance Guru - SMA Crossover Strategy

Simple Moving Average crossover trend-following strategy.

STRATEGY RULES:
- BUY when fast SMA crosses ABOVE slow SMA (golden cross)
- SELL when fast SMA crosses BELOW slow SMA (death cross)

PARAMETERS:
- fast_period: Short-term SMA period (default: 20)
- slow_period: Long-term SMA period (default: 50)

Author: HealerAgent Development Team
"""

import numpy as np

from src.agents.tools.finance_guru.models.backtest import (
    TradeSignal,
    TradeAction,
    StrategyType,
)
from src.agents.tools.finance_guru.strategies.base_strategy import (
    BaseStrategy,
    StrategyContext,
)


class SMAcrossoverStrategy(BaseStrategy):
    """SMA Crossover trend-following strategy.

    EDUCATIONAL NOTE:
    This is one of the oldest and simplest trend-following strategies.
    The idea: prices moving above their average indicates upward momentum.

    GOLDEN CROSS: Fast MA crosses above Slow MA → BUY signal
    DEATH CROSS: Fast MA crosses below Slow MA → SELL signal

    PROS:
    - Simple to understand and implement
    - Captures major trends
    - Works well in trending markets

    CONS:
    - Generates false signals in sideways/choppy markets
    - Lagging indicator (signals come after the trend starts)
    - Multiple whipsaws can erode capital
    """

    name = "SMA Crossover"
    strategy_type = StrategyType.SMA_CROSSOVER
    description = "Buy on golden cross, sell on death cross"
    min_data_points = 60  # Need enough for slow SMA

    def generate_signals(self, context: StrategyContext) -> list[TradeSignal]:
        """Generate signals based on SMA crossovers.

        Args:
            context: StrategyContext with price data

        Returns:
            List of TradeSignal objects
        """
        self.validate_context(context)

        # Get parameters
        fast_period = context.parameters.get("fast_period", 20)
        slow_period = context.parameters.get("slow_period", 50)

        # Calculate SMAs
        fast_sma = self.calculate_sma(context.closes, fast_period)
        slow_sma = self.calculate_sma(context.closes, slow_period)

        signals = []
        position_open = False

        # Start from where we have valid SMA values
        start_idx = slow_period

        for i in range(start_idx, len(context)):
            if np.isnan(fast_sma[i]) or np.isnan(slow_sma[i]):
                continue

            if np.isnan(fast_sma[i - 1]) or np.isnan(slow_sma[i - 1]):
                continue

            # Check for golden cross (fast crosses above slow)
            if not position_open:
                if fast_sma[i - 1] <= slow_sma[i - 1] and fast_sma[i] > slow_sma[i]:
                    signals.append(self._create_signal(
                        date_val=context.dates[i],
                        symbol=context.symbol,
                        action=TradeAction.BUY,
                        price=context.closes[i],
                        reason=f"Golden cross: SMA{fast_period} crossed above SMA{slow_period}",
                        strength=min(1.0, (fast_sma[i] - slow_sma[i]) / slow_sma[i] * 100),
                    ))
                    position_open = True

            # Check for death cross (fast crosses below slow)
            elif position_open:
                if fast_sma[i - 1] >= slow_sma[i - 1] and fast_sma[i] < slow_sma[i]:
                    signals.append(self._create_signal(
                        date_val=context.dates[i],
                        symbol=context.symbol,
                        action=TradeAction.SELL,
                        price=context.closes[i],
                        reason=f"Death cross: SMA{fast_period} crossed below SMA{slow_period}",
                    ))
                    position_open = False

        return signals
