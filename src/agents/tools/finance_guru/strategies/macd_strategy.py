"""
Finance Guru - MACD Signal Strategy

Moving Average Convergence Divergence momentum strategy.

STRATEGY RULES:
- BUY when MACD line crosses ABOVE signal line (bullish crossover)
- SELL when MACD line crosses BELOW signal line (bearish crossover)

PARAMETERS:
- fast_period: Fast EMA period (default: 12)
- slow_period: Slow EMA period (default: 26)
- signal_period: Signal line period (default: 9)

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


class MACDSignalStrategy(BaseStrategy):
    """MACD signal line crossover strategy.

    EDUCATIONAL NOTE:
    MACD is a momentum indicator that shows the relationship between two EMAs.

    CALCULATION:
    - MACD Line = EMA(12) - EMA(26)
    - Signal Line = EMA(9) of MACD Line
    - Histogram = MACD Line - Signal Line

    BULLISH CROSSOVER: MACD crosses above Signal → momentum turning positive
    BEARISH CROSSOVER: MACD crosses below Signal → momentum turning negative

    PROS:
    - Combines trend and momentum
    - More responsive than simple MA crossovers
    - Histogram shows strength of momentum

    CONS:
    - Still a lagging indicator
    - Can give false signals in low-volatility periods
    - Multiple parameters to tune
    """

    name = "MACD Signal"
    strategy_type = StrategyType.MACD_SIGNAL
    description = "Buy on bullish MACD crossover, sell on bearish crossover"
    min_data_points = 50

    def generate_signals(self, context: StrategyContext) -> list[TradeSignal]:
        """Generate signals based on MACD/signal crossovers.

        Args:
            context: StrategyContext with price data

        Returns:
            List of TradeSignal objects
        """
        self.validate_context(context)

        # Get parameters
        fast_period = context.parameters.get("fast_period", 12)
        slow_period = context.parameters.get("slow_period", 26)
        signal_period = context.parameters.get("signal_period", 9)

        # Calculate MACD
        macd_line, signal_line, histogram = self.calculate_macd(
            context.closes, fast_period, slow_period, signal_period
        )

        signals = []
        position_open = False

        # Start from where we have valid MACD values
        start_idx = slow_period + signal_period

        for i in range(start_idx, len(context)):
            if np.isnan(macd_line[i]) or np.isnan(signal_line[i]):
                continue
            if np.isnan(macd_line[i - 1]) or np.isnan(signal_line[i - 1]):
                continue

            # Bullish crossover: MACD crosses above signal
            if not position_open:
                if macd_line[i - 1] <= signal_line[i - 1] and macd_line[i] > signal_line[i]:
                    # Calculate histogram strength
                    hist_strength = abs(histogram[i]) if not np.isnan(histogram[i]) else 0

                    signals.append(self._create_signal(
                        date_val=context.dates[i],
                        symbol=context.symbol,
                        action=TradeAction.BUY,
                        price=context.closes[i],
                        reason=f"MACD bullish crossover: MACD({macd_line[i]:.3f}) > Signal({signal_line[i]:.3f})",
                        strength=min(1.0, hist_strength * 10),  # Scale histogram to 0-1
                    ))
                    position_open = True

            # Bearish crossover: MACD crosses below signal
            elif position_open:
                if macd_line[i - 1] >= signal_line[i - 1] and macd_line[i] < signal_line[i]:
                    signals.append(self._create_signal(
                        date_val=context.dates[i],
                        symbol=context.symbol,
                        action=TradeAction.SELL,
                        price=context.closes[i],
                        reason=f"MACD bearish crossover: MACD({macd_line[i]:.3f}) < Signal({signal_line[i]:.3f})",
                    ))
                    position_open = False

        return signals
