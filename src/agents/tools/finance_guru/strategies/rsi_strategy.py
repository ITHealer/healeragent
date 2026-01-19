"""
Finance Guru - RSI Mean Reversion Strategy

Relative Strength Index based mean reversion strategy.

STRATEGY RULES:
- BUY when RSI drops below oversold threshold (default: 30)
- SELL when RSI rises above overbought threshold (default: 70)

PARAMETERS:
- period: RSI calculation period (default: 14)
- oversold_threshold: Buy below this level (default: 30)
- overbought_threshold: Sell above this level (default: 70)

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


class RSIMeanReversionStrategy(BaseStrategy):
    """RSI-based mean reversion strategy.

    EDUCATIONAL NOTE:
    RSI (Relative Strength Index) measures momentum on a 0-100 scale.
    The strategy assumes prices will "revert to the mean" after extremes.

    OVERSOLD (RSI < 30): Price has fallen too fast, likely to bounce
    OVERBOUGHT (RSI > 70): Price has risen too fast, likely to pull back

    PROS:
    - Works well in ranging/sideways markets
    - Clear entry/exit rules
    - Catches reversals

    CONS:
    - Fails in strong trends (keeps fighting the trend)
    - "Oversold can get more oversold" in crashes
    - May miss big moves by selling too early
    """

    name = "RSI Mean Reversion"
    strategy_type = StrategyType.RSI_MEAN_REVERSION
    description = "Buy when oversold (RSI<30), sell when overbought (RSI>70)"
    min_data_points = 30

    def generate_signals(self, context: StrategyContext) -> list[TradeSignal]:
        """Generate signals based on RSI levels.

        Args:
            context: StrategyContext with price data

        Returns:
            List of TradeSignal objects
        """
        self.validate_context(context)

        # Get parameters (ensure period is integer, thresholds are floats)
        period = int(context.parameters.get("period", 14))
        oversold = float(context.parameters.get("oversold_threshold", 30.0))
        overbought = float(context.parameters.get("overbought_threshold", 70.0))

        # Calculate RSI
        rsi = self.calculate_rsi(context.closes, period)

        signals = []
        position_open = False

        # Start from where RSI is valid
        start_idx = period + 1

        for i in range(start_idx, len(context)):
            if np.isnan(rsi[i]) or np.isnan(rsi[i - 1]):
                continue

            # Buy signal: RSI crosses below oversold then back above
            if not position_open:
                # RSI was oversold and is now rising
                if rsi[i - 1] < oversold and rsi[i] >= oversold:
                    signals.append(self._create_signal(
                        date_val=context.dates[i],
                        symbol=context.symbol,
                        action=TradeAction.BUY,
                        price=context.closes[i],
                        reason=f"RSI bouncing from oversold: {rsi[i]:.1f} crossing above {oversold}",
                        strength=min(1.0, (oversold - rsi[i - 1]) / oversold),
                    ))
                    position_open = True

            # Sell signal: RSI reaches overbought
            elif position_open:
                if rsi[i] > overbought:
                    signals.append(self._create_signal(
                        date_val=context.dates[i],
                        symbol=context.symbol,
                        action=TradeAction.SELL,
                        price=context.closes[i],
                        reason=f"RSI overbought: {rsi[i]:.1f} above {overbought}",
                    ))
                    position_open = False

        return signals
