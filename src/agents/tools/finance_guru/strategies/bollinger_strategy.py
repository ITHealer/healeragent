"""
Finance Guru - Bollinger Bands Strategy

Mean reversion strategy using Bollinger Bands.

STRATEGY RULES:
- BUY when price touches/crosses below lower band (oversold)
- SELL when price touches/crosses above upper band (overbought)

PARAMETERS:
- period: SMA period for middle band (default: 20)
- num_std: Number of standard deviations (default: 2.0)

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


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands mean reversion strategy.

    EDUCATIONAL NOTE:
    Bollinger Bands create a price envelope around a moving average.

    CALCULATION:
    - Middle Band = SMA(20)
    - Upper Band = Middle + (2 × Standard Deviation)
    - Lower Band = Middle - (2 × Standard Deviation)

    THEORY: ~95% of price action stays within 2 standard deviations.
    Touches of the bands are "extreme" and likely to revert.

    PROS:
    - Self-adjusting (bands widen/narrow with volatility)
    - Clear visual representation
    - Works well in ranging markets

    CONS:
    - In strong trends, price can "ride the band" → multiple losses
    - Requires confirmation signals for best results
    - Volatility squeezes can precede breakouts
    """

    name = "Bollinger Bands"
    strategy_type = StrategyType.BOLLINGER_BANDS
    description = "Buy at lower band, sell at upper band (mean reversion)"
    min_data_points = 30

    def generate_signals(self, context: StrategyContext) -> list[TradeSignal]:
        """Generate signals based on Bollinger Band touches.

        Args:
            context: StrategyContext with price data

        Returns:
            List of TradeSignal objects
        """
        self.validate_context(context)

        # Get parameters
        period = context.parameters.get("period", 20)
        num_std = context.parameters.get("num_std", 2.0)

        # Calculate Bollinger Bands
        upper, middle, lower = self.calculate_bollinger_bands(
            context.closes, period, num_std
        )

        signals = []
        position_open = False

        # Start from where we have valid band values
        start_idx = period

        for i in range(start_idx, len(context)):
            if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]):
                continue

            price = context.closes[i]
            bandwidth = (upper[i] - lower[i]) / middle[i]  # Normalized bandwidth

            # Buy signal: Price touches or crosses below lower band
            if not position_open:
                if price <= lower[i]:
                    # Calculate how far below lower band (strength indicator)
                    deviation = (lower[i] - price) / (upper[i] - lower[i])

                    signals.append(self._create_signal(
                        date_val=context.dates[i],
                        symbol=context.symbol,
                        action=TradeAction.BUY,
                        price=price,
                        reason=f"Price ({price:.2f}) touched lower Bollinger Band ({lower[i]:.2f})",
                        strength=min(1.0, 0.5 + deviation),
                    ))
                    position_open = True

            # Sell signal: Price touches or crosses above upper band
            elif position_open:
                if price >= upper[i]:
                    signals.append(self._create_signal(
                        date_val=context.dates[i],
                        symbol=context.symbol,
                        action=TradeAction.SELL,
                        price=price,
                        reason=f"Price ({price:.2f}) touched upper Bollinger Band ({upper[i]:.2f})",
                    ))
                    position_open = False

                # Also sell if price crosses back to middle band (take profit)
                elif price >= middle[i] and context.closes[i - 1] < middle[i]:
                    signals.append(self._create_signal(
                        date_val=context.dates[i],
                        symbol=context.symbol,
                        action=TradeAction.SELL,
                        price=price,
                        reason=f"Price ({price:.2f}) reached middle band ({middle[i]:.2f})",
                    ))
                    position_open = False

        return signals
