"""
Finance Guru - Buy and Hold Strategy

Simple buy and hold benchmark strategy.

STRATEGY RULES:
- BUY on first day
- HOLD until end of backtest period

This serves as a benchmark to compare active strategies against.
If your strategy can't beat buy-and-hold, why bother trading?

Author: HealerAgent Development Team
"""

from src.agents.tools.finance_guru.models.backtest import (
    TradeSignal,
    TradeAction,
    StrategyType,
)
from src.agents.tools.finance_guru.strategies.base_strategy import (
    BaseStrategy,
    StrategyContext,
)


class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy and hold benchmark strategy.

    EDUCATIONAL NOTE:
    Buy and hold is the simplest investment strategy:
    1. Buy the asset
    2. Hold it forever (or until you need the money)

    This is the benchmark every active strategy should beat.
    Studies show most active traders underperform buy-and-hold
    after accounting for:
    - Transaction costs (commissions, slippage)
    - Taxes (short-term capital gains)
    - Time spent managing trades
    - Emotional stress and bad decisions

    Warren Buffett recommends this for most investors!
    """

    name = "Buy and Hold"
    strategy_type = StrategyType.BUY_AND_HOLD
    description = "Buy on first day, hold until end (benchmark)"
    min_data_points = 2

    def generate_signals(self, context: StrategyContext) -> list[TradeSignal]:
        """Generate buy signal on first day only.

        Args:
            context: StrategyContext with price data

        Returns:
            List with single BUY signal
        """
        self.validate_context(context)

        # Only one signal: buy on first day
        signals = [
            self._create_signal(
                date_val=context.dates[0],
                symbol=context.symbol,
                action=TradeAction.BUY,
                price=context.closes[0],
                reason="Buy and hold: Initial purchase",
                strength=1.0,
            )
        ]

        return signals
