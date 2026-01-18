"""
Finance Guru - Trading Strategies (Phase 5)

This module provides pre-built trading strategies for backtesting:
- SMA Crossover: Trend-following moving average strategy
- RSI Mean Reversion: Buy oversold, sell overbought
- MACD Signal: Momentum crossover strategy
- Bollinger Bands: Mean reversion within bands
- Buy and Hold: Benchmark strategy

All strategies inherit from BaseStrategy and implement generate_signals().

Author: HealerAgent Development Team
"""

from src.agents.tools.finance_guru.strategies.base_strategy import (
    BaseStrategy,
    StrategyContext,
)
from src.agents.tools.finance_guru.strategies.sma_crossover import SMAcrossoverStrategy
from src.agents.tools.finance_guru.strategies.rsi_strategy import RSIMeanReversionStrategy
from src.agents.tools.finance_guru.strategies.macd_strategy import MACDSignalStrategy
from src.agents.tools.finance_guru.strategies.bollinger_strategy import BollingerBandsStrategy
from src.agents.tools.finance_guru.strategies.buy_hold_strategy import BuyAndHoldStrategy

__all__ = [
    # Base
    "BaseStrategy",
    "StrategyContext",
    # Strategies
    "SMAcrossoverStrategy",
    "RSIMeanReversionStrategy",
    "MACDSignalStrategy",
    "BollingerBandsStrategy",
    "BuyAndHoldStrategy",
]
