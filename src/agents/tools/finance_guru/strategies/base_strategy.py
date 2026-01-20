"""
Finance Guru - Base Strategy Interface

Abstract base class for all trading strategies.

All strategies must implement:
- generate_signals(): Convert price data into buy/sell signals

Author: HealerAgent Development Team
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Optional

import numpy as np

from src.agents.tools.finance_guru.models.backtest import (
    TradeSignal,
    TradeAction,
    StrategyType,
)


class StrategyContext:
    """Context for strategy execution.

    Holds price data and current state for signal generation.

    Attributes:
        symbol: Asset symbol
        dates: List of dates
        opens: Open prices
        highs: High prices
        lows: Low prices
        closes: Close prices
        volumes: Trading volumes
        parameters: Strategy-specific parameters
    """

    def __init__(
        self,
        symbol: str,
        dates: list[date],
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: Optional[list[float]] = None,
        parameters: Optional[dict[str, Any]] = None,
    ):
        self.symbol = symbol
        self.dates = dates
        self.opens = np.array(opens)
        self.highs = np.array(highs)
        self.lows = np.array(lows)
        self.closes = np.array(closes)
        self.volumes = np.array(volumes) if volumes else np.zeros(len(closes))
        self.parameters = parameters or {}

    def __len__(self) -> int:
        return len(self.dates)


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    All strategies must implement generate_signals() which returns
    a list of TradeSignal objects based on price data.

    Attributes:
        name: Strategy name
        strategy_type: Strategy type enum
        description: Strategy description
        min_data_points: Minimum data points required
    """

    name: str = "Base Strategy"
    strategy_type: StrategyType = StrategyType.CUSTOM
    description: str = "Abstract base strategy"
    min_data_points: int = 50

    @abstractmethod
    def generate_signals(self, context: StrategyContext) -> list[TradeSignal]:
        """Generate trading signals from price data.

        Args:
            context: StrategyContext with price data and parameters

        Returns:
            List of TradeSignal objects
        """
        pass

    def validate_context(self, context: StrategyContext) -> bool:
        """Validate that context has enough data.

        Args:
            context: StrategyContext to validate

        Returns:
            True if valid, raises ValueError otherwise
        """
        if len(context) < self.min_data_points:
            raise ValueError(
                f"{self.name} requires at least {self.min_data_points} data points, "
                f"got {len(context)}"
            )
        return True

    @staticmethod
    def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average.

        Args:
            prices: Price array
            period: SMA period

        Returns:
            SMA array (NaN for first period-1 values)
        """
        sma = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            sma[i] = np.mean(prices[i - period + 1:i + 1])
        return sma

    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average.

        Args:
            prices: Price array
            period: EMA period

        Returns:
            EMA array
        """
        ema = np.full(len(prices), np.nan)
        multiplier = 2 / (period + 1)

        # Initialize with SMA
        ema[period - 1] = np.mean(prices[:period])

        # Calculate EMA
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]

        return ema

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index.

        Args:
            prices: Price array
            period: RSI period

        Returns:
            RSI array (0-100 scale)
        """
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        rsi = np.full(len(prices), np.nan)

        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss == 0:
            rsi[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[period] = 100 - (100 / (1 + rs))

        # Calculate rest using smoothing
        for i in range(period + 1, len(prices)):
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_macd(
        prices: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD indicator.

        Args:
            prices: Price array
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        fast_ema = BaseStrategy.calculate_ema(prices, fast_period)
        slow_ema = BaseStrategy.calculate_ema(prices, slow_period)

        macd_line = fast_ema - slow_ema
        signal_line = BaseStrategy.calculate_ema(macd_line[~np.isnan(macd_line)], signal_period)

        # Align signal line with macd line
        full_signal = np.full(len(prices), np.nan)
        start_idx = slow_period - 1 + signal_period - 1
        if start_idx < len(prices):
            full_signal[start_idx:start_idx + len(signal_line)] = signal_line

        histogram = macd_line - full_signal

        return macd_line, full_signal, histogram

    @staticmethod
    def calculate_bollinger_bands(
        prices: np.ndarray,
        period: int = 20,
        num_std: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands.

        Args:
            prices: Price array
            period: SMA period
            num_std: Number of standard deviations

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = BaseStrategy.calculate_sma(prices, period)

        std = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            std[i] = np.std(prices[i - period + 1:i + 1])

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        return upper, middle, lower

    def _create_signal(
        self,
        date_val: date,
        symbol: str,
        action: TradeAction,
        price: float,
        reason: str,
        strength: Optional[float] = None,
    ) -> TradeSignal:
        """Helper to create a TradeSignal.

        Args:
            date_val: Signal date
            symbol: Asset symbol
            action: Trade action
            price: Price at signal
            reason: Signal reason
            strength: Optional signal strength

        Returns:
            TradeSignal object
        """
        return TradeSignal(
            signal_date=date_val,
            symbol=symbol,
            action=action,
            price=price,
            signal_strength=strength,
            reason=reason,
        )
