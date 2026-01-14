"""
Technical Analysis Constants and Configuration

This module centralizes all technical analysis parameters and thresholds
to follow Open/Closed Principle and eliminate magic numbers.

Usage:
    from src.agents.tools.technical.technical_constants import TECHNICAL_CONFIG

    rsi = ta.rsi(prices, length=TECHNICAL_CONFIG.RSI_PERIOD)
    if rsi > TECHNICAL_CONFIG.RSI_OVERBOUGHT:
        signal = "SELL"
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class TechnicalAnalysisConfig:
    """
    Configuration class for technical analysis parameters.

    All values are immutable (frozen=True) to prevent accidental modification.
    Change values here to affect all indicator calculations system-wide.
    """

    # =========================================================================
    # RSI Configuration
    # =========================================================================
    RSI_PERIOD: int = 14
    RSI_OVERBOUGHT: float = 70.0
    RSI_OVERSOLD: float = 30.0
    RSI_STRONG_THRESHOLD: float = 60.0  # Above this = strong momentum
    RSI_WEAK_THRESHOLD: float = 40.0    # Below this = weak momentum

    # =========================================================================
    # Moving Average Configuration
    # =========================================================================
    SMA_SHORT_PERIOD: int = 20
    SMA_MEDIUM_PERIOD: int = 50
    SMA_LONG_PERIOD: int = 200
    EMA_FAST_PERIOD: int = 12
    EMA_SLOW_PERIOD: int = 26
    EMA_PERIOD: int = 21  # Default EMA

    # =========================================================================
    # MACD Configuration
    # =========================================================================
    MACD_FAST_PERIOD: int = 12
    MACD_SLOW_PERIOD: int = 26
    MACD_SIGNAL_PERIOD: int = 9

    # =========================================================================
    # Bollinger Bands Configuration
    # =========================================================================
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD_DEV: float = 2.0
    BOLLINGER_SQUEEZE_THRESHOLD: float = 0.05  # Bandwidth below this = squeeze

    # =========================================================================
    # Stochastic Oscillator Configuration
    # =========================================================================
    STOCH_K_PERIOD: int = 14
    STOCH_D_PERIOD: int = 3
    STOCH_SMOOTH: int = 3  # Default smoothing
    STOCH_OVERBOUGHT: float = 80.0
    STOCH_OVERSOLD: float = 20.0

    # =========================================================================
    # Volume Analysis Configuration
    # =========================================================================
    VOLUME_SMA_PERIOD: int = 20
    VOLUME_VERY_HIGH_THRESHOLD: float = 2.0   # 2.0x average = very high
    VOLUME_HIGH_THRESHOLD: float = 1.5        # 1.5x average = high
    VOLUME_LOW_THRESHOLD: float = 0.5         # 0.5x average = low

    # =========================================================================
    # ATR Configuration
    # =========================================================================
    ATR_PERIOD: int = 14
    ADR_PERIOD: int = 20  # Average Daily Range period

    # =========================================================================
    # ADX Configuration
    # =========================================================================
    ADX_PERIOD: int = 14
    ADX_VERY_STRONG_TREND: float = 50.0  # Very strong trend
    ADX_STRONG_TREND: float = 25.0       # Strong trend
    ADX_MODERATE_TREND: float = 20.0     # Moderate trend

    # =========================================================================
    # CCI Configuration (for future use)
    # =========================================================================
    CCI_PERIOD: int = 20
    CCI_OVERBOUGHT: float = 100.0
    CCI_OVERSOLD: float = -100.0

    # =========================================================================
    # Williams %R Configuration (for future use)
    # =========================================================================
    WILLIAMS_R_PERIOD: int = 14
    WILLIAMS_R_OVERBOUGHT: float = -20.0
    WILLIAMS_R_OVERSOLD: float = -80.0

    # =========================================================================
    # Chart Pattern Configuration
    # =========================================================================
    PATTERN_LOOKBACK: int = 50             # Days to look back for patterns
    PATTERN_MIN_BARS: int = 20             # Minimum bars for pattern detection
    PATTERN_TOLERANCE: float = 0.02        # 2% tolerance for double top/bottom
    PATTERN_EXTREMA_ORDER: int = 5         # Order for local extrema detection
    BULL_FLAG_MIN_MOVE: float = 0.05       # 5% minimum move for flag pole
    BULL_FLAG_MAX_CONSOLIDATION: float = 0.03  # 3% max consolidation range
    PATTERN_CONFIDENCE_HIGH: float = 0.75  # High confidence threshold
    PATTERN_CONFIDENCE_MEDIUM: float = 0.70  # Medium confidence threshold

    # =========================================================================
    # Support and Resistance Configuration
    # =========================================================================
    SR_LOOKBACK: int = 5                   # Order for extrema detection
    SR_NUM_LEVELS: int = 5                 # Number of S/R levels to return
    SR_CLUSTER_TOLERANCE: float = 1.0      # 1% tolerance for clustering
    SR_DEFAULT_STRENGTH: float = 0.7       # Default strength for levels
    SR_STRENGTH_INCREMENT: float = 0.1     # Strength increment on merge

    # =========================================================================
    # Signal Generation Thresholds
    # =========================================================================
    OUTLOOK_STRONG_THRESHOLD: float = 0.70  # 70%+ for strong outlook


# Global configuration instance (singleton)
TECHNICAL_CONFIG: Final[TechnicalAnalysisConfig] = TechnicalAnalysisConfig()


@dataclass(frozen=True)
class ScreeningConfig:
    """
    Configuration for stock screening strategies.

    Used by screening tools to filter stocks based on technical criteria.
    """

    # =========================================================================
    # General Filtering
    # =========================================================================
    MIN_VOLUME: int = 1_000_000           # Minimum daily volume
    MIN_PRICE: float = 5.0                # Minimum stock price
    MAX_PRICE: float = 500.0              # Maximum stock price
    MIN_MARKET_CAP: float = 100_000_000   # $100M minimum market cap
    EXCLUDE_PENNY_STOCKS: bool = True
    EXCLUDE_ETFS: bool = False
    MAX_RESULTS_PER_STRATEGY: int = 50

    # =========================================================================
    # Maverick Bullish Strategy
    # =========================================================================
    RSI_MIN_BULLISH: float = 30.0
    RSI_MAX_BULLISH: float = 70.0
    VOLUME_SPIKE_THRESHOLD: float = 1.5   # 1.5x average volume
    MA_CROSSOVER_PERIOD: int = 5          # Days to check for crossover

    # =========================================================================
    # Bear Strategy Thresholds
    # =========================================================================
    RSI_MAX_BEARISH: float = 30.0
    PRICE_DECLINE_THRESHOLD: float = -0.10  # 10% decline

    # =========================================================================
    # Trending Breakout Strategy
    # =========================================================================
    BREAKOUT_VOLUME_MULTIPLIER: float = 2.0
    BREAKOUT_PRICE_THRESHOLD: float = 0.05  # 5% price increase


# Global screening configuration instance
SCREENING_CONFIG: Final[ScreeningConfig] = ScreeningConfig()
