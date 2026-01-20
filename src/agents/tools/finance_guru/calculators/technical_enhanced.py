"""
Finance Guru - Enhanced Technical Indicators Calculator

Implements 5 advanced technical indicators:
1. Ichimoku Cloud - Japanese trend/momentum system (5 lines)
2. Fibonacci Retracement/Extension - Key price levels
3. Williams %R - Momentum oscillator
4. CCI (Commodity Channel Index) - Trend strength
5. Parabolic SAR - Stop and Reverse trailing stop

WHAT: Calculators for enhanced technical analysis beyond basic indicators
WHY: Provides advanced insights for strategy, risk, and timing decisions
ARCHITECTURE: Layer 2 of 3-layer type-safe architecture

Author: HealerAgent Development Team
Created: 2025-01-18
"""

import logging
import math
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.agents.tools.finance_guru.calculators.base import (
    BaseCalculator,
    CalculationError,
    InsufficientDataError,
)
from src.agents.tools.finance_guru.models.base import (
    FinanceValidationError,
    SignalType,
    TrendDirection,
)
from src.agents.tools.finance_guru.models.technical_enhanced import (
    # Input
    OHLCDataInput,
    # Ichimoku
    IchimokuConfig,
    IchimokuLineOutput,
    IchimokuCloudOutput,
    IchimokuSignals,
    IchimokuOutput,
    # Fibonacci
    FibonacciConfig,
    FibonacciLevel,
    FibonacciOutput,
    # Williams %R
    WilliamsRConfig,
    WilliamsROutput,
    # CCI
    CCIConfig,
    CCIOutput,
    # Parabolic SAR
    ParabolicSARConfig,
    ParabolicSAROutput,
    # Combined
    EnhancedTechnicalOutput,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ICHIMOKU CLOUD CALCULATOR
# =============================================================================

class IchimokuCalculator(BaseCalculator[OHLCDataInput, IchimokuOutput, IchimokuConfig]):
    """
    WHAT: Calculates Ichimoku Kinko Hyo (Ichimoku Cloud) indicator
    WHY: Complete trend/momentum analysis system with 5 lines

    EDUCATIONAL NOTE:
    Ichimoku was developed by Goichi Hosoda in Japan before WW2.
    "Ichimoku Kinko Hyo" = "One Glance Equilibrium Chart"

    The 5 Lines:
    1. Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
       - Fast signal line, like a fast MA
       - Shows short-term momentum

    2. Kijun-sen (Base Line): (26-period high + 26-period low) / 2
       - Standard signal line, like a slower MA
       - Acts as support/resistance

    3. Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
       - One edge of the cloud (kumo)

    4. Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, 26 periods ahead
       - Other edge of the cloud

    5. Chikou Span (Lagging Span): Close plotted 26 periods behind
       - Confirms trend by comparing to past price

    The Cloud (Kumo):
    - Area between Senkou A and B
    - Green (bullish) when A > B
    - Red (bearish) when B > A
    - Acts as support/resistance
    - Thicker cloud = stronger trend
    """

    def __init__(self, config: Optional[IchimokuConfig] = None):
        """Initialize with configuration."""
        super().__init__(config or IchimokuConfig())

    def _get_minimum_data_points(self) -> int:
        """Ichimoku needs at least 52 + 26 = 78 periods for full calculation."""
        return self.config.senkou_b_period + self.config.displacement

    def calculate(self, data: OHLCDataInput, **kwargs) -> IchimokuOutput:
        """
        Calculate Ichimoku Cloud indicator.

        Args:
            data: OHLC price data

        Returns:
            IchimokuOutput with all 5 lines and signals
        """
        # Convert to DataFrame
        df = pd.DataFrame({
            "date": data.dates,
            "open": data.open,
            "high": data.high,
            "low": data.low,
            "close": data.close,
        })
        df = df.set_index("date")

        # Get config values
        tenkan_period = self.config.tenkan_period
        kijun_period = self.config.kijun_period
        senkou_b_period = self.config.senkou_b_period
        displacement = self.config.displacement

        # Calculate Tenkan-sen (Conversion Line)
        tenkan_high = df["high"].rolling(window=tenkan_period).max()
        tenkan_low = df["low"].rolling(window=tenkan_period).min()
        df["tenkan_sen"] = (tenkan_high + tenkan_low) / 2

        # Calculate Kijun-sen (Base Line)
        kijun_high = df["high"].rolling(window=kijun_period).max()
        kijun_low = df["low"].rolling(window=kijun_period).min()
        df["kijun_sen"] = (kijun_high + kijun_low) / 2

        # Calculate Senkou Span A (shifted forward by displacement)
        df["senkou_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(displacement)

        # Calculate Senkou Span B (shifted forward by displacement)
        senkou_b_high = df["high"].rolling(window=senkou_b_period).max()
        senkou_b_low = df["low"].rolling(window=senkou_b_period).min()
        df["senkou_b"] = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)

        # Calculate Chikou Span (current close shifted back)
        df["chikou_span"] = df["close"].shift(-displacement)

        # Get current values (latest row)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        current_price = float(latest["close"])
        tenkan_val = float(latest["tenkan_sen"])
        kijun_val = float(latest["kijun_sen"])
        senkou_a_val = float(latest["senkou_a"]) if not pd.isna(latest["senkou_a"]) else tenkan_val
        senkou_b_val = float(latest["senkou_b"]) if not pd.isna(latest["senkou_b"]) else kijun_val
        chikou_val = float(df["close"].iloc[-1])  # Current close for chikou reference

        prev_tenkan = float(prev["tenkan_sen"]) if not pd.isna(prev["tenkan_sen"]) else tenkan_val
        prev_kijun = float(prev["kijun_sen"]) if not pd.isna(prev["kijun_sen"]) else kijun_val

        # Determine cloud properties
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        cloud_thickness = cloud_top - cloud_bottom
        cloud_color = "green" if senkou_a_val > senkou_b_val else "red"
        cloud_thickness_percent = (cloud_thickness / current_price * 100) if current_price > 0 else 0

        # Determine price vs cloud position
        if current_price > cloud_top:
            price_vs_cloud = "above"
        elif current_price < cloud_bottom:
            price_vs_cloud = "below"
        else:
            price_vs_cloud = "inside"

        # Detect TK Cross
        tk_cross = "none"
        tk_cross_location = None
        if prev_tenkan <= prev_kijun and tenkan_val > kijun_val:
            tk_cross = "bullish"
        elif prev_tenkan >= prev_kijun and tenkan_val < kijun_val:
            tk_cross = "bearish"

        if tk_cross != "none":
            cross_price = (tenkan_val + kijun_val) / 2
            if cross_price > cloud_top:
                tk_cross_location = "above_cloud"
            elif cross_price < cloud_bottom:
                tk_cross_location = "below_cloud"
            else:
                tk_cross_location = "in_cloud"

        # Chikou confirmation (compare to price 26 periods ago)
        chikou_confirmation = "neutral"
        if len(df) > displacement:
            price_26_ago = float(df["close"].iloc[-displacement])
            if chikou_val > price_26_ago:
                chikou_confirmation = "bullish"
            elif chikou_val < price_26_ago:
                chikou_confirmation = "bearish"

        # Check for Kumo twist ahead (future cloud color change)
        kumo_twist_ahead = False
        # Look at future senkou spans (if calculated)
        future_senkou_a = (tenkan_val + kijun_val) / 2
        future_senkou_b = (senkou_b_high.iloc[-1] + senkou_b_low.iloc[-1]) / 2 if not pd.isna(senkou_b_high.iloc[-1]) else senkou_b_val
        if (senkou_a_val > senkou_b_val) != (future_senkou_a > future_senkou_b):
            kumo_twist_ahead = True

        # Determine trend strength
        bullish_count = 0
        if price_vs_cloud == "above":
            bullish_count += 1
        if tenkan_val > kijun_val:
            bullish_count += 1
        if chikou_confirmation == "bullish":
            bullish_count += 1
        if cloud_color == "green":
            bullish_count += 1

        if bullish_count >= 4:
            trend_strength = "strong"
            signal = SignalType.BULLISH
        elif bullish_count == 3:
            trend_strength = "moderate"
            signal = SignalType.BULLISH
        elif bullish_count <= 1:
            trend_strength = "strong" if bullish_count == 0 else "moderate"
            signal = SignalType.BEARISH
        else:
            trend_strength = "conflicting"
            signal = SignalType.NEUTRAL

        # Generate interpretation
        interpretation_parts = []
        if price_vs_cloud == "above":
            interpretation_parts.append("Price above cloud (bullish)")
        elif price_vs_cloud == "below":
            interpretation_parts.append("Price below cloud (bearish)")
        else:
            interpretation_parts.append("Price inside cloud (indecisive)")

        if tk_cross == "bullish":
            interpretation_parts.append(f"TK bullish cross {tk_cross_location}")
        elif tk_cross == "bearish":
            interpretation_parts.append(f"TK bearish cross {tk_cross_location}")

        if kumo_twist_ahead:
            interpretation_parts.append("Kumo twist ahead (potential reversal)")

        interpretation = ". ".join(interpretation_parts) + "."

        return IchimokuOutput(
            ticker=data.ticker,
            calculation_date=data.dates[-1],
            calculation_method="Ichimoku Kinko Hyo",
            data_points_used=len(data.dates),
            tenkan_sen=IchimokuLineOutput(
                value=tenkan_val,
                previous_value=prev_tenkan,
            ),
            kijun_sen=IchimokuLineOutput(
                value=kijun_val,
                previous_value=prev_kijun,
            ),
            chikou_span=chikou_val,
            cloud=IchimokuCloudOutput(
                senkou_a=senkou_a_val,
                senkou_b=senkou_b_val,
                cloud_top=cloud_top,
                cloud_bottom=cloud_bottom,
                cloud_color=cloud_color,
                cloud_thickness=cloud_thickness,
                cloud_thickness_percent=cloud_thickness_percent,
            ),
            current_price=current_price,
            signals=IchimokuSignals(
                price_vs_cloud=price_vs_cloud,
                tk_cross=tk_cross,
                tk_cross_location=tk_cross_location,
                chikou_confirmation=chikou_confirmation,
                kumo_twist_ahead=kumo_twist_ahead,
                trend_strength=trend_strength,
                signal=signal,
            ),
            interpretation=interpretation,
        )


# =============================================================================
# FIBONACCI CALCULATOR
# =============================================================================

class FibonacciCalculator(BaseCalculator[OHLCDataInput, FibonacciOutput, FibonacciConfig]):
    """
    WHAT: Calculates Fibonacci retracement and extension levels
    WHY: Identifies key support/resistance levels based on mathematical ratios

    EDUCATIONAL NOTE:
    Fibonacci ratios are derived from the Fibonacci sequence:
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144...

    Key ratios:
    - 23.6% = ratio of a number to the number 2 places higher
    - 38.2% = ratio of a number to the number 1 place higher
    - 61.8% = The Golden Ratio (phi) = limit of Fn/Fn-1
    - 78.6% = Square root of 61.8%

    HOW TO USE:
    1. Identify swing high and swing low
    2. In uptrend: Measure from low to high
       - Retracement levels show potential support during pullback
    3. In downtrend: Measure from high to low
       - Retracement levels show potential resistance during bounce
    4. Extension levels show potential targets beyond the original move
    """

    def __init__(self, config: Optional[FibonacciConfig] = None):
        """Initialize with configuration."""
        super().__init__(config or FibonacciConfig())

    def _get_minimum_data_points(self) -> int:
        """Need enough data to identify swings."""
        return self.config.swing_lookback

    def _detect_swing_points(
        self, df: pd.DataFrame, lookback: int
    ) -> Tuple[float, float, date, date, TrendDirection]:
        """
        Detect the most significant swing high and low.

        Uses a simple approach: Find highest high and lowest low,
        then determine which came first to establish trend direction.

        Returns:
            Tuple of (swing_high, swing_low, high_date, low_date, trend)
        """
        # Find highest high and lowest low in lookback period
        recent_df = df.tail(lookback)

        high_idx = recent_df["high"].idxmax()
        low_idx = recent_df["low"].idxmin()

        swing_high = float(recent_df.loc[high_idx, "high"])
        swing_low = float(recent_df.loc[low_idx, "low"])

        # Determine trend based on which came first
        if high_idx > low_idx:
            # Low came first, then high = uptrend
            trend = TrendDirection.UP
        else:
            # High came first, then low = downtrend
            trend = TrendDirection.DOWN

        return swing_high, swing_low, high_idx, low_idx, trend

    def calculate(self, data: OHLCDataInput, **kwargs) -> FibonacciOutput:
        """
        Calculate Fibonacci retracement and extension levels.

        Args:
            data: OHLC price data

        Returns:
            FibonacciOutput with all Fibonacci levels
        """
        # Convert to DataFrame
        df = pd.DataFrame({
            "date": data.dates,
            "open": data.open,
            "high": data.high,
            "low": data.low,
            "close": data.close,
        })
        df = df.set_index("date")

        # Detect swing points
        if self.config.auto_detect_swings:
            swing_high, swing_low, high_date, low_date, trend = self._detect_swing_points(
                df, self.config.swing_lookback
            )
        else:
            # Use full range (manual would need swing points passed in)
            swing_high = float(df["high"].max())
            swing_low = float(df["low"].min())
            high_date = df["high"].idxmax()
            low_date = df["low"].idxmin()
            trend = TrendDirection.UP if high_date > low_date else TrendDirection.DOWN

        current_price = float(df["close"].iloc[-1])
        price_range = swing_high - swing_low

        # Calculate retracement levels
        retracement_levels = []
        for ratio in self.config.retracement_levels:
            if trend == TrendDirection.UP:
                # In uptrend, retracement is from high downward
                price = swing_high - (price_range * ratio)
            else:
                # In downtrend, retracement is from low upward
                price = swing_low + (price_range * ratio)

            distance = price - current_price
            distance_percent = (distance / current_price * 100) if current_price > 0 else 0

            level = FibonacciLevel(
                ratio=ratio,
                price=round(price, 2),
                label=f"{ratio * 100:.1f}%",
                is_broken=current_price < price if trend == TrendDirection.UP else current_price > price,
                distance_from_price=round(distance, 2),
                distance_percent=round(distance_percent, 2),
            )
            retracement_levels.append(level)

        # Calculate extension levels
        extension_levels = []
        for ratio in self.config.extension_levels:
            if trend == TrendDirection.UP:
                # Extension projects above the swing high
                price = swing_low + (price_range * ratio)
            else:
                # Extension projects below the swing low
                price = swing_high - (price_range * ratio)

            distance = price - current_price
            distance_percent = (distance / current_price * 100) if current_price > 0 else 0

            level = FibonacciLevel(
                ratio=ratio,
                price=round(price, 2),
                label=f"{ratio * 100:.1f}%",
                is_broken=current_price > price if trend == TrendDirection.UP else current_price < price,
                distance_from_price=round(distance, 2),
                distance_percent=round(distance_percent, 2),
            )
            extension_levels.append(level)

        # Find nearest support and resistance
        all_levels = retracement_levels + extension_levels
        supports = [l for l in all_levels if l.distance_from_price < 0]
        resistances = [l for l in all_levels if l.distance_from_price > 0]

        nearest_support = max(supports, key=lambda x: x.price) if supports else None
        nearest_resistance = min(resistances, key=lambda x: x.price) if resistances else None

        # Determine current Fib zone
        current_fib_zone = "outside levels"
        for i in range(len(retracement_levels) - 1):
            l1 = retracement_levels[i]
            l2 = retracement_levels[i + 1]
            low_price = min(l1.price, l2.price)
            high_price = max(l1.price, l2.price)
            if low_price <= current_price <= high_price:
                current_fib_zone = f"between {l1.label} and {l2.label}"
                break

        # Calculate retracement percentage
        retracement_percent = None
        if trend == TrendDirection.UP and price_range > 0:
            # How much of the up move has been retraced
            retracement_percent = ((swing_high - current_price) / price_range) * 100
        elif trend == TrendDirection.DOWN and price_range > 0:
            # How much of the down move has been retraced
            retracement_percent = ((current_price - swing_low) / price_range) * 100

        return FibonacciOutput(
            ticker=data.ticker,
            calculation_date=data.dates[-1],
            calculation_method="Fibonacci Retracement/Extension",
            data_points_used=len(data.dates),
            swing_high=swing_high,
            swing_low=swing_low,
            swing_high_date=high_date,
            swing_low_date=low_date,
            current_price=current_price,
            trend_direction=trend,
            retracement_levels=retracement_levels,
            extension_levels=extension_levels,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            current_fib_zone=current_fib_zone,
            retracement_percent=round(retracement_percent, 1) if retracement_percent else None,
        )


# =============================================================================
# WILLIAMS %R CALCULATOR
# =============================================================================

class WilliamsRCalculator(BaseCalculator[OHLCDataInput, WilliamsROutput, WilliamsRConfig]):
    """
    WHAT: Calculates Williams %R momentum oscillator
    WHY: Identifies overbought/oversold conditions for entry/exit timing

    EDUCATIONAL NOTE:
    Williams %R was developed by Larry Williams.

    Formula:
    %R = (Highest High - Close) / (Highest High - Lowest Low) × -100

    Scale: 0 to -100
    - 0 to -20: Overbought (price near highs)
    - -80 to -100: Oversold (price near lows)
    - -50: Equilibrium

    KEY INSIGHT:
    Williams %R is the inverse of Stochastic %K:
    %R = -100 + Stochastic %K

    TRADING SIGNALS:
    - Buy when %R rises above -80 (exits oversold)
    - Sell when %R falls below -20 (exits overbought)
    - Divergence between price and %R signals potential reversal
    """

    def __init__(self, config: Optional[WilliamsRConfig] = None):
        """Initialize with configuration."""
        super().__init__(config or WilliamsRConfig())

    def _get_minimum_data_points(self) -> int:
        """Need period + buffer for calculations."""
        return self.config.period + 5

    def calculate(self, data: OHLCDataInput, **kwargs) -> WilliamsROutput:
        """
        Calculate Williams %R indicator.

        Args:
            data: OHLC price data

        Returns:
            WilliamsROutput with %R value and signals
        """
        period = self.config.period

        # Convert to DataFrame
        df = pd.DataFrame({
            "date": data.dates,
            "high": data.high,
            "low": data.low,
            "close": data.close,
        })
        df = df.set_index("date")

        # Calculate rolling highest high and lowest low
        highest_high = df["high"].rolling(window=period).max()
        lowest_low = df["low"].rolling(window=period).min()

        # Calculate Williams %R
        df["williams_r"] = ((highest_high - df["close"]) / (highest_high - lowest_low)) * -100

        # Get current and previous values
        current_price = float(df["close"].iloc[-1])
        williams_r = float(df["williams_r"].iloc[-1])
        prev_williams_r = float(df["williams_r"].iloc[-2]) if len(df) > 1 else williams_r
        hh = float(highest_high.iloc[-1])
        ll = float(lowest_low.iloc[-1])

        # Determine zone
        if williams_r >= self.config.overbought_level:
            zone = "overbought"
            signal = SignalType.OVERBOUGHT
        elif williams_r <= self.config.oversold_level:
            zone = "oversold"
            signal = SignalType.OVERSOLD
        else:
            zone = "neutral"
            signal = SignalType.NEUTRAL

        # Determine momentum
        if williams_r > prev_williams_r + 2:
            momentum = "increasing"
        elif williams_r < prev_williams_r - 2:
            momentum = "decreasing"
        else:
            momentum = "flat"

        # Simple divergence detection (compare recent price/indicator trends)
        divergence = "none"
        if len(df) >= 10:
            # Look at last 10 periods
            price_trend = df["close"].iloc[-1] - df["close"].iloc[-10]
            wr_trend = df["williams_r"].iloc[-1] - df["williams_r"].iloc[-10]

            # Bullish divergence: price making lower lows, %R making higher lows
            if price_trend < 0 and wr_trend > 5:
                divergence = "bullish"
            # Bearish divergence: price making higher highs, %R making lower highs
            elif price_trend > 0 and wr_trend < -5:
                divergence = "bearish"

        return WilliamsROutput(
            ticker=data.ticker,
            calculation_date=data.dates[-1],
            calculation_method=f"Williams %R ({period})",
            data_points_used=len(data.dates),
            williams_r=round(williams_r, 2),
            previous_williams_r=round(prev_williams_r, 2),
            current_price=current_price,
            highest_high=hh,
            lowest_low=ll,
            zone=zone,
            signal=signal,
            momentum=momentum,
            divergence=divergence,
        )


# =============================================================================
# CCI CALCULATOR
# =============================================================================

class CCICalculator(BaseCalculator[OHLCDataInput, CCIOutput, CCIConfig]):
    """
    WHAT: Calculates Commodity Channel Index (CCI)
    WHY: Identifies cyclical trends and extreme price movements

    EDUCATIONAL NOTE:
    CCI was developed by Donald Lambert in 1980.

    Formula:
    Typical Price (TP) = (High + Low + Close) / 3
    CCI = (TP - SMA(TP, n)) / (0.015 × Mean Deviation)

    The 0.015 constant scales CCI so that ~70-80% of values
    fall between -100 and +100 in a normal market.

    INTERPRETATION:
    - CCI > +100: Strong uptrend (price above average)
    - CCI > +200: Extremely overbought
    - CCI < -100: Strong downtrend (price below average)
    - CCI < -200: Extremely oversold
    - CCI crossing 0: Momentum shift

    TRADING STRATEGIES:
    1. +100/-100 breakout: Trade in direction of breakout
    2. Zero-line cross: Buy above 0, sell below 0
    3. Divergence: CCI diverging from price signals reversal
    """

    def __init__(self, config: Optional[CCIConfig] = None):
        """Initialize with configuration."""
        super().__init__(config or CCIConfig())

    def _get_minimum_data_points(self) -> int:
        """Need period + buffer for calculations."""
        return self.config.period + 5

    def calculate(self, data: OHLCDataInput, **kwargs) -> CCIOutput:
        """
        Calculate Commodity Channel Index (CCI).

        Args:
            data: OHLC price data

        Returns:
            CCIOutput with CCI value and signals
        """
        period = self.config.period

        # Convert to DataFrame
        df = pd.DataFrame({
            "date": data.dates,
            "high": data.high,
            "low": data.low,
            "close": data.close,
        })
        df = df.set_index("date")

        # Calculate Typical Price
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

        # Calculate SMA of Typical Price
        df["sma_tp"] = df["typical_price"].rolling(window=period).mean()

        # Calculate Mean Deviation
        def mean_deviation(tp_series):
            mean = tp_series.mean()
            return (tp_series - mean).abs().mean()

        df["mean_dev"] = df["typical_price"].rolling(window=period).apply(
            mean_deviation, raw=False
        )

        # Calculate CCI
        df["cci"] = (df["typical_price"] - df["sma_tp"]) / (0.015 * df["mean_dev"])

        # Get current and previous values
        current_price = float(df["close"].iloc[-1])
        typical_price = float(df["typical_price"].iloc[-1])
        sma_tp = float(df["sma_tp"].iloc[-1])
        cci = float(df["cci"].iloc[-1])
        prev_cci = float(df["cci"].iloc[-2]) if len(df) > 1 else cci

        # Determine zone
        if cci >= self.config.extreme_overbought:
            zone = "extreme_overbought"
            signal = SignalType.OVERBOUGHT
        elif cci >= self.config.overbought_level:
            zone = "overbought"
            signal = SignalType.BULLISH
        elif cci <= self.config.extreme_oversold:
            zone = "extreme_oversold"
            signal = SignalType.OVERSOLD
        elif cci <= self.config.oversold_level:
            zone = "oversold"
            signal = SignalType.BEARISH
        else:
            zone = "neutral"
            signal = SignalType.NEUTRAL

        # Check for zero line cross
        zero_cross = "none"
        if prev_cci <= 0 < cci:
            zero_cross = "bullish"
        elif prev_cci >= 0 > cci:
            zero_cross = "bearish"

        # Determine trend strength
        abs_cci = abs(cci)
        if abs_cci > 200:
            trend_strength = "strong"
        elif abs_cci > 100:
            trend_strength = "moderate"
        else:
            trend_strength = "weak"

        # Simple divergence detection
        divergence = "none"
        if len(df) >= 10:
            price_trend = df["close"].iloc[-1] - df["close"].iloc[-10]
            cci_trend = df["cci"].iloc[-1] - df["cci"].iloc[-10]

            if price_trend < 0 and cci_trend > 20:
                divergence = "bullish"
            elif price_trend > 0 and cci_trend < -20:
                divergence = "bearish"

        return CCIOutput(
            ticker=data.ticker,
            calculation_date=data.dates[-1],
            calculation_method=f"CCI ({period})",
            data_points_used=len(data.dates),
            cci=round(cci, 2),
            previous_cci=round(prev_cci, 2),
            current_price=current_price,
            typical_price=round(typical_price, 2),
            sma_typical_price=round(sma_tp, 2),
            zone=zone,
            signal=signal,
            zero_cross=zero_cross,
            trend_strength=trend_strength,
            divergence=divergence,
        )


# =============================================================================
# PARABOLIC SAR CALCULATOR
# =============================================================================

class ParabolicSARCalculator(BaseCalculator[OHLCDataInput, ParabolicSAROutput, ParabolicSARConfig]):
    """
    WHAT: Calculates Parabolic SAR (Stop And Reverse)
    WHY: Provides dynamic trailing stop levels and trend direction

    EDUCATIONAL NOTE:
    Parabolic SAR was developed by J. Welles Wilder Jr.
    (same creator as RSI, ATR, and ADX).

    Formula:
    SAR(tomorrow) = SAR(today) + AF × (EP - SAR(today))

    Where:
    - AF = Acceleration Factor (starts at 0.02, increases by 0.02 each new extreme)
    - EP = Extreme Point (highest high in uptrend, lowest low in downtrend)
    - Max AF is typically 0.20

    HOW IT WORKS:
    1. In uptrend (SAR below price):
       - SAR rises each day, accelerating towards price
       - When price closes below SAR → flip to downtrend

    2. In downtrend (SAR above price):
       - SAR falls each day, accelerating towards price
       - When price closes above SAR → flip to uptrend

    USE CASES:
    - Trailing stop: Use SAR as stop loss level
    - Trend direction: SAR position shows trend
    - Entry/exit: SAR flip signals trend change
    """

    def __init__(self, config: Optional[ParabolicSARConfig] = None):
        """Initialize with configuration."""
        super().__init__(config or ParabolicSARConfig())

    def _get_minimum_data_points(self) -> int:
        """Need sufficient data for SAR calculation."""
        return 30

    def calculate(self, data: OHLCDataInput, **kwargs) -> ParabolicSAROutput:
        """
        Calculate Parabolic SAR indicator.

        Args:
            data: OHLC price data

        Returns:
            ParabolicSAROutput with SAR value and signals
        """
        # Get config
        initial_af = self.config.initial_af
        step_af = self.config.step_af
        max_af = self.config.max_af

        # Convert to arrays for calculation
        highs = np.array(data.high)
        lows = np.array(data.low)
        closes = np.array(data.close)
        n = len(closes)

        # Initialize arrays
        sar = np.zeros(n)
        af = np.zeros(n)
        ep = np.zeros(n)
        trend = np.zeros(n)  # 1 = uptrend, -1 = downtrend

        # Initial values (assume uptrend starting)
        trend[0] = 1
        sar[0] = lows[0]
        ep[0] = highs[0]
        af[0] = initial_af

        bars_since_flip = 0
        last_flip_idx = 0

        for i in range(1, n):
            # Calculate new SAR
            if trend[i - 1] == 1:  # Uptrend
                sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
                # SAR cannot be above prior two lows
                sar[i] = min(sar[i], lows[i - 1])
                if i >= 2:
                    sar[i] = min(sar[i], lows[i - 2])

                # Check for reversal
                if lows[i] < sar[i]:
                    # Flip to downtrend
                    trend[i] = -1
                    sar[i] = ep[i - 1]  # SAR becomes previous EP
                    ep[i] = lows[i]
                    af[i] = initial_af
                    last_flip_idx = i
                else:
                    trend[i] = 1
                    # Update EP and AF
                    if highs[i] > ep[i - 1]:
                        ep[i] = highs[i]
                        af[i] = min(af[i - 1] + step_af, max_af)
                    else:
                        ep[i] = ep[i - 1]
                        af[i] = af[i - 1]

            else:  # Downtrend
                sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
                # SAR cannot be below prior two highs
                sar[i] = max(sar[i], highs[i - 1])
                if i >= 2:
                    sar[i] = max(sar[i], highs[i - 2])

                # Check for reversal
                if highs[i] > sar[i]:
                    # Flip to uptrend
                    trend[i] = 1
                    sar[i] = ep[i - 1]  # SAR becomes previous EP
                    ep[i] = highs[i]
                    af[i] = initial_af
                    last_flip_idx = i
                else:
                    trend[i] = -1
                    # Update EP and AF
                    if lows[i] < ep[i - 1]:
                        ep[i] = lows[i]
                        af[i] = min(af[i - 1] + step_af, max_af)
                    else:
                        ep[i] = ep[i - 1]
                        af[i] = af[i - 1]

        # Get current values
        current_sar = float(sar[-1])
        prev_sar = float(sar[-2]) if n > 1 else current_sar
        current_price = float(closes[-1])
        current_af = float(af[-1])
        current_ep = float(ep[-1])
        current_trend = int(trend[-1])
        bars_since_flip = n - 1 - last_flip_idx

        # Determine SAR position and trend direction
        sar_position = "below" if current_sar < current_price else "above"
        trend_direction = TrendDirection.UP if current_trend == 1 else TrendDirection.DOWN

        # Check for reversal
        reversal = "none"
        prev_trend = int(trend[-2]) if n > 1 else current_trend
        if prev_trend == -1 and current_trend == 1:
            reversal = "bullish"
        elif prev_trend == 1 and current_trend == -1:
            reversal = "bearish"

        # Calculate distance to SAR
        distance = abs(current_price - current_sar)
        distance_percent = (distance / current_price * 100) if current_price > 0 else 0

        # Signal based on position and trend
        if current_trend == 1:
            signal = SignalType.BULLISH
        else:
            signal = SignalType.BEARISH

        return ParabolicSAROutput(
            ticker=data.ticker,
            calculation_date=data.dates[-1],
            calculation_method=f"Parabolic SAR (AF: {initial_af}-{max_af})",
            data_points_used=n,
            sar=round(current_sar, 2),
            previous_sar=round(prev_sar, 2),
            current_price=current_price,
            sar_position=sar_position,
            trend=trend_direction,
            reversal=reversal,
            current_af=round(current_af, 4),
            extreme_point=round(current_ep, 2),
            distance_to_sar=round(distance, 2),
            distance_to_sar_percent=round(distance_percent, 2),
            bars_since_flip=bars_since_flip,
            signal=signal,
            stop_loss_price=round(current_sar, 2),
        )


# =============================================================================
# COMBINED ENHANCED TECHNICAL CALCULATOR
# =============================================================================

class EnhancedTechnicalCalculator:
    """
    WHAT: Master calculator for all enhanced technical indicators
    WHY: Single entry point for comprehensive technical analysis

    This class orchestrates all 5 enhanced technical indicators
    and provides consensus analysis.
    """

    def __init__(
        self,
        ichimoku_config: Optional[IchimokuConfig] = None,
        fibonacci_config: Optional[FibonacciConfig] = None,
        williams_r_config: Optional[WilliamsRConfig] = None,
        cci_config: Optional[CCIConfig] = None,
        parabolic_sar_config: Optional[ParabolicSARConfig] = None,
    ):
        """Initialize with optional configurations for each indicator."""
        self.ichimoku_calc = IchimokuCalculator(ichimoku_config)
        self.fibonacci_calc = FibonacciCalculator(fibonacci_config)
        self.williams_r_calc = WilliamsRCalculator(williams_r_config)
        self.cci_calc = CCICalculator(cci_config)
        self.parabolic_sar_calc = ParabolicSARCalculator(parabolic_sar_config)
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_all(
        self,
        data: OHLCDataInput,
        include_ichimoku: bool = True,
        include_fibonacci: bool = True,
        include_williams_r: bool = True,
        include_cci: bool = True,
        include_parabolic_sar: bool = True,
    ) -> EnhancedTechnicalOutput:
        """
        Calculate all requested enhanced technical indicators.

        Args:
            data: OHLC price data
            include_*: Flags to enable/disable specific indicators

        Returns:
            EnhancedTechnicalOutput with all calculated indicators and consensus
        """
        results = {}
        signals = []

        # Calculate each indicator
        if include_ichimoku:
            try:
                results["ichimoku"] = self.ichimoku_calc.safe_calculate(data)
                signals.append(results["ichimoku"].signals.signal)
            except Exception as e:
                self.logger.warning(f"Ichimoku calculation failed: {e}")
                results["ichimoku"] = None

        if include_fibonacci:
            try:
                results["fibonacci"] = self.fibonacci_calc.safe_calculate(data)
                # Fibonacci doesn't have a direct signal, infer from trend
                if results["fibonacci"].trend_direction == TrendDirection.UP:
                    signals.append(SignalType.BULLISH)
                else:
                    signals.append(SignalType.BEARISH)
            except Exception as e:
                self.logger.warning(f"Fibonacci calculation failed: {e}")
                results["fibonacci"] = None

        if include_williams_r:
            try:
                results["williams_r"] = self.williams_r_calc.safe_calculate(data)
                signals.append(results["williams_r"].signal)
            except Exception as e:
                self.logger.warning(f"Williams %R calculation failed: {e}")
                results["williams_r"] = None

        if include_cci:
            try:
                results["cci"] = self.cci_calc.safe_calculate(data)
                signals.append(results["cci"].signal)
            except Exception as e:
                self.logger.warning(f"CCI calculation failed: {e}")
                results["cci"] = None

        if include_parabolic_sar:
            try:
                results["parabolic_sar"] = self.parabolic_sar_calc.safe_calculate(data)
                signals.append(results["parabolic_sar"].signal)
            except Exception as e:
                self.logger.warning(f"Parabolic SAR calculation failed: {e}")
                results["parabolic_sar"] = None

        # Calculate consensus
        consensus_signal, agreement_percent = self._calculate_consensus(signals)

        # Generate summary
        summary = self._generate_summary(results, consensus_signal, agreement_percent)

        return EnhancedTechnicalOutput(
            ticker=data.ticker,
            calculation_date=data.dates[-1],
            calculation_method="Enhanced Technical Analysis (Phase 2)",
            data_points_used=len(data.dates),
            ichimoku=results.get("ichimoku"),
            fibonacci=results.get("fibonacci"),
            williams_r=results.get("williams_r"),
            cci=results.get("cci"),
            parabolic_sar=results.get("parabolic_sar"),
            consensus_signal=consensus_signal,
            signal_agreement_percent=agreement_percent,
            summary=summary,
        )

    def _calculate_consensus(
        self, signals: List[SignalType]
    ) -> Tuple[SignalType, float]:
        """Calculate consensus signal from multiple indicators."""
        if not signals:
            return SignalType.NEUTRAL, 0.0

        # Count bullish/bearish signals
        bullish_signals = {SignalType.BULLISH, SignalType.STRONG_BUY}
        bearish_signals = {SignalType.BEARISH, SignalType.STRONG_SELL, SignalType.OVERBOUGHT}
        oversold_signals = {SignalType.OVERSOLD}

        bullish = sum(1 for s in signals if s in bullish_signals)
        bearish = sum(1 for s in signals if s in bearish_signals)
        oversold = sum(1 for s in signals if s in oversold_signals)

        total = len(signals)

        if bullish > bearish and bullish > oversold:
            consensus = SignalType.BULLISH
            agreement = (bullish / total) * 100
        elif bearish > bullish and bearish > oversold:
            consensus = SignalType.BEARISH
            agreement = (bearish / total) * 100
        elif oversold > 0 and oversold >= bullish and oversold >= bearish:
            consensus = SignalType.OVERSOLD
            agreement = (oversold / total) * 100
        else:
            consensus = SignalType.NEUTRAL
            agreement = ((total - bullish - bearish - oversold) / total) * 100 if total > 0 else 0

        return consensus, round(agreement, 1)

    def _generate_summary(
        self,
        results: dict,
        consensus: SignalType,
        agreement: float,
    ) -> str:
        """Generate human-readable summary of analysis."""
        parts = []

        # Overall consensus
        # Safely get enum value (handles both Enum and string)
        consensus_str = consensus.value if hasattr(consensus, 'value') else str(consensus)
        parts.append(f"Consensus: {consensus_str.upper()} ({agreement:.0f}% agreement)")

        # Key insights from each indicator
        if results.get("ichimoku"):
            ich = results["ichimoku"]
            parts.append(f"Ichimoku: {ich.signals.trend_strength} trend, price {ich.signals.price_vs_cloud} cloud")

        if results.get("fibonacci"):
            fib = results["fibonacci"]
            parts.append(f"Fibonacci: {fib.current_fib_zone}")

        if results.get("williams_r"):
            wr = results["williams_r"]
            parts.append(f"Williams %R: {wr.zone} ({wr.williams_r:.1f})")

        if results.get("cci"):
            cci = results["cci"]
            parts.append(f"CCI: {cci.zone} ({cci.cci:.1f})")

        if results.get("parabolic_sar"):
            sar = results["parabolic_sar"]
            parts.append(f"SAR: {sar.trend.value}trend, stop at ${sar.stop_loss_price:.2f}")

        return " | ".join(parts)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_ichimoku(
    data: OHLCDataInput,
    config: Optional[IchimokuConfig] = None,
) -> IchimokuOutput:
    """Calculate Ichimoku Cloud indicator."""
    calc = IchimokuCalculator(config)
    return calc.safe_calculate(data)


def calculate_fibonacci(
    data: OHLCDataInput,
    config: Optional[FibonacciConfig] = None,
) -> FibonacciOutput:
    """Calculate Fibonacci retracement/extension levels."""
    calc = FibonacciCalculator(config)
    return calc.safe_calculate(data)


def calculate_williams_r(
    data: OHLCDataInput,
    config: Optional[WilliamsRConfig] = None,
) -> WilliamsROutput:
    """Calculate Williams %R oscillator."""
    calc = WilliamsRCalculator(config)
    return calc.safe_calculate(data)


def calculate_cci(
    data: OHLCDataInput,
    config: Optional[CCIConfig] = None,
) -> CCIOutput:
    """Calculate Commodity Channel Index."""
    calc = CCICalculator(config)
    return calc.safe_calculate(data)


def calculate_parabolic_sar(
    data: OHLCDataInput,
    config: Optional[ParabolicSARConfig] = None,
) -> ParabolicSAROutput:
    """Calculate Parabolic SAR indicator."""
    calc = ParabolicSARCalculator(config)
    return calc.safe_calculate(data)


def calculate_enhanced_technical(
    data: OHLCDataInput,
    include_ichimoku: bool = True,
    include_fibonacci: bool = True,
    include_williams_r: bool = True,
    include_cci: bool = True,
    include_parabolic_sar: bool = True,
) -> EnhancedTechnicalOutput:
    """Calculate all enhanced technical indicators."""
    calc = EnhancedTechnicalCalculator()
    return calc.calculate_all(
        data,
        include_ichimoku=include_ichimoku,
        include_fibonacci=include_fibonacci,
        include_williams_r=include_williams_r,
        include_cci=include_cci,
        include_parabolic_sar=include_parabolic_sar,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Calculators
    "IchimokuCalculator",
    "FibonacciCalculator",
    "WilliamsRCalculator",
    "CCICalculator",
    "ParabolicSARCalculator",
    "EnhancedTechnicalCalculator",
    # Convenience functions
    "calculate_ichimoku",
    "calculate_fibonacci",
    "calculate_williams_r",
    "calculate_cci",
    "calculate_parabolic_sar",
    "calculate_enhanced_technical",
]
