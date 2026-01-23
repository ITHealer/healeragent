"""
Finance Guru - Enhanced Technical Indicator Models

Pydantic models for advanced technical analysis indicators:
- Ichimoku Cloud (Japanese trend/momentum system)
- Fibonacci Retracement/Extension
- Williams %R (Momentum oscillator)
- CCI (Commodity Channel Index)
- Parabolic SAR (Stop and Reverse)

WHAT: Data models for enhanced technical indicator inputs, configuration, and outputs
WHY: Type-safe, validated data structures for Finance Guru agents
ARCHITECTURE: Layer 1 of 3-layer type-safe architecture

Used by: Strategy Advisor, Risk Assessment, Position Sizing workflows
Author: HealerAgent Development Team
Created: 2025-01-18
"""

from datetime import date
from typing import List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from src.agents.tools.finance_guru.models.base import (
    BaseCalculationResult,
    BaseFinanceModel,
    FinanceValidationError,
    SignalType,
    TrendDirection,
)


# =============================================================================
# COMMON INPUT MODEL
# =============================================================================

class OHLCDataInput(BaseFinanceModel):
    """
    WHAT: Historical OHLCV data for technical indicator calculations
    WHY: Ensures valid price data before running any technical indicator

    VALIDATES:
      - Prices are positive
      - Dates are chronological
      - Arrays are aligned in length
      - Sufficient data points for calculations

    EDUCATIONAL NOTE:
    Most technical indicators need OHLC (Open-High-Low-Close) data:
    - Open: First trade price of the day
    - High: Highest price during the trading day
    - Low: Lowest price during the trading day
    - Close: Final price at market close
    - Volume: Total shares/contracts traded (optional for some indicators)

    This gives a complete picture of price action throughout each session.
    """

    ticker: str = Field(
        ...,
        description="Stock ticker symbol (e.g., 'AAPL', 'TSLA')",
        pattern=r"^[A-Z0-9\-\.]{1,20}$",
    )
    dates: List[date] = Field(
        ...,
        min_length=26,  # Minimum for Ichimoku (26-period)
        description="Trading dates (min 26 days for Ichimoku)"
    )
    open: List[float] = Field(
        ...,
        min_length=26,
        description="Daily opening prices"
    )
    high: List[float] = Field(
        ...,
        min_length=26,
        description="Daily high prices"
    )
    low: List[float] = Field(
        ...,
        min_length=26,
        description="Daily low prices"
    )
    close: List[float] = Field(
        ...,
        min_length=26,
        description="Daily closing prices"
    )
    volume: Optional[List[float]] = Field(
        default=None,
        description="Daily trading volumes (optional)"
    )

    @field_validator("open", "high", "low", "close")
    @classmethod
    def prices_must_be_positive(cls, v: List[float], info) -> List[float]:
        """Validate all prices are positive numbers."""
        for i, price in enumerate(v):
            if price <= 0:
                raise FinanceValidationError(
                    f"Price at index {i} must be positive",
                    field=info.field_name,
                    value=price,
                    suggestion="Check for data errors or adjust for stock splits",
                )
        return v

    @field_validator("dates")
    @classmethod
    def dates_must_be_sorted(cls, v: List[date]) -> List[date]:
        """Validate dates are in chronological order."""
        if v != sorted(v):
            raise FinanceValidationError(
                "Dates must be in chronological order",
                field="dates",
                suggestion="Sort data by date ascending before analysis",
            )
        return v

    @field_validator("volume")
    @classmethod
    def volumes_must_be_non_negative(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate volumes are non-negative if provided."""
        if v is not None:
            for i, vol in enumerate(v):
                if vol < 0:
                    raise FinanceValidationError(
                        f"Volume at index {i} cannot be negative",
                        field="volume",
                        value=vol,
                    )
        return v

    @model_validator(mode="after")
    def validate_array_alignment(self) -> "OHLCDataInput":
        """Ensure all price arrays have the same length as dates."""
        n = len(self.dates)
        fields = {"open": self.open, "high": self.high, "low": self.low, "close": self.close}

        for field_name, values in fields.items():
            if len(values) != n:
                raise FinanceValidationError(
                    f"Length mismatch: {n} dates but {len(values)} {field_name} prices",
                    field=field_name,
                    suggestion="Ensure all price arrays match dates length",
                )

        if self.volume is not None and len(self.volume) != n:
            raise FinanceValidationError(
                f"Length mismatch: {n} dates but {len(self.volume)} volume values",
                field="volume",
                suggestion="Ensure volume array matches dates length",
            )

        return self

    @model_validator(mode="after")
    def validate_ohlc_logic(self) -> "OHLCDataInput":
        """Validate OHLC price relationships (High >= Low, etc.)."""
        for i in range(len(self.dates)):
            high, low = self.high[i], self.low[i]
            open_p, close_p = self.open[i], self.close[i]

            if high < low:
                raise FinanceValidationError(
                    f"High ({high}) cannot be less than Low ({low}) at index {i}",
                    field="high/low",
                    suggestion="Check for data quality issues",
                )

            if high < open_p or high < close_p:
                raise FinanceValidationError(
                    f"High ({high}) must be >= Open ({open_p}) and Close ({close_p}) at index {i}",
                    field="high",
                    suggestion="Check for data quality issues",
                )

            if low > open_p or low > close_p:
                raise FinanceValidationError(
                    f"Low ({low}) must be <= Open ({open_p}) and Close ({close_p}) at index {i}",
                    field="low",
                    suggestion="Check for data quality issues",
                )

        return self

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "ticker": "AAPL",
                "dates": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "open": [150.0, 151.5, 152.0],
                "high": [152.5, 153.0, 154.5],
                "low": [149.0, 150.5, 151.0],
                "close": [151.0, 152.0, 153.5],
                "volume": [1000000, 1200000, 900000]
            }]
        }
    }


# =============================================================================
# ICHIMOKU CLOUD
# =============================================================================

class IchimokuConfig(BaseFinanceModel):
    """
    WHAT: Configuration for Ichimoku Cloud calculations
    WHY: Allows customization while maintaining sensible defaults

    EDUCATIONAL NOTE:
    Ichimoku Kinko Hyo ("equilibrium chart at a glance") was developed by
    Goichi Hosoda in Japan. The traditional periods were designed for
    6-day trading weeks (9, 26, 52 periods).

    For modern 5-day weeks, some traders adjust:
    - Traditional: 9, 26, 52
    - Modern: 8, 22, 44 (scaled for 5-day weeks)

    Default settings use traditional values as they are most widely used.
    """

    tenkan_period: int = Field(
        default=9,
        ge=5,
        le=20,
        description="Tenkan-sen (Conversion Line) period. Default: 9"
    )
    kijun_period: int = Field(
        default=26,
        ge=15,
        le=40,
        description="Kijun-sen (Base Line) period. Default: 26"
    )
    senkou_b_period: int = Field(
        default=52,
        ge=30,
        le=80,
        description="Senkou Span B period. Default: 52"
    )
    displacement: int = Field(
        default=26,
        ge=15,
        le=40,
        description="Cloud displacement (Chikou Span shift). Default: 26"
    )


class IchimokuLineOutput(BaseFinanceModel):
    """Output for a single Ichimoku line at current date."""

    value: float = Field(..., description="Current line value")
    previous_value: Optional[float] = Field(
        default=None,
        description="Previous period value (for crossover detection)"
    )


class IchimokuCloudOutput(BaseFinanceModel):
    """
    WHAT: Senkou Span A & B forming the Kumo (Cloud)
    WHY: The cloud is the key feature of Ichimoku for trend/support analysis

    EDUCATIONAL NOTE:
    The cloud (Kumo) shows:
    - Support/Resistance zones
    - Trend direction (price above cloud = bullish)
    - Trend strength (thicker cloud = stronger trend)

    Green Cloud: Senkou A > Senkou B (bullish)
    Red Cloud: Senkou A < Senkou B (bearish)
    """

    senkou_a: float = Field(..., description="Senkou Span A (leading span)")
    senkou_b: float = Field(..., description="Senkou Span B (lagging span)")
    cloud_top: float = Field(..., description="Upper edge of cloud")
    cloud_bottom: float = Field(..., description="Lower edge of cloud")
    cloud_color: Literal["green", "red"] = Field(
        ...,
        description="Cloud color: green (bullish) or red (bearish)"
    )
    cloud_thickness: float = Field(
        ...,
        ge=0,
        description="Cloud thickness (absolute value)"
    )
    cloud_thickness_percent: float = Field(
        ...,
        ge=0,
        description="Cloud thickness as % of price"
    )


class IchimokuSignals(BaseFinanceModel):
    """
    WHAT: Trading signals derived from Ichimoku analysis
    WHY: Provides actionable insights from the complex indicator

    EDUCATIONAL NOTE:
    Key Ichimoku signals (ranked by strength):

    1. TK Cross (Tenkan crosses Kijun)
       - Bullish: Tenkan crosses above Kijun
       - Bearish: Tenkan crosses below Kijun
       - Strongest when above/in cloud

    2. Kumo Breakout (Price vs Cloud)
       - Bullish: Price breaks above cloud
       - Bearish: Price breaks below cloud

    3. Chikou Span Confirmation
       - Bullish: Chikou above price 26 periods ago
       - Bearish: Chikou below price 26 periods ago

    4. Kumo Twist (Cloud color change)
       - Indicates potential trend change ahead
    """

    # Price position relative to cloud
    price_vs_cloud: Literal["above", "inside", "below"] = Field(
        ...,
        description="Price position relative to Kumo"
    )

    # TK Cross signals
    tk_cross: Optional[Literal["bullish", "bearish", "none"]] = Field(
        default="none",
        description="Tenkan/Kijun crossover signal"
    )
    tk_cross_location: Optional[Literal["above_cloud", "in_cloud", "below_cloud"]] = Field(
        default=None,
        description="Where TK cross occurred (affects signal strength)"
    )

    # Chikou confirmation
    chikou_confirmation: Literal["bullish", "bearish", "neutral"] = Field(
        ...,
        description="Chikou Span confirmation signal"
    )

    # Cloud signals
    kumo_twist_ahead: bool = Field(
        default=False,
        description="Cloud color change coming (trend reversal warning)"
    )

    # Overall assessment
    trend_strength: Literal["strong", "moderate", "weak", "conflicting"] = Field(
        ...,
        description="Overall trend strength based on all 5 lines"
    )

    signal: SignalType = Field(
        ...,
        description="Overall trading signal"
    )


class IchimokuOutput(BaseCalculationResult):
    """
    WHAT: Complete Ichimoku Cloud analysis output
    WHY: Provides all 5 lines, cloud analysis, and trading signals

    AGENT USE CASES:
    - Strategy Advisor: Uses for trend direction and strength
    - Risk Assessment: Uses cloud as dynamic support/resistance
    - Entry Timing: Uses TK cross and price/cloud relationship

    EDUCATIONAL NOTE:
    Ichimoku is a "one glance" system designed to show:
    - Trend direction (price vs cloud)
    - Trend strength (cloud thickness)
    - Support/Resistance (cloud edges)
    - Momentum (TK relationship)
    - Confirmation (Chikou position)

    It works best in trending markets and may give false signals in ranges.
    """

    # Individual lines
    tenkan_sen: IchimokuLineOutput = Field(
        ...,
        description="Tenkan-sen (Conversion Line): (9-high + 9-low) / 2"
    )
    kijun_sen: IchimokuLineOutput = Field(
        ...,
        description="Kijun-sen (Base Line): (26-high + 26-low) / 2"
    )
    chikou_span: float = Field(
        ...,
        description="Chikou Span (Lagging Span): Close plotted 26 periods back"
    )

    # Cloud components
    cloud: IchimokuCloudOutput = Field(
        ...,
        description="Kumo (Cloud) analysis"
    )

    # Current price context
    current_price: float = Field(
        ...,
        gt=0,
        description="Latest closing price"
    )

    # Trading signals
    signals: IchimokuSignals = Field(
        ...,
        description="Trading signals derived from Ichimoku"
    )

    # Educational summary
    interpretation: str = Field(
        ...,
        description="Human-readable interpretation of current Ichimoku state"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "ticker": "AAPL",
                "calculation_date": "2025-01-18",
                "tenkan_sen": {"value": 152.5, "previous_value": 151.0},
                "kijun_sen": {"value": 150.0, "previous_value": 150.5},
                "chikou_span": 148.0,
                "cloud": {
                    "senkou_a": 151.25,
                    "senkou_b": 149.0,
                    "cloud_top": 151.25,
                    "cloud_bottom": 149.0,
                    "cloud_color": "green",
                    "cloud_thickness": 2.25,
                    "cloud_thickness_percent": 1.5
                },
                "current_price": 153.0,
                "signals": {
                    "price_vs_cloud": "above",
                    "tk_cross": "bullish",
                    "tk_cross_location": "above_cloud",
                    "chikou_confirmation": "bullish",
                    "kumo_twist_ahead": False,
                    "trend_strength": "strong",
                    "signal": "bullish"
                },
                "interpretation": "Strong bullish trend: Price above cloud, TK bullish cross above cloud, Chikou confirms."
            }]
        }
    }


# =============================================================================
# FIBONACCI RETRACEMENT/EXTENSION
# =============================================================================

class FibonacciConfig(BaseFinanceModel):
    """
    WHAT: Configuration for Fibonacci calculations
    WHY: Allows selection of retracement vs extension and custom levels

    EDUCATIONAL NOTE:
    Fibonacci ratios come from the Fibonacci sequence (0, 1, 1, 2, 3, 5, 8, 13...):
    - 23.6% = 1 - (13/8 × 13/21) (inverse of golden ratio squared)
    - 38.2% = 1 - 61.8%
    - 50.0% = Not Fibonacci, but psychologically important
    - 61.8% = Golden Ratio (phi) = 1/1.618
    - 78.6% = Square root of 61.8%

    Extension levels:
    - 127.2% = Square root of 161.8%
    - 161.8% = Golden Ratio extension
    - 261.8% = 161.8% × 1.618
    """

    # Standard retracement levels
    retracement_levels: List[float] = Field(
        default=[0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0],
        description="Fibonacci retracement levels (as decimals)"
    )

    # Extension levels
    extension_levels: List[float] = Field(
        default=[1.0, 1.272, 1.618, 2.0, 2.618],
        description="Fibonacci extension levels (as decimals)"
    )

    # Auto-detect swing points or use provided
    auto_detect_swings: bool = Field(
        default=True,
        description="Auto-detect swing high/low or use provided prices"
    )

    swing_lookback: int = Field(
        default=50,
        ge=20,
        le=200,
        description="Lookback period for swing detection"
    )


class FibonacciLevel(BaseFinanceModel):
    """Single Fibonacci level with price and context."""

    ratio: float = Field(..., description="Fibonacci ratio (e.g., 0.618)")
    price: float = Field(..., gt=0, description="Price at this level")
    label: str = Field(..., description="Label (e.g., '61.8%')")
    is_broken: bool = Field(
        default=False,
        description="Whether price has broken this level"
    )
    distance_from_price: float = Field(
        ...,
        description="Distance from current price (positive = above, negative = below)"
    )
    distance_percent: float = Field(
        ...,
        description="Distance as percentage of current price"
    )


class FibonacciOutput(BaseCalculationResult):
    """
    WHAT: Fibonacci retracement and extension level analysis
    WHY: Provides key price levels for support, resistance, and targets

    AGENT USE CASES:
    - Entry Planning: Identifies potential support levels for buying
    - Target Setting: Extension levels for profit targets
    - Stop Placement: Levels below entry for stop loss

    EDUCATIONAL NOTE:
    How to use Fibonacci:

    In UPTREND (retracing from high):
    - 38.2%: Shallow retracement (strong trend)
    - 50.0%: Moderate retracement (healthy trend)
    - 61.8%: Deep retracement (trend may be weakening)
    - 78.6%: Very deep (trend reversal possible)

    In DOWNTREND (bouncing from low):
    - Same levels act as resistance

    Extensions:
    - 127.2%: First profit target
    - 161.8%: Primary profit target
    - 261.8%: Extended target (strong trends only)
    """

    # Swing points used
    swing_high: float = Field(..., gt=0, description="Swing high price")
    swing_low: float = Field(..., gt=0, description="Swing low price")
    swing_high_date: date = Field(..., description="Date of swing high")
    swing_low_date: date = Field(..., description="Date of swing low")

    # Current price context
    current_price: float = Field(..., gt=0, description="Current closing price")

    # Trend direction
    trend_direction: TrendDirection = Field(
        ...,
        description="Trend based on swing points"
    )

    # Retracement levels
    retracement_levels: List[FibonacciLevel] = Field(
        ...,
        description="Fibonacci retracement levels"
    )

    # Extension levels
    extension_levels: List[FibonacciLevel] = Field(
        ...,
        description="Fibonacci extension levels"
    )

    # Key levels
    nearest_support: Optional[FibonacciLevel] = Field(
        default=None,
        description="Nearest Fibonacci support level"
    )
    nearest_resistance: Optional[FibonacciLevel] = Field(
        default=None,
        description="Nearest Fibonacci resistance level"
    )

    # Current position
    current_fib_zone: str = Field(
        ...,
        description="Current price zone (e.g., 'between 38.2% and 50%')"
    )

    # Retracement depth (for uptrend)
    retracement_percent: Optional[float] = Field(
        default=None,
        description="How much of the move has been retraced (0-100%)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "ticker": "AAPL",
                "calculation_date": "2025-01-18",
                "swing_high": 200.0,
                "swing_low": 150.0,
                "swing_high_date": "2025-01-10",
                "swing_low_date": "2024-12-01",
                "current_price": 175.0,
                "trend_direction": "up",
                "retracement_levels": [
                    {"ratio": 0.382, "price": 180.9, "label": "38.2%", "is_broken": True, "distance_from_price": 5.9, "distance_percent": 3.37},
                    {"ratio": 0.5, "price": 175.0, "label": "50.0%", "is_broken": False, "distance_from_price": 0.0, "distance_percent": 0.0}
                ],
                "extension_levels": [
                    {"ratio": 1.618, "price": 230.9, "label": "161.8%", "is_broken": False, "distance_from_price": 55.9, "distance_percent": 31.9}
                ],
                "current_fib_zone": "At 50% retracement",
                "retracement_percent": 50.0
            }]
        }
    }


# =============================================================================
# WILLIAMS %R
# =============================================================================

class WilliamsRConfig(BaseFinanceModel):
    """
    WHAT: Configuration for Williams %R calculation
    WHY: Allows period customization

    EDUCATIONAL NOTE:
    Williams %R was developed by Larry Williams.
    It's similar to Stochastic but inverted (0 to -100 scale).

    Default period of 14 is standard, but traders often use:
    - 10: More sensitive (more signals, more noise)
    - 14: Standard (balanced)
    - 20: Less sensitive (fewer but more reliable signals)
    """

    period: int = Field(
        default=14,
        ge=5,
        le=30,
        description="Lookback period (default: 14)"
    )
    overbought_level: float = Field(
        default=-20.0,
        ge=-30.0,
        le=-10.0,
        description="Overbought threshold (default: -20)"
    )
    oversold_level: float = Field(
        default=-80.0,
        ge=-90.0,
        le=-70.0,
        description="Oversold threshold (default: -80)"
    )


class WilliamsROutput(BaseCalculationResult):
    """
    WHAT: Williams %R oscillator analysis
    WHY: Momentum indicator for overbought/oversold conditions

    AGENT USE CASES:
    - Entry Timing: Buy when %R exits oversold (-80 crossing up)
    - Exit Timing: Sell when %R exits overbought (-20 crossing down)
    - Divergence Detection: Price makes new high but %R doesn't

    EDUCATIONAL NOTE:
    Williams %R formula: %R = (Highest High - Close) / (Highest High - Lowest Low) × -100

    Scale: 0 to -100
    - 0 to -20: Overbought (price near recent highs)
    - -80 to -100: Oversold (price near recent lows)
    - -50: Middle ground

    KEY DIFFERENCE FROM STOCHASTIC:
    - Stochastic: 0-100 scale, %K and %D lines
    - Williams %R: 0 to -100 scale, single line
    - Williams %R = -100 + Stochastic %K

    TRADING TIPS:
    - In strong uptrend, %R can stay overbought for extended periods
    - In strong downtrend, %R can stay oversold for extended periods
    - Best signals come from divergences, not just overbought/oversold
    """

    # Current reading
    williams_r: float = Field(
        ...,
        ge=-100,
        le=0,
        description="Williams %R value (-100 to 0)"
    )

    # Previous reading for trend
    previous_williams_r: Optional[float] = Field(
        default=None,
        ge=-100,
        le=0,
        description="Previous period %R value"
    )

    # Current price context
    current_price: float = Field(..., gt=0, description="Current close price")
    highest_high: float = Field(..., gt=0, description="Highest high in period")
    lowest_low: float = Field(..., gt=0, description="Lowest low in period")

    # Signal analysis
    zone: Literal["overbought", "neutral", "oversold"] = Field(
        ...,
        description="Current zone classification"
    )

    signal: SignalType = Field(
        ...,
        description="Trading signal based on %R"
    )

    # Momentum direction
    momentum: Literal["increasing", "decreasing", "flat"] = Field(
        ...,
        description="Momentum direction"
    )

    # Divergence detection
    divergence: Optional[Literal["bullish", "bearish", "none"]] = Field(
        default="none",
        description="Divergence with price (if detected)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "ticker": "AAPL",
                "calculation_date": "2025-01-18",
                "williams_r": -25.5,
                "previous_williams_r": -30.0,
                "current_price": 153.0,
                "highest_high": 155.0,
                "lowest_low": 145.0,
                "zone": "overbought",
                "signal": "overbought",
                "momentum": "increasing",
                "divergence": "none"
            }]
        }
    }


# =============================================================================
# CCI (COMMODITY CHANNEL INDEX)
# =============================================================================

class CCIConfig(BaseFinanceModel):
    """
    WHAT: Configuration for CCI calculation
    WHY: Allows period and threshold customization

    EDUCATIONAL NOTE:
    CCI was developed by Donald Lambert for commodity trading but works
    well on any market. It measures how far price deviates from its
    statistical mean.

    The 0.015 constant was chosen so that 70-80% of CCI values
    fall between -100 and +100 in a normal market.
    """

    period: int = Field(
        default=20,
        ge=10,
        le=50,
        description="CCI calculation period (default: 20)"
    )
    overbought_level: float = Field(
        default=100.0,
        ge=50.0,
        le=200.0,
        description="Overbought threshold (default: 100)"
    )
    oversold_level: float = Field(
        default=-100.0,
        ge=-200.0,
        le=-50.0,
        description="Oversold threshold (default: -100)"
    )
    extreme_overbought: float = Field(
        default=200.0,
        ge=150.0,
        le=300.0,
        description="Extreme overbought level (default: 200)"
    )
    extreme_oversold: float = Field(
        default=-200.0,
        ge=-300.0,
        le=-150.0,
        description="Extreme oversold level (default: -200)"
    )


class CCIOutput(BaseCalculationResult):
    """
    WHAT: Commodity Channel Index analysis
    WHY: Identifies cyclical trends and extreme price movements

    AGENT USE CASES:
    - Trend Identification: CCI > 100 = strong uptrend
    - Mean Reversion: CCI > 200 = extreme, expect pullback
    - Zero Line Cross: Momentum shift signal

    EDUCATIONAL NOTE:
    CCI Formula:
    Typical Price (TP) = (High + Low + Close) / 3
    CCI = (TP - SMA(TP)) / (0.015 × Mean Deviation)

    Interpretation:
    - CCI > +100: Strong uptrend (consider buying)
    - CCI > +200: Extremely overbought (take profits)
    - CCI < -100: Strong downtrend (consider selling)
    - CCI < -200: Extremely oversold (look for bounce)
    - CCI crossing 0: Momentum shift

    TRADING STRATEGIES:
    1. Zero-line cross: Buy when CCI crosses above 0
    2. +100/-100 breakout: Buy when CCI breaks above +100
    3. Divergence: CCI diverging from price signals reversal
    """

    # Current reading
    cci: float = Field(
        ...,
        description="CCI value (typically -300 to +300)"
    )

    # Previous for trend
    previous_cci: Optional[float] = Field(
        default=None,
        description="Previous period CCI"
    )

    # Price context
    current_price: float = Field(..., gt=0, description="Current close")
    typical_price: float = Field(..., gt=0, description="Current typical price")
    sma_typical_price: float = Field(..., gt=0, description="SMA of typical price")

    # Zone classification
    zone: Literal["extreme_overbought", "overbought", "neutral", "oversold", "extreme_oversold"] = Field(
        ...,
        description="Current CCI zone"
    )

    # Signal
    signal: SignalType = Field(
        ...,
        description="Trading signal"
    )

    # Zero line cross
    zero_cross: Optional[Literal["bullish", "bearish", "none"]] = Field(
        default="none",
        description="Zero line crossover signal"
    )

    # Trend strength
    trend_strength: Literal["strong", "moderate", "weak"] = Field(
        ...,
        description="Trend strength based on CCI magnitude"
    )

    # Divergence
    divergence: Optional[Literal["bullish", "bearish", "none"]] = Field(
        default="none",
        description="Divergence signal"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "ticker": "AAPL",
                "calculation_date": "2025-01-18",
                "cci": 125.5,
                "previous_cci": 110.0,
                "current_price": 153.0,
                "typical_price": 152.5,
                "sma_typical_price": 148.0,
                "zone": "overbought",
                "signal": "bullish",
                "zero_cross": "none",
                "trend_strength": "strong",
                "divergence": "none"
            }]
        }
    }


# =============================================================================
# PARABOLIC SAR
# =============================================================================

class ParabolicSARConfig(BaseFinanceModel):
    """
    WHAT: Configuration for Parabolic SAR calculation
    WHY: Allows AF (Acceleration Factor) customization

    EDUCATIONAL NOTE:
    Parabolic SAR was developed by J. Welles Wilder Jr.
    (same creator as RSI and ATR).

    SAR = "Stop And Reverse"

    The Acceleration Factor (AF):
    - Starts at initial_af (usually 0.02)
    - Increases by step (usually 0.02) each time new high/low is made
    - Capped at max_af (usually 0.20)

    This makes SAR accelerate towards price as trend continues.
    """

    initial_af: float = Field(
        default=0.02,
        ge=0.01,
        le=0.05,
        description="Initial Acceleration Factor (default: 0.02)"
    )
    step_af: float = Field(
        default=0.02,
        ge=0.01,
        le=0.05,
        description="AF step increase (default: 0.02)"
    )
    max_af: float = Field(
        default=0.20,
        ge=0.10,
        le=0.30,
        description="Maximum AF (default: 0.20)"
    )


class ParabolicSAROutput(BaseCalculationResult):
    """
    WHAT: Parabolic SAR trend following and stop-loss indicator
    WHY: Provides dynamic stop levels and trend direction

    AGENT USE CASES:
    - Stop Loss Placement: SAR provides trailing stop level
    - Trend Direction: SAR below price = uptrend, above = downtrend
    - Entry Timing: SAR flip signals trend change

    EDUCATIONAL NOTE:
    How Parabolic SAR works:

    UPTREND (SAR below price):
    - SAR value = trailing stop level (sell if price drops below)
    - SAR rises each bar, never falls
    - If price closes below SAR → flip to downtrend

    DOWNTREND (SAR above price):
    - SAR value = trailing stop level (buy if price rises above)
    - SAR falls each bar, never rises
    - If price closes above SAR → flip to uptrend

    TRADING STRATEGY:
    - Use SAR as trailing stop in trending markets
    - Combine with trend filter (e.g., ADX) to avoid whipsaws
    - SAR works poorly in ranging/choppy markets

    STOP CALCULATION:
    SAR(tomorrow) = SAR(today) + AF × (EP - SAR(today))
    Where EP = Extreme Point (highest high in uptrend / lowest low in downtrend)
    """

    # Current SAR value
    sar: float = Field(..., gt=0, description="Current SAR value")
    previous_sar: Optional[float] = Field(
        default=None,
        gt=0,
        description="Previous SAR value"
    )

    # Current price
    current_price: float = Field(..., gt=0, description="Current close price")

    # SAR position relative to price
    sar_position: Literal["below", "above"] = Field(
        ...,
        description="SAR position: below = uptrend, above = downtrend"
    )

    # Trend based on SAR
    trend: TrendDirection = Field(
        ...,
        description="Current trend direction"
    )

    # Reversal detection
    reversal: Optional[Literal["bullish", "bearish", "none"]] = Field(
        default="none",
        description="SAR reversal signal (flip)"
    )

    # Current AF and EP
    current_af: float = Field(
        ...,
        ge=0,
        le=0.30,
        description="Current Acceleration Factor"
    )
    extreme_point: float = Field(
        ...,
        gt=0,
        description="Extreme Point (highest high or lowest low)"
    )

    # Distance to SAR (for stop loss)
    distance_to_sar: float = Field(
        ...,
        description="Absolute distance from price to SAR"
    )
    distance_to_sar_percent: float = Field(
        ...,
        description="Distance to SAR as % of price (stop loss %)"
    )

    # Bars since flip
    bars_since_flip: int = Field(
        ...,
        ge=0,
        description="Number of bars since last SAR reversal"
    )

    # Signal
    signal: SignalType = Field(
        ...,
        description="Trading signal based on SAR"
    )

    # Recommended stop
    stop_loss_price: float = Field(
        ...,
        gt=0,
        description="Recommended stop loss price (SAR value)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "ticker": "AAPL",
                "calculation_date": "2025-01-18",
                "sar": 148.5,
                "previous_sar": 147.8,
                "current_price": 153.0,
                "sar_position": "below",
                "trend": "up",
                "reversal": "none",
                "current_af": 0.06,
                "extreme_point": 155.0,
                "distance_to_sar": 4.5,
                "distance_to_sar_percent": 2.94,
                "bars_since_flip": 12,
                "signal": "bullish",
                "stop_loss_price": 148.5
            }]
        }
    }


# =============================================================================
# COMBINED ENHANCED TECHNICAL OUTPUT
# =============================================================================

class EnhancedTechnicalOutput(BaseCalculationResult):
    """
    WHAT: Combined output for all enhanced technical indicators
    WHY: Provides comprehensive technical analysis in single response

    AGENT USE CASES:
    - Full Technical Analysis: All 5 indicators in one call
    - Multi-indicator Confirmation: Cross-check signals across indicators
    - Dashboard Display: Complete technical overview
    """

    # Individual indicator outputs (optional - may not all be requested)
    ichimoku: Optional[IchimokuOutput] = Field(
        default=None,
        description="Ichimoku Cloud analysis"
    )
    fibonacci: Optional[FibonacciOutput] = Field(
        default=None,
        description="Fibonacci levels analysis"
    )
    williams_r: Optional[WilliamsROutput] = Field(
        default=None,
        description="Williams %R analysis"
    )
    cci: Optional[CCIOutput] = Field(
        default=None,
        description="CCI analysis"
    )
    parabolic_sar: Optional[ParabolicSAROutput] = Field(
        default=None,
        description="Parabolic SAR analysis"
    )

    # Consensus signal
    consensus_signal: SignalType = Field(
        ...,
        description="Consensus signal from all calculated indicators"
    )

    # Signal agreement
    signal_agreement_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage of indicators agreeing with consensus"
    )

    # Summary
    summary: str = Field(
        ...,
        description="Human-readable summary of enhanced technical analysis"
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Input Models
    "OHLCDataInput",
    # Ichimoku
    "IchimokuConfig",
    "IchimokuLineOutput",
    "IchimokuCloudOutput",
    "IchimokuSignals",
    "IchimokuOutput",
    # Fibonacci
    "FibonacciConfig",
    "FibonacciLevel",
    "FibonacciOutput",
    # Williams %R
    "WilliamsRConfig",
    "WilliamsROutput",
    # CCI
    "CCIConfig",
    "CCIOutput",
    # Parabolic SAR
    "ParabolicSARConfig",
    "ParabolicSAROutput",
    # Combined
    "EnhancedTechnicalOutput",
]
