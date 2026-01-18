"""
Finance Guru - Enhanced Technical Analysis Tools

Layer 3 of the 3-layer architecture: Agent-callable tool interfaces.

These tools provide the agent-facing interface for enhanced technical indicators:
- GetIchimokuCloudTool
- GetFibonacciLevelsTool
- GetWilliamsRTool
- GetCCITool
- GetParabolicSARTool
- GetEnhancedTechnicalsTool (all indicators combined)

WHAT: Agent-callable tools for enhanced technical analysis
WHY: Provides standardized interface for LLM agents to access advanced indicators
ARCHITECTURE: Layer 3 of 3-layer type-safe architecture

Author: HealerAgent Development Team
Created: 2025-01-18
"""

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from src.agents.tools.base import (
    BaseTool,
    ToolOutput,
    ToolParameter,
    ToolSchema,
)
from src.agents.tools.finance_guru.calculators.technical_enhanced import (
    IchimokuCalculator,
    FibonacciCalculator,
    WilliamsRCalculator,
    CCICalculator,
    ParabolicSARCalculator,
    EnhancedTechnicalCalculator,
)
from src.agents.tools.finance_guru.models.technical_enhanced import (
    OHLCDataInput,
    IchimokuConfig,
    FibonacciConfig,
    WilliamsRConfig,
    CCIConfig,
    ParabolicSARConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_success_output(
    tool_name: str,
    data: Dict[str, Any],
    formatted_context: Optional[str] = None,
    execution_time_ms: int = 0,
) -> ToolOutput:
    """Create a successful ToolOutput."""
    return ToolOutput(
        tool_name=tool_name,
        status="success",
        data=data,
        formatted_context=formatted_context,
        execution_time_ms=execution_time_ms,
    )


def create_error_output(
    tool_name: str,
    error: str,
    execution_time_ms: int = 0,
) -> ToolOutput:
    """Create an error ToolOutput."""
    return ToolOutput(
        tool_name=tool_name,
        status="error",
        error=error,
        execution_time_ms=execution_time_ms,
    )


async def fetch_ohlc_data(
    symbol: str,
    days: int = 100,
    fmp_api_key: Optional[str] = None,
) -> OHLCDataInput:
    """
    Fetch OHLC data from FMP API.

    This is a placeholder - in production, integrate with your FMP service.

    Args:
        symbol: Stock symbol
        days: Number of days of data
        fmp_api_key: FMP API key

    Returns:
        OHLCDataInput with price data
    """
    # Import FMP service (lazy import to avoid circular deps)
    try:
        from src.services.fmp_service import FMPService

        fmp = FMPService(api_key=fmp_api_key)
        historical = await fmp.get_historical_price(symbol, days=days)

        if not historical or len(historical) < 26:
            raise ValueError(f"Insufficient data for {symbol}: need 26+ days, got {len(historical) if historical else 0}")

        # FMP returns newest first, reverse for chronological order
        historical = list(reversed(historical))

        return OHLCDataInput(
            ticker=symbol.upper(),
            dates=[date.fromisoformat(d["date"]) for d in historical],
            open=[float(d["open"]) for d in historical],
            high=[float(d["high"]) for d in historical],
            low=[float(d["low"]) for d in historical],
            close=[float(d["close"]) for d in historical],
            volume=[float(d.get("volume", 0)) for d in historical],
        )

    except ImportError:
        logger.warning("FMPService not available, using mock data")
        # Return mock data for testing
        return _create_mock_ohlc_data(symbol, days)


def _create_mock_ohlc_data(symbol: str, days: int = 100) -> OHLCDataInput:
    """Create mock OHLC data for testing when FMP is not available."""
    import random

    base_price = 150.0
    dates = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    today = date.today()
    for i in range(days):
        d = today - timedelta(days=days - i - 1)
        # Skip weekends
        if d.weekday() < 5:
            dates.append(d)

            # Random walk price
            change = random.uniform(-2, 2)
            open_p = base_price + random.uniform(-1, 1)
            close_p = base_price + change
            high_p = max(open_p, close_p) + random.uniform(0, 2)
            low_p = min(open_p, close_p) - random.uniform(0, 2)

            opens.append(round(open_p, 2))
            highs.append(round(high_p, 2))
            lows.append(round(low_p, 2))
            closes.append(round(close_p, 2))
            volumes.append(random.randint(500000, 2000000))

            base_price = close_p

    return OHLCDataInput(
        ticker=symbol.upper(),
        dates=dates,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        volume=volumes,
    )


# =============================================================================
# ICHIMOKU CLOUD TOOL
# =============================================================================

class GetIchimokuCloudTool(BaseTool):
    """
    Tool to calculate Ichimoku Cloud (Ichimoku Kinko Hyo) indicator.

    EDUCATIONAL NOTE:
    Ichimoku is a complete trading system that shows:
    - Trend direction (price vs cloud)
    - Support/Resistance (cloud edges)
    - Momentum (TK relationship)
    - Confirmation (Chikou span)

    Best used for: Trend identification, entry/exit timing, stop placement
    """

    def __init__(self, fmp_api_key: Optional[str] = None):
        super().__init__()
        self.fmp_api_key = fmp_api_key
        self.calculator = IchimokuCalculator()

        self.schema = ToolSchema(
            name="getIchimokuCloud",
            category="technical_enhanced",
            description="Calculate Ichimoku Cloud indicator with all 5 lines (Tenkan, Kijun, Senkou A/B, Chikou) and trading signals",
            capabilities=[
                "✅ Full Ichimoku Cloud analysis (5 lines)",
                "✅ TK cross detection with location context",
                "✅ Price vs Cloud position analysis",
                "✅ Chikou Span confirmation",
                "✅ Kumo twist ahead detection",
                "✅ Trend strength assessment",
                "✅ Trading signals (bullish/bearish/neutral)",
            ],
            limitations=[
                "❌ Does NOT provide basic indicators (RSI, MACD)",
                "❌ Does NOT show historical Ichimoku values",
                "❌ Best for trending markets, may give false signals in ranges",
            ],
            usage_hints=[
                "Use when analyzing trend direction and strength",
                "Use for identifying dynamic support/resistance (cloud)",
                "Combine with volume and other indicators for confirmation",
                "Best for swing trading and position trading timeframes",
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol (e.g., 'AAPL', 'TSLA')",
                    required=True,
                    pattern=r"^[A-Z]{1,10}$",
                    examples=["AAPL", "TSLA", "NVDA"],
                ),
                ToolParameter(
                    name="tenkan_period",
                    type="integer",
                    description="Tenkan-sen period (default: 9)",
                    required=False,
                    default=9,
                    min_value=5,
                    max_value=20,
                ),
                ToolParameter(
                    name="kijun_period",
                    type="integer",
                    description="Kijun-sen period (default: 26)",
                    required=False,
                    default=26,
                    min_value=15,
                    max_value=40,
                ),
            ],
            returns={
                "ticker": "string",
                "tenkan_sen": "object (value, previous_value)",
                "kijun_sen": "object (value, previous_value)",
                "chikou_span": "number",
                "cloud": "object (senkou_a, senkou_b, cloud_top, cloud_bottom, color, thickness)",
                "signals": "object (price_vs_cloud, tk_cross, chikou_confirmation, trend_strength, signal)",
                "interpretation": "string",
            },
            examples=[
                {
                    "input": {"symbol": "AAPL"},
                    "output": {
                        "signals": {
                            "price_vs_cloud": "above",
                            "tk_cross": "bullish",
                            "trend_strength": "strong",
                            "signal": "bullish"
                        }
                    }
                }
            ],
            typical_execution_time_ms=2000,
        )

    async def execute(self, **params) -> ToolOutput:
        """Execute Ichimoku Cloud calculation."""
        import time
        start_time = time.time()

        try:
            # Validate input
            validated = self.validate_input(params)
            symbol = validated["symbol"]

            # Fetch OHLC data
            data = await fetch_ohlc_data(symbol, days=100, fmp_api_key=self.fmp_api_key)

            # Configure calculator
            config = IchimokuConfig(
                tenkan_period=validated.get("tenkan_period", 9),
                kijun_period=validated.get("kijun_period", 26),
            )
            calculator = IchimokuCalculator(config)

            # Calculate
            result = calculator.safe_calculate(data)

            # Convert to dict
            result_dict = result.to_dict()

            # Create formatted context for agent
            formatted = f"""
**Ichimoku Cloud Analysis for {symbol}**
- Current Price: ${result.current_price:.2f}
- Tenkan-sen: ${result.tenkan_sen.value:.2f}
- Kijun-sen: ${result.kijun_sen.value:.2f}
- Cloud: {result.cloud.cloud_color.upper()} (${result.cloud.cloud_bottom:.2f} - ${result.cloud.cloud_top:.2f})
- Price Position: {result.signals.price_vs_cloud.upper()} cloud
- TK Cross: {result.signals.tk_cross}
- Trend Strength: {result.signals.trend_strength.upper()}
- Signal: {result.signals.signal.value.upper()}
- Interpretation: {result.interpretation}
"""

            execution_time = int((time.time() - start_time) * 1000)

            return create_success_output(
                tool_name=self.schema.name,
                data=result_dict,
                formatted_context=formatted,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Ichimoku calculation failed: {e}")
            return create_error_output(
                tool_name=self.schema.name,
                error=str(e),
                execution_time_ms=execution_time,
            )


# =============================================================================
# FIBONACCI TOOL
# =============================================================================

class GetFibonacciLevelsTool(BaseTool):
    """
    Tool to calculate Fibonacci retracement and extension levels.

    EDUCATIONAL NOTE:
    Fibonacci levels are based on mathematical ratios found throughout nature.
    Key levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%

    Best used for: Identifying support/resistance, setting profit targets
    """

    def __init__(self, fmp_api_key: Optional[str] = None):
        super().__init__()
        self.fmp_api_key = fmp_api_key
        self.calculator = FibonacciCalculator()

        self.schema = ToolSchema(
            name="getFibonacciLevels",
            category="technical_enhanced",
            description="Calculate Fibonacci retracement and extension levels based on swing highs/lows",
            capabilities=[
                "✅ Auto-detect swing high and low points",
                "✅ Standard retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)",
                "✅ Extension levels (127.2%, 161.8%, 200%, 261.8%)",
                "✅ Current price position within Fib levels",
                "✅ Nearest support and resistance levels",
                "✅ Retracement percentage calculation",
            ],
            limitations=[
                "❌ Requires clear swing points (may be less accurate in choppy markets)",
                "❌ Does NOT predict price direction",
                "❌ Historical swing detection only (not forward-looking)",
            ],
            usage_hints=[
                "Use for identifying potential support during pullbacks",
                "Use 61.8% retracement as key decision level",
                "Use extension levels for profit targets",
                "Combine with other indicators for confluence",
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol",
                    required=True,
                    pattern=r"^[A-Z]{1,10}$",
                ),
                ToolParameter(
                    name="lookback",
                    type="integer",
                    description="Lookback period for swing detection (default: 50)",
                    required=False,
                    default=50,
                    min_value=20,
                    max_value=200,
                ),
            ],
            returns={
                "ticker": "string",
                "swing_high": "number",
                "swing_low": "number",
                "trend_direction": "string (up/down)",
                "retracement_levels": "array of Fib levels",
                "extension_levels": "array of extension levels",
                "current_fib_zone": "string",
                "nearest_support": "object (level)",
                "nearest_resistance": "object (level)",
            },
            typical_execution_time_ms=1500,
        )

    async def execute(self, **params) -> ToolOutput:
        """Execute Fibonacci calculation."""
        import time
        start_time = time.time()

        try:
            validated = self.validate_input(params)
            symbol = validated["symbol"]

            data = await fetch_ohlc_data(symbol, days=100, fmp_api_key=self.fmp_api_key)

            config = FibonacciConfig(
                swing_lookback=validated.get("lookback", 50),
            )
            calculator = FibonacciCalculator(config)

            result = calculator.safe_calculate(data)
            result_dict = result.to_dict()

            # Format key levels
            key_levels = []
            for level in result.retracement_levels:
                if level.ratio in [0.382, 0.5, 0.618]:
                    key_levels.append(f"{level.label}: ${level.price:.2f}")

            formatted = f"""
**Fibonacci Analysis for {symbol}**
- Swing High: ${result.swing_high:.2f} ({result.swing_high_date})
- Swing Low: ${result.swing_low:.2f} ({result.swing_low_date})
- Trend: {result.trend_direction.value.upper()}
- Current Price: ${result.current_price:.2f}
- Current Zone: {result.current_fib_zone}
- Retracement: {result.retracement_percent:.1f}% of move
- Key Levels: {', '.join(key_levels)}
- Nearest Support: ${result.nearest_support.price:.2f} ({result.nearest_support.label}) if result.nearest_support else 'N/A'
- Nearest Resistance: ${result.nearest_resistance.price:.2f} ({result.nearest_resistance.label}) if result.nearest_resistance else 'N/A'
"""

            execution_time = int((time.time() - start_time) * 1000)

            return create_success_output(
                tool_name=self.schema.name,
                data=result_dict,
                formatted_context=formatted,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Fibonacci calculation failed: {e}")
            return create_error_output(
                tool_name=self.schema.name,
                error=str(e),
                execution_time_ms=execution_time,
            )


# =============================================================================
# WILLIAMS %R TOOL
# =============================================================================

class GetWilliamsRTool(BaseTool):
    """
    Tool to calculate Williams %R momentum oscillator.

    EDUCATIONAL NOTE:
    Williams %R shows where current price is relative to the high-low range.
    Scale: 0 to -100 (-20 = overbought, -80 = oversold)

    Best used for: Timing entries/exits, divergence detection
    """

    def __init__(self, fmp_api_key: Optional[str] = None):
        super().__init__()
        self.fmp_api_key = fmp_api_key
        self.calculator = WilliamsRCalculator()

        self.schema = ToolSchema(
            name="getWilliamsR",
            category="technical_enhanced",
            description="Calculate Williams %R momentum oscillator for overbought/oversold analysis",
            capabilities=[
                "✅ Williams %R value (-100 to 0 scale)",
                "✅ Overbought/Oversold zone classification",
                "✅ Momentum direction (increasing/decreasing)",
                "✅ Basic divergence detection",
                "✅ Trading signal generation",
            ],
            limitations=[
                "❌ Can stay overbought/oversold in strong trends",
                "❌ Works best in ranging markets",
                "❌ Should be confirmed with other indicators",
            ],
            usage_hints=[
                "Buy when %R exits oversold (-80 crossing up)",
                "Sell when %R exits overbought (-20 crossing down)",
                "Look for divergences for reversal signals",
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol",
                    required=True,
                    pattern=r"^[A-Z]{1,10}$",
                ),
                ToolParameter(
                    name="period",
                    type="integer",
                    description="Lookback period (default: 14)",
                    required=False,
                    default=14,
                    min_value=5,
                    max_value=30,
                ),
            ],
            returns={
                "ticker": "string",
                "williams_r": "number (-100 to 0)",
                "zone": "string (overbought/neutral/oversold)",
                "signal": "string",
                "momentum": "string (increasing/decreasing/flat)",
                "divergence": "string (bullish/bearish/none)",
            },
            typical_execution_time_ms=1000,
        )

    async def execute(self, **params) -> ToolOutput:
        """Execute Williams %R calculation."""
        import time
        start_time = time.time()

        try:
            validated = self.validate_input(params)
            symbol = validated["symbol"]

            data = await fetch_ohlc_data(symbol, days=50, fmp_api_key=self.fmp_api_key)

            config = WilliamsRConfig(period=validated.get("period", 14))
            calculator = WilliamsRCalculator(config)

            result = calculator.safe_calculate(data)
            result_dict = result.to_dict()

            formatted = f"""
**Williams %R for {symbol}**
- %R Value: {result.williams_r:.1f}
- Zone: {result.zone.upper()}
- Signal: {result.signal.value.upper()}
- Momentum: {result.momentum.upper()}
- Divergence: {result.divergence.upper() if result.divergence else 'None'}
- Price Range: ${result.lowest_low:.2f} - ${result.highest_high:.2f}
- Current Price: ${result.current_price:.2f}
"""

            execution_time = int((time.time() - start_time) * 1000)

            return create_success_output(
                tool_name=self.schema.name,
                data=result_dict,
                formatted_context=formatted,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Williams %R calculation failed: {e}")
            return create_error_output(
                tool_name=self.schema.name,
                error=str(e),
                execution_time_ms=execution_time,
            )


# =============================================================================
# CCI TOOL
# =============================================================================

class GetCCITool(BaseTool):
    """
    Tool to calculate Commodity Channel Index (CCI).

    EDUCATIONAL NOTE:
    CCI measures deviation from average price. >100 = strong trend, >200 = extreme.
    Works on any asset despite "Commodity" in the name.

    Best used for: Identifying strong trends and extreme conditions
    """

    def __init__(self, fmp_api_key: Optional[str] = None):
        super().__init__()
        self.fmp_api_key = fmp_api_key
        self.calculator = CCICalculator()

        self.schema = ToolSchema(
            name="getCCI",
            category="technical_enhanced",
            description="Calculate Commodity Channel Index for trend strength and extreme conditions",
            capabilities=[
                "✅ CCI value (typically -300 to +300)",
                "✅ Zone classification (overbought, oversold, extreme)",
                "✅ Zero-line cross detection",
                "✅ Trend strength assessment",
                "✅ Divergence detection",
            ],
            limitations=[
                "❌ No upper/lower bounds (can exceed ±200)",
                "❌ Can stay extreme in strong trends",
                "❌ Requires confirmation from price action",
            ],
            usage_hints=[
                "CCI > +100 = strong uptrend",
                "CCI < -100 = strong downtrend",
                "Zero-line cross signals momentum shift",
                "Extreme readings (±200) often precede reversals",
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol",
                    required=True,
                    pattern=r"^[A-Z]{1,10}$",
                ),
                ToolParameter(
                    name="period",
                    type="integer",
                    description="CCI period (default: 20)",
                    required=False,
                    default=20,
                    min_value=10,
                    max_value=50,
                ),
            ],
            returns={
                "ticker": "string",
                "cci": "number",
                "zone": "string",
                "signal": "string",
                "zero_cross": "string (bullish/bearish/none)",
                "trend_strength": "string (strong/moderate/weak)",
            },
            typical_execution_time_ms=1000,
        )

    async def execute(self, **params) -> ToolOutput:
        """Execute CCI calculation."""
        import time
        start_time = time.time()

        try:
            validated = self.validate_input(params)
            symbol = validated["symbol"]

            data = await fetch_ohlc_data(symbol, days=50, fmp_api_key=self.fmp_api_key)

            config = CCIConfig(period=validated.get("period", 20))
            calculator = CCICalculator(config)

            result = calculator.safe_calculate(data)
            result_dict = result.to_dict()

            formatted = f"""
**CCI Analysis for {symbol}**
- CCI Value: {result.cci:.1f}
- Zone: {result.zone.upper()}
- Signal: {result.signal.value.upper()}
- Trend Strength: {result.trend_strength.upper()}
- Zero Cross: {result.zero_cross.upper() if result.zero_cross else 'None'}
- Divergence: {result.divergence.upper() if result.divergence else 'None'}
- Current Price: ${result.current_price:.2f}
"""

            execution_time = int((time.time() - start_time) * 1000)

            return create_success_output(
                tool_name=self.schema.name,
                data=result_dict,
                formatted_context=formatted,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"CCI calculation failed: {e}")
            return create_error_output(
                tool_name=self.schema.name,
                error=str(e),
                execution_time_ms=execution_time,
            )


# =============================================================================
# PARABOLIC SAR TOOL
# =============================================================================

class GetParabolicSARTool(BaseTool):
    """
    Tool to calculate Parabolic SAR (Stop And Reverse).

    EDUCATIONAL NOTE:
    SAR provides trailing stop levels and trend direction.
    SAR below price = uptrend, SAR above price = downtrend.

    Best used for: Trailing stops, trend direction confirmation
    """

    def __init__(self, fmp_api_key: Optional[str] = None):
        super().__init__()
        self.fmp_api_key = fmp_api_key
        self.calculator = ParabolicSARCalculator()

        self.schema = ToolSchema(
            name="getParabolicSAR",
            category="technical_enhanced",
            description="Calculate Parabolic SAR for trailing stops and trend direction",
            capabilities=[
                "✅ Current SAR stop level",
                "✅ Trend direction (up/down)",
                "✅ SAR reversal detection",
                "✅ Distance to stop (dollars and %)",
                "✅ Acceleration factor tracking",
                "✅ Recommended stop loss price",
            ],
            limitations=[
                "❌ Whipsaws in sideways markets",
                "❌ No volume consideration",
                "❌ Best used with trend filter (like ADX)",
            ],
            usage_hints=[
                "Use SAR as dynamic trailing stop in trends",
                "SAR flip signals potential trend reversal",
                "Combine with ADX to filter sideways markets",
                "Distance to SAR shows current risk level",
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol",
                    required=True,
                    pattern=r"^[A-Z]{1,10}$",
                ),
                ToolParameter(
                    name="initial_af",
                    type="number",
                    description="Initial Acceleration Factor (default: 0.02)",
                    required=False,
                    default=0.02,
                    min_value=0.01,
                    max_value=0.05,
                ),
                ToolParameter(
                    name="max_af",
                    type="number",
                    description="Maximum Acceleration Factor (default: 0.20)",
                    required=False,
                    default=0.20,
                    min_value=0.10,
                    max_value=0.30,
                ),
            ],
            returns={
                "ticker": "string",
                "sar": "number (stop level)",
                "trend": "string (up/down)",
                "reversal": "string (bullish/bearish/none)",
                "distance_to_sar": "number (dollars)",
                "distance_to_sar_percent": "number",
                "stop_loss_price": "number",
                "bars_since_flip": "number",
            },
            typical_execution_time_ms=1000,
        )

    async def execute(self, **params) -> ToolOutput:
        """Execute Parabolic SAR calculation."""
        import time
        start_time = time.time()

        try:
            validated = self.validate_input(params)
            symbol = validated["symbol"]

            data = await fetch_ohlc_data(symbol, days=100, fmp_api_key=self.fmp_api_key)

            config = ParabolicSARConfig(
                initial_af=validated.get("initial_af", 0.02),
                max_af=validated.get("max_af", 0.20),
            )
            calculator = ParabolicSARCalculator(config)

            result = calculator.safe_calculate(data)
            result_dict = result.to_dict()

            formatted = f"""
**Parabolic SAR for {symbol}**
- SAR Value: ${result.sar:.2f}
- Trend: {result.trend.value.upper()}TREND
- Current Price: ${result.current_price:.2f}
- SAR Position: {result.sar_position.upper()} price
- Distance to SAR: ${result.distance_to_sar:.2f} ({result.distance_to_sar_percent:.1f}%)
- Recommended Stop: ${result.stop_loss_price:.2f}
- Reversal Signal: {result.reversal.upper() if result.reversal else 'None'}
- Bars Since Flip: {result.bars_since_flip}
- Signal: {result.signal.value.upper()}
"""

            execution_time = int((time.time() - start_time) * 1000)

            return create_success_output(
                tool_name=self.schema.name,
                data=result_dict,
                formatted_context=formatted,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Parabolic SAR calculation failed: {e}")
            return create_error_output(
                tool_name=self.schema.name,
                error=str(e),
                execution_time_ms=execution_time,
            )


# =============================================================================
# COMBINED ENHANCED TECHNICALS TOOL
# =============================================================================

class GetEnhancedTechnicalsTool(BaseTool):
    """
    Tool to calculate all enhanced technical indicators in one call.

    Combines: Ichimoku, Fibonacci, Williams %R, CCI, Parabolic SAR
    and provides consensus signal.
    """

    def __init__(self, fmp_api_key: Optional[str] = None):
        super().__init__()
        self.fmp_api_key = fmp_api_key
        self.calculator = EnhancedTechnicalCalculator()

        self.schema = ToolSchema(
            name="getEnhancedTechnicals",
            category="technical_enhanced",
            description="Calculate all enhanced technical indicators (Ichimoku, Fibonacci, Williams %R, CCI, Parabolic SAR) with consensus signal",
            capabilities=[
                "✅ Complete Ichimoku Cloud analysis",
                "✅ Fibonacci retracement/extension levels",
                "✅ Williams %R oscillator",
                "✅ CCI (Commodity Channel Index)",
                "✅ Parabolic SAR trailing stops",
                "✅ Consensus signal from all indicators",
                "✅ Signal agreement percentage",
            ],
            limitations=[
                "❌ Higher latency (calculates 5 indicators)",
                "❌ Returns larger payload",
                "❌ May have conflicting signals",
            ],
            usage_hints=[
                "Use for comprehensive technical analysis",
                "Check consensus signal and agreement %",
                "Higher agreement = stronger signal confidence",
                "Use individual tools if only one indicator needed",
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol",
                    required=True,
                    pattern=r"^[A-Z]{1,10}$",
                ),
                ToolParameter(
                    name="include_ichimoku",
                    type="boolean",
                    description="Include Ichimoku Cloud (default: true)",
                    required=False,
                    default=True,
                ),
                ToolParameter(
                    name="include_fibonacci",
                    type="boolean",
                    description="Include Fibonacci levels (default: true)",
                    required=False,
                    default=True,
                ),
                ToolParameter(
                    name="include_williams_r",
                    type="boolean",
                    description="Include Williams %R (default: true)",
                    required=False,
                    default=True,
                ),
                ToolParameter(
                    name="include_cci",
                    type="boolean",
                    description="Include CCI (default: true)",
                    required=False,
                    default=True,
                ),
                ToolParameter(
                    name="include_parabolic_sar",
                    type="boolean",
                    description="Include Parabolic SAR (default: true)",
                    required=False,
                    default=True,
                ),
            ],
            returns={
                "ticker": "string",
                "ichimoku": "object (if included)",
                "fibonacci": "object (if included)",
                "williams_r": "object (if included)",
                "cci": "object (if included)",
                "parabolic_sar": "object (if included)",
                "consensus_signal": "string",
                "signal_agreement_percent": "number",
                "summary": "string",
            },
            typical_execution_time_ms=3000,
        )

    async def execute(self, **params) -> ToolOutput:
        """Execute all enhanced technical calculations."""
        import time
        start_time = time.time()

        try:
            validated = self.validate_input(params)
            symbol = validated["symbol"]

            data = await fetch_ohlc_data(symbol, days=100, fmp_api_key=self.fmp_api_key)

            result = self.calculator.calculate_all(
                data,
                include_ichimoku=validated.get("include_ichimoku", True),
                include_fibonacci=validated.get("include_fibonacci", True),
                include_williams_r=validated.get("include_williams_r", True),
                include_cci=validated.get("include_cci", True),
                include_parabolic_sar=validated.get("include_parabolic_sar", True),
            )

            result_dict = result.to_dict()

            formatted = f"""
**Enhanced Technical Analysis for {symbol}**

CONSENSUS: {result.consensus_signal.value.upper()} ({result.signal_agreement_percent:.0f}% agreement)

{result.summary}
"""

            execution_time = int((time.time() - start_time) * 1000)

            return create_success_output(
                tool_name=self.schema.name,
                data=result_dict,
                formatted_context=formatted,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Enhanced technicals calculation failed: {e}")
            return create_error_output(
                tool_name=self.schema.name,
                error=str(e),
                execution_time_ms=execution_time,
            )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "GetIchimokuCloudTool",
    "GetFibonacciLevelsTool",
    "GetWilliamsRTool",
    "GetCCITool",
    "GetParabolicSARTool",
    "GetEnhancedTechnicalsTool",
]
