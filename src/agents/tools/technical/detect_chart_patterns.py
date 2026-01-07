import json
import httpx
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    create_success_output,
    create_error_output
)
from src.helpers.redis_cache import get_redis_client_llm


class DetectChartPatternsTool(BaseTool):
    """
    Candlestick & Chart Pattern Detection Tool

    Self-contained implementation with 7 essential candlestick patterns:
    1. Doji - Indecision
    2. Hammer - Bullish reversal
    3. Shooting Star - Bearish reversal
    4. Bullish Engulfing - Bullish reversal
    5. Bearish Engulfing - Bearish reversal
    6. Morning Star - Bullish reversal (3-candle)
    7. Evening Star - Bearish reversal (3-candle)

    Features:
    - Self-contained calculations (no TA-Lib dependency)
    - Redis caching (5-min TTL)
    - Formatted context with trading insights

    Usage:
        tool = DetectChartPatternsTool(api_key="your_fmp_key")
        result = await tool.safe_execute(symbol="AAPL", lookback_days=30)
    """

    FMP_BASE_URL = "https://financialmodelingprep.com/api"
    CACHE_TTL = 300  # 5 minutes

    def __init__(self, api_key: Optional[str] = None):
        """Initialize tool"""
        super().__init__()

        # Get API key from env if not provided
        if api_key is None:
            import os
            api_key = os.environ.get("FMP_API_KEY")

        if not api_key:
            raise ValueError("FMP_API_KEY not provided and not found in environment")

        self.api_key = api_key
        self.logger = logging.getLogger(__name__)

        # Define schema
        self.schema = ToolSchema(
            name="detectChartPatterns",
            category="technical",
            description=(
                "Detect candlestick patterns in stock price data (Doji, Hammer, Shooting Star, "
                "Engulfing, Morning/Evening Star). Returns identified patterns with confidence scores. "
                "Use when user asks about chart patterns, candlestick patterns, or reversal signals."
            ),
            capabilities=[
                "✅ Single-candle patterns (Doji, Hammer, Shooting Star)",
                "✅ Double-candle patterns (Bullish/Bearish Engulfing)",
                "✅ Triple-candle patterns (Morning Star, Evening Star)",
                "✅ Pattern confidence scoring (0-100%)",
                "✅ Bullish vs Bearish classification",
                "✅ Trading signal recommendations"
            ],
            limitations=[
                "❌ Requires minimum 30 days of data",
                "❌ Pattern recognition is probabilistic (not 100% accurate)",
                "❌ One symbol at a time",
                "❌ Daily timeframe only"
            ],
            usage_hints=[
                # English
                "User asks: 'Chart patterns for Apple' → USE THIS with symbol=AAPL",
                "User asks: 'Candlestick patterns in TSLA' → USE THIS with symbol=TSLA",
                "User asks: 'Is there a hammer in Microsoft?' → USE THIS with symbol=MSFT",
                "User asks: 'Show me NVDA reversal patterns' → USE THIS with symbol=NVDA",

                # Vietnamese
                "User asks: 'Mô hình nến của Apple' → USE THIS with symbol=AAPL",
                "User asks: 'Pattern nến Tesla' → USE THIS with symbol=TSLA",
                "User asks: 'NVDA có doji không?' → USE THIS with symbol=NVDA",

                # When NOT to use
                "User asks for technical INDICATORS → DO NOT USE (use getTechnicalIndicators)",
                "User asks for S/R levels → DO NOT USE (use getSupportResistance)"
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol",
                    required=True,
                    pattern="^[A-Z]{1,7}$"
                ),
                ToolParameter(
                    name="lookback_days",
                    type="integer",
                    description="Number of recent days to analyze (default: 30)",
                    required=False,
                    default=30
                )
            ],
            returns={
                "symbol": "string",
                "patterns_detected": "array - List of detected patterns with details",
                "pattern_count": "number",
                "bullish_patterns": "array",
                "bearish_patterns": "array",
                "recent_signal": "object - Most recent actionable pattern",
                "timestamp": "string"
            },
            typical_execution_time_ms=1500,
            requires_symbol=True
        )

    async def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get cached data from Redis"""
        try:
            redis_client = await get_redis_client_llm()
            if redis_client:
                cached = await redis_client.get(cache_key)
                if cached:
                    if isinstance(cached, bytes):
                        cached = cached.decode('utf-8')
                    return json.loads(cached)
        except Exception as e:
            self.logger.warning(f"[CACHE] Read error: {e}")
        return None

    async def _set_cached_data(self, cache_key: str, data: Any, ttl: int) -> None:
        """Set data in Redis cache"""
        try:
            redis_client = await get_redis_client_llm()
            if redis_client:
                await redis_client.set(cache_key, json.dumps(data), ex=ttl)
        except Exception as e:
            self.logger.warning(f"[CACHE] Write error: {e}")

    async def execute(
        self,
        symbol: str,
        lookback_days: int = 30,
        **kwargs
    ) -> ToolOutput:
        """
        Execute candlestick pattern detection

        Args:
            symbol: Stock symbol
            lookback_days: Recent days to analyze (default: 30)

        Returns:
            ToolOutput with detected patterns
        """
        symbol_upper = symbol.upper()
        start_time = datetime.now()

        # Validate lookback_days
        lookback_days = max(10, min(90, lookback_days))

        self.logger.info(
            f"[detectChartPatterns] {symbol_upper} | lookback={lookback_days} days"
        )

        try:
            # Check cache
            cache_key = f"stock_patterns:{symbol_upper}:{lookback_days}"
            cached_result = await self._get_cached_data(cache_key)

            if cached_result:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                cached_result["from_cache"] = True
                return create_success_output(
                    tool_name=self.schema.name,
                    data=cached_result,
                    formatted_context=self._build_formatted_context(cached_result),
                    metadata={
                        "source": "Redis Cache",
                        "execution_time_ms": int(execution_time),
                        "cache_hit": True
                    }
                )

            # Fetch historical data
            historical_data = await self._fetch_historical_data(
                symbol_upper,
                lookback_days + 10  # Extra buffer for pattern detection
            )

            if not historical_data or len(historical_data) < 10:
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"Insufficient historical data for {symbol_upper}"
                )

            # Detect patterns
            pattern_data = self._detect_all_patterns(historical_data, symbol_upper)

            # Cache the result
            await self._set_cached_data(cache_key, pattern_data, self.CACHE_TTL)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return create_success_output(
                tool_name=self.schema.name,
                data=pattern_data,
                formatted_context=self._build_formatted_context(pattern_data),
                metadata={
                    "source": "FMP API + Self-calculated",
                    "symbol_queried": symbol_upper,
                    "lookback_days": lookback_days,
                    "data_points": len(historical_data),
                    "execution_time_ms": int(execution_time),
                    "cache_hit": False,
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            self.logger.error(
                f"[detectChartPatterns] Error for {symbol_upper}: {e}",
                exc_info=True
            )
            return create_error_output(
                tool_name=self.schema.name,
                error=f"Pattern detection failed: {str(e)}"
            )

    async def _fetch_historical_data(
        self,
        symbol: str,
        lookback_days: int
    ) -> Optional[List[Dict]]:
        """Fetch historical OHLC data from FMP"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 30)

        url = f"{self.FMP_BASE_URL}/v3/historical-price-full/{symbol}"

        params = {
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "apikey": self.api_key
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()

                data = response.json()

                if isinstance(data, dict) and "historical" in data:
                    historical = data["historical"]
                    # Reverse to get chronological order (oldest first)
                    return list(reversed(historical))

                return None

        except Exception as e:
            self.logger.error(f"FMP request error: {e}")
            return None

    def _detect_all_patterns(
        self,
        historical_data: List[Dict],
        symbol: str
    ) -> Dict[str, Any]:
        """
        Detect all candlestick patterns in historical data.

        Patterns detected:
        1. Doji - body/range < 0.1
        2. Hammer - small body at top, lower shadow >= 2x body
        3. Shooting Star - small body at bottom, upper shadow >= 2x body
        4. Bullish Engulfing - prev bearish, current bullish engulfs
        5. Bearish Engulfing - prev bullish, current bearish engulfs
        6. Morning Star - 3-candle bullish reversal
        7. Evening Star - 3-candle bearish reversal
        """
        patterns_detected = []
        current_price = float(historical_data[-1]['close'])
        latest_date = historical_data[-1].get("date", "N/A")

        # Iterate through candles (start from index 2 for 3-candle patterns)
        for i in range(2, len(historical_data)):
            candle = historical_data[i]
            prev_candle = historical_data[i - 1]
            prev2_candle = historical_data[i - 2]

            date = candle.get("date", f"Day {i}")

            # Extract OHLC
            o = float(candle['open'])
            h = float(candle['high'])
            l = float(candle['low'])
            c = float(candle['close'])

            prev_o = float(prev_candle['open'])
            prev_h = float(prev_candle['high'])
            prev_l = float(prev_candle['low'])
            prev_c = float(prev_candle['close'])

            prev2_o = float(prev2_candle['open'])
            prev2_c = float(prev2_candle['close'])

            # Calculate body and shadows
            body = abs(c - o)
            total_range = h - l if h > l else 0.0001  # Avoid division by zero
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l

            is_bullish = c > o
            is_bearish = c < o

            prev_body = abs(prev_c - prev_o)
            prev_is_bullish = prev_c > prev_o
            prev_is_bearish = prev_c < prev_o

            # === PATTERN DETECTION ===

            # 1. DOJI - Indecision pattern
            if body / total_range < 0.1:
                confidence = self._calculate_doji_confidence(body, total_range)
                patterns_detected.append({
                    "type": "DOJI",
                    "signal": "NEUTRAL",
                    "date": date,
                    "confidence": confidence,
                    "description": "Indecision candle - open and close nearly equal",
                    "trading_hint": "Wait for next candle confirmation before trading"
                })

            # 2. HAMMER - Bullish reversal
            if self._is_hammer(o, h, l, c, body, upper_shadow, lower_shadow):
                confidence = self._calculate_hammer_confidence(
                    body, lower_shadow, upper_shadow, total_range
                )
                patterns_detected.append({
                    "type": "HAMMER",
                    "signal": "BULLISH",
                    "date": date,
                    "confidence": confidence,
                    "description": "Bullish reversal - rejection of lower prices",
                    "trading_hint": "Potential long entry if confirmed by bullish candle"
                })

            # 3. SHOOTING STAR - Bearish reversal
            if self._is_shooting_star(o, h, l, c, body, upper_shadow, lower_shadow):
                confidence = self._calculate_shooting_star_confidence(
                    body, upper_shadow, lower_shadow, total_range
                )
                patterns_detected.append({
                    "type": "SHOOTING_STAR",
                    "signal": "BEARISH",
                    "date": date,
                    "confidence": confidence,
                    "description": "Bearish reversal - rejection of higher prices",
                    "trading_hint": "Potential short entry if confirmed by bearish candle"
                })

            # 4. BULLISH ENGULFING
            if self._is_bullish_engulfing(o, c, prev_o, prev_c, prev_is_bearish):
                confidence = self._calculate_engulfing_confidence(
                    body, prev_body, is_bullish
                )
                patterns_detected.append({
                    "type": "BULLISH_ENGULFING",
                    "signal": "BULLISH",
                    "date": date,
                    "confidence": confidence,
                    "description": "Strong bullish reversal - buyers overwhelm sellers",
                    "trading_hint": "Consider long entry with stop below engulfing low"
                })

            # 5. BEARISH ENGULFING
            if self._is_bearish_engulfing(o, c, prev_o, prev_c, prev_is_bullish):
                confidence = self._calculate_engulfing_confidence(
                    body, prev_body, is_bullish
                )
                patterns_detected.append({
                    "type": "BEARISH_ENGULFING",
                    "signal": "BEARISH",
                    "date": date,
                    "confidence": confidence,
                    "description": "Strong bearish reversal - sellers overwhelm buyers",
                    "trading_hint": "Consider short entry with stop above engulfing high"
                })

            # 6. MORNING STAR - 3-candle bullish reversal
            if self._is_morning_star(
                prev2_o, prev2_c,
                prev_o, prev_c, prev_body,
                o, c, is_bullish
            ):
                confidence = self._calculate_star_confidence(
                    prev2_o, prev2_c, prev_body, c, is_bullish
                )
                patterns_detected.append({
                    "type": "MORNING_STAR",
                    "signal": "BULLISH",
                    "date": date,
                    "confidence": confidence,
                    "description": "3-candle bullish reversal - trend exhaustion + reversal",
                    "trading_hint": "Strong buy signal, enter long with stop below star low"
                })

            # 7. EVENING STAR - 3-candle bearish reversal
            if self._is_evening_star(
                prev2_o, prev2_c,
                prev_o, prev_c, prev_body,
                o, c, is_bearish
            ):
                confidence = self._calculate_star_confidence(
                    prev2_o, prev2_c, prev_body, c, is_bearish
                )
                patterns_detected.append({
                    "type": "EVENING_STAR",
                    "signal": "BEARISH",
                    "date": date,
                    "confidence": confidence,
                    "description": "3-candle bearish reversal - trend exhaustion + reversal",
                    "trading_hint": "Strong sell signal, enter short with stop above star high"
                })

        # Separate bullish/bearish patterns
        bullish_patterns = [
            p for p in patterns_detected if p["signal"] == "BULLISH"
        ]
        bearish_patterns = [
            p for p in patterns_detected if p["signal"] == "BEARISH"
        ]
        neutral_patterns = [
            p for p in patterns_detected if p["signal"] == "NEUTRAL"
        ]

        # Find most recent high-confidence pattern
        recent_signal = None
        high_confidence_patterns = [
            p for p in patterns_detected if p["confidence"] >= 70
        ]
        if high_confidence_patterns:
            recent_signal = high_confidence_patterns[-1]

        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "latest_date": latest_date,
            "patterns_detected": patterns_detected[-20:],  # Last 20 patterns
            "pattern_count": len(patterns_detected),
            "bullish_patterns": bullish_patterns[-10:],
            "bearish_patterns": bearish_patterns[-10:],
            "neutral_patterns": neutral_patterns[-5:],
            "bullish_count": len(bullish_patterns),
            "bearish_count": len(bearish_patterns),
            "recent_signal": recent_signal,
            "pattern_summary": self._generate_pattern_summary(
                bullish_patterns, bearish_patterns, neutral_patterns
            ),
            "analysis_period": f"{len(historical_data)} days",
            "timestamp": datetime.now().isoformat()
        }

    # =========================================================================
    # Pattern Detection Helpers
    # =========================================================================

    def _is_hammer(
        self,
        o: float, h: float, l: float, c: float,
        body: float, upper_shadow: float, lower_shadow: float
    ) -> bool:
        """
        HAMMER criteria:
        - Small body at TOP of candle
        - Lower shadow >= 2x body size
        - Upper shadow is tiny (< 10% of total range)
        """
        total_range = h - l if h > l else 0.0001

        # Body should be small (< 30% of range)
        body_ratio = body / total_range
        if body_ratio > 0.3:
            return False

        # Lower shadow should be >= 2x body
        if body > 0 and lower_shadow < 2 * body:
            return False

        # Upper shadow should be small (< 10% of range)
        upper_shadow_ratio = upper_shadow / total_range
        if upper_shadow_ratio > 0.1:
            return False

        # Body should be at top of range
        body_position = (min(o, c) - l) / total_range
        if body_position < 0.6:
            return False

        return True

    def _is_shooting_star(
        self,
        o: float, h: float, l: float, c: float,
        body: float, upper_shadow: float, lower_shadow: float
    ) -> bool:
        """
        SHOOTING STAR criteria:
        - Small body at BOTTOM of candle
        - Upper shadow >= 2x body size
        - Lower shadow is tiny (< 10% of total range)
        """
        total_range = h - l if h > l else 0.0001

        # Body should be small (< 30% of range)
        body_ratio = body / total_range
        if body_ratio > 0.3:
            return False

        # Upper shadow should be >= 2x body
        if body > 0 and upper_shadow < 2 * body:
            return False

        # Lower shadow should be small (< 10% of range)
        lower_shadow_ratio = lower_shadow / total_range
        if lower_shadow_ratio > 0.1:
            return False

        # Body should be at bottom of range
        body_position = (min(o, c) - l) / total_range
        if body_position > 0.4:
            return False

        return True

    def _is_bullish_engulfing(
        self,
        o: float, c: float,
        prev_o: float, prev_c: float,
        prev_is_bearish: bool
    ) -> bool:
        """
        BULLISH ENGULFING criteria:
        - Previous candle is bearish (red)
        - Current candle is bullish (green)
        - Current body completely engulfs previous body
        """
        is_bullish = c > o

        if not prev_is_bearish or not is_bullish:
            return False

        # Current body must engulf previous body
        if o <= prev_c and c >= prev_o:
            return True

        return False

    def _is_bearish_engulfing(
        self,
        o: float, c: float,
        prev_o: float, prev_c: float,
        prev_is_bullish: bool
    ) -> bool:
        """
        BEARISH ENGULFING criteria:
        - Previous candle is bullish (green)
        - Current candle is bearish (red)
        - Current body completely engulfs previous body
        """
        is_bearish = c < o

        if not prev_is_bullish or not is_bearish:
            return False

        # Current body must engulf previous body
        if o >= prev_c and c <= prev_o:
            return True

        return False

    def _is_morning_star(
        self,
        prev2_o: float, prev2_c: float,
        prev_o: float, prev_c: float, prev_body: float,
        o: float, c: float, is_bullish: bool
    ) -> bool:
        """
        MORNING STAR criteria:
        1. First candle: Large bearish (prev2)
        2. Second candle: Small body star (prev) - gap down preferred
        3. Third candle: Large bullish (current) - closes above midpoint of first
        """
        # First candle must be bearish
        prev2_is_bearish = prev2_c < prev2_o
        if not prev2_is_bearish:
            return False

        # First candle should be large
        prev2_body = abs(prev2_c - prev2_o)
        if prev2_body < 0.001:  # Skip if too small
            return False

        # Second candle should be small (star)
        star_ratio = prev_body / prev2_body if prev2_body > 0 else 0
        if star_ratio > 0.5:  # Star body should be < 50% of first body
            return False

        # Third candle must be bullish
        if not is_bullish:
            return False

        # Third candle should close above midpoint of first candle
        first_midpoint = (prev2_o + prev2_c) / 2
        if c < first_midpoint:
            return False

        return True

    def _is_evening_star(
        self,
        prev2_o: float, prev2_c: float,
        prev_o: float, prev_c: float, prev_body: float,
        o: float, c: float, is_bearish: bool
    ) -> bool:
        """
        EVENING STAR criteria:
        1. First candle: Large bullish (prev2)
        2. Second candle: Small body star (prev) - gap up preferred
        3. Third candle: Large bearish (current) - closes below midpoint of first
        """
        # First candle must be bullish
        prev2_is_bullish = prev2_c > prev2_o
        if not prev2_is_bullish:
            return False

        # First candle should be large
        prev2_body = abs(prev2_c - prev2_o)
        if prev2_body < 0.001:
            return False

        # Second candle should be small (star)
        star_ratio = prev_body / prev2_body if prev2_body > 0 else 0
        if star_ratio > 0.5:
            return False

        # Third candle must be bearish
        if not is_bearish:
            return False

        # Third candle should close below midpoint of first candle
        first_midpoint = (prev2_o + prev2_c) / 2
        if c > first_midpoint:
            return False

        return True

    # =========================================================================
    # Confidence Calculations
    # =========================================================================

    def _calculate_doji_confidence(self, body: float, total_range: float) -> int:
        """Calculate Doji confidence based on body/range ratio"""
        if total_range == 0:
            return 50

        ratio = body / total_range

        if ratio < 0.02:
            return 95  # Perfect doji
        elif ratio < 0.05:
            return 85
        elif ratio < 0.08:
            return 75
        else:
            return 65

    def _calculate_hammer_confidence(
        self,
        body: float, lower_shadow: float, upper_shadow: float, total_range: float
    ) -> int:
        """Calculate Hammer confidence"""
        if body == 0:
            return 50

        # Key factor: lower_shadow / body ratio
        shadow_ratio = lower_shadow / body if body > 0 else 0

        if shadow_ratio >= 3:
            base = 90
        elif shadow_ratio >= 2.5:
            base = 80
        elif shadow_ratio >= 2:
            base = 70
        else:
            base = 60

        # Bonus for small upper shadow
        if total_range > 0:
            upper_ratio = upper_shadow / total_range
            if upper_ratio < 0.05:
                base += 5

        return min(95, base)

    def _calculate_shooting_star_confidence(
        self,
        body: float, upper_shadow: float, lower_shadow: float, total_range: float
    ) -> int:
        """Calculate Shooting Star confidence"""
        if body == 0:
            return 50

        shadow_ratio = upper_shadow / body if body > 0 else 0

        if shadow_ratio >= 3:
            base = 90
        elif shadow_ratio >= 2.5:
            base = 80
        elif shadow_ratio >= 2:
            base = 70
        else:
            base = 60

        # Bonus for small lower shadow
        if total_range > 0:
            lower_ratio = lower_shadow / total_range
            if lower_ratio < 0.05:
                base += 5

        return min(95, base)

    def _calculate_engulfing_confidence(
        self,
        body: float, prev_body: float, is_bullish: bool
    ) -> int:
        """Calculate Engulfing pattern confidence"""
        if prev_body == 0:
            return 60

        # Key factor: how much current body is larger than previous
        engulf_ratio = body / prev_body if prev_body > 0 else 0

        if engulf_ratio >= 2:
            return 90
        elif engulf_ratio >= 1.5:
            return 80
        elif engulf_ratio >= 1.2:
            return 70
        else:
            return 60

    def _calculate_star_confidence(
        self,
        prev2_o: float, prev2_c: float, prev_body: float,
        c: float, is_reversal_direction: bool
    ) -> int:
        """Calculate Morning/Evening Star confidence"""
        prev2_body = abs(prev2_c - prev2_o)

        if prev2_body == 0:
            return 60

        # How small is the star?
        star_ratio = prev_body / prev2_body if prev2_body > 0 else 1

        base = 70

        if star_ratio < 0.2:
            base = 90  # Very small star = strong signal
        elif star_ratio < 0.3:
            base = 80

        # Check if third candle closed convincingly
        midpoint = (prev2_o + prev2_c) / 2
        if is_reversal_direction:
            base += 5

        return min(95, base)

    def _generate_pattern_summary(
        self,
        bullish_patterns: List[Dict],
        bearish_patterns: List[Dict],
        neutral_patterns: List[Dict]
    ) -> str:
        """Generate a brief summary of pattern analysis"""
        total = len(bullish_patterns) + len(bearish_patterns) + len(neutral_patterns)

        if total == 0:
            return "No significant candlestick patterns detected in this period."

        bullish_pct = (len(bullish_patterns) / total) * 100 if total > 0 else 0
        bearish_pct = (len(bearish_patterns) / total) * 100 if total > 0 else 0

        if bullish_pct > 60:
            bias = "BULLISH BIAS - More bullish reversal patterns detected"
        elif bearish_pct > 60:
            bias = "BEARISH BIAS - More bearish reversal patterns detected"
        else:
            bias = "NEUTRAL - Mixed pattern signals"

        return (
            f"{total} patterns detected: {len(bullish_patterns)} bullish, "
            f"{len(bearish_patterns)} bearish, {len(neutral_patterns)} neutral. {bias}"
        )

    def _build_formatted_context(self, data: Dict[str, Any]) -> str:
        """Build formatted context for LLM interpretation"""
        symbol = data.get("symbol", "N/A")
        current_price = data.get("current_price", 0)
        pattern_count = data.get("pattern_count", 0)
        bullish_count = data.get("bullish_count", 0)
        bearish_count = data.get("bearish_count", 0)
        recent_signal = data.get("recent_signal")
        pattern_summary = data.get("pattern_summary", "")
        latest_date = data.get("latest_date", "N/A")

        lines = []
        lines.append(f"=== {symbol} CANDLESTICK PATTERN ANALYSIS ===")
        lines.append(f"Current Price: ${current_price:,.2f}")
        lines.append(f"Analysis Date: {latest_date}")
        lines.append("")

        # Summary
        lines.append("📊 PATTERN OVERVIEW:")
        lines.append(f"  Total Patterns: {pattern_count}")
        lines.append(f"  Bullish Patterns: {bullish_count}")
        lines.append(f"  Bearish Patterns: {bearish_count}")
        lines.append(f"  Summary: {pattern_summary}")
        lines.append("")

        # Recent Signal
        if recent_signal:
            lines.append("🎯 MOST RECENT HIGH-CONFIDENCE SIGNAL:")
            lines.append(f"  Pattern: {recent_signal['type']}")
            lines.append(f"  Signal: {recent_signal['signal']}")
            lines.append(f"  Date: {recent_signal['date']}")
            lines.append(f"  Confidence: {recent_signal['confidence']}%")
            lines.append(f"  Description: {recent_signal['description']}")
            lines.append(f"  Trading Hint: {recent_signal['trading_hint']}")
            lines.append("")
        else:
            lines.append("🎯 RECENT SIGNAL: No high-confidence patterns detected recently")
            lines.append("")

        # Recent Bullish Patterns
        bullish_patterns = data.get("bullish_patterns", [])
        if bullish_patterns:
            lines.append("🟢 RECENT BULLISH PATTERNS:")
            for p in bullish_patterns[-5:]:
                lines.append(
                    f"  [{p['date']}] {p['type']} ({p['confidence']}% confidence)"
                )
            lines.append("")

        # Recent Bearish Patterns
        bearish_patterns = data.get("bearish_patterns", [])
        if bearish_patterns:
            lines.append("🔴 RECENT BEARISH PATTERNS:")
            for p in bearish_patterns[-5:]:
                lines.append(
                    f"  [{p['date']}] {p['type']} ({p['confidence']}% confidence)"
                )
            lines.append("")

        # Trading Insights
        lines.append("💡 TRADING INSIGHTS:")
        insights = self._generate_pattern_insights(data)
        for insight in insights:
            lines.append(f"  • {insight}")
        lines.append("")

        # Disclaimer
        lines.append("⚠️ DISCLAIMER:")
        lines.append("  Candlestick patterns are probabilistic, not guaranteed.")
        lines.append("  Always use with other indicators and proper risk management.")

        return "\n".join(lines)

    def _generate_pattern_insights(self, data: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from pattern analysis"""
        insights = []

        bullish_count = data.get("bullish_count", 0)
        bearish_count = data.get("bearish_count", 0)
        recent_signal = data.get("recent_signal")
        bullish_patterns = data.get("bullish_patterns", [])
        bearish_patterns = data.get("bearish_patterns", [])

        # Overall bias
        if bullish_count > bearish_count * 2:
            insights.append(
                "Strong bullish pattern activity - market showing buying pressure"
            )
        elif bearish_count > bullish_count * 2:
            insights.append(
                "Strong bearish pattern activity - market showing selling pressure"
            )
        elif bullish_count > 0 and bearish_count > 0:
            insights.append(
                "Mixed pattern signals - wait for clearer direction"
            )

        # Recent high-confidence signal
        if recent_signal:
            if recent_signal["confidence"] >= 85:
                insights.append(
                    f"High-confidence {recent_signal['type']} pattern - "
                    f"strong {recent_signal['signal'].lower()} signal"
                )
            elif recent_signal["confidence"] >= 70:
                insights.append(
                    f"Moderate-confidence {recent_signal['type']} pattern - "
                    f"consider {recent_signal['signal'].lower()} setup"
                )

        # Look for pattern clusters
        if bullish_patterns:
            recent_bullish = [
                p for p in bullish_patterns
                if p.get("confidence", 0) >= 75
            ]
            if len(recent_bullish) >= 2:
                insights.append(
                    "Multiple bullish patterns detected - potential reversal zone"
                )

        if bearish_patterns:
            recent_bearish = [
                p for p in bearish_patterns
                if p.get("confidence", 0) >= 75
            ]
            if len(recent_bearish) >= 2:
                insights.append(
                    "Multiple bearish patterns detected - potential top forming"
                )

        # 3-candle patterns are stronger
        star_patterns = [
            p for p in (bullish_patterns + bearish_patterns)
            if "STAR" in p.get("type", "")
        ]
        if star_patterns:
            insights.append(
                "Morning/Evening Star patterns detected - strong reversal signals"
            )

        if not insights:
            insights.append(
                "No significant pattern clusters. Monitor for new signals."
            )

        return insights


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import os

    async def test_tool():
        """Standalone test for DetectChartPatternsTool"""

        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            print("❌ ERROR: FMP_API_KEY not found in environment")
            return

        print("=" * 80)
        print("TESTING [DetectChartPatternsTool]")
        print("=" * 80)

        tool = DetectChartPatternsTool(api_key=api_key)

        # Test 1: Valid symbol
        print("\n📊 Test 1: AAPL Candlestick Patterns (30 days)")
        print("-" * 40)
        result = await tool.safe_execute(symbol="AAPL", lookback_days=30)

        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_ms}ms")

        if result.is_success():
            print("✅ SUCCESS")
            print(f"\nCurrent Price: ${result.data['current_price']:,.2f}")
            print(f"Patterns Detected: {result.data['pattern_count']}")
            print(f"Bullish: {result.data['bullish_count']}")
            print(f"Bearish: {result.data['bearish_count']}")

            if result.data.get('recent_signal'):
                sig = result.data['recent_signal']
                print(f"\n🎯 Recent Signal: {sig['type']} ({sig['signal']})")
                print(f"   Confidence: {sig['confidence']}%")
                print(f"   Date: {sig['date']}")

            print("\n" + "-" * 40)
            print("FORMATTED CONTEXT:")
            print(result.formatted_context)
        else:
            print(f"❌ ERROR: {result.error}")

        print("\n" + "=" * 80)
        print("TESTING COMPLETE")
        print("=" * 80)

    asyncio.run(test_tool())
