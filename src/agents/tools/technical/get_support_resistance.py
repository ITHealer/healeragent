import json
import httpx
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from scipy.signal import argrelextrema

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    create_success_output,
    create_error_output
)
from src.helpers.redis_cache import get_redis_client_llm


class GetSupportResistanceTool(BaseTool):
    """
    Support/Resistance Level Calculator

    Methods:
    1. Standard Pivot Points (P0 - 90% traders use this)
    2. Fibonacci Retracement (P0 - Key S/R levels)
    3. Swing Highs/Lows (local extrema clustering)
    4. Key MA levels (20/50/200 SMA)

    Features:
    - Redis caching (5-min TTL)
    - Formatted context with insights
    - Clear interpretations for each level

    Usage:
        tool = GetSupportResistanceTool()
        result = await tool.safe_execute(symbol="AAPL", lookback_days=60)
    """

    FMP_BASE_URL = "https://financialmodelingprep.com/api"
    CACHE_TTL = 300  # 5 minutes
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize tool
        
        Args:
            api_key: FMP API key (fallback to env var if not provided)
        """
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
            name="getSupportResistance",
            category="technical",
            description=(
                "Identify key support and resistance levels based on historical price action. "
                "Returns major support/resistance zones with strength ratings. "
                "Use when user asks about support levels, resistance levels, or key price levels."
            ),
            capabilities=[
                "Standard Pivot Points (P, R1, R2, R3, S1, S2, S3)",
                "Fibonacci Retracement (23.6%, 38.2%, 50%, 61.8%, 78.6%)",
                "Swing Highs/Lows from price action",
                "Key Moving Averages (20/50/200 SMA)",
                "Level strength ratings and distance from price",
                "Trading range analysis with position %"
            ],
            limitations=[
                "❌ Requires minimum 6 months historical data",
                "❌ Levels based on past price action (not predictive)",
                "❌ Dynamic levels may change with new data",
                "❌ One symbol at a time"
            ],
            usage_hints=[
                # English
                "User asks: 'Apple support levels' → USE THIS with symbol=AAPL",
                "User asks: 'Where is TSLA resistance?' → USE THIS with symbol=TSLA",
                "User asks: 'Key levels for NVDA' → USE THIS with symbol=NVDA",
                "User asks: 'Microsoft support and resistance' → USE THIS with symbol=MSFT",
                
                # Vietnamese
                "User asks: 'Ngưỡng hỗ trợ của Apple' → USE THIS with symbol=AAPL",
                "User asks: 'Kháng cự Tesla ở đâu?' → USE THIS with symbol=TSLA",
                "User asks: 'Các mức giá quan trọng NVDA' → USE THIS with symbol=NVDA",
                
                # When NOT to use
                "User asks for current PRICE → DO NOT USE (use getStockPrice)",
                "User asks about price TARGETS → DO NOT USE (use getPriceTargets)"
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
                    description="Number of days to analyze for levels",
                    required=False,
                    default=180
                )
            ],
            returns={
                "symbol": "string",
                "current_price": "number",
                "support_levels": "array - List of support levels",
                "resistance_levels": "array - List of resistance levels",
                "nearest_support": "object",
                "nearest_resistance": "object",
                "trading_range": "object",
                "timestamp": "string"
            },
            typical_execution_time_ms=1300,
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

    async def execute(self, symbol: str, lookback_days: int = 60) -> ToolOutput:
        """
        Execute S/R calculation

        Args:
            symbol: Stock symbol
            lookback_days: Days of historical data to analyze

        Returns:
            ToolOutput with S/R levels including Pivot Points and Fibonacci
        """
        symbol_upper = symbol.upper()
        start_time = datetime.now()

        # Validate lookback_days
        lookback_days = max(30, min(252, lookback_days))

        self.logger.info(
            f"[getSupportResistance] {symbol_upper} | lookback={lookback_days} days"
        )

        try:
            # Check cache
            cache_key = f"stock_sr:{symbol_upper}:{lookback_days}"
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
                lookback_days
            )

            if not historical_data or len(historical_data) < 30:
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"Insufficient historical data for {symbol_upper}"
                )

            # Calculate S/R levels
            sr_data = self._calculate_support_resistance(
                historical_data,
                symbol_upper
            )

            # Cache the result
            await self._set_cached_data(cache_key, sr_data, self.CACHE_TTL)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return create_success_output(
                tool_name=self.schema.name,
                data=sr_data,
                formatted_context=self._build_formatted_context(sr_data),
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
                f"[getSupportResistance] Error for {symbol_upper}: {e}",
                exc_info=True
            )
            return create_error_output(
                tool_name=self.schema.name,
                error=f"Failed to calculate S/R: {str(e)}"
            )
    
    async def _fetch_historical_data(
        self,
        symbol: str,
        lookback_days: int
    ) -> Optional[List[Dict]]:
        """Fetch historical price data from FMP"""
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer
        
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
                
                # Extract historical data
                if isinstance(data, dict) and "historical" in data:
                    historical = data["historical"]
                    # Reverse to get chronological order (oldest first)
                    return list(reversed(historical))
                
                return None
                
        except Exception as e:
            self.logger.error(f"FMP request error: {e}")
            return None
    
    def _calculate_support_resistance(
        self,
        historical_data: List[Dict],
        symbol: str
    ) -> Dict[str, Any]:
        """
        Calculate support and resistance levels

        Methods:
        1. Standard Pivot Points (from yesterday's OHLC)
        2. Fibonacci Retracement (from swing high/low)
        3. Swing Highs/Lows (local extrema)
        4. Key Moving Averages
        """
        # Extract OHLC data
        closes = np.array([float(d['close']) for d in historical_data])
        highs = np.array([float(d['high']) for d in historical_data])
        lows = np.array([float(d['low']) for d in historical_data])

        current_price = closes[-1]
        latest_date = historical_data[-1].get("date", "N/A")

        # 1. Calculate Pivot Points (from yesterday's data)
        pivot_points = self._calculate_pivot_points(
            high=highs[-2],
            low=lows[-2],
            close=closes[-2]
        )

        # 2. Find Swing Highs/Lows for Fibonacci
        swing_highs = self._find_swing_highs(highs)
        swing_lows = self._find_swing_lows(lows)

        # Get recent swing high and low for Fibonacci
        recent_high = max(highs[-30:]) if len(highs) >= 30 else max(highs)
        recent_low = min(lows[-30:]) if len(lows) >= 30 else min(lows)

        # Determine trend direction
        if current_price > (recent_high + recent_low) / 2:
            trend_direction = "UP"
        else:
            trend_direction = "DOWN"

        # 3. Calculate Fibonacci Retracement
        fibonacci = self._calculate_fibonacci_retracement(
            swing_high=recent_high,
            swing_low=recent_low,
            trend_direction=trend_direction
        )

        # 4. Calculate Key Moving Averages
        ma_levels = self._calculate_ma_levels(closes)
        
        # 4. Aggregate and rank resistance levels
        all_resistances = []
        
        # Add pivot resistances
        all_resistances.extend([
            {"price": pivot_points['r1'], "type": "Pivot R1", "strength": 0.8},
            {"price": pivot_points['r2'], "type": "Pivot R2", "strength": 0.6},
            {"price": pivot_points['r3'], "type": "Pivot R3", "strength": 0.4}
        ])
        
        # Add swing highs as resistance
        for swing_high in swing_highs:
            if swing_high > current_price:
                all_resistances.append({
                    "price": swing_high,
                    "type": "Swing High",
                    "strength": 0.7
                })
        
        # Add MA resistances
        for ma_name, ma_value in ma_levels.items():
            if ma_value > current_price:
                all_resistances.append({
                    "price": ma_value,
                    "type": f"MA-{ma_name}",
                    "strength": 0.6
                })
        
        # Filter and cluster resistances
        resistance_levels = self._cluster_levels(
            all_resistances,
            current_price,
            direction="above"
        )
        
        # 5. Aggregate and rank support levels
        all_supports = []
        
        # Add pivot supports
        all_supports.extend([
            {"price": pivot_points['s1'], "type": "Pivot S1", "strength": 0.8},
            {"price": pivot_points['s2'], "type": "Pivot S2", "strength": 0.6},
            {"price": pivot_points['s3'], "type": "Pivot S3", "strength": 0.4}
        ])
        
        # Add swing lows as support
        for swing_low in swing_lows:
            if swing_low < current_price:
                all_supports.append({
                    "price": swing_low,
                    "type": "Swing Low",
                    "strength": 0.7
                })
        
        # Add MA supports
        for ma_name, ma_value in ma_levels.items():
            if ma_value < current_price:
                all_supports.append({
                    "price": ma_value,
                    "type": f"MA-{ma_name}",
                    "strength": 0.6
                })
        
        # Filter and cluster supports
        support_levels = self._cluster_levels(
            all_supports,
            current_price,
            direction="below"
        )
        
        # Find nearest levels
        nearest_resistance = resistance_levels[0] if resistance_levels else None
        nearest_support = support_levels[0] if support_levels else None
        
        trading_range = {
            "current_price": round(current_price, 2),
            "nearest_support": nearest_support["price"] if nearest_support else None,
            "nearest_resistance": nearest_resistance["price"] if nearest_resistance else None,
            "range_width": 0.0,
            "position_in_range_pct": 0.0
        }

        if nearest_support and nearest_resistance:
            support_price = nearest_support["price"]
            resistance_price = nearest_resistance["price"]
            
            # Calculate range width
            range_width = resistance_price - support_price
            trading_range["range_width"] = round(range_width, 2)
            
            # Calculate range percentage
            range_pct = (range_width / support_price) * 100
            trading_range["range_pct"] = round(range_pct, 2)
            
            # Calculate position in range (0% = at support, 100% = at resistance)
            if range_width > 0:
                position = ((current_price - support_price) / range_width) * 100
                trading_range["position_in_range_pct"] = round(position, 2)

        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "latest_date": latest_date,
            "pivot_points": pivot_points,
            "fibonacci_retracement": fibonacci,
            "resistance_levels": resistance_levels[:5],
            "support_levels": support_levels[:5],
            "nearest_resistance": nearest_resistance,
            "nearest_support": nearest_support,
            "trading_range": trading_range,
            "key_moving_averages": ma_levels,
            "analysis_period": f"{len(historical_data)} days",
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_pivot_points(
        self,
        high: float,
        low: float,
        close: float
    ) -> Dict[str, float]:
        """Calculate classic pivot points"""
        pivot = (high + low + close) / 3
        
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            "pivot": round(pivot, 2),
            "r1": round(r1, 2),
            "r2": round(r2, 2),
            "r3": round(r3, 2),
            "s1": round(s1, 2),
            "s2": round(s2, 2),
            "s3": round(s3, 2)
        }

    def _calculate_fibonacci_retracement(
        self,
        swing_high: float,
        swing_low: float,
        trend_direction: str = "UP"
    ) -> Dict[str, Any]:
        """
        Calculate Fibonacci Retracement levels.

        Key levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%

        For UPTREND: Retracement from high going down (looking for support)
        For DOWNTREND: Retracement from low going up (looking for resistance)

        Args:
            swing_high: Recent swing high price
            swing_low: Recent swing low price
            trend_direction: "UP" or "DOWN"

        Returns:
            Dict with Fibonacci levels and interpretation
        """
        diff = swing_high - swing_low

        # Standard Fibonacci ratios
        fib_ratios = {
            "0.0": 0.0,
            "23.6": 0.236,
            "38.2": 0.382,
            "50.0": 0.500,
            "61.8": 0.618,  # Golden ratio - most important
            "78.6": 0.786,
            "100.0": 1.0
        }

        if trend_direction == "UP":
            # In uptrend, we measure from high going down
            levels = {}
            for name, ratio in fib_ratios.items():
                levels[f"{name}%"] = round(swing_high - diff * ratio, 2)
            key_level = levels["61.8%"]
            interpretation = "In UPTREND: Watch 38.2%, 50%, 61.8% for pullback support"
        else:
            # In downtrend, we measure from low going up
            levels = {}
            for name, ratio in fib_ratios.items():
                levels[f"{name}%"] = round(swing_low + diff * ratio, 2)
            key_level = levels["38.2%"]
            interpretation = "In DOWNTREND: Watch 38.2%, 50%, 61.8% for bounce resistance"

        return {
            "levels": levels,
            "swing_high": round(swing_high, 2),
            "swing_low": round(swing_low, 2),
            "trend_direction": trend_direction,
            "key_level": key_level,
            "golden_ratio_level": levels["61.8%"],
            "interpretation": interpretation
        }

    def _find_swing_highs(self, highs: np.ndarray, order: int = 5) -> List[float]:
        """Find swing highs (local maxima)"""
        # Find local maxima
        maxima_indices = argrelextrema(highs, np.greater, order=order)[0]
        
        # Get unique swing highs
        swing_highs = [highs[i] for i in maxima_indices]
        
        return swing_highs[-10:]  # Last 10 swing highs
    
    def _find_swing_lows(self, lows: np.ndarray, order: int = 5) -> List[float]:
        """Find swing lows (local minima)"""
        # Find local minima
        minima_indices = argrelextrema(lows, np.less, order=order)[0]
        
        # Get unique swing lows
        swing_lows = [lows[i] for i in minima_indices]
        
        return swing_lows[-10:]  # Last 10 swing lows
    
    def _calculate_ma_levels(self, closes: np.ndarray) -> Dict[str, float]:
        """Calculate key moving average levels"""
        ma_levels = {}
        
        periods = {
            "20": 20,
            "50": 50,
            "200": 200
        }
        
        for name, period in periods.items():
            if len(closes) >= period:
                ma_value = np.mean(closes[-period:])
                ma_levels[name] = round(ma_value, 2)
        
        return ma_levels
    
    def _cluster_levels(
        self,
        levels: List[Dict],
        current_price: float,
        direction: str,
        tolerance_pct: float = 0.5
    ) -> List[Dict]:
        """
        Cluster nearby levels and rank by proximity

        Args:
            levels: List of level dicts
            current_price: Current stock price
            direction: 'above' or 'below'
            tolerance_pct: % tolerance for clustering
        """
        if not levels:
            return []

        # Filter by direction
        if direction == "above":
            filtered = [l for l in levels if l['price'] > current_price]
        else:
            filtered = [l for l in levels if l['price'] < current_price]

        if not filtered:
            return []

        # Sort by distance from current price
        filtered.sort(key=lambda x: abs(x['price'] - current_price))

        # Cluster nearby levels
        clustered = []
        tolerance = current_price * (tolerance_pct / 100)

        for level in filtered:
            # Check if similar level already exists
            is_duplicate = False
            for existing in clustered:
                if abs(level['price'] - existing['price']) <= tolerance:
                    # Merge into existing (keep stronger)
                    if level['strength'] > existing['strength']:
                        existing['price'] = level['price']
                        existing['type'] = level['type']
                        existing['strength'] = level['strength']
                    is_duplicate = True
                    break

            if not is_duplicate:
                clustered.append(level)

        # Add distance_pct field
        for level in clustered:
            distance_pct = abs(
                (level['price'] - current_price) / current_price * 100
            )
            level['distance_pct'] = round(distance_pct, 2)
            level['price'] = round(level['price'], 2)

        return clustered

    def _build_formatted_context(self, data: Dict[str, Any]) -> str:
        """
        Build formatted context for LLM interpretation.

        Provides clear, actionable insights for trading decisions.
        """
        symbol = data.get("symbol", "N/A")
        current_price = data.get("current_price", 0)
        pivot_points = data.get("pivot_points", {})
        fibonacci = data.get("fibonacci_retracement", {})
        trading_range = data.get("trading_range", {})
        resistance_levels = data.get("resistance_levels", [])
        support_levels = data.get("support_levels", [])
        nearest_resistance = data.get("nearest_resistance")
        nearest_support = data.get("nearest_support")
        ma_levels = data.get("key_moving_averages", {})

        lines = []
        lines.append(f"=== {symbol} SUPPORT/RESISTANCE ANALYSIS ===")
        lines.append(f"Current Price: ${current_price:,.2f}")
        lines.append("")

        # Position Analysis
        lines.append("📍 PRICE POSITION:")
        if trading_range.get("position_in_range_pct") is not None:
            position = trading_range["position_in_range_pct"]
            if position < 30:
                position_desc = "NEAR SUPPORT (potential bounce zone)"
            elif position > 70:
                position_desc = "NEAR RESISTANCE (potential reversal zone)"
            else:
                position_desc = "MID-RANGE (neutral zone)"
            lines.append(f"  Position in range: {position:.1f}% - {position_desc}")
        lines.append("")

        # Pivot Points
        lines.append("🎯 PIVOT POINTS (Yesterday's OHLC):")
        if pivot_points:
            lines.append(f"  R3: ${pivot_points.get('r3', 0):,.2f}")
            lines.append(f"  R2: ${pivot_points.get('r2', 0):,.2f}")
            lines.append(f"  R1: ${pivot_points.get('r1', 0):,.2f}")
            lines.append(f"  P:  ${pivot_points.get('pivot', 0):,.2f} (Central Pivot)")
            lines.append(f"  S1: ${pivot_points.get('s1', 0):,.2f}")
            lines.append(f"  S2: ${pivot_points.get('s2', 0):,.2f}")
            lines.append(f"  S3: ${pivot_points.get('s3', 0):,.2f}")
        lines.append("")

        # Fibonacci Retracement
        lines.append("📐 FIBONACCI RETRACEMENT:")
        if fibonacci:
            fib_levels = fibonacci.get("levels", {})
            trend = fibonacci.get("trend_direction", "N/A")
            lines.append(f"  Trend: {trend}")
            lines.append(f"  Swing Range: ${fibonacci.get('swing_low', 0):,.2f} - ${fibonacci.get('swing_high', 0):,.2f}")

            # Show key levels
            for key in ["23.6%", "38.2%", "50.0%", "61.8%", "78.6%"]:
                level_val = fib_levels.get(key, 0)
                golden_marker = " ⭐ (Golden Ratio)" if key == "61.8%" else ""
                lines.append(f"  {key}: ${level_val:,.2f}{golden_marker}")

            lines.append(f"  → {fibonacci.get('interpretation', '')}")
        lines.append("")

        # Key Resistance Levels
        lines.append("🔴 RESISTANCE LEVELS (Above Current Price):")
        if resistance_levels:
            for r in resistance_levels[:5]:
                lines.append(
                    f"  ${r['price']:,.2f} ({r['type']}) - {r['distance_pct']:.1f}% away"
                )
        else:
            lines.append("  No resistance levels identified")
        lines.append("")

        # Key Support Levels
        lines.append("🟢 SUPPORT LEVELS (Below Current Price):")
        if support_levels:
            for s in support_levels[:5]:
                lines.append(
                    f"  ${s['price']:,.2f} ({s['type']}) - {s['distance_pct']:.1f}% away"
                )
        else:
            lines.append("  No support levels identified")
        lines.append("")

        # Moving Average Levels
        lines.append("📊 KEY MOVING AVERAGES:")
        if ma_levels:
            for period, value in sorted(ma_levels.items(), key=lambda x: int(x[0])):
                ma_position = "above" if value > current_price else "below"
                lines.append(f"  SMA-{period}: ${value:,.2f} ({ma_position} price)")
        lines.append("")

        # Trading Insights
        lines.append("💡 TRADING INSIGHTS:")
        insights = self._generate_sr_insights(data)
        for insight in insights:
            lines.append(f"  • {insight}")
        lines.append("")

        # Disclaimer
        lines.append("⚠️ DISCLAIMER:")
        lines.append("  S/R levels are based on historical price action.")
        lines.append("  Levels may be broken. Always use stop-loss and proper risk management.")

        return "\n".join(lines)

    def _generate_sr_insights(self, data: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from S/R analysis"""
        insights = []

        current_price = data.get("current_price", 0)
        trading_range = data.get("trading_range", {})
        nearest_support = data.get("nearest_support")
        nearest_resistance = data.get("nearest_resistance")
        pivot_points = data.get("pivot_points", {})
        fibonacci = data.get("fibonacci_retracement", {})
        ma_levels = data.get("key_moving_averages", {})

        # Position in range insight
        position = trading_range.get("position_in_range_pct", 50)
        if position < 20:
            insights.append(
                "Price is very close to support - potential bounce opportunity. "
                "Watch for bullish reversal patterns."
            )
        elif position < 40:
            insights.append(
                "Price in lower half of range - leaning towards support. "
                "Good entry zone for longs if support holds."
            )
        elif position > 80:
            insights.append(
                "Price near resistance - caution for longs. "
                "Consider taking profits or wait for breakout confirmation."
            )
        elif position > 60:
            insights.append(
                "Price in upper half of range - approaching resistance. "
                "Potential short opportunity if resistance holds."
            )
        else:
            insights.append(
                "Price in middle of range - wait for clearer direction. "
                "Watch for break of support or resistance."
            )

        # Pivot level insight
        pivot = pivot_points.get("pivot", 0)
        if pivot > 0:
            if current_price > pivot:
                insights.append(
                    f"Trading above central pivot (${pivot:,.2f}) - short-term bullish bias."
                )
            else:
                insights.append(
                    f"Trading below central pivot (${pivot:,.2f}) - short-term bearish bias."
                )

        # Fibonacci insight
        golden_ratio = fibonacci.get("golden_ratio_level", 0)
        if golden_ratio > 0:
            distance_pct = abs((current_price - golden_ratio) / current_price * 100)
            if distance_pct < 2:
                insights.append(
                    f"Price near 61.8% Fibonacci level (${golden_ratio:,.2f}) - "
                    "key level for potential reversal."
                )

        # MA alignment insight
        if ma_levels:
            above_mas = sum(1 for v in ma_levels.values() if current_price > v)
            total_mas = len(ma_levels)
            if above_mas == total_mas:
                insights.append(
                    "Price above all key MAs (20/50/200) - strong bullish structure."
                )
            elif above_mas == 0:
                insights.append(
                    "Price below all key MAs (20/50/200) - bearish structure."
                )
            else:
                insights.append(
                    f"Price above {above_mas}/{total_mas} key MAs - mixed signals."
                )

        # Risk/Reward insight
        if nearest_support and nearest_resistance:
            support_dist = abs(current_price - nearest_support["price"])
            resistance_dist = abs(nearest_resistance["price"] - current_price)

            if resistance_dist > 0:
                rr_ratio = support_dist / resistance_dist
                if rr_ratio < 0.5:
                    insights.append(
                        f"Risk/Reward favorable for longs (R:R = 1:{1/rr_ratio:.1f})."
                    )
                elif rr_ratio > 2:
                    insights.append(
                        f"Risk/Reward unfavorable for longs (R:R = 1:{1/rr_ratio:.1f}). "
                        "Consider waiting for better entry."
                    )

        return insights


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import os
    
    async def test_tool():
        """Standalone test for GetSupportResistanceTool"""
        
        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            print("❌ ERROR: FMP_API_KEY not found in environment")
            return
        
        print("=" * 80)
        print("TESTING [GetSupportResistanceTool]")
        print("=" * 80)
        
        tool = GetSupportResistanceTool(api_key=api_key)
        
        # Test 1: Valid symbol
        print("\n📊 Test 1: AAPL S/R levels (60 days)")
        print("-" * 40)
        result = await tool.safe_execute(symbol="AAPL", lookback_days=60)
        
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_ms}ms")
        
        if result.is_success():
            print("✅ SUCCESS")
            print(f"\nCurrent Price: ${result.data['current_price']:,.2f}")
            
            print(f"\n🔴 Resistance Levels:")
            for r in result.data['resistance_levels']:
                print(f"  ${r['price']:,.2f} ({r['type']}) - {r['distance_pct']:.1f}% away")
            
            print(f"\n🟢 Support Levels:")
            for s in result.data['support_levels']:
                print(f"  ${s['price']:,.2f} ({s['type']}) - {s['distance_pct']:.1f}% away")
            
            print(f"\n📍 Nearest Resistance: ${result.data['nearest_resistance']['price']:,.2f}")
            print(f"📍 Nearest Support: ${result.data['nearest_support']['price']:,.2f}")
            
            print(f"\n🎯 Pivot Points:")
            pp = result.data['pivot_points']
            print(f"  R3: ${pp['r3']:,.2f}")
            print(f"  R2: ${pp['r2']:,.2f}")
            print(f"  R1: ${pp['r1']:,.2f}")
            print(f"  P:  ${pp['pivot']:,.2f}")
            print(f"  S1: ${pp['s1']:,.2f}")
            print(f"  S2: ${pp['s2']:,.2f}")
            print(f"  S3: ${pp['s3']:,.2f}")
        else:
            print(f"❌ ERROR: {result.error}")
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETE")
        print("=" * 80)
    
    asyncio.run(test_tool())