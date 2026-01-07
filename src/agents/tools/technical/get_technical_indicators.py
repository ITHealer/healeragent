import json
import httpx
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    create_success_output,
    create_error_output
)
from src.helpers.redis_cache import get_redis_client_llm


class GetTechnicalIndicatorsTool(BaseTool):
    """
    Stock Technical Indicators Tool

    Calculates core technical indicators for stock analysis:
    - RSI (14-period): Momentum oscillator for overbought/oversold
    - MACD (12,26,9): Trend momentum and signal crossovers
    - SMA (20,50,200): Trend direction and support/resistance
    - EMA (12,26): Faster trend detection
    - Bollinger Bands (20,2): Volatility and price channels
    - VWAP: Volume Weighted Average Price (institutional benchmark)
    - Stochastic (14,3,3): Momentum oscillator
    - OBV: On-Balance Volume for trend confirmation
    - ADX (14): Trend strength measurement

    Features:
    - Redis caching for historical data (5-min TTL for intraday freshness)
    - Clear timeframe specifications (daily data)
    - Accurate calculations with proper lookback periods
    """

    FMP_BASE_URL = "https://financialmodelingprep.com/api"
    FMP_STABLE_URL = "https://financialmodelingprep.com/stable"
    DEFAULT_INDICATORS = ["RSI", "MACD", "SMA", "EMA", "BB", "VWAP", "STOCH", "OBV", "ADX", "ATR", "SUPERTREND"]

    # Cache TTL settings
    CACHE_TTL_HISTORICAL = 300   # 5 minutes for historical OHLCV data
    CACHE_TTL_INDICATORS = 300   # 5 minutes for calculated indicators
    CACHE_TTL_FMP_INDICATOR = 600  # 10 minutes for FMP API indicators

    # Indicator aliases mapping
    INDICATOR_ALIASES = {
        "BOLLINGER": "BB",
        "BOLLINGER_BANDS": "BB",
        "BOLLINGERBANDS": "BB",
        "BOLL": "BB",
        "RSI_14": "RSI",
        "RSI14": "RSI",
        "SIMPLE_MOVING_AVERAGE": "SMA",
        "EXPONENTIAL_MOVING_AVERAGE": "EMA",
        "STOCHASTIC": "STOCH",
        "STOCHASTIC_OSCILLATOR": "STOCH",
        "SLOW_STOCHASTIC": "STOCH",
        "ON_BALANCE_VOLUME": "OBV",
        "VOLUME_WEIGHTED_AVERAGE_PRICE": "VWAP",
        "AVERAGE_DIRECTIONAL_INDEX": "ADX",
        "AVERAGE_TRUE_RANGE": "ATR",
        "ATR_14": "ATR",
        "SUPER_TREND": "SUPERTREND"
    }

    # Data source tracking
    INDICATOR_DATA_SOURCE = {
        "RSI": "self-calc (Wilder's smoothing)",
        "MACD": "self-calc",
        "SMA": "self-calc (requires 260+ days for SMA-200)",
        "EMA": "self-calc",
        "BB": "self-calc",
        "VWAP": "FMP intraday 1h + daily fallback",
        "STOCH": "self-calc",
        "OBV": "self-calc",
        "ADX": "FMP primary + self-calc fallback",
        "ATR": "self-calc",
        "SUPERTREND": "self-calc",
        "52W_RANGE": "self-calc from historical data"
    }

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()

        if api_key is None:
            import os
            api_key = os.environ.get("FMP_API_KEY")

        if not api_key:
            raise ValueError("FMP_API_KEY required")

        self.api_key = api_key
        self.logger = logging.getLogger(__name__)

        # Define tool schema
        self.schema = ToolSchema(
            name="getTechnicalIndicators",
            category="technical",
            description=(
                "Calculate technical indicators (RSI, MACD, SMA, EMA, Bollinger Bands) "
                "from daily stock price data. Returns indicator values with buy/sell signals."
            ),
            capabilities=[
                "RSI (14-period): Overbought (>70) / Oversold (<30) detection",
                "MACD (12,26,9): Trend momentum with signal line crossovers",
                "SMA (20,50,200): Short/medium/long-term trend direction",
                "EMA (12,26): Exponential moving averages for faster signals",
                "Bollinger Bands (20,2): Volatility channels and breakout detection",
                "VWAP: Volume Weighted Average Price (institutional benchmark)",
                "Stochastic (14,3,3): Momentum oscillator with %K and %D",
                "OBV: On-Balance Volume for volume-price trend confirmation",
                "ADX (14): Average Directional Index for trend strength (FMP + self-calc)",
                "ATR (14): Average True Range for volatility measurement",
                "Supertrend (10,3): Trend following indicator with buy/sell signals"
            ],
            limitations=[
                "Daily timeframe only (EOD data)",
                "Requires minimum 50 data points for accurate SMA/BB",
                "Requires 200 data points for SMA-200",
                "Data cached for 5 minutes"
            ],
            usage_hints=[
                "User asks: 'Apple technical analysis' → USE THIS with symbol=AAPL",
                "User asks: 'TSLA RSI and MACD' → USE THIS with symbol=TSLA",
                "User asks: 'Is NVDA overbought?' → USE THIS with symbol=NVDA"
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol (e.g., AAPL, TSLA, NVDA)",
                    required=True,
                    pattern="^[A-Z]{1,7}$"
                ),
                ToolParameter(
                    name="timeframe",
                    type="string",
                    description="Analysis period: 1M (30 days), 3M (90 days), 6M (180 days), 1Y (252 days)",
                    required=False,
                    default="3M",
                    allowed_values=["1M", "3M", "6M", "1Y"]
                ),
                ToolParameter(
                    name="indicators",
                    type="array",
                    description="Indicators to calculate: RSI, MACD, SMA, EMA, BB, VWAP, STOCH, OBV, ADX, ATR, SUPERTREND (default: all)",
                    required=False,
                    default=["RSI", "MACD", "SMA", "EMA", "BB", "VWAP", "STOCH", "OBV", "ADX", "ATR", "SUPERTREND"]
                )
            ],
            returns={
                "symbol": "string",
                "timeframe": "string",
                "current_price": "number",
                "indicators": "object",
                "signals": "array",
                "data_points": "number",
                "timestamp": "string"
            },
            typical_execution_time_ms=1200,
            requires_symbol=True
        )

    # ════════════════════════════════════════════════════════════════════════
    # REDIS CACHING METHODS
    # ════════════════════════════════════════════════════════════════════════

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
            self.logger.warning(f"[CACHE] Read error for {cache_key}: {e}")
        return None

    async def _set_cached_data(self, cache_key: str, data: Any, ttl: int) -> None:
        """Set data in Redis cache"""
        try:
            redis_client = await get_redis_client_llm()
            if redis_client:
                await redis_client.set(cache_key, json.dumps(data), ex=ttl)
                self.logger.debug(f"[CACHE SET] {cache_key} TTL={ttl}s")
        except Exception as e:
            self.logger.warning(f"[CACHE] Write error for {cache_key}: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # MAIN EXECUTE METHOD
    # ════════════════════════════════════════════════════════════════════════

    async def execute(
        self,
        symbol: str,
        indicators: Optional[List[str]] = None,
        timeframe: str = "3M",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate technical indicators for a stock symbol.

        Args:
            symbol: Stock ticker (e.g., AAPL, TSLA)
            indicators: List of indicators to calculate (RSI, MACD, SMA, EMA, BB)
            timeframe: Analysis period (1M, 3M, 6M, 1Y)

        Returns:
            Dict with calculated indicators and signals
        """
        start_time = datetime.now()
        symbol = symbol.upper()

        # Map timeframe to lookback days
        # IMPORTANT: Always fetch >= 260 days for SMA-200 calculation
        timeframe_map = {
            "1M": 260,   # Need 260 for SMA-200 even on 1M view
            "3M": 260,   # Need 260 for SMA-200
            "6M": 280,   # 180 days + buffer
            "1Y": 320    # 252 days + buffer
        }
        lookback_days = timeframe_map.get(timeframe, 260)

        try:
            # ════════════════════════════════════════════════════════════
            # STEP 1: Normalize indicators
            # ════════════════════════════════════════════════════════════
            if indicators is None or len(indicators) == 0:
                indicators = self.DEFAULT_INDICATORS.copy()
            else:
                normalized = []
                for ind in indicators:
                    ind_upper = ind.upper().strip()
                    normalized_name = self.INDICATOR_ALIASES.get(ind_upper, ind_upper)
                    if normalized_name not in normalized:
                        normalized.append(normalized_name)
                indicators = normalized

            self.logger.info(
                f"[getTechnicalIndicators] {symbol} | timeframe={timeframe} | "
                f"indicators={indicators}"
            )

            # ════════════════════════════════════════════════════════════
            # STEP 2: Check cache for calculated indicators
            # ════════════════════════════════════════════════════════════
            indicator_cache_key = f"stock_technicals:{symbol}:{timeframe}:{'-'.join(sorted(indicators))}"
            cached_result = await self._get_cached_data(indicator_cache_key)

            if cached_result:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                self.logger.info(f"[{symbol}] CACHE HIT for indicators")
                cached_result["from_cache"] = True
                return create_success_output(
                    tool_name="getTechnicalIndicators",
                    data=cached_result,
                    metadata={
                        "source": "Redis Cache",
                        "execution_time_ms": int(execution_time),
                        "cache_hit": True
                    }
                )

            # ════════════════════════════════════════════════════════════
            # STEP 3: Fetch historical data (with caching)
            # ════════════════════════════════════════════════════════════
            historical_data = await self._fetch_historical_data_cached(symbol, lookback_days)

            if not historical_data or len(historical_data) < 50:
                return create_error_output(
                    tool_name="getTechnicalIndicators",
                    error=f"Insufficient historical data for {symbol} (need 50+ days, got {len(historical_data) if historical_data else 0})",
                    metadata={"symbol": symbol}
                )

            # ════════════════════════════════════════════════════════════
            # STEP 4: Convert to DataFrame
            # ════════════════════════════════════════════════════════════
            df = pd.DataFrame(historical_data)
            df = df.rename(columns={
                "date": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume"
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            current_price = float(df["close"].iloc[-1])
            latest_date = df["timestamp"].iloc[-1]

            # ════════════════════════════════════════════════════════════
            # STEP 5: Calculate Indicators
            # ════════════════════════════════════════════════════════════
            result_indicators = {}
            indicators_calculated = []

            # --- RSI (14-period) ---
            if "RSI" in indicators:
                rsi_result = self._calculate_rsi(df, period=14)
                result_indicators["rsi"] = rsi_result
                indicators_calculated.append("RSI")

            # --- MACD (12, 26, 9) ---
            if "MACD" in indicators:
                macd_result = self._calculate_macd(df)
                result_indicators["macd"] = macd_result
                indicators_calculated.append("MACD")

            # --- SMA (20, 50, 200) ---
            if "SMA" in indicators:
                sma_result = self._calculate_sma(df, current_price)
                result_indicators["sma"] = sma_result
                indicators_calculated.append("SMA")

            # --- EMA (12, 26) ---
            if "EMA" in indicators:
                ema_result = self._calculate_ema(df, current_price)
                result_indicators["ema"] = ema_result
                indicators_calculated.append("EMA")

            # --- Bollinger Bands (20, 2) ---
            if "BB" in indicators:
                bb_result = self._calculate_bollinger_bands(df, current_price)
                result_indicators["bollinger_bands"] = bb_result
                indicators_calculated.append("BB")

            # --- VWAP (Volume Weighted Average Price) - Using intraday data ---
            if "VWAP" in indicators:
                # Try intraday VWAP first (accurate), fallback to daily
                vwap_result = await self._calculate_vwap_intraday(symbol, current_price)
                if vwap_result.get("value") is None:
                    # Fallback to daily typical price
                    vwap_result = self._calculate_vwap_daily_fallback(df, current_price)
                result_indicators["vwap"] = vwap_result
                indicators_calculated.append("VWAP")

            # --- Stochastic Oscillator (14, 3, 3) ---
            if "STOCH" in indicators:
                stoch_result = self._calculate_stochastic(df)
                result_indicators["stochastic"] = stoch_result
                indicators_calculated.append("STOCH")

            # --- OBV (On-Balance Volume) ---
            if "OBV" in indicators:
                obv_result = self._calculate_obv(df)
                result_indicators["obv"] = obv_result
                indicators_calculated.append("OBV")

            # --- ADX (Average Directional Index) - FMP primary + self-calc fallback ---
            if "ADX" in indicators:
                adx_result = await self._calculate_adx_with_fmp_fallback(symbol, df)
                result_indicators["adx"] = adx_result
                indicators_calculated.append("ADX")

            # --- ATR (Average True Range) ---
            if "ATR" in indicators:
                atr_result = self._calculate_atr(df)
                result_indicators["atr"] = atr_result
                indicators_calculated.append("ATR")

            # --- Supertrend (10, 3) ---
            if "SUPERTREND" in indicators:
                supertrend_result = self._calculate_supertrend(df, current_price)
                result_indicators["supertrend"] = supertrend_result
                indicators_calculated.append("SUPERTREND")

            # ════════════════════════════════════════════════════════════
            # STEP 5.5: Calculate 52-week range (always included)
            # ════════════════════════════════════════════════════════════
            range_52w_result = self._calculate_52_week_range(df, current_price)
            result_indicators["range_52w"] = range_52w_result

            # ════════════════════════════════════════════════════════════
            # STEP 6: Generate Signals
            # ════════════════════════════════════════════════════════════
            signals = self._generate_signals(result_indicators, current_price)

            # ════════════════════════════════════════════════════════════
            # STEP 7: Build Result
            # ════════════════════════════════════════════════════════════
            result_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": round(current_price, 2),
                "data_points": len(df),
                "latest_date": latest_date.strftime("%Y-%m-%d"),
                "indicators": result_indicators,
                "signals": signals,
                "indicators_calculated": indicators_calculated,
                "timestamp": datetime.now().isoformat()
            }

            # Cache the result
            await self._set_cached_data(
                indicator_cache_key,
                result_data,
                self.CACHE_TTL_INDICATORS
            )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(f"[{symbol}] SUCCESS ({int(execution_time)}ms)")

            return create_success_output(
                tool_name="getTechnicalIndicators",
                data=result_data,
                formatted_context=self._build_formatted_context(result_data),
                metadata={
                    "source": "FMP API + Self-calculated",
                    "execution_time_ms": int(execution_time),
                    "cache_hit": False
                }
            )

        except Exception as e:
            self.logger.error(f"[getTechnicalIndicators] Error: {e}", exc_info=True)
            return create_error_output(
                tool_name="getTechnicalIndicators",
                error=str(e)
            )

    # ════════════════════════════════════════════════════════════════════════
    # DATA FETCHING WITH CACHE
    # ════════════════════════════════════════════════════════════════════════

    async def _fetch_historical_data_cached(
        self,
        symbol: str,
        lookback_days: int
    ) -> List[Dict]:
        """
        Fetch historical OHLCV data from FMP with Redis caching.

        Cache Key: stock_ohlcv:{symbol}:{lookback_days}
        Cache TTL: 5 minutes (intraday data can change)
        """
        cache_key = f"stock_ohlcv:{symbol}:{lookback_days}"

        # Try cache first
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            self.logger.debug(f"[{symbol}] OHLCV cache hit ({len(cached_data)} candles)")
            return cached_data

        # Fetch from FMP API
        url = f"{self.FMP_BASE_URL}/v3/historical-price-full/{symbol}"
        params = {"apikey": self.api_key, "timeseries": lookback_days}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                historical = data.get("historical", [])

                if historical:
                    # Cache the data
                    await self._set_cached_data(
                        cache_key,
                        historical,
                        self.CACHE_TTL_HISTORICAL
                    )
                    self.logger.debug(f"[{symbol}] Fetched {len(historical)} candles from FMP")

                return historical

        except httpx.HTTPStatusError as e:
            self.logger.error(f"[{symbol}] FMP API error: {e.response.status_code}")
            return []
        except Exception as e:
            self.logger.error(f"[{symbol}] Fetch error: {e}")
            return []

    # ════════════════════════════════════════════════════════════════════════
    # INDICATOR CALCULATIONS
    # ════════════════════════════════════════════════════════════════════════

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """
        RSI (Relative Strength Index) - 14-period using WILDER'S SMOOTHING

        Formula:
            RSI = 100 - (100 / (1 + RS))
            RS = Average Gain / Average Loss (over 14 periods)

        IMPORTANT: Uses Wilder's smoothing (alpha=1/period) to match:
        - TradingView
        - Bloomberg
        - FMP API
        - Most professional platforms

        Interpretation:
            > 70: Overbought (potential reversal down)
            < 30: Oversold (potential reversal up)
            50: Neutral
        """
        close = df['close']
        delta = close.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        # WILDER'S SMOOTHING: alpha = 1/period
        # This matches TradingView, Bloomberg, FMP, and industry standard
        # Previously used: ewm(span=period) which gives different results
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        current_rsi = float(rsi.iloc[-1])

        # Determine signal
        if current_rsi > 70:
            signal = "overbought"
        elif current_rsi < 30:
            signal = "oversold"
        else:
            signal = "neutral"

        return {
            "value": round(current_rsi, 2),
            "period": period,
            "signal": signal,
            "calculation_method": "wilder_smoothing",
            "description": f"RSI({period}) = {current_rsi:.1f} ({signal})"
        }

    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        MACD (Moving Average Convergence Divergence)

        Formula:
            MACD Line = EMA(12) - EMA(26)
            Signal Line = EMA(9) of MACD Line
            Histogram = MACD Line - Signal Line

        Interpretation:
            Histogram > 0: Bullish momentum
            Histogram < 0: Bearish momentum
            Crossover (MACD crosses Signal): Buy/Sell signal
        """
        close = df['close']

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        current_macd = float(macd_line.iloc[-1])
        current_signal = float(signal_line.iloc[-1])
        current_histogram = float(histogram.iloc[-1])
        prev_histogram = float(histogram.iloc[-2]) if len(histogram) > 1 else 0

        # Determine signal
        if current_histogram > 0 and prev_histogram <= 0:
            signal = "bullish_crossover"
        elif current_histogram < 0 and prev_histogram >= 0:
            signal = "bearish_crossover"
        elif current_histogram > 0:
            signal = "bullish"
        else:
            signal = "bearish"

        return {
            "macd_line": round(current_macd, 4),
            "signal_line": round(current_signal, 4),
            "histogram": round(current_histogram, 4),
            "parameters": "(12, 26, 9)",
            "signal": signal,
            "description": f"MACD = {current_macd:.4f}, Signal = {current_signal:.4f}, Histogram = {current_histogram:.4f} ({signal})"
        }

    def _calculate_sma(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        SMA (Simple Moving Average) - 20, 50, 200 periods

        Formula:
            SMA(n) = Sum of last n closing prices / n

        Interpretation:
            Price > SMA: Bullish trend
            Price < SMA: Bearish trend
            SMA-20: Short-term trend
            SMA-50: Medium-term trend
            SMA-200: Long-term trend (Golden/Death Cross with SMA-50)
        """
        close = df['close']

        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]

        # SMA-200 may not be available if insufficient data
        sma_200 = None
        if len(close) >= 200:
            sma_200_val = close.rolling(200).mean().iloc[-1]
            if pd.notna(sma_200_val):
                sma_200 = float(sma_200_val)

        result = {
            "sma_20": round(float(sma_20), 2) if pd.notna(sma_20) else None,
            "sma_50": round(float(sma_50), 2) if pd.notna(sma_50) else None,
            "sma_200": round(sma_200, 2) if sma_200 else None,
            "price_vs_sma": {}
        }

        # Price position relative to SMAs
        if result["sma_20"]:
            pct_20 = ((current_price - result["sma_20"]) / result["sma_20"]) * 100
            result["price_vs_sma"]["vs_sma20"] = {
                "position": "above" if current_price > result["sma_20"] else "below",
                "distance_pct": round(pct_20, 2)
            }

        if result["sma_50"]:
            pct_50 = ((current_price - result["sma_50"]) / result["sma_50"]) * 100
            result["price_vs_sma"]["vs_sma50"] = {
                "position": "above" if current_price > result["sma_50"] else "below",
                "distance_pct": round(pct_50, 2)
            }

        if result["sma_200"]:
            pct_200 = ((current_price - result["sma_200"]) / result["sma_200"]) * 100
            result["price_vs_sma"]["vs_sma200"] = {
                "position": "above" if current_price > result["sma_200"] else "below",
                "distance_pct": round(pct_200, 2)
            }

        return result

    def _calculate_ema(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        EMA (Exponential Moving Average) - 12, 26 periods

        Formula:
            EMA = Price(t) * k + EMA(y) * (1 - k)
            k = 2 / (N + 1)

        More responsive to recent price changes than SMA.
        """
        close = df['close']

        ema_12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = close.ewm(span=26, adjust=False).mean().iloc[-1]

        result = {
            "ema_12": round(float(ema_12), 2),
            "ema_26": round(float(ema_26), 2),
            "price_vs_ema": {}
        }

        pct_12 = ((current_price - result["ema_12"]) / result["ema_12"]) * 100
        pct_26 = ((current_price - result["ema_26"]) / result["ema_26"]) * 100

        result["price_vs_ema"]["vs_ema12"] = {
            "position": "above" if current_price > result["ema_12"] else "below",
            "distance_pct": round(pct_12, 2)
        }
        result["price_vs_ema"]["vs_ema26"] = {
            "position": "above" if current_price > result["ema_26"] else "below",
            "distance_pct": round(pct_26, 2)
        }

        return result

    def _calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        current_price: float,
        period: int = 20,
        std_dev: int = 2
    ) -> Dict[str, Any]:
        """
        Bollinger Bands (20, 2)

        Formula:
            Middle Band = SMA(20)
            Upper Band = SMA(20) + 2 * StdDev(20)
            Lower Band = SMA(20) - 2 * StdDev(20)
            %B = (Price - Lower) / (Upper - Lower)
            Bandwidth = (Upper - Lower) / Middle * 100

        Interpretation:
            Price near Upper: Overbought
            Price near Lower: Oversold
            %B > 1: Above upper band
            %B < 0: Below lower band
            Narrow bandwidth: Low volatility (breakout coming)
        """
        close = df['close']

        sma = close.rolling(period).mean()
        std = close.rolling(period).std()

        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)

        current_upper = float(upper.iloc[-1])
        current_middle = float(sma.iloc[-1])
        current_lower = float(lower.iloc[-1])

        # %B calculation
        band_width = current_upper - current_lower
        percent_b = (current_price - current_lower) / band_width if band_width > 0 else 0.5

        # Bandwidth percentage
        bandwidth_pct = (band_width / current_middle) * 100 if current_middle > 0 else 0

        # Determine position
        if percent_b > 1:
            position = "above_upper"
            signal = "overbought"
        elif percent_b < 0:
            position = "below_lower"
            signal = "oversold"
        elif percent_b > 0.8:
            position = "near_upper"
            signal = "neutral_high"
        elif percent_b < 0.2:
            position = "near_lower"
            signal = "neutral_low"
        else:
            position = "middle"
            signal = "neutral"

        return {
            "upper": round(current_upper, 2),
            "middle": round(current_middle, 2),
            "lower": round(current_lower, 2),
            "parameters": f"({period}, {std_dev})",
            "percent_b": round(percent_b, 4),
            "bandwidth_pct": round(bandwidth_pct, 2),
            "position": position,
            "signal": signal,
            "description": f"BB: Upper={current_upper:.2f}, Middle={current_middle:.2f}, Lower={current_lower:.2f}, %B={percent_b:.2f}"
        }

    async def _fetch_intraday_data(self, symbol: str) -> Optional[List[Dict]]:
        """
        Fetch 1-hour intraday data from FMP for VWAP calculation.

        Endpoint: /stable/historical-chart/1hour
        Returns today's (or most recent trading day's) hourly bars.
        """
        url = f"{self.FMP_STABLE_URL}/historical-chart/1hour"
        params = {
            "symbol": symbol,
            "apikey": self.api_key
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        return data
        except Exception as e:
            self.logger.warning(f"[{symbol}] Intraday fetch error: {e}")

        return None

    async def _calculate_vwap_intraday(
        self,
        symbol: str,
        current_price: float
    ) -> Dict[str, Any]:
        """
        VWAP (Volume Weighted Average Price) using INTRADAY data

        Formula:
            VWAP = Σ(Typical Price × Volume) / Σ(Volume)
            Typical Price = (High + Low + Close) / 3

        IMPORTANT: Standard VWAP resets daily and uses intraday data.
        This method fetches 1-hour bars from FMP for accurate calculation.

        Reference: https://site.financialmodelingprep.com/how-to/how-to-use-vwap-with-hour-stock-chart-api

        Interpretation:
            Price > VWAP: Bullish (buyers in control, institutional buying)
            Price < VWAP: Bearish (sellers in control, institutional selling)
        """
        # Try to fetch intraday data
        intraday_data = await self._fetch_intraday_data(symbol)

        if not intraday_data:
            # Fallback: Return None to indicate VWAP unavailable
            return {
                "value": None,
                "price_vs_vwap": "N/A",
                "distance_pct": None,
                "signal": "unavailable",
                "data_source": "failed",
                "description": "VWAP unavailable - could not fetch intraday data"
            }

        # Get today's date (or most recent trading day)
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")

        # Filter to today's data only
        today_data = [d for d in intraday_data if d.get("date", "").startswith(today)]

        if not today_data:
            # Use most recent trading day if today has no data (weekend/holiday)
            dates = set(d.get("date", "")[:10] for d in intraday_data if d.get("date"))
            if dates:
                latest_date = max(dates)
                today_data = [d for d in intraday_data if d.get("date", "").startswith(latest_date)]
            else:
                today_data = intraday_data[:10]  # Use most recent 10 bars as fallback

        if not today_data:
            return {
                "value": None,
                "price_vs_vwap": "N/A",
                "distance_pct": None,
                "signal": "unavailable",
                "data_source": "no_data",
                "description": "VWAP unavailable - no recent trading data"
            }

        # Calculate VWAP from intraday bars
        total_tp_vol = 0.0
        total_vol = 0

        for bar in today_data:
            high = float(bar.get("high", 0))
            low = float(bar.get("low", 0))
            close = float(bar.get("close", 0))
            volume = int(bar.get("volume", 0))

            typical_price = (high + low + close) / 3
            total_tp_vol += typical_price * volume
            total_vol += volume

        if total_vol == 0:
            return {
                "value": None,
                "price_vs_vwap": "N/A",
                "distance_pct": None,
                "signal": "unavailable",
                "data_source": "zero_volume",
                "description": "VWAP unavailable - zero volume"
            }

        vwap = total_tp_vol / total_vol

        # Price vs VWAP
        distance_pct = ((current_price - vwap) / vwap) * 100 if vwap > 0 else 0
        position = "above" if current_price > vwap else "below"

        return {
            "value": round(vwap, 2),
            "price_vs_vwap": position,
            "distance_pct": round(distance_pct, 2),
            "signal": "bullish" if position == "above" else "bearish",
            "data_source": "FMP_intraday_1h",
            "bars_used": len(today_data),
            "description": f"VWAP = ${vwap:.2f}, Price {position} ({distance_pct:+.2f}%)"
        }

    def _calculate_vwap_daily_fallback(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> Dict[str, Any]:
        """
        VWAP fallback using daily data (less accurate).

        This is a FALLBACK only - for accurate VWAP use _calculate_vwap_intraday.
        Daily VWAP = Today's Typical Price (not cumulative).
        """
        # Use only the most recent day's data
        high = float(df['high'].iloc[-1])
        low = float(df['low'].iloc[-1])
        close = float(df['close'].iloc[-1])

        # Typical Price for today
        vwap = (high + low + close) / 3

        # Price vs VWAP
        distance_pct = ((current_price - vwap) / vwap) * 100 if vwap > 0 else 0
        position = "above" if current_price > vwap else "below"

        return {
            "value": round(vwap, 2),
            "price_vs_vwap": position,
            "distance_pct": round(distance_pct, 2),
            "signal": "bullish" if position == "above" else "bearish",
            "data_source": "daily_typical_price",
            "note": "Approximation - for accurate VWAP use intraday data",
            "description": f"VWAP (daily) = ${vwap:.2f}, Price {position} ({distance_pct:+.2f}%)"
        }

    def _calculate_52_week_range(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Calculate 52-week high/low and position within range.

        Formula:
            Position = (Current - Low) / (High - Low) × 100%

        Interpretation:
            0% = At 52-week low
            100% = At 52-week high
            50% = Exactly in middle
            >80% = Near yearly highs (extended)
            <20% = Near yearly lows (potential value)
        """
        # Need at least 252 days for 52-week range (or use all available)
        if len(df) >= 252:
            high_52w = float(df['high'].tail(252).max())
            low_52w = float(df['low'].tail(252).min())
        else:
            # Use all available data if less than 252 days
            high_52w = float(df['high'].max())
            low_52w = float(df['low'].min())

        # Calculate position in range
        range_52w = high_52w - low_52w
        if range_52w > 0:
            position_pct = ((current_price - low_52w) / range_52w) * 100
        else:
            position_pct = 50.0

        # Determine position description
        if position_pct >= 95:
            position_desc = "at_52w_high"
        elif position_pct >= 80:
            position_desc = "near_highs"
        elif position_pct <= 5:
            position_desc = "at_52w_low"
        elif position_pct <= 20:
            position_desc = "near_lows"
        elif position_pct >= 60:
            position_desc = "upper_half"
        elif position_pct <= 40:
            position_desc = "lower_half"
        else:
            position_desc = "middle"

        # Distance from high/low
        distance_from_high_pct = ((high_52w - current_price) / high_52w) * 100 if high_52w > 0 else 0
        distance_from_low_pct = ((current_price - low_52w) / low_52w) * 100 if low_52w > 0 else 0

        return {
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "range": round(range_52w, 2),
            "position_pct": round(position_pct, 1),
            "position_desc": position_desc,
            "distance_from_high_pct": round(distance_from_high_pct, 1),
            "distance_from_low_pct": round(distance_from_low_pct, 1),
            "description": f"52W Range: ${low_52w:.2f} - ${high_52w:.2f}, Position: {position_pct:.1f}% ({position_desc})"
        }

    def _calculate_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3
    ) -> Dict[str, Any]:
        """
        Stochastic Oscillator (Slow Stochastic)

        Formula:
            %K = 100 × (Close - Lowest Low) / (Highest High - Lowest Low)
            %D = 3-period SMA of %K

        For Slow Stochastic: %K is smoothed with 3-period SMA first.

        Interpretation:
            > 80: Overbought (potential reversal down)
            < 20: Oversold (potential reversal up)
            %K crosses above %D: Buy signal
            %K crosses below %D: Sell signal
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # Lowest low and highest high over k_period
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        # Guard against division by zero
        denominator = highest_high - lowest_low
        denominator = denominator.replace(0, np.nan)

        # Fast %K
        fast_k = 100 * (close - lowest_low) / denominator

        # Slow %K (smoothed)
        slow_k = fast_k.rolling(window=smooth_k).mean()

        # %D (signal line)
        slow_d = slow_k.rolling(window=d_period).mean()

        current_k = float(slow_k.iloc[-1])
        current_d = float(slow_d.iloc[-1])
        prev_k = float(slow_k.iloc[-2]) if len(slow_k) > 1 else current_k
        prev_d = float(slow_d.iloc[-2]) if len(slow_d) > 1 else current_d

        # Determine signal
        if current_k > 80:
            signal = "overbought"
        elif current_k < 20:
            signal = "oversold"
        elif current_k > current_d and prev_k <= prev_d:
            signal = "bullish_crossover"
        elif current_k < current_d and prev_k >= prev_d:
            signal = "bearish_crossover"
        else:
            signal = "neutral"

        return {
            "k": round(current_k, 2),
            "d": round(current_d, 2),
            "parameters": f"({k_period}, {d_period}, {smooth_k})",
            "signal": signal,
            "description": f"Stoch %K={current_k:.1f}, %D={current_d:.1f} ({signal})"
        }

    def _calculate_obv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        OBV (On-Balance Volume)

        Formula:
            If Close > Close(prev): OBV = OBV(prev) + Volume
            If Close < Close(prev): OBV = OBV(prev) - Volume
            If Close = Close(prev): OBV = OBV(prev)

        Interpretation:
            OBV rising with price: Confirms uptrend (accumulation)
            OBV falling with price: Confirms downtrend (distribution)
            OBV diverging from price: Potential trend reversal
        """
        close = df['close']
        volume = df['volume']

        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        current_obv = float(obv.iloc[-1])

        # Trend detection using 20-period SMA
        obv_sma = obv.rolling(window=20).mean()
        trend = "increasing" if current_obv > obv_sma.iloc[-1] else "decreasing"

        # Check OBV vs price trend (divergence detection)
        price_change = close.iloc[-1] - close.iloc[-20] if len(close) > 20 else 0
        obv_change = obv.iloc[-1] - obv.iloc[-20] if len(obv) > 20 else 0

        if price_change > 0 and obv_change < 0:
            divergence = "bearish_divergence"
        elif price_change < 0 and obv_change > 0:
            divergence = "bullish_divergence"
        else:
            divergence = "none"

        return {
            "value": int(current_obv),
            "trend": trend,
            "divergence": divergence,
            "signal": "bullish" if trend == "increasing" else "bearish",
            "description": f"OBV trend: {trend}, divergence: {divergence}"
        }

    async def _fetch_adx_from_fmp(self, symbol: str, period: int = 14) -> Optional[Dict]:
        """
        Fetch ADX from FMP API (primary source).

        Endpoint: /stable/technical-indicators/adx
        Cache TTL: 10 minutes
        """
        cache_key = f"fmp_adx:{symbol}:{period}"

        # Check cache
        cached = await self._get_cached_data(cache_key)
        if cached:
            self.logger.debug(f"[{symbol}] FMP ADX cache hit")
            return cached

        url = f"{self.FMP_STABLE_URL}/technical-indicators/adx"
        params = {
            "symbol": symbol,
            "periodLength": period,
            "timeframe": "daily",
            "apikey": self.api_key
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        result = data[0]  # Get latest
                        await self._set_cached_data(cache_key, result, self.CACHE_TTL_FMP_INDICATOR)
                        return result
        except Exception as e:
            self.logger.warning(f"[{symbol}] FMP ADX fetch error: {e}")

        return None

    async def _calculate_adx_with_fmp_fallback(
        self,
        symbol: str,
        df: pd.DataFrame,
        period: int = 14
    ) -> Dict[str, Any]:
        """
        ADX with FMP API primary and self-calc fallback.

        Strategy: Try FMP first, fallback to self-calc if API fails.
        """
        data_source = "self-calc"

        # Try FMP API first
        fmp_adx = await self._fetch_adx_from_fmp(symbol, period)

        if fmp_adx and "adx" in fmp_adx:
            # Use FMP value, but still calculate +DI/-DI ourselves
            current_adx = float(fmp_adx["adx"])
            data_source = "FMP API"

            # Calculate +DI/-DI for direction using Wilder's smoothing
            high = df['high']
            low = df['low']
            close = df['close']

            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # WILDER'S SMOOTHING: alpha = 1/period (NOT span=period!)
            # This matches industry standard (TradingView, Bloomberg, Polygon)
            atr = tr.ewm(alpha=1/period, adjust=False).mean()
            plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
            minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)

            current_plus_di = float(plus_di.iloc[-1])
            current_minus_di = float(minus_di.iloc[-1])
        else:
            # Fallback to self-calc
            self.logger.debug(f"[{symbol}] Using self-calc ADX (FMP unavailable)")
            result = self._calculate_adx_selfcalc(df, period)
            result["data_source"] = data_source
            return result

        # Determine trend strength
        if current_adx < 20:
            strength = "weak"
        elif current_adx < 40:
            strength = "developing"
        elif current_adx < 60:
            strength = "strong"
        else:
            strength = "very_strong"

        direction = "bullish" if current_plus_di > current_minus_di else "bearish"

        return {
            "value": round(current_adx, 2),
            "plus_di": round(current_plus_di, 2),
            "minus_di": round(current_minus_di, 2),
            "period": period,
            "strength": strength,
            "direction": direction,
            "signal": f"{strength}_{direction}",
            "data_source": data_source,
            "description": f"ADX={current_adx:.1f} ({strength}), +DI={current_plus_di:.1f}, -DI={current_minus_di:.1f} ({direction})"
        }

    def _calculate_adx_selfcalc(self, df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """
        ADX (Average Directional Index) - Self-calculated

        Formula:
            +DM = High - High(prev) if positive and > -DM, else 0
            -DM = Low(prev) - Low if positive and > +DM, else 0
            TR = max(High-Low, |High-Close(prev)|, |Low-Close(prev)|)
            +DI = 100 × EMA(+DM) / EMA(TR)
            -DI = 100 × EMA(-DM) / EMA(TR)
            DX = 100 × |+DI - -DI| / (+DI + -DI)
            ADX = EMA(DX)

        Interpretation:
            ADX < 20: Weak trend (ranging market)
            ADX 20-40: Developing trend
            ADX 40-60: Strong trend
            ADX > 60: Very strong trend
            +DI > -DI: Bullish trend
            +DI < -DI: Bearish trend
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        # Apply conditions
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # WILDER'S SMOOTHING: alpha = 1/period
        # This matches TradingView, Bloomberg, Polygon, FMP
        # Previously used: ewm(span=period) which gives different results
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)

        # DX and ADX using Wilder's smoothing
        di_sum = plus_di + minus_di
        di_diff = (plus_di - minus_di).abs()
        dx = 100 * di_diff / di_sum.replace(0, np.nan)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        current_adx = float(adx.iloc[-1])
        current_plus_di = float(plus_di.iloc[-1])
        current_minus_di = float(minus_di.iloc[-1])

        # Determine trend strength
        if current_adx < 20:
            strength = "weak"
        elif current_adx < 40:
            strength = "developing"
        elif current_adx < 60:
            strength = "strong"
        else:
            strength = "very_strong"

        # Trend direction
        direction = "bullish" if current_plus_di > current_minus_di else "bearish"

        return {
            "value": round(current_adx, 2),
            "plus_di": round(current_plus_di, 2),
            "minus_di": round(current_minus_di, 2),
            "period": period,
            "strength": strength,
            "direction": direction,
            "signal": f"{strength}_{direction}",
            "data_source": "self-calc",
            "description": f"ADX={current_adx:.1f} ({strength}), +DI={current_plus_di:.1f}, -DI={current_minus_di:.1f} ({direction})"
        }

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """
        ATR (Average True Range) - Volatility indicator

        Formula:
            TR = max(High-Low, |High-Close(prev)|, |Low-Close(prev)|)
            ATR = EMA(TR, period)

        Interpretation:
            High ATR: High volatility, larger price moves expected
            Low ATR: Low volatility, smaller price moves expected
            ATR can be used for position sizing and stop-loss placement
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR using WILDER'S SMOOTHING: alpha = 1/period
        # This matches TradingView, Bloomberg, Polygon
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        current_atr = float(atr.iloc[-1])
        current_price = float(close.iloc[-1])

        # ATR as percentage of price
        atr_pct = (current_atr / current_price) * 100 if current_price > 0 else 0

        # Historical ATR comparison (is current ATR high or low?)
        atr_sma = atr.rolling(window=50).mean()
        atr_relative = "high" if current_atr > atr_sma.iloc[-1] else "low"

        return {
            "value": round(current_atr, 2),
            "period": period,
            "atr_pct": round(atr_pct, 2),
            "relative": atr_relative,
            "description": f"ATR({period}) = ${current_atr:.2f} ({atr_pct:.2f}% of price, {atr_relative} volatility)"
        }

    def _calculate_supertrend(
        self,
        df: pd.DataFrame,
        current_price: float,
        period: int = 10,
        multiplier: float = 3.0
    ) -> Dict[str, Any]:
        """
        Supertrend - Trend following indicator

        Formula:
            Basic Upper Band = (High + Low) / 2 + Multiplier × ATR
            Basic Lower Band = (High + Low) / 2 - Multiplier × ATR
            Final bands follow persistence rules.
            Supertrend = Lower Band in uptrend, Upper Band in downtrend

        Interpretation:
            Price > Supertrend: Uptrend (Buy signal)
            Price < Supertrend: Downtrend (Sell signal)
            Trend changes when price crosses Supertrend
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate ATR using WILDER'S SMOOTHING
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()

        # HL2 (median price)
        hl2 = (high + low) / 2

        # Basic bands
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)

        # Final bands with persistence
        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)

        # Initialize
        final_upper.iloc[0] = basic_upper.iloc[0]
        final_lower.iloc[0] = basic_lower.iloc[0]
        supertrend.iloc[0] = final_lower.iloc[0]
        direction.iloc[0] = 1  # Start bullish

        for i in range(1, len(close)):
            # Upper band persistence
            if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i-1]

            # Lower band persistence
            if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i-1]

            # Direction and Supertrend value
            if direction.iloc[i-1] == 1:  # Was bullish
                if close.iloc[i] < final_lower.iloc[i]:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = final_upper.iloc[i]
                else:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = final_lower.iloc[i]
            else:  # Was bearish
                if close.iloc[i] > final_upper.iloc[i]:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = final_lower.iloc[i]
                else:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = final_upper.iloc[i]

        current_supertrend = float(supertrend.iloc[-1])
        current_direction = int(direction.iloc[-1])
        prev_direction = int(direction.iloc[-2]) if len(direction) > 1 else current_direction

        # Check for trend change
        trend_changed = current_direction != prev_direction

        return {
            "value": round(current_supertrend, 2),
            "parameters": f"({period}, {multiplier})",
            "direction": "bullish" if current_direction == 1 else "bearish",
            "trend_changed": trend_changed,
            "signal": "buy" if current_direction == 1 else "sell",
            "description": f"Supertrend = ${current_supertrend:.2f}, {'BULLISH' if current_direction == 1 else 'BEARISH'}" + (" (TREND CHANGED!)" if trend_changed else "")
        }

    # ════════════════════════════════════════════════════════════════════════
    # SIGNAL GENERATION
    # ════════════════════════════════════════════════════════════════════════

    def _generate_signals(
        self,
        indicators: Dict[str, Any],
        current_price: float
    ) -> List[Dict[str, str]]:
        """
        Generate trading signals from calculated indicators.

        Returns list of signals with type and description.
        """
        signals = []

        # RSI Signals
        if "rsi" in indicators:
            rsi = indicators["rsi"]
            if rsi["signal"] == "overbought":
                signals.append({
                    "type": "RSI_OVERBOUGHT",
                    "strength": "strong",
                    "description": f"RSI at {rsi['value']:.1f} - stock may be overbought"
                })
            elif rsi["signal"] == "oversold":
                signals.append({
                    "type": "RSI_OVERSOLD",
                    "strength": "strong",
                    "description": f"RSI at {rsi['value']:.1f} - stock may be oversold"
                })

        # MACD Signals
        if "macd" in indicators:
            macd = indicators["macd"]
            if macd["signal"] == "bullish_crossover":
                signals.append({
                    "type": "MACD_BULLISH_CROSS",
                    "strength": "strong",
                    "description": "MACD crossed above signal line - bullish momentum"
                })
            elif macd["signal"] == "bearish_crossover":
                signals.append({
                    "type": "MACD_BEARISH_CROSS",
                    "strength": "strong",
                    "description": "MACD crossed below signal line - bearish momentum"
                })
            elif macd["signal"] == "bullish":
                signals.append({
                    "type": "MACD_BULLISH",
                    "strength": "moderate",
                    "description": "MACD histogram positive - bullish momentum"
                })
            elif macd["signal"] == "bearish":
                signals.append({
                    "type": "MACD_BEARISH",
                    "strength": "moderate",
                    "description": "MACD histogram negative - bearish momentum"
                })

        # SMA Trend Signals
        if "sma" in indicators:
            sma = indicators["sma"]
            price_vs = sma.get("price_vs_sma", {})

            # Long-term trend (SMA-200)
            if "vs_sma200" in price_vs:
                pos = price_vs["vs_sma200"]["position"]
                signals.append({
                    "type": f"TREND_LONG_TERM_{'BULLISH' if pos == 'above' else 'BEARISH'}",
                    "strength": "moderate",
                    "description": f"Price {pos} SMA-200 - long-term trend {'bullish' if pos == 'above' else 'bearish'}"
                })

            # Medium-term trend (SMA-50)
            if "vs_sma50" in price_vs:
                pos = price_vs["vs_sma50"]["position"]
                signals.append({
                    "type": f"TREND_MEDIUM_TERM_{'BULLISH' if pos == 'above' else 'BEARISH'}",
                    "strength": "moderate",
                    "description": f"Price {pos} SMA-50 - medium-term trend {'bullish' if pos == 'above' else 'bearish'}"
                })

        # Bollinger Band Signals
        if "bollinger_bands" in indicators:
            bb = indicators["bollinger_bands"]
            if bb["signal"] == "overbought":
                signals.append({
                    "type": "BB_OVERBOUGHT",
                    "strength": "strong",
                    "description": f"Price above upper Bollinger Band (%B={bb['percent_b']:.2f})"
                })
            elif bb["signal"] == "oversold":
                signals.append({
                    "type": "BB_OVERSOLD",
                    "strength": "strong",
                    "description": f"Price below lower Bollinger Band (%B={bb['percent_b']:.2f})"
                })

            # Low volatility signal (potential breakout)
            if bb["bandwidth_pct"] < 5:
                signals.append({
                    "type": "BB_SQUEEZE",
                    "strength": "moderate",
                    "description": f"Bollinger Bands squeeze (bandwidth={bb['bandwidth_pct']:.1f}%) - potential breakout"
                })

        # VWAP Signals
        if "vwap" in indicators:
            vwap = indicators["vwap"]
            signals.append({
                "type": f"VWAP_{'BULLISH' if vwap['signal'] == 'bullish' else 'BEARISH'}",
                "strength": "moderate",
                "description": f"Price {vwap['price_vs_vwap']} VWAP ({vwap['distance_pct']:+.1f}%) - institutional benchmark"
            })

        # Stochastic Signals
        if "stochastic" in indicators:
            stoch = indicators["stochastic"]
            if stoch["signal"] == "overbought":
                signals.append({
                    "type": "STOCH_OVERBOUGHT",
                    "strength": "strong",
                    "description": f"Stochastic at {stoch['k']:.1f} - stock may be overbought"
                })
            elif stoch["signal"] == "oversold":
                signals.append({
                    "type": "STOCH_OVERSOLD",
                    "strength": "strong",
                    "description": f"Stochastic at {stoch['k']:.1f} - stock may be oversold"
                })
            elif stoch["signal"] == "bullish_crossover":
                signals.append({
                    "type": "STOCH_BULLISH_CROSS",
                    "strength": "strong",
                    "description": "Stochastic %K crossed above %D - bullish signal"
                })
            elif stoch["signal"] == "bearish_crossover":
                signals.append({
                    "type": "STOCH_BEARISH_CROSS",
                    "strength": "strong",
                    "description": "Stochastic %K crossed below %D - bearish signal"
                })

        # OBV Signals
        if "obv" in indicators:
            obv = indicators["obv"]
            if obv["divergence"] == "bullish_divergence":
                signals.append({
                    "type": "OBV_BULLISH_DIVERGENCE",
                    "strength": "strong",
                    "description": "OBV rising while price falling - potential bullish reversal"
                })
            elif obv["divergence"] == "bearish_divergence":
                signals.append({
                    "type": "OBV_BEARISH_DIVERGENCE",
                    "strength": "strong",
                    "description": "OBV falling while price rising - potential bearish reversal"
                })

        # ADX Signals
        if "adx" in indicators:
            adx = indicators["adx"]
            if adx["strength"] in ["strong", "very_strong"]:
                signals.append({
                    "type": f"ADX_STRONG_TREND_{adx['direction'].upper()}",
                    "strength": "strong",
                    "description": f"ADX at {adx['value']:.1f} - {adx['strength'].replace('_', ' ')} {adx['direction']} trend"
                })
            elif adx["strength"] == "weak":
                signals.append({
                    "type": "ADX_WEAK_TREND",
                    "strength": "moderate",
                    "description": f"ADX at {adx['value']:.1f} - ranging market, no clear trend"
                })

        # ATR Signals (volatility)
        if "atr" in indicators:
            atr = indicators["atr"]
            if atr["relative"] == "high":
                signals.append({
                    "type": "ATR_HIGH_VOLATILITY",
                    "strength": "moderate",
                    "description": f"ATR at ${atr['value']:.2f} ({atr['atr_pct']:.1f}%) - high volatility, expect larger moves"
                })

        # Supertrend Signals
        if "supertrend" in indicators:
            st = indicators["supertrend"]
            if st["trend_changed"]:
                signals.append({
                    "type": f"SUPERTREND_{st['signal'].upper()}_SIGNAL",
                    "strength": "strong",
                    "description": f"Supertrend TREND CHANGE to {st['direction'].upper()} - potential entry point"
                })
            else:
                signals.append({
                    "type": f"SUPERTREND_{st['direction'].upper()}",
                    "strength": "moderate",
                    "description": f"Supertrend indicates {st['direction']} trend"
                })

        # 52-Week Range Signals
        if "range_52w" in indicators:
            r52w = indicators["range_52w"]
            position = r52w.get("position_pct", 50)
            if position >= 95:
                signals.append({
                    "type": "52W_AT_HIGH",
                    "strength": "strong",
                    "description": f"At 52-week high ({position:.1f}%) - extended, potential profit-taking"
                })
            elif position >= 80:
                signals.append({
                    "type": "52W_NEAR_HIGH",
                    "strength": "moderate",
                    "description": f"Near 52-week high ({position:.1f}%) - strong momentum but extended"
                })
            elif position <= 5:
                signals.append({
                    "type": "52W_AT_LOW",
                    "strength": "strong",
                    "description": f"At 52-week low ({position:.1f}%) - potential capitulation or value"
                })
            elif position <= 20:
                signals.append({
                    "type": "52W_NEAR_LOW",
                    "strength": "moderate",
                    "description": f"Near 52-week low ({position:.1f}%) - potential value or continued weakness"
                })

        return signals

    # ════════════════════════════════════════════════════════════════════════
    # FORMATTED CONTEXT FOR LLM
    # ════════════════════════════════════════════════════════════════════════

    def _build_formatted_context(self, result_data: Dict[str, Any]) -> str:
        """
        Build human-readable formatted context for LLM interpretation.

        Provides clear, structured output of technical indicators.
        """
        symbol = result_data.get("symbol", "Unknown")
        timeframe = result_data.get("timeframe", "3M")
        price = result_data.get("current_price", 0)
        latest_date = result_data.get("latest_date", "N/A")
        indicators = result_data.get("indicators", {})
        signals = result_data.get("signals", [])

        lines = [
            f"TECHNICAL ANALYSIS: {symbol}",
            f"Date: {latest_date} | Timeframe: {timeframe} | Price: ${price:,.2f}",
            "=" * 60
        ]

        # RSI Section
        if "rsi" in indicators:
            rsi = indicators["rsi"]
            lines.append("")
            lines.append(f"RSI (14-period): {rsi['value']:.1f}")
            lines.append(f"  Signal: {rsi['signal'].upper()}")
            lines.append(f"  Interpretation: >70 overbought, <30 oversold")

        # MACD Section
        if "macd" in indicators:
            macd = indicators["macd"]
            lines.append("")
            lines.append(f"MACD {macd['parameters']}:")
            lines.append(f"  MACD Line: {macd['macd_line']:.4f}")
            lines.append(f"  Signal Line: {macd['signal_line']:.4f}")
            lines.append(f"  Histogram: {macd['histogram']:.4f}")
            lines.append(f"  Signal: {macd['signal'].upper().replace('_', ' ')}")

        # Moving Averages Section
        if "sma" in indicators:
            sma = indicators["sma"]
            lines.append("")
            lines.append("SMA (Simple Moving Averages):")
            if sma.get("sma_20"):
                pos = sma.get("price_vs_sma", {}).get("vs_sma20", {})
                lines.append(f"  SMA-20: ${sma['sma_20']:,.2f} (price {pos.get('position', 'N/A')}, {pos.get('distance_pct', 0):+.1f}%)")
            if sma.get("sma_50"):
                pos = sma.get("price_vs_sma", {}).get("vs_sma50", {})
                lines.append(f"  SMA-50: ${sma['sma_50']:,.2f} (price {pos.get('position', 'N/A')}, {pos.get('distance_pct', 0):+.1f}%)")
            if sma.get("sma_200"):
                pos = sma.get("price_vs_sma", {}).get("vs_sma200", {})
                lines.append(f"  SMA-200: ${sma['sma_200']:,.2f} (price {pos.get('position', 'N/A')}, {pos.get('distance_pct', 0):+.1f}%)")

        if "ema" in indicators:
            ema = indicators["ema"]
            lines.append("")
            lines.append("EMA (Exponential Moving Averages):")
            lines.append(f"  EMA-12: ${ema['ema_12']:,.2f}")
            lines.append(f"  EMA-26: ${ema['ema_26']:,.2f}")

        # Bollinger Bands Section
        if "bollinger_bands" in indicators:
            bb = indicators["bollinger_bands"]
            lines.append("")
            lines.append(f"Bollinger Bands {bb['parameters']}:")
            lines.append(f"  Upper: ${bb['upper']:,.2f}")
            lines.append(f"  Middle: ${bb['middle']:,.2f}")
            lines.append(f"  Lower: ${bb['lower']:,.2f}")
            lines.append(f"  %B: {bb['percent_b']:.2f} | Bandwidth: {bb['bandwidth_pct']:.1f}%")
            lines.append(f"  Position: {bb['position'].upper().replace('_', ' ')}")

        # VWAP Section
        if "vwap" in indicators:
            vwap = indicators["vwap"]
            lines.append("")
            lines.append(f"VWAP (Volume Weighted Average Price):")
            if vwap.get('value') is not None:
                lines.append(f"  Value: ${vwap['value']:,.2f}")
                lines.append(f"  Price vs VWAP: {vwap['price_vs_vwap'].upper()} ({vwap.get('distance_pct', 0):+.2f}%)")
                lines.append(f"  Data Source: {vwap.get('data_source', 'N/A')}")
                if vwap.get('bars_used'):
                    lines.append(f"  Bars Used: {vwap['bars_used']} (intraday)")
                lines.append(f"  Note: Institutional benchmark - price above VWAP = bullish")
            else:
                lines.append(f"  Status: {vwap.get('description', 'Unavailable')}")

        # 52-Week Range Section
        if "range_52w" in indicators:
            r52w = indicators["range_52w"]
            lines.append("")
            lines.append(f"52-WEEK RANGE:")
            lines.append(f"  High: ${r52w['high_52w']:,.2f}")
            lines.append(f"  Low: ${r52w['low_52w']:,.2f}")
            lines.append(f"  Current Position: {r52w['position_pct']:.1f}% ({r52w['position_desc'].upper().replace('_', ' ')})")
            lines.append(f"  Distance from High: {r52w['distance_from_high_pct']:.1f}%")
            lines.append(f"  Distance from Low: {r52w['distance_from_low_pct']:.1f}%")
            lines.append(f"  Interpretation: 0%=at low, 100%=at high, >80%=near highs")

        # Stochastic Section
        if "stochastic" in indicators:
            stoch = indicators["stochastic"]
            lines.append("")
            lines.append(f"Stochastic Oscillator {stoch['parameters']}:")
            lines.append(f"  %K: {stoch['k']:.1f}")
            lines.append(f"  %D: {stoch['d']:.1f}")
            lines.append(f"  Signal: {stoch['signal'].upper().replace('_', ' ')}")
            lines.append(f"  Interpretation: >80 overbought, <20 oversold")

        # OBV Section
        if "obv" in indicators:
            obv = indicators["obv"]
            lines.append("")
            lines.append(f"OBV (On-Balance Volume):")
            lines.append(f"  Value: {obv['value']:,}")
            lines.append(f"  Trend: {obv['trend'].upper()}")
            if obv['divergence'] != "none":
                lines.append(f"  Divergence: {obv['divergence'].upper().replace('_', ' ')}")

        # ADX Section
        if "adx" in indicators:
            adx = indicators["adx"]
            lines.append("")
            lines.append(f"ADX (Average Directional Index):")
            lines.append(f"  ADX: {adx['value']:.1f} ({adx['strength'].upper().replace('_', ' ')})")
            lines.append(f"  +DI: {adx['plus_di']:.1f} | -DI: {adx['minus_di']:.1f}")
            lines.append(f"  Direction: {adx['direction'].upper()}")
            lines.append(f"  Data Source: {adx.get('data_source', 'self-calc')}")
            lines.append(f"  Interpretation: <20 weak, 20-40 developing, >40 strong trend")

        # ATR Section
        if "atr" in indicators:
            atr = indicators["atr"]
            lines.append("")
            lines.append(f"ATR (Average True Range):")
            lines.append(f"  Value: ${atr['value']:.2f} ({atr['atr_pct']:.2f}% of price)")
            lines.append(f"  Volatility: {atr['relative'].upper()}")
            lines.append(f"  Use: Position sizing, stop-loss placement")

        # Supertrend Section
        if "supertrend" in indicators:
            st = indicators["supertrend"]
            lines.append("")
            lines.append(f"Supertrend {st['parameters']}:")
            lines.append(f"  Value: ${st['value']:.2f}")
            lines.append(f"  Direction: {st['direction'].upper()}")
            lines.append(f"  Signal: {st['signal'].upper()}")
            if st.get("trend_changed"):
                lines.append(f"  *** TREND JUST CHANGED! ***")

        # Signals Summary
        if signals:
            lines.append("")
            lines.append("=" * 60)
            lines.append("SIGNALS:")
            for sig in signals:
                strength = sig.get("strength", "moderate")
                icon = "[!]" if strength == "strong" else "[>]"
                lines.append(f"  {icon} {sig['type']}: {sig['description']}")

        # Generate comprehensive insights
        lines.append("")
        lines.append("=" * 60)
        lines.append("TECHNICAL INSIGHTS SUMMARY:")
        insights = self._generate_insights_summary(indicators, signals, price)
        for insight in insights:
            lines.append(f"  • {insight}")

        # Data scope note
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"DATA CONTEXT:")
        lines.append(f"  Analysis Date: {latest_date}")
        lines.append(f"  Timeframe: {timeframe} ({result_data.get('data_points', 0)} trading days)")
        lines.append(f"  Price: ${price:,.2f}")
        lines.append("")
        lines.append(f"DISCLAIMER: This is technical analysis data only - not investment")
        lines.append(f"advice. Always consider fundamentals, news, and market context.")
        lines.append(f"Past performance does not guarantee future results.")

        return "\n".join(lines)

    def _generate_insights_summary(
        self,
        indicators: Dict[str, Any],
        signals: List[Dict],
        current_price: float
    ) -> List[str]:
        """
        Generate comprehensive insights based on all indicators.
        """
        insights = []

        # Count bullish vs bearish signals
        bullish_count = sum(1 for s in signals if "BULLISH" in s["type"] or "BUY" in s["type"] or "OVERSOLD" in s["type"])
        bearish_count = sum(1 for s in signals if "BEARISH" in s["type"] or "SELL" in s["type"] or "OVERBOUGHT" in s["type"])

        # Overall sentiment
        if bullish_count > bearish_count + 2:
            insights.append(f"OVERALL: Technical indicators lean BULLISH ({bullish_count} bullish vs {bearish_count} bearish signals)")
        elif bearish_count > bullish_count + 2:
            insights.append(f"OVERALL: Technical indicators lean BEARISH ({bearish_count} bearish vs {bullish_count} bullish signals)")
        else:
            insights.append(f"OVERALL: Mixed signals - NEUTRAL ({bullish_count} bullish, {bearish_count} bearish)")

        # Trend analysis
        if "adx" in indicators:
            adx = indicators["adx"]
            if adx["strength"] in ["strong", "very_strong"]:
                insights.append(f"TREND: Strong {adx['direction']} trend confirmed (ADX={adx['value']:.1f})")
            elif adx["strength"] == "weak":
                insights.append("TREND: No clear trend - ranging/consolidating market")

        # Momentum analysis
        if "rsi" in indicators:
            rsi = indicators["rsi"]
            if rsi["signal"] == "overbought":
                insights.append(f"MOMENTUM: RSI overbought ({rsi['value']:.1f}) - potential pullback ahead")
            elif rsi["signal"] == "oversold":
                insights.append(f"MOMENTUM: RSI oversold ({rsi['value']:.1f}) - potential bounce ahead")

        # Volume confirmation
        if "obv" in indicators:
            obv = indicators["obv"]
            if obv["divergence"] != "none":
                insights.append(f"VOLUME: {obv['divergence'].replace('_', ' ').upper()} - watch for reversal")

        # Supertrend
        if "supertrend" in indicators:
            st = indicators["supertrend"]
            if st.get("trend_changed"):
                insights.append(f"SUPERTREND: Fresh {st['signal'].upper()} signal - potential entry point")

        # Volatility
        if "atr" in indicators:
            atr = indicators["atr"]
            if atr["relative"] == "high":
                insights.append(f"VOLATILITY: High ({atr['atr_pct']:.1f}% ATR) - use wider stops")

        # Support/Resistance from Bollinger Bands
        if "bollinger_bands" in indicators:
            bb = indicators["bollinger_bands"]
            if bb["bandwidth_pct"] < 5:
                insights.append("VOLATILITY: BB squeeze detected - potential breakout imminent")

        # 52-Week Range Position
        if "range_52w" in indicators:
            r52w = indicators["range_52w"]
            position = r52w.get("position_pct", 50)
            if position >= 80:
                insights.append(f"52W POSITION: Trading near yearly highs ({position:.1f}%) - extended but strong momentum")
            elif position <= 20:
                insights.append(f"52W POSITION: Trading near yearly lows ({position:.1f}%) - potential value or continued weakness")

        return insights if insights else ["No significant patterns detected"]


# ============================================================================
# Standalone Testing
# ============================================================================
if __name__ == "__main__":
    import asyncio
    import os

    async def test_tool():
        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            print("FMP_API_KEY not set")
            return

        tool = GetTechnicalIndicatorsTool(api_key=api_key)

        print("\n" + "=" * 60)
        print("Testing getTechnicalIndicators - Stock Analysis")
        print("=" * 60)

        # Test with AAPL
        print("\n--- Test 1: AAPL with all indicators ---")
        result = await tool.execute(symbol="AAPL", timeframe="3M")

        if result.get("status") == "success":
            data = result.get("data", {})
            print(f"Symbol: {data.get('symbol')}")
            print(f"Price: ${data.get('current_price', 0):,.2f}")
            print(f"Data Points: {data.get('data_points')}")
            print(f"\nIndicators calculated: {data.get('indicators_calculated')}")
            print(f"\nSignals: {len(data.get('signals', []))}")
            for sig in data.get("signals", []):
                print(f"  - {sig['type']}: {sig['description']}")

            # Print formatted context
            if "formatted_context" in result:
                print("\n--- Formatted Context ---")
                print(result["formatted_context"][:500] + "...")
        else:
            print(f"ERROR: {result.get('error')}")

        # Test with different timeframe
        print("\n--- Test 2: TSLA with 1Y timeframe ---")
        result2 = await tool.execute(symbol="TSLA", timeframe="1Y")

        if result2.get("status") == "success":
            data = result2.get("data", {})
            print(f"Symbol: {data.get('symbol')}")
            print(f"Price: ${data.get('current_price', 0):,.2f}")
            print(f"Data Points: {data.get('data_points')}")
            sma = data.get("indicators", {}).get("sma", {})
            if sma.get("sma_200"):
                print(f"SMA-200: ${sma['sma_200']:,.2f}")
        else:
            print(f"ERROR: {result2.get('error')}")

    asyncio.run(test_tool())