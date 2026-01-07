"""
Get Crypto Technical Indicators Tool (Enhanced)

Fetches comprehensive technical indicators for cryptocurrencies.
Uses internal API for OHLCV data and internal calculators.

Features:
- Multi-timeframe analysis (15m, 1h, 4h, 1d)
- Indicator filtering (select specific indicators)
- Internal API integration
- Full technical indicator suite

Internal API: http://10.10.0.2:20073/api/v1/market/crypto
"""

import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

import httpx

from src.agents.tools.base import BaseTool, ToolOutput, ToolSchema, ToolParameter
from src.agents.tools.crypto.base_crypto_tool import BaseCryptoTool
from src.agents.tools.crypto.calculators import technicals


class GetCryptoTechnicalsTool(BaseCryptoTool):
    """
    Tool: getCryptoTechnicals (Enhanced)

    Fetch comprehensive technical indicators for cryptocurrencies.
    Uses internal API for OHLCV data and internal calculators.

    Features:
    - Multi-timeframe: 15m, 1h, 4h, 1d
    - Full indicator suite: RSI, MACD, Stochastic, ADX, ATR, Bollinger, Ichimoku, etc.
    - Indicator filtering: Request only specific indicators
    - Internal API for fast, reliable data

    FIXED:
    - Accepts multiple symbol formats (BTCUSD, BTCUSDT, BTC)
    - Auto-normalizes symbols
    """

    CACHE_TTL = 300  # 5 minutes for technicals

    # Known crypto base symbols
    KNOWN_CRYPTO = {
        'BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI', 'AAVE',
        'XRP', 'LTC', 'BCH', 'EOS', 'TRX', 'XLM', 'VET', 'ALGO', 'ATOM', 'LUNA',
        'NEAR', 'FTM', 'CRO', 'SAND', 'MANA', 'AXS', 'GALA', 'ENJ', 'CHZ', 'BAT',
        'ZEC', 'DASH', 'XMR', 'DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BNB', 'TON', 'ICP',
        'HBAR', 'THETA', 'FIL', 'ETC', 'MKR', 'APT', 'LDO', 'OP', 'ARB', 'SUI',
        'IMX', 'GRT', 'RUNE', 'FLOW', 'EGLD', 'XTZ', 'MINA', 'ROSE', 'KAVA',
        'INJ', 'SEI', 'TIA', 'JUP', 'BONK', 'WIF', 'ORDI', 'STX', 'RENDER'
    }

    # Supported timeframes from internal API
    SUPPORTED_TIMEFRAMES = ["15m", "1h", "4h", "1d"]

    # Timeframe to hours mapping for candle calculation
    TIMEFRAME_HOURS = {"15m": 0.25, "1h": 1, "4h": 4, "1d": 24}

    # Available indicator categories (expanded with new indicators)
    INDICATOR_CATEGORIES = {
        "momentum": ["rsi", "stoch_rsi", "stochastic", "macd", "cci", "williams_r", "awesome_osc"],
        "trend": ["adx", "supertrend", "ichimoku", "parabolic_sar", "aroon"],
        "volatility": ["atr", "bollinger", "keltner", "donchian"],
        "volume": ["obv", "vwap", "mfi", "cmf", "ad_line"],
        "moving_averages": ["sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "ema_50"],
        "all": None,  # All indicators
    }

    # FMP API (fallback source)
    FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize GetCryptoTechnicalsTool

        Args:
            api_key: FMP API key (optional, for fallback)
        """
        super().__init__()
        self.fmp_api_key = api_key

        # Define enhanced schema
        self.schema = ToolSchema(
            name="getCryptoTechnicals",
            category="crypto",
            description=(
                "Fetch comprehensive technical indicators for cryptocurrencies. "
                "Supports multi-timeframe analysis (15m, 1h, 4h, 1d) with full indicator suite. "
                "Returns RSI, MACD, Stochastic, ADX, ATR, Bollinger Bands, Ichimoku, VWAP, and more. "
                "Use when user asks about crypto technical analysis, indicators, or trading signals."
            ),
            capabilities=[
                "RSI, Stochastic RSI, Stochastic (%K, %D)",
                "MACD (Line, Signal, Histogram)",
                "ADX (+DI, -DI) for trend strength",
                "ATR for volatility measurement",
                "Bollinger Bands (%B, bandwidth)",
                "Ichimoku Cloud (full 5-line system)",
                "VWAP, OBV for volume analysis",
                "Supertrend indicator",
                "Moving Averages (SMA, EMA) multiple periods",
                "Multi-timeframe: 15m, 1h, 4h, 1d",
                "Accepts BTC, BTCUSD, BTCUSDT formats",
            ],
            limitations=[
                "Requires sufficient historical data (min 200 candles)",
                "Internal API timeframes: 15m, 1h, 4h, 1d only",
                "5-minute cache for performance",
            ],
            usage_hints=[
                "User asks: 'Bitcoin RSI' → USE THIS with symbol=BTC",
                "User asks: 'ETH technical analysis' → USE THIS with symbol=ETH",
                "User asks: 'BTC all timeframes' → USE THIS with timeframes=['15m','1h','4h','1d']",
                "User asks: 'SOL momentum indicators' → USE THIS with indicators=['momentum']",
                "User asks: 'Phân tích kỹ thuật BTC' → USE THIS with symbol=BTC",
                "User wants SMC analysis → DO NOT USE (use getCryptoSMCAnalysis)",
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description=(
                        "Crypto symbol in any format: "
                        "BTC, BTCUSD, BTCUSDT, ETH, ETHUSD, ETHUSDT, etc. "
                        "Will be automatically normalized."
                    ),
                    required=True,
                    pattern=r"^[A-Za-z]{2,15}(USD[T]?)?$"
                ),
                ToolParameter(
                    name="timeframe",
                    type="string",
                    description="Primary timeframe for analysis (default: 1h)",
                    required=False,
                    default="1h",
                    enum=["15m", "1h", "4h", "1d"]
                ),
                ToolParameter(
                    name="timeframes",
                    type="array",
                    description="Multiple timeframes for multi-TF analysis (optional)",
                    required=False,
                    default=None,
                ),
                ToolParameter(
                    name="indicators",
                    type="array",
                    description=(
                        "List of indicator categories to calculate. "
                        "Options: 'momentum', 'trend', 'volatility', 'volume', 'moving_averages', 'all'. "
                        "Default: ['all']"
                    ),
                    required=False,
                    default=["all"],
                ),
                ToolParameter(
                    name="lookback_days",
                    type="integer",
                    description=(
                        "Number of days to look back for analysis. "
                        "Example: '7 ngày' → lookback_days=7. "
                        "If not specified, uses smart defaults based on timeframe."
                    ),
                    required=False,
                    default=None,
                ),
            ],
            returns={
                "symbol": "string - Normalized symbol",
                "timeframe": "string - Primary timeframe",
                "current_price": "number - Current price",
                "data_range": "object - Time range of data {from, to, candle_count, duration_days}",
                "indicators": "object - All calculated indicators",
                "signal": "object - Overall signal summary",
                "multi_tf": "object - Multi-timeframe data (if requested)",
                "timestamp": "string - Data timestamp",
            },
            typical_execution_time_ms=1000,
            requires_symbol=True,
        )

    def _normalize_crypto_symbol(self, symbol: str) -> str:
        """Normalize crypto symbol to standard format"""
        symbol = symbol.upper().strip()

        if symbol.endswith('USDT'):
            return symbol[:-4]  # Remove USDT suffix
        elif symbol.endswith('USD'):
            return symbol[:-3]  # Remove USD suffix
        elif symbol.endswith('BUSD'):
            return symbol[:-4]
        return symbol

    def _extract_base_symbol(self, symbol: str) -> str:
        """Extract base symbol (same as normalize for crypto)"""
        return self._normalize_crypto_symbol(symbol)

    def _calculate_candle_limit(
        self,
        timeframe: str,
        lookback_days: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> int:
        """
        Calculate optimal candle limit for technical analysis

        Priority:
        1. Explicit limit (if provided)
        2. lookback_days converted to candles
        3. Smart default based on timeframe

        Returns:
            Number of candles to fetch
        """
        hours = self.TIMEFRAME_HOURS.get(timeframe, 1)

        # If explicit limit provided
        if limit:
            return min(limit, 1000)  # Cap at 1000

        # If lookback_days provided
        if lookback_days:
            candles = int(lookback_days * 24 / hours)
            return min(candles, 1000)

        # Smart defaults for technical analysis
        # RSI needs 14 periods, MACD needs 26+9, Ichimoku needs 52
        defaults = {
            "15m": 200,   # ~2 days
            "1h": 200,    # ~8 days
            "4h": 200,    # ~33 days
            "1d": 200,    # ~200 days
        }
        return defaults.get(timeframe, 200)

    def _calculate_data_range(
        self,
        ohlcv_data: List[Dict],
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Calculate data range information from OHLCV data

        Returns:
            Dict with from, to, candle_count, duration_days, duration_hours
        """
        if not ohlcv_data:
            return {}

        candle_count = len(ohlcv_data)
        hours = self.TIMEFRAME_HOURS.get(timeframe, 1)
        duration_hours = candle_count * hours
        duration_days = duration_hours / 24

        # Get time range from data
        first_candle = ohlcv_data[0]
        last_candle = ohlcv_data[-1]

        # Extract timestamps
        from_time = first_candle.get("time", first_candle.get("timestamp", ""))
        to_time = last_candle.get("time", last_candle.get("timestamp", ""))

        return {
            "from": from_time,
            "to": to_time,
            "candle_count": candle_count,
            "duration_hours": round(duration_hours, 1),
            "duration_days": round(duration_days, 1),
            "timeframe": timeframe,
        }

    async def execute(
        self,
        symbol: str,
        timeframe: str = "1h",
        timeframes: Optional[List[str]] = None,
        indicators: Optional[List[str]] = None,
        lookback_days: Optional[int] = None,
        **kwargs
    ) -> ToolOutput:
        """
        Execute getCryptoTechnicals

        Args:
            symbol: Crypto symbol (any format)
            timeframe: Primary timeframe for analysis
            timeframes: Optional list for multi-TF analysis
            indicators: List of indicator categories to calculate
            lookback_days: Number of days to look back (optional)

        Returns:
            ToolOutput with comprehensive technical indicators
        """
        start_time = time.time()
        original_symbol = symbol

        try:
            # Normalize symbol
            symbol = self._normalize_crypto_symbol(symbol)

            # Validate timeframe
            if timeframe not in self.SUPPORTED_TIMEFRAMES:
                timeframe = "1h"

            # Calculate candle limit based on lookback_days or smart default
            candle_limit = self._calculate_candle_limit(timeframe, lookback_days)

            # Default indicators to all
            if indicators is None or "all" in indicators:
                indicator_categories = list(self.INDICATOR_CATEGORIES.keys())
                indicator_categories.remove("all")
            else:
                indicator_categories = indicators

            self.logger.info(f"  ┌─ 🔧 TOOL: {self.schema.name}")
            self.logger.info(f"  │  Input: {{symbol={original_symbol}→{symbol}, tf={timeframe}, limit={candle_limit}}}")

            # Check cache (include candle_limit in key)
            cache_key = f"crypto:technicals:{symbol}:{timeframe}:{candle_limit}:{','.join(sorted(indicator_categories))}"
            cached_result = await self._get_cached_result(cache_key)

            if cached_result:
                execution_time = int((time.time() - start_time) * 1000)
                self.logger.info(f"  │  🎯 [CACHE HIT]")
                self.logger.info(f"  └─ ✅ SUCCESS ({execution_time}ms)")

                return ToolOutput(
                    tool_name=self.schema.name,
                    status="success",
                    data=cached_result,
                    formatted_context=self._build_formatted_context(cached_result),
                    execution_time_ms=execution_time,
                    metadata={
                        "symbol": symbol,
                        "original_symbol": original_symbol,
                        "timeframe": timeframe,
                        "from_cache": True,
                    },
                )

            # Fetch OHLCV data with fallback chain: Internal Klines API (primary) → FMP (fallback)
            # Binance data (via internal API) is more accurate and up-to-date
            ohlcv_data = await self._fetch_ohlcv_internal_klines(symbol, timeframe, limit=candle_limit)

            if not ohlcv_data or len(ohlcv_data) < 50:
                # Fallback: FMP API
                self.logger.warning(f"[{self.schema.name}] Internal API failed, trying FMP fallback")
                ohlcv_data = await self._fetch_ohlcv_fmp(symbol, timeframe, limit=candle_limit)

            if not ohlcv_data or len(ohlcv_data) < 50:
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error=f"Insufficient data for {symbol}. Need at least 50 candles.",
                    metadata={"symbol": symbol, "original_symbol": original_symbol},
                )

            # Extract OHLCV arrays
            opens = [c.get("open", 0) for c in ohlcv_data]
            highs = [c.get("high", 0) for c in ohlcv_data]
            lows = [c.get("low", 0) for c in ohlcv_data]
            closes = [c.get("close", 0) for c in ohlcv_data]
            volumes = [c.get("volume", 0) for c in ohlcv_data]

            # Calculate indicators using internal calculators
            calculated_indicators = self._calculate_indicators(
                opens, highs, lows, closes, volumes, indicator_categories
            )

            # Get current price
            current_price = closes[-1] if closes else None

            # Generate signal summary
            signal_summary = self._generate_signal_summary(calculated_indicators, current_price)

            # Calculate data range for context
            data_range = self._calculate_data_range(ohlcv_data, timeframe)

            # Multi-timeframe analysis if requested
            multi_tf_data = None
            if timeframes and len(timeframes) > 1:
                multi_tf_data = await self._calculate_multi_tf(
                    symbol, timeframes, indicator_categories
                )

            # Compile result
            result_data = {
                "symbol": symbol,
                "original_input": original_symbol,
                "timeframe": timeframe,
                "current_price": current_price,
                "data_range": data_range,
                "indicators": calculated_indicators,
                "signal": signal_summary,
                "multi_tf": multi_tf_data,
                "candle_count": len(ohlcv_data),
                "timestamp": datetime.now().isoformat(),
            }

            # Cache result
            await self._set_cached_result(cache_key, result_data, ttl=self.CACHE_TTL)

            execution_time = int((time.time() - start_time) * 1000)

            rsi_val = calculated_indicators.get("rsi", {}).get("value", "N/A")
            self.logger.info(f"  │  Result: RSI={rsi_val}, Signal={signal_summary.get('overall', 'N/A')}")
            self.logger.info(f"  └─ ✅ SUCCESS ({execution_time}ms)")

            return ToolOutput(
                tool_name=self.schema.name,
                status="success",
                data=result_data,
                formatted_context=self._build_formatted_context(result_data),
                execution_time_ms=execution_time,
                metadata={
                    "symbol": symbol,
                    "original_symbol": original_symbol,
                    "timeframe": timeframe,
                    "indicator_categories": indicator_categories,
                    "from_cache": False,
                },
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"  │  Error: {str(e)[:100]}")
            self.logger.info(f"  └─ ❌ FAILED ({execution_time}ms)")

            return ToolOutput(
                tool_name=self.schema.name,
                status="error",
                error=f"Error calculating technicals: {str(e)}",
                execution_time_ms=execution_time,
                metadata={"symbol": symbol, "original_symbol": original_symbol},
            )

    async def _fetch_ohlcv_internal_klines(
        self, symbol: str, timeframe: str, limit: int = 200
    ) -> Optional[List[Dict]]:
        """
        Fallback: Fetch OHLCV data from Internal Klines API
        Endpoint: /api/v1/market/crypto/{symbol}/klines
        """
        try:
            # Map timeframe to internal API interval format
            tf_map = {
                "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"
            }
            interval = tf_map.get(timeframe, "1h")

            # Build klines endpoint URL
            # Uses: {base_url}{api_prefix}/{symbol}/klines
            # Internal API expects base symbol (e.g., BTC, not BTCUSD)
            internal_symbol = symbol.replace("USD", "").replace("USDT", "")
            klines_endpoint = f"/{internal_symbol}/klines"

            self.logger.debug(f"[OHLCV-InternalKlines] Fetching {internal_symbol} ({interval})")

            response = await self._fetch_api(
                endpoint=klines_endpoint,
                params={
                    "interval": interval,
                    "limit": limit,
                },
            )

            # Handle various response formats
            if response is None:
                return None

            # If response is a string, try to parse as JSON
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    self.logger.warning(f"[OHLCV-InternalKlines] Invalid JSON response for {symbol}")
                    return None

            # Extract data from dict response
            if isinstance(response, dict):
                data = response.get("data", [])
                if isinstance(data, dict):
                    data = data.get("data", [])
            elif isinstance(response, list):
                data = response
            else:
                self.logger.warning(f"[OHLCV-InternalKlines] Unexpected response type: {type(response)}")
                return None

            if data and isinstance(data, list) and len(data) > 0:
                # Transform internal klines format to standard OHLCV format
                # Handle both formats:
                # 1. List of dicts: [{openTime, open, high, low, close, volume, ...}, ...]
                # 2. List of lists (Binance-like): [[openTime, open, high, low, close, volume, ...], ...]
                ohlcv_data = []
                for kline in data:
                    if isinstance(kline, dict):
                        # Dict format: {openTime, closeTime, open, high, low, close, volume, ...}
                        ohlcv_data.append({
                            "time": kline.get("openTime", ""),
                            "timestamp": kline.get("openTime", ""),
                            "open": float(kline.get("open", 0)),
                            "high": float(kline.get("high", 0)),
                            "low": float(kline.get("low", 0)),
                            "close": float(kline.get("close", 0)),
                            "volume": float(kline.get("volume", 0)),
                        })
                    elif isinstance(kline, (list, tuple)) and len(kline) >= 6:
                        # List format (Binance-like): [openTime, open, high, low, close, volume, ...]
                        ohlcv_data.append({
                            "time": kline[0],
                            "timestamp": kline[0],
                            "open": float(kline[1]),
                            "high": float(kline[2]),
                            "low": float(kline[3]),
                            "close": float(kline[4]),
                            "volume": float(kline[5]),
                        })

                # Sort by openTime ascending
                ohlcv_data.sort(key=lambda x: x.get("time", ""))

                self.logger.info(f"[OHLCV-InternalKlines] Fetched {len(ohlcv_data)} candles for {symbol}")
                return ohlcv_data

            return None

        except Exception as e:
            self.logger.warning(f"[OHLCV-InternalKlines] Fetch error for {symbol}: {e}")
            return None

    async def _fetch_ohlcv_fmp(
        self, symbol: str, timeframe: str, limit: int = 200
    ) -> Optional[List[Dict]]:
        """Primary: Fetch OHLCV data from FMP API"""
        if not self.fmp_api_key:
            self.logger.warning(f"[OHLCV-FMP] No FMP API key configured")
            return None

        try:
            # Map timeframe to FMP format
            tf_map = {
                "15m": "15min", "1h": "1hour", "4h": "4hour", "1d": "1day"
            }
            fmp_tf = tf_map.get(timeframe, "1hour")

            # FMP requires USD suffix (e.g., BTCUSD)
            fmp_symbol = f"{symbol}USD" if not symbol.endswith("USD") else symbol

            url = f"{self.FMP_BASE_URL}/historical-chart/{fmp_tf}/{fmp_symbol}"
            params = {"apikey": self.fmp_api_key}

            self.logger.debug(f"[OHLCV-FMP] Fetching {fmp_symbol} ({fmp_tf}) from {url}")

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            if data and isinstance(data, list) and len(data) > 0:
                # Normalize FMP data to standard OHLCV format
                # FMP format: {date, open, high, low, close, volume}
                ohlcv_data = []
                for candle in data:
                    if isinstance(candle, dict):
                        ohlcv_data.append({
                            "time": candle.get("date", ""),
                            "timestamp": candle.get("date", ""),
                            "open": float(candle.get("open", 0)),
                            "high": float(candle.get("high", 0)),
                            "low": float(candle.get("low", 0)),
                            "close": float(candle.get("close", 0)),
                            "volume": float(candle.get("volume", 0)),
                        })

                # Sort by date ascending and limit
                ohlcv_data.sort(key=lambda x: x.get("time", ""))
                self.logger.info(f"[OHLCV-FMP] Fetched {len(ohlcv_data)} candles for {symbol}")
                return ohlcv_data[-limit:]

            return None

        except Exception as e:
            self.logger.warning(f"[OHLCV-FMP] Fetch error for {symbol}: {e}")
            return None

    def _calculate_indicators(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        categories: List[str],
    ) -> Dict[str, Any]:
        """Calculate selected indicators using internal calculators"""
        result = {}

        # Momentum indicators
        if "momentum" in categories:
            # RSI
            rsi_result = technicals.calculate_rsi(closes, period=14)
            result["rsi"] = rsi_result

            # Stochastic RSI
            stoch_rsi_result = technicals.calculate_stoch_rsi(closes)
            result["stoch_rsi"] = stoch_rsi_result

            # Stochastic
            stoch_result = technicals.calculate_stochastic(highs, lows, closes)
            result["stochastic"] = stoch_result

            # MACD
            macd_result = technicals.calculate_macd(closes)
            result["macd"] = macd_result

            # CCI (Commodity Channel Index)
            cci_result = technicals.calculate_cci(highs, lows, closes)
            result["cci"] = cci_result

            # Williams %R
            williams_r_result = technicals.calculate_williams_r(highs, lows, closes)
            result["williams_r"] = williams_r_result

            # Awesome Oscillator
            ao_result = technicals.calculate_awesome_oscillator(highs, lows)
            result["awesome_osc"] = ao_result

        # Trend indicators
        if "trend" in categories:
            # ADX
            adx_result = technicals.calculate_adx(highs, lows, closes)
            result["adx"] = adx_result

            # Supertrend
            supertrend_result = technicals.calculate_supertrend(highs, lows, closes)
            result["supertrend"] = supertrend_result

            # Ichimoku
            ichimoku_result = technicals.calculate_ichimoku(highs, lows, closes)
            result["ichimoku"] = ichimoku_result

            # Parabolic SAR
            psar_result = technicals.calculate_parabolic_sar(highs, lows, closes)
            result["parabolic_sar"] = psar_result

            # Aroon
            aroon_result = technicals.calculate_aroon(highs, lows)
            result["aroon"] = aroon_result

        # Volatility indicators
        if "volatility" in categories:
            # ATR
            atr_result = technicals.calculate_atr(highs, lows, closes)
            result["atr"] = atr_result

            # Bollinger Bands
            bb_result = technicals.calculate_bollinger(closes)
            result["bollinger"] = bb_result

            # Keltner Channels
            keltner_result = technicals.calculate_keltner_channels(highs, lows, closes)
            result["keltner"] = keltner_result

            # Donchian Channels
            donchian_result = technicals.calculate_donchian_channels(highs, lows, closes)
            result["donchian"] = donchian_result

        # Volume indicators
        if "volume" in categories:
            # OBV
            obv_result = technicals.calculate_obv(closes, volumes)
            result["obv"] = obv_result

            # VWAP
            vwap_result = technicals.calculate_vwap(highs, lows, closes, volumes)
            result["vwap"] = vwap_result

            # MFI (Money Flow Index)
            mfi_result = technicals.calculate_mfi(highs, lows, closes, volumes)
            result["mfi"] = mfi_result

            # CMF (Chaikin Money Flow)
            cmf_result = technicals.calculate_cmf(highs, lows, closes, volumes)
            result["cmf"] = cmf_result

            # A/D Line (Accumulation/Distribution)
            ad_result = technicals.calculate_ad_line(highs, lows, closes, volumes)
            result["ad_line"] = ad_result

        # Moving Averages
        if "moving_averages" in categories:
            result["sma_20"] = technicals.calculate_sma(closes, 20)
            result["sma_50"] = technicals.calculate_sma(closes, 50)
            result["sma_200"] = technicals.calculate_sma(closes, 200)
            result["ema_12"] = technicals.calculate_ema(closes, 12)
            result["ema_26"] = technicals.calculate_ema(closes, 26)
            result["ema_50"] = technicals.calculate_ema(closes, 50)

        return result

    def _generate_signal_summary(
        self, indicators: Dict[str, Any], current_price: Optional[float]
    ) -> Dict[str, Any]:
        """Generate overall signal summary from indicators"""
        bullish_signals = 0
        bearish_signals = 0
        neutral_signals = 0

        signal_details = []

        # RSI signal
        rsi = indicators.get("rsi", {})
        if rsi and rsi.get("value"):
            rsi_val = rsi["value"]
            if rsi_val <= 30:
                bullish_signals += 1
                signal_details.append(f"RSI oversold ({rsi_val:.1f})")
            elif rsi_val >= 70:
                bearish_signals += 1
                signal_details.append(f"RSI overbought ({rsi_val:.1f})")
            else:
                neutral_signals += 1

        # MACD signal
        macd = indicators.get("macd", {})
        if macd and macd.get("histogram") is not None:
            hist = macd["histogram"]
            if hist > 0:
                bullish_signals += 1
                signal_details.append("MACD bullish")
            else:
                bearish_signals += 1
                signal_details.append("MACD bearish")

        # ADX signal
        adx = indicators.get("adx", {})
        if adx and adx.get("adx"):
            adx_val = adx["adx"]
            if adx_val >= 25:
                # ADX calculator returns signal: "strong_uptrend", "strong_downtrend", etc.
                adx_signal = adx.get("signal", "")
                if "uptrend" in adx_signal:
                    bullish_signals += 1
                    signal_details.append(f"Strong uptrend (ADX={adx_val:.1f})")
                elif "downtrend" in adx_signal:
                    bearish_signals += 1
                    signal_details.append(f"Strong downtrend (ADX={adx_val:.1f})")

        # Supertrend signal
        supertrend = indicators.get("supertrend", {})
        if supertrend and supertrend.get("direction"):
            if supertrend["direction"] == "bullish":
                bullish_signals += 1
                signal_details.append("Supertrend bullish")
            else:
                bearish_signals += 1
                signal_details.append("Supertrend bearish")

        # Price vs MAs
        if current_price:
            sma_50_data = indicators.get("sma_50", {})
            # Handle both dict (from calculator) and raw value
            sma_50_val = sma_50_data.get("value") if isinstance(sma_50_data, dict) else sma_50_data
            if sma_50_val and isinstance(sma_50_val, (int, float)):
                if current_price > sma_50_val:
                    bullish_signals += 1
                else:
                    bearish_signals += 1

        # Determine overall signal
        total_signals = bullish_signals + bearish_signals + neutral_signals
        if total_signals == 0:
            overall = "neutral"
            strength = 0
        else:
            if bullish_signals > bearish_signals:
                overall = "bullish"
                strength = (bullish_signals / total_signals) * 100
            elif bearish_signals > bullish_signals:
                overall = "bearish"
                strength = (bearish_signals / total_signals) * 100
            else:
                overall = "neutral"
                strength = 50

        return {
            "overall": overall,
            "strength": round(strength, 1),
            "bullish_count": bullish_signals,
            "bearish_count": bearish_signals,
            "neutral_count": neutral_signals,
            "details": signal_details[:5],  # Top 5 signals
        }

    async def _calculate_multi_tf(
        self,
        symbol: str,
        timeframes: List[str],
        indicator_categories: List[str],
    ) -> Dict[str, Any]:
        """Calculate indicators for multiple timeframes"""
        multi_tf_result = {}

        for tf in timeframes:
            if tf not in self.SUPPORTED_TIMEFRAMES:
                continue

            try:
                # Internal klines primary, FMP fallback
                ohlcv = await self._fetch_ohlcv_internal_klines(symbol, tf)
                if not ohlcv or len(ohlcv) < 50:
                    ohlcv = await self._fetch_ohlcv_fmp(symbol, tf)

                if ohlcv and len(ohlcv) >= 50:
                    opens = [c.get("open", 0) for c in ohlcv]
                    highs = [c.get("high", 0) for c in ohlcv]
                    lows = [c.get("low", 0) for c in ohlcv]
                    closes = [c.get("close", 0) for c in ohlcv]
                    volumes = [c.get("volume", 0) for c in ohlcv]

                    indicators = self._calculate_indicators(
                        opens, highs, lows, closes, volumes, indicator_categories
                    )

                    signal = self._generate_signal_summary(indicators, closes[-1])

                    multi_tf_result[tf] = {
                        "signal": signal["overall"],
                        "strength": signal["strength"],
                        "rsi": indicators.get("rsi", {}).get("value"),
                        "macd_hist": indicators.get("macd", {}).get("histogram"),
                    }

            except Exception as e:
                self.logger.warning(f"[Multi-TF] Error for {tf}: {e}")
                continue

        return multi_tf_result

    def _build_formatted_context(self, data: Dict[str, Any]) -> str:
        """Build human-readable formatted context for LLM with comprehensive indicators"""
        symbol = data.get("symbol", "Unknown")
        timeframe = data.get("timeframe", "N/A")
        price = data.get("current_price", 0)
        data_range = data.get("data_range", {})
        timestamp = data.get("timestamp", "")
        indicators = data.get("indicators", {})
        signal = data.get("signal", {})
        multi_tf = data.get("multi_tf")

        # Build data range description
        candle_count = data_range.get("candle_count", data.get("candle_count", 0))
        duration_days = data_range.get("duration_days", 0)
        duration_hours = data_range.get("duration_hours", 0)
        from_time = data_range.get("from", "")
        to_time = data_range.get("to", "")

        # Format time range description
        if duration_days >= 1:
            duration_str = f"{duration_days:.1f} ngày ({duration_hours:.0f} giờ)"
        else:
            duration_str = f"{duration_hours:.1f} giờ"

        # Format timestamp
        timestamp_display = timestamp[:19].replace("T", " ") if timestamp else "N/A"

        lines = [
            f"📊 CRYPTO TECHNICALS - {symbol} ({timeframe}):",
            "",
            "=" * 55,
            f"💵 CURRENT PRICE: {self._format_price(price)} USD" if price else "💵 CURRENT PRICE: N/A",
            f"⏰ Data fetched at: {timestamp_display}",
            "=" * 55,
            "",
            f"📅 Data Range: {candle_count} candles ({duration_str})",
        ]

        # Add time range if available
        if from_time and to_time:
            from_display = from_time[:16].replace("T", " ") if "T" in str(from_time) else str(from_time)[:16]
            to_display = to_time[:16].replace("T", " ") if "T" in str(to_time) else str(to_time)[:16]
            lines.append(f"   From: {from_display}")
            lines.append(f"   To: {to_display}")
        lines.append("")

        # Overall Signal Summary
        overall = signal.get("overall", "neutral").upper()
        strength = signal.get("strength", 0)
        emoji = "🟢" if overall == "BULLISH" else ("🔴" if overall == "BEARISH" else "⚪")
        lines.append(f"📈 OVERALL SIGNAL: {emoji} {overall} ({strength:.0f}% confidence)")
        lines.append(f"   Bullish signals: {signal.get('bullish_count', 0)} | Bearish signals: {signal.get('bearish_count', 0)}")
        if signal.get('details'):
            for detail in signal['details'][:3]:
                lines.append(f"   • {detail}")
        lines.append("")

        # ═══════════════════════════════════════════════════════════════
        # MOMENTUM INDICATORS
        # ═══════════════════════════════════════════════════════════════
        momentum_indicators = ["rsi", "stoch_rsi", "stochastic", "macd", "cci", "williams_r", "awesome_osc"]
        has_momentum = any(indicators.get(ind) for ind in momentum_indicators)
        if has_momentum:
            lines.append("📉 MOMENTUM INDICATORS:")
            lines.append("-" * 40)

            # RSI
            rsi = indicators.get("rsi", {})
            if rsi and rsi.get("value") is not None:
                rsi_emoji = "🔴" if rsi.get("signal") == "overbought" else ("🟢" if rsi.get("signal") == "oversold" else "⚪")
                lines.append(f"   RSI(14): {rsi.get('value')} {rsi_emoji} [{rsi.get('signal', 'N/A').upper()}]")

            # Stochastic RSI
            stoch_rsi = indicators.get("stoch_rsi", {})
            if stoch_rsi and stoch_rsi.get("k") is not None:
                lines.append(f"   StochRSI: K={stoch_rsi.get('k')} D={stoch_rsi.get('d')} [{stoch_rsi.get('signal', 'N/A').upper()}]")

            # Stochastic
            stoch = indicators.get("stochastic", {})
            if stoch and stoch.get("k") is not None:
                stoch_emoji = "🔴" if stoch.get("signal") == "overbought" else ("🟢" if stoch.get("signal") == "oversold" else "⚪")
                lines.append(f"   Stochastic: %K={stoch.get('k')} %D={stoch.get('d')} {stoch_emoji} [{stoch.get('signal', 'N/A').upper()}]")

            # MACD
            macd = indicators.get("macd", {})
            if macd and macd.get("macd_line") is not None:
                macd_emoji = "🟢" if macd.get("histogram", 0) > 0 else "🔴"
                lines.append(f"   MACD: Line={macd.get('macd_line'):.4f} Signal={macd.get('signal_line'):.4f} Hist={macd.get('histogram'):.4f} {macd_emoji}")

            # CCI
            cci = indicators.get("cci", {})
            if cci and cci.get("value") is not None:
                cci_emoji = "🔴" if cci.get("signal") == "overbought" else ("🟢" if cci.get("signal") == "oversold" else "⚪")
                lines.append(f"   CCI(20): {cci.get('value')} {cci_emoji} [{cci.get('interpretation', 'N/A')}]")

            # Williams %R
            williams_r = indicators.get("williams_r", {})
            if williams_r and williams_r.get("value") is not None:
                wr_emoji = "🔴" if williams_r.get("signal") == "overbought" else ("🟢" if williams_r.get("signal") == "oversold" else "⚪")
                lines.append(f"   Williams %R(14): {williams_r.get('value')} {wr_emoji} [{williams_r.get('interpretation', 'N/A')}]")

            # Awesome Oscillator
            ao = indicators.get("awesome_osc", {})
            if ao and ao.get("value") is not None:
                ao_emoji = "🟢" if ao.get("value", 0) > 0 else "🔴"
                ao_trend = "↑" if ao.get("increasing") else "↓"
                lines.append(f"   Awesome Osc: {ao.get('value'):.4f} {ao_trend} {ao_emoji} [{ao.get('interpretation', 'N/A')}]")

            lines.append("")

        # ═══════════════════════════════════════════════════════════════
        # TREND INDICATORS
        # ═══════════════════════════════════════════════════════════════
        trend_indicators = ["adx", "supertrend", "ichimoku", "parabolic_sar", "aroon"]
        has_trend = any(indicators.get(ind) for ind in trend_indicators)
        if has_trend:
            lines.append("📈 TREND INDICATORS:")
            lines.append("-" * 40)

            # ADX
            adx = indicators.get("adx", {})
            if adx and adx.get("adx") is not None:
                adx_emoji = "💪" if adx.get("adx", 0) >= 25 else "😐"
                lines.append(f"   ADX(14): {adx.get('adx'):.1f} {adx_emoji} +DI={adx.get('plus_di'):.1f} -DI={adx.get('minus_di'):.1f}")
                lines.append(f"   └─ Trend: {adx.get('signal', 'N/A').upper().replace('_', ' ')}")

            # Supertrend
            supertrend = indicators.get("supertrend", {})
            if supertrend and supertrend.get("value") is not None:
                st_emoji = "🟢" if supertrend.get("direction") == "bullish" else "🔴"
                lines.append(f"   Supertrend: {supertrend.get('value'):.2f} {st_emoji} [{supertrend.get('direction', 'N/A').upper()}]")

            # Parabolic SAR
            psar = indicators.get("parabolic_sar", {})
            if psar and psar.get("value") is not None:
                psar_emoji = "🟢" if psar.get("direction") == "bullish" else "🔴"
                trend_change = " ⚠️REVERSAL" if psar.get("trend_changed") else ""
                lines.append(f"   Parabolic SAR: {psar.get('value'):.2f} {psar_emoji} [{psar.get('interpretation', 'N/A')}]{trend_change}")

            # Aroon
            aroon = indicators.get("aroon", {})
            if aroon and aroon.get("aroon_up") is not None:
                aroon_emoji = "🟢" if aroon.get("oscillator", 0) > 0 else "🔴"
                lines.append(f"   Aroon(25): Up={aroon.get('aroon_up'):.0f} Down={aroon.get('aroon_down'):.0f} Osc={aroon.get('oscillator'):.0f} {aroon_emoji}")
                lines.append(f"   └─ [{aroon.get('interpretation', 'N/A')}]")

            # Ichimoku
            ichimoku = indicators.get("ichimoku", {})
            if ichimoku and ichimoku.get("tenkan") is not None:
                ich_emoji = "🟢" if "bullish" in ichimoku.get("signal", "").lower() else ("🔴" if "bearish" in ichimoku.get("signal", "").lower() else "⚪")
                lines.append(f"   Ichimoku: {ich_emoji} [{ichimoku.get('signal', 'N/A').upper().replace('_', ' ')}]")
                lines.append(f"   └─ Tenkan={ichimoku.get('tenkan'):.2f} Kijun={ichimoku.get('kijun'):.2f}")
                lines.append(f"   └─ Cloud: {ichimoku.get('cloud_bottom'):.2f} - {ichimoku.get('cloud_top'):.2f}")
                lines.append(f"   └─ Price vs Cloud: {ichimoku.get('price_vs_cloud', 'N/A').upper()}")

            lines.append("")

        # ═══════════════════════════════════════════════════════════════
        # VOLATILITY INDICATORS
        # ═══════════════════════════════════════════════════════════════
        volatility_indicators = ["atr", "bollinger", "keltner", "donchian"]
        has_volatility = any(indicators.get(ind) for ind in volatility_indicators)
        if has_volatility:
            lines.append("📊 VOLATILITY INDICATORS:")
            lines.append("-" * 40)

            # ATR
            atr = indicators.get("atr", {})
            if atr and atr.get("value") is not None:
                vol_emoji = "🔥" if atr.get("volatility") == "high" else ("📈" if atr.get("volatility") == "medium" else "😴")
                lines.append(f"   ATR(14): {atr.get('value'):.4f} ({atr.get('percent'):.2f}%) {vol_emoji} [{atr.get('volatility', 'N/A').upper()}]")

            # Bollinger Bands
            bb = indicators.get("bollinger", {})
            if bb and bb.get("upper") is not None:
                bb_emoji = "🔴" if bb.get("signal") == "overbought" else ("🟢" if bb.get("signal") == "oversold" else "⚪")
                lines.append(f"   Bollinger Bands: {bb_emoji} [{bb.get('signal', 'N/A').upper().replace('_', ' ')}]")
                lines.append(f"   └─ Upper={bb.get('upper'):.2f} Middle={bb.get('middle'):.2f} Lower={bb.get('lower'):.2f}")
                lines.append(f"   └─ %B={bb.get('percent_b'):.2f}% Bandwidth={bb.get('bandwidth'):.2f}%")

            # Keltner Channels
            keltner = indicators.get("keltner", {})
            if keltner and keltner.get("upper") is not None:
                kc_emoji = "🔴" if keltner.get("signal") == "overbought" else ("🟢" if keltner.get("signal") == "oversold" else "⚪")
                lines.append(f"   Keltner Channels: {kc_emoji} [{keltner.get('interpretation', 'N/A')}]")
                lines.append(f"   └─ Upper={keltner.get('upper'):.2f} Middle={keltner.get('middle'):.2f} Lower={keltner.get('lower'):.2f}")

            # Donchian Channels
            donchian = indicators.get("donchian", {})
            if donchian and donchian.get("upper") is not None:
                dc_emoji = "🚀" if donchian.get("signal") == "breakout_high" else ("💥" if donchian.get("signal") == "breakout_low" else "⚪")
                lines.append(f"   Donchian Channels(20): {dc_emoji} [{donchian.get('interpretation', 'N/A')}]")
                lines.append(f"   └─ Upper={donchian.get('upper'):.2f} Middle={donchian.get('middle'):.2f} Lower={donchian.get('lower'):.2f}")
                lines.append(f"   └─ Width={donchian.get('width'):.4f}")

            lines.append("")

        # ═══════════════════════════════════════════════════════════════
        # VOLUME INDICATORS
        # ═══════════════════════════════════════════════════════════════
        volume_indicators = ["obv", "vwap", "mfi", "cmf", "ad_line"]
        has_volume = any(indicators.get(ind) for ind in volume_indicators)
        if has_volume:
            lines.append("📦 VOLUME INDICATORS:")
            lines.append("-" * 40)

            # OBV
            obv = indicators.get("obv", {})
            if obv and obv.get("value") is not None:
                obv_emoji = "🟢" if obv.get("signal") == "accumulation" else ("🔴" if obv.get("signal") == "distribution" else "⚪")
                lines.append(f"   OBV: {obv.get('value'):,.0f} {obv_emoji} [{obv.get('signal', 'N/A').upper()}]")

            # VWAP
            vwap = indicators.get("vwap", {})
            if vwap and vwap.get("value") is not None:
                vwap_emoji = "🟢" if vwap.get("signal") == "above_vwap" else "🔴"
                lines.append(f"   VWAP: {vwap.get('value'):.2f} {vwap_emoji} Price {vwap.get('signal', 'N/A').replace('_', ' ').upper()}")
                lines.append(f"   └─ Deviation: {vwap.get('deviation'):.2f} ({vwap.get('deviation_percent'):.2f}%)")

            # MFI
            mfi = indicators.get("mfi", {})
            if mfi and mfi.get("value") is not None:
                mfi_emoji = "🔴" if mfi.get("signal") == "overbought" else ("🟢" if mfi.get("signal") == "oversold" else "⚪")
                lines.append(f"   MFI(14): {mfi.get('value'):.1f} {mfi_emoji} [{mfi.get('interpretation', 'N/A')}]")

            # CMF
            cmf = indicators.get("cmf", {})
            if cmf and cmf.get("value") is not None:
                cmf_emoji = "🟢" if cmf.get("signal") == "accumulation" else ("🔴" if cmf.get("signal") == "distribution" else "⚪")
                lines.append(f"   CMF(21): {cmf.get('value'):.4f} {cmf_emoji} [{cmf.get('interpretation', 'N/A')}]")

            # A/D Line
            ad_line = indicators.get("ad_line", {})
            if ad_line and ad_line.get("value") is not None:
                ad_emoji = "🟢" if ad_line.get("signal") == "accumulation" else ("🔴" if ad_line.get("signal") == "distribution" else "⚪")
                lines.append(f"   A/D Line: {ad_line.get('value'):,.0f} {ad_emoji} [{ad_line.get('interpretation', 'N/A')}]")

            lines.append("")

        # ═══════════════════════════════════════════════════════════════
        # MOVING AVERAGES
        # ═══════════════════════════════════════════════════════════════
        ma_indicators = ["sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "ema_50"]
        has_ma = any(indicators.get(ind) for ind in ma_indicators)
        if has_ma:
            lines.append("📏 MOVING AVERAGES:")
            lines.append("-" * 40)

            # SMAs
            sma_20 = indicators.get("sma_20", {})
            sma_50 = indicators.get("sma_50", {})
            sma_200 = indicators.get("sma_200", {})

            if sma_20 and sma_20.get("value"):
                sma20_emoji = "🟢" if sma_20.get("trend") == "above" else "🔴"
                lines.append(f"   SMA(20): {sma_20.get('value'):.2f} {sma20_emoji} Price {sma_20.get('trend', 'N/A').upper()}")
            if sma_50 and sma_50.get("value"):
                sma50_emoji = "🟢" if sma_50.get("trend") == "above" else "🔴"
                lines.append(f"   SMA(50): {sma_50.get('value'):.2f} {sma50_emoji} Price {sma_50.get('trend', 'N/A').upper()}")
            if sma_200 and sma_200.get("value"):
                sma200_emoji = "🟢" if sma_200.get("trend") == "above" else "🔴"
                lines.append(f"   SMA(200): {sma_200.get('value'):.2f} {sma200_emoji} Price {sma_200.get('trend', 'N/A').upper()}")

            # EMAs
            ema_12 = indicators.get("ema_12", {})
            ema_26 = indicators.get("ema_26", {})
            ema_50 = indicators.get("ema_50", {})

            if ema_12 and ema_12.get("value"):
                ema12_emoji = "🟢" if ema_12.get("trend") == "above" else "🔴"
                lines.append(f"   EMA(12): {ema_12.get('value'):.2f} {ema12_emoji} Price {ema_12.get('trend', 'N/A').upper()}")
            if ema_26 and ema_26.get("value"):
                ema26_emoji = "🟢" if ema_26.get("trend") == "above" else "🔴"
                lines.append(f"   EMA(26): {ema_26.get('value'):.2f} {ema26_emoji} Price {ema_26.get('trend', 'N/A').upper()}")
            if ema_50 and ema_50.get("value"):
                ema50_emoji = "🟢" if ema_50.get("trend") == "above" else "🔴"
                lines.append(f"   EMA(50): {ema_50.get('value'):.2f} {ema50_emoji} Price {ema_50.get('trend', 'N/A').upper()}")

            lines.append("")

        # ═══════════════════════════════════════════════════════════════
        # MULTI-TIMEFRAME SUMMARY
        # ═══════════════════════════════════════════════════════════════
        if multi_tf:
            lines.append("⏱️ MULTI-TIMEFRAME ANALYSIS:")
            lines.append("-" * 40)
            for tf, tf_data in multi_tf.items():
                tf_signal = tf_data.get("signal", "N/A").upper()
                tf_emoji = "🟢" if tf_signal == "BULLISH" else ("🔴" if tf_signal == "BEARISH" else "⚪")
                tf_rsi = tf_data.get("rsi", "N/A")
                tf_rsi_str = f"RSI={tf_rsi:.1f}" if isinstance(tf_rsi, (int, float)) else ""
                lines.append(f"   {tf}: {tf_emoji} {tf_signal} ({tf_data.get('strength', 0):.0f}%) {tf_rsi_str}")
            lines.append("")

        # ═══════════════════════════════════════════════════════════════
        # DATA SCOPE NOTE FOR LLM
        # ═══════════════════════════════════════════════════════════════
        lines.append("=" * 55)
        lines.append("⚠️ IMPORTANT NOTE FOR RESPONSE:")
        lines.append(f"• This analysis is based ONLY on {symbol} technical data shown above")
        lines.append(f"• Current price: {self._format_price(price)} USD (as of {timestamp_display})")
        lines.append("• DO NOT make claims about external markets (gold, stocks, altcoins)")
        lines.append("• Always use specific indicator values when providing analysis")
        lines.append("• Cite exact numbers and timeframes from this data")
        lines.append("=" * 55)

        return "\n".join(lines)


# Standalone test
if __name__ == "__main__":
    import asyncio
    import os

    async def test():
        api_key = os.getenv("FMP_API_KEY")
        tool = GetCryptoTechnicalsTool(api_key=api_key)

        print("\n" + "=" * 60)
        print("Testing getCryptoTechnicals Tool (Enhanced)")
        print("=" * 60)

        # Test with BTC
        print("\nTest: BTC 1h timeframe")
        result = await tool.execute(
            symbol="BTC",
            timeframe="1h",
            indicators=["momentum", "trend"],
        )

        if result.status == "success":
            print("✅ Success")
            print(f"Symbol: {result.data['symbol']}")
            print(f"Signal: {result.data['signal']}")
            print(f"\nFormatted Context:")
            print(result.formatted_context)
        else:
            print(f"❌ Error: {result.error}")

    asyncio.run(test())
