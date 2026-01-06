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

    # Available indicator categories
    INDICATOR_CATEGORIES = {
        "momentum": ["rsi", "stoch_rsi", "stochastic", "macd"],
        "trend": ["adx", "supertrend", "ichimoku"],
        "volatility": ["atr", "bollinger"],
        "volume": ["obv", "vwap"],
        "moving_averages": ["sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "ema_50"],
        "all": None,  # All indicators
    }

    # FMP API fallback
    FMP_BASE_URL = "https://financialmodelingprep.com/stable"

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
            ],
            returns={
                "symbol": "string - Normalized symbol",
                "timeframe": "string - Primary timeframe",
                "current_price": "number - Current price",
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

    async def execute(
        self,
        symbol: str,
        timeframe: str = "1h",
        timeframes: Optional[List[str]] = None,
        indicators: Optional[List[str]] = None,
        **kwargs
    ) -> ToolOutput:
        """
        Execute getCryptoTechnicals

        Args:
            symbol: Crypto symbol (any format)
            timeframe: Primary timeframe for analysis
            timeframes: Optional list for multi-TF analysis
            indicators: List of indicator categories to calculate

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

            # Default indicators to all
            if indicators is None or "all" in indicators:
                indicator_categories = list(self.INDICATOR_CATEGORIES.keys())
                indicator_categories.remove("all")
            else:
                indicator_categories = indicators

            self.logger.info(f"  ┌─ 🔧 TOOL: {self.schema.name}")
            self.logger.info(f"  │  Input: {{symbol={original_symbol}→{symbol}, tf={timeframe}}}")

            # Check cache
            cache_key = f"crypto:technicals:{symbol}:{timeframe}:{','.join(sorted(indicator_categories))}"
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

            # Fetch OHLCV data from internal API
            ohlcv_data = await self._fetch_ohlcv_internal(symbol, timeframe)

            if not ohlcv_data or len(ohlcv_data) < 50:
                # Fallback to FMP if internal API fails
                self.logger.warning(f"[{self.schema.name}] Internal API failed, trying FMP fallback")
                ohlcv_data = await self._fetch_ohlcv_fmp(symbol, timeframe)

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

    async def _fetch_ohlcv_internal(
        self, symbol: str, timeframe: str, limit: int = 200
    ) -> Optional[List[Dict]]:
        """Fetch OHLCV data from internal API"""
        try:
            response = await self._fetch_api(
                endpoint="/ohlcv",
                params={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "limit": limit,
                },
            )

            data = response.get("data", []) if isinstance(response, dict) else response

            if data and len(data) > 0:
                # Ensure sorted by time ascending
                sorted_data = sorted(data, key=lambda x: x.get("time", x.get("timestamp", 0)))
                return sorted_data

            return None

        except Exception as e:
            self.logger.warning(f"[OHLCV-Internal] Fetch error for {symbol}: {e}")
            return None

    async def _fetch_ohlcv_fmp(
        self, symbol: str, timeframe: str, limit: int = 200
    ) -> Optional[List[Dict]]:
        """Fallback: Fetch OHLCV data from FMP API"""
        if not self.fmp_api_key:
            return None

        try:
            # Map timeframe to FMP format
            tf_map = {
                "15m": "15min", "1h": "1hour", "4h": "4hour", "1d": "1day"
            }
            fmp_tf = tf_map.get(timeframe, "1hour")

            # Add USD suffix for FMP
            fmp_symbol = f"{symbol}USD"

            url = f"{self.FMP_BASE_URL}/historical-chart/{fmp_tf}/{fmp_symbol}"
            params = {"apikey": self.fmp_api_key}

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            if data and isinstance(data, list) and len(data) > 0:
                # Sort by date ascending
                sorted_data = sorted(data, key=lambda x: x.get("date", ""))
                return sorted_data[-limit:]

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

        # Volatility indicators
        if "volatility" in categories:
            # ATR
            atr_result = technicals.calculate_atr(highs, lows, closes)
            result["atr"] = atr_result

            # Bollinger Bands
            bb_result = technicals.calculate_bollinger(closes)
            result["bollinger"] = bb_result

        # Volume indicators
        if "volume" in categories:
            # OBV
            obv_result = technicals.calculate_obv(closes, volumes)
            result["obv"] = obv_result

            # VWAP
            vwap_result = technicals.calculate_vwap(highs, lows, closes, volumes)
            result["vwap"] = vwap_result

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
                trend_dir = adx.get("trend_direction", "neutral")
                if trend_dir == "bullish":
                    bullish_signals += 1
                    signal_details.append(f"Strong uptrend (ADX={adx_val:.1f})")
                elif trend_dir == "bearish":
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
            sma_50 = indicators.get("sma_50")
            if sma_50 and isinstance(sma_50, (int, float)):
                if current_price > sma_50:
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
                ohlcv = await self._fetch_ohlcv_internal(symbol, tf)
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
        """Build human-readable formatted context for LLM"""
        symbol = data.get("symbol", "Unknown")
        timeframe = data.get("timeframe", "N/A")
        price = data.get("current_price", 0)
        indicators = data.get("indicators", {})
        signal = data.get("signal", {})
        multi_tf = data.get("multi_tf")

        lines = [
            f"📊 CRYPTO TECHNICALS - {symbol} ({timeframe}):",
            "",
            f"💵 Current Price: {self._format_price(price)}" if price else "💵 Price: N/A",
            "",
        ]

        # Overall Signal
        overall = signal.get("overall", "neutral").upper()
        strength = signal.get("strength", 0)
        emoji = "🟢" if overall == "BULLISH" else ("🔴" if overall == "BEARISH" else "⚪")
        lines.append(f"📈 Overall Signal: {emoji} {overall} ({strength:.0f}%)")
        lines.append(f"   Bullish: {signal.get('bullish_count', 0)} | Bearish: {signal.get('bearish_count', 0)}")
        lines.append("")

        # Momentum
        rsi = indicators.get("rsi", {})
        macd = indicators.get("macd", {})
        if rsi or macd:
            lines.append("📉 Momentum:")
            if rsi:
                lines.append(f"   • RSI(14): {rsi.get('value', 'N/A')} - {rsi.get('signal', 'N/A')}")
            if macd:
                lines.append(f"   • MACD: {macd.get('macd', 'N/A'):.4f}" if macd.get('macd') else "   • MACD: N/A")
                lines.append(f"   • Signal: {macd.get('signal', 'N/A'):.4f}" if macd.get('signal') else "   • Signal: N/A")
            lines.append("")

        # Trend
        adx = indicators.get("adx", {})
        supertrend = indicators.get("supertrend", {})
        if adx or supertrend:
            lines.append("📈 Trend:")
            if adx:
                lines.append(f"   • ADX: {adx.get('adx', 'N/A'):.1f} - {adx.get('strength', 'N/A')}" if adx.get('adx') else "   • ADX: N/A")
            if supertrend:
                st_dir = supertrend.get("direction", "N/A").upper()
                lines.append(f"   • Supertrend: {st_dir}")
            lines.append("")

        # Volatility
        atr = indicators.get("atr", {})
        bb = indicators.get("bollinger", {})
        if atr or bb:
            lines.append("📊 Volatility:")
            if atr:
                lines.append(f"   • ATR(14): {atr.get('atr', 'N/A'):.4f}" if atr.get('atr') else "   • ATR: N/A")
            if bb:
                lines.append(f"   • BB %B: {bb.get('percent_b', 'N/A'):.2f}" if bb.get('percent_b') else "   • BB %B: N/A")
            lines.append("")

        # Multi-TF Summary
        if multi_tf:
            lines.append("⏱️ Multi-Timeframe:")
            for tf, tf_data in multi_tf.items():
                tf_signal = tf_data.get("signal", "N/A").upper()
                tf_emoji = "🟢" if tf_signal == "BULLISH" else ("🔴" if tf_signal == "BEARISH" else "⚪")
                lines.append(f"   • {tf}: {tf_emoji} {tf_signal}")

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
