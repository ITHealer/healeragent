"""
Get Crypto SMC Analysis Tool

Comprehensive Smart Money Concepts (SMC) analysis for cryptocurrencies.
Uses internal API for OHLCV data and internal SMC calculators.

Features:
- Market Structure (HH/HL/LH/LL, BOS, CHoCH)
- Order Blocks (Bullish/Bearish)
- Fair Value Gaps (FVG/Imbalances)
- Liquidity Zones (Equal Highs/Lows)
- Premium/Discount Zones
- OTE (Optimal Trade Entry)
- Multi-timeframe SMC analysis

Internal API: http://10.10.0.2:20073/api/v1/market/crypto
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime

import httpx

from src.agents.tools.base import ToolOutput, ToolSchema, ToolParameter
from src.agents.tools.crypto.base_crypto_tool import BaseCryptoTool
from src.agents.tools.crypto.calculators import smc


class GetCryptoSMCAnalysisTool(BaseCryptoTool):
    """
    Tool: getCryptoSMCAnalysis

    Comprehensive Smart Money Concepts (SMC) analysis for cryptocurrencies.
    Detects institutional trading patterns and key price zones.

    Features:
    - Market Structure: HH/HL/LH/LL detection, BOS/CHoCH
    - Order Blocks: Bullish/Bearish OB identification
    - Fair Value Gaps: FVG/Imbalance detection
    - Liquidity Zones: Equal highs/lows identification
    - Premium/Discount: Zone classification with OTE
    - Multi-timeframe analysis

    Use when user asks about:
    - SMC analysis / Smart Money
    - Order Blocks
    - Fair Value Gaps (FVG)
    - Market Structure (BOS, CHoCH)
    - Liquidity zones
    - ICT concepts
    """

    CACHE_TTL = 300  # 5 minutes

    # Supported timeframes
    SUPPORTED_TIMEFRAMES = ["15m", "1h", "4h", "1d"]

    # FMP API fallback
    FMP_BASE_URL = "https://financialmodelingprep.com/stable"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize GetCryptoSMCAnalysisTool

        Args:
            api_key: FMP API key (optional, for fallback)
        """
        super().__init__()
        self.fmp_api_key = api_key

        # Define schema
        self.schema = ToolSchema(
            name="getCryptoSMCAnalysis",
            category="crypto",
            description=(
                "Comprehensive Smart Money Concepts (SMC) analysis for cryptocurrencies. "
                "Detects Market Structure (BOS/CHoCH), Order Blocks, Fair Value Gaps, "
                "Liquidity Zones, and Premium/Discount zones. "
                "Use when user asks about SMC, ICT concepts, order blocks, FVG, or market structure."
            ),
            capabilities=[
                "Market Structure: HH/HL/LH/LL pattern detection",
                "Break of Structure (BOS) / Change of Character (CHoCH)",
                "Order Block detection (Bullish/Bearish)",
                "Fair Value Gap (FVG/Imbalance) identification",
                "Liquidity Zone detection (Equal Highs/Lows)",
                "Premium/Discount zone classification",
                "Optimal Trade Entry (OTE) levels",
                "Fibonacci retracement levels",
                "Multi-timeframe SMC analysis",
            ],
            limitations=[
                "Requires sufficient historical data (min 100 candles)",
                "Internal API timeframes: 15m, 1h, 4h, 1d only",
                "SMC patterns are subjective interpretations",
                "5-minute cache for performance",
            ],
            usage_hints=[
                "User asks: 'BTC order blocks' → USE THIS",
                "User asks: 'ETH fair value gaps' → USE THIS",
                "User asks: 'SOL market structure' → USE THIS",
                "User asks: 'SMC analysis Bitcoin' → USE THIS",
                "User asks: 'ICT analysis ETH' → USE THIS",
                "User asks: 'Phân tích SMC BTC' → USE THIS",
                "User wants traditional technicals → DO NOT USE (use getCryptoTechnicals)",
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description=(
                        "Crypto symbol in any format: "
                        "BTC, BTCUSD, BTCUSDT, ETH, SOL, etc."
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
                    description="Multiple timeframes for multi-TF SMC analysis (optional)",
                    required=False,
                    default=None,
                ),
                ToolParameter(
                    name="components",
                    type="array",
                    description=(
                        "SMC components to analyze. "
                        "Options: 'structure', 'order_blocks', 'fvg', 'liquidity', 'zones', 'all'. "
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
                "market_structure": "object - HH/HL/LH/LL, BOS, CHoCH",
                "order_blocks": "object - Bullish/Bearish OBs",
                "fair_value_gaps": "object - FVG/Imbalances",
                "liquidity_zones": "object - EQH/EQL",
                "premium_discount": "object - Zone and OTE levels",
                "bias": "object - Overall trading bias",
                "multi_tf": "object - Multi-timeframe SMC (if requested)",
                "timestamp": "string - Data timestamp",
            },
            typical_execution_time_ms=1200,
            requires_symbol=True,
        )

    def _normalize_crypto_symbol(self, symbol: str) -> str:
        """Normalize crypto symbol to standard format"""
        symbol = symbol.upper().strip()

        if symbol.endswith('USDT'):
            return symbol[:-4]
        elif symbol.endswith('USD'):
            return symbol[:-3]
        elif symbol.endswith('BUSD'):
            return symbol[:-4]
        return symbol

    async def execute(
        self,
        symbol: str,
        timeframe: str = "1h",
        timeframes: Optional[List[str]] = None,
        components: Optional[List[str]] = None,
        **kwargs
    ) -> ToolOutput:
        """
        Execute getCryptoSMCAnalysis

        Args:
            symbol: Crypto symbol (any format)
            timeframe: Primary timeframe for analysis
            timeframes: Optional list for multi-TF analysis
            components: SMC components to analyze

        Returns:
            ToolOutput with comprehensive SMC analysis
        """
        start_time = time.time()
        original_symbol = symbol

        try:
            # Normalize symbol
            symbol = self._normalize_crypto_symbol(symbol)

            # Validate timeframe
            if timeframe not in self.SUPPORTED_TIMEFRAMES:
                timeframe = "1h"

            # Default components to all
            if components is None or "all" in components:
                selected_components = ["structure", "order_blocks", "fvg", "liquidity", "zones"]
            else:
                selected_components = components

            self.logger.info(f"  ┌─ 🔧 TOOL: {self.schema.name}")
            self.logger.info(f"  │  Input: {{symbol={original_symbol}→{symbol}, tf={timeframe}}}")

            # Check cache
            cache_key = f"crypto:smc:{symbol}:{timeframe}:{','.join(sorted(selected_components))}"
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

            if not ohlcv_data or len(ohlcv_data) < 100:
                # Fallback to FMP if internal API fails
                self.logger.warning(f"[{self.schema.name}] Internal API failed, trying FMP fallback")
                ohlcv_data = await self._fetch_ohlcv_fmp(symbol, timeframe)

            if not ohlcv_data or len(ohlcv_data) < 100:
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error=f"Insufficient data for SMC analysis on {symbol}. Need at least 100 candles.",
                    metadata={"symbol": symbol, "original_symbol": original_symbol},
                )

            # Prepare OHLCV for SMC calculators
            ohlcv_list = [
                {
                    "open": c.get("open", 0),
                    "high": c.get("high", 0),
                    "low": c.get("low", 0),
                    "close": c.get("close", 0),
                    "volume": c.get("volume", 0),
                }
                for c in ohlcv_data
            ]

            # Calculate SMC components
            smc_analysis = self._calculate_smc(ohlcv_list, selected_components)

            # Get current price
            current_price = ohlcv_list[-1]["close"] if ohlcv_list else None

            # Multi-timeframe SMC analysis if requested
            multi_tf_data = None
            if timeframes and len(timeframes) > 1:
                multi_tf_data = await self._calculate_multi_tf_smc(
                    symbol, timeframes, selected_components
                )

            # Compile result
            result_data = {
                "symbol": symbol,
                "original_input": original_symbol,
                "timeframe": timeframe,
                "current_price": current_price,
                **smc_analysis,
                "multi_tf": multi_tf_data,
                "candle_count": len(ohlcv_data),
                "timestamp": datetime.now().isoformat(),
            }

            # Cache result
            await self._set_cached_result(cache_key, result_data, ttl=self.CACHE_TTL)

            execution_time = int((time.time() - start_time) * 1000)

            bias = smc_analysis.get("bias", {}).get("direction", "N/A")
            self.logger.info(f"  │  Result: Bias={bias}")
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
                    "components": selected_components,
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
                error=f"Error performing SMC analysis: {str(e)}",
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
            tf_map = {
                "15m": "15min", "1h": "1hour", "4h": "4hour", "1d": "1day"
            }
            fmp_tf = tf_map.get(timeframe, "1hour")
            fmp_symbol = f"{symbol}USD"

            url = f"{self.FMP_BASE_URL}/historical-chart/{fmp_tf}/{fmp_symbol}"
            params = {"apikey": self.fmp_api_key}

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            if data and isinstance(data, list) and len(data) > 0:
                sorted_data = sorted(data, key=lambda x: x.get("date", ""))
                return sorted_data[-limit:]

            return None

        except Exception as e:
            self.logger.warning(f"[OHLCV-FMP] Fetch error for {symbol}: {e}")
            return None

    def _calculate_smc(
        self,
        ohlcv: List[Dict[str, float]],
        components: List[str],
    ) -> Dict[str, Any]:
        """Calculate selected SMC components"""
        result = {}

        # Full SMC analysis using calculator
        full_analysis = smc.analyze_smc(ohlcv)

        # Market Structure
        if "structure" in components:
            structure = smc.detect_market_structure(ohlcv)
            result["market_structure"] = structure

        # Order Blocks
        if "order_blocks" in components:
            order_blocks = smc.detect_order_blocks(ohlcv)
            result["order_blocks"] = order_blocks

        # Fair Value Gaps
        if "fvg" in components:
            fvgs = smc.detect_fair_value_gaps(ohlcv)
            result["fair_value_gaps"] = fvgs

        # Liquidity Zones
        if "liquidity" in components:
            liquidity = smc.detect_liquidity_zones(ohlcv)
            result["liquidity_zones"] = liquidity

        # Premium/Discount Zones
        if "zones" in components:
            zones = smc.calculate_premium_discount(ohlcv)
            result["premium_discount"] = zones

        # Add bias from full analysis
        result["bias"] = full_analysis.get("bias", {
            "direction": "neutral",
            "confidence": 0,
            "reasoning": [],
        })

        return result

    async def _calculate_multi_tf_smc(
        self,
        symbol: str,
        timeframes: List[str],
        components: List[str],
    ) -> Dict[str, Any]:
        """Calculate SMC for multiple timeframes"""
        multi_tf_result = {}

        for tf in timeframes:
            if tf not in self.SUPPORTED_TIMEFRAMES:
                continue

            try:
                ohlcv = await self._fetch_ohlcv_internal(symbol, tf)
                if not ohlcv or len(ohlcv) < 100:
                    ohlcv = await self._fetch_ohlcv_fmp(symbol, tf)

                if ohlcv and len(ohlcv) >= 100:
                    ohlcv_list = [
                        {
                            "open": c.get("open", 0),
                            "high": c.get("high", 0),
                            "low": c.get("low", 0),
                            "close": c.get("close", 0),
                            "volume": c.get("volume", 0),
                        }
                        for c in ohlcv
                    ]

                    # Quick analysis for multi-TF
                    structure = smc.detect_market_structure(ohlcv_list)
                    full_analysis = smc.analyze_smc(ohlcv_list)

                    multi_tf_result[tf] = {
                        "trend": structure.get("trend", "unclear"),
                        "bias": full_analysis.get("bias", {}).get("direction", "neutral"),
                        "last_structure": structure.get("last_structure_break", {}),
                        "bullish_obs": len(structure.get("order_blocks", {}).get("bullish", []) or []),
                        "bearish_obs": len(structure.get("order_blocks", {}).get("bearish", []) or []),
                    }

            except Exception as e:
                self.logger.warning(f"[Multi-TF SMC] Error for {tf}: {e}")
                continue

        return multi_tf_result

    def _build_formatted_context(self, data: Dict[str, Any]) -> str:
        """Build human-readable formatted context for LLM"""
        symbol = data.get("symbol", "Unknown")
        timeframe = data.get("timeframe", "N/A")
        price = data.get("current_price", 0)
        bias = data.get("bias", {})
        structure = data.get("market_structure", {})
        obs = data.get("order_blocks", {})
        fvgs = data.get("fair_value_gaps", {})
        liquidity = data.get("liquidity_zones", {})
        zones = data.get("premium_discount", {})
        multi_tf = data.get("multi_tf")

        lines = [
            f"🎯 SMC ANALYSIS - {symbol} ({timeframe}):",
            "",
            f"💵 Current Price: {self._format_price(price)}" if price else "💵 Price: N/A",
            "",
        ]

        # Trading Bias
        bias_dir = bias.get("direction", "neutral").upper()
        confidence = bias.get("confidence", 0)
        bias_emoji = "🟢" if bias_dir == "BULLISH" else ("🔴" if bias_dir == "BEARISH" else "⚪")
        lines.append(f"📈 Trading Bias: {bias_emoji} {bias_dir} ({confidence:.0f}% confidence)")

        if bias.get("reasoning"):
            for reason in bias["reasoning"][:3]:
                lines.append(f"   • {reason}")
        lines.append("")

        # Market Structure
        if structure:
            trend = structure.get("trend", "unclear").upper()
            trend_emoji = "📈" if trend == "BULLISH" else ("📉" if trend == "BEARISH" else "↔️")
            lines.append(f"📊 Market Structure: {trend_emoji} {trend}")

            swing_highs = structure.get("swing_highs", [])
            swing_lows = structure.get("swing_lows", [])
            lines.append(f"   Swing Highs: {len(swing_highs)} | Swing Lows: {len(swing_lows)}")

            last_break = structure.get("last_structure_break", {})
            if last_break:
                break_type = last_break.get("type", "N/A")
                lines.append(f"   Last Break: {break_type}")
            lines.append("")

        # Order Blocks
        if obs:
            bullish_obs = obs.get("bullish", []) or []
            bearish_obs = obs.get("bearish", []) or []
            lines.append(f"📦 Order Blocks:")
            lines.append(f"   🟢 Bullish OBs: {len(bullish_obs)}")

            for ob in bullish_obs[:2]:
                if isinstance(ob, dict):
                    lines.append(f"      • {self._format_price(ob.get('low', 0))} - {self._format_price(ob.get('high', 0))}")

            lines.append(f"   🔴 Bearish OBs: {len(bearish_obs)}")
            for ob in bearish_obs[:2]:
                if isinstance(ob, dict):
                    lines.append(f"      • {self._format_price(ob.get('low', 0))} - {self._format_price(ob.get('high', 0))}")
            lines.append("")

        # Fair Value Gaps
        if fvgs:
            bullish_fvgs = fvgs.get("bullish", []) or []
            bearish_fvgs = fvgs.get("bearish", []) or []
            lines.append(f"⚡ Fair Value Gaps:")
            lines.append(f"   🟢 Bullish FVGs: {len(bullish_fvgs)}")
            lines.append(f"   🔴 Bearish FVGs: {len(bearish_fvgs)}")
            lines.append("")

        # Liquidity Zones
        if liquidity:
            eqh = liquidity.get("equal_highs", []) or []
            eql = liquidity.get("equal_lows", []) or []
            lines.append(f"💧 Liquidity Zones:")
            lines.append(f"   Equal Highs (EQH): {len(eqh)}")
            lines.append(f"   Equal Lows (EQL): {len(eql)}")
            lines.append("")

        # Premium/Discount
        if zones:
            current_zone = zones.get("current_zone", "N/A")
            zone_emoji = "🔵" if current_zone == "discount" else ("🔴" if current_zone == "premium" else "⚪")
            lines.append(f"💎 Premium/Discount: {zone_emoji} {current_zone.upper()}")

            ote = zones.get("ote", {})
            if ote:
                lines.append(f"   OTE Zone: {self._format_price(ote.get('low', 0))} - {self._format_price(ote.get('high', 0))}")
            lines.append("")

        # Multi-TF Summary
        if multi_tf:
            lines.append("⏱️ Multi-Timeframe SMC:")
            for tf, tf_data in multi_tf.items():
                tf_bias = tf_data.get("bias", "neutral").upper()
                tf_emoji = "🟢" if tf_bias == "BULLISH" else ("🔴" if tf_bias == "BEARISH" else "⚪")
                lines.append(f"   • {tf}: {tf_emoji} {tf_bias} | OBs: {tf_data.get('bullish_obs', 0)}B/{tf_data.get('bearish_obs', 0)}S")

        return "\n".join(lines)


# Standalone test
if __name__ == "__main__":
    import asyncio
    import os

    async def test():
        api_key = os.getenv("FMP_API_KEY")
        tool = GetCryptoSMCAnalysisTool(api_key=api_key)

        print("\n" + "=" * 60)
        print("Testing getCryptoSMCAnalysis Tool")
        print("=" * 60)

        # Test with BTC
        print("\nTest: BTC 1h SMC analysis")
        result = await tool.execute(
            symbol="BTC",
            timeframe="1h",
            components=["structure", "order_blocks", "fvg"],
        )

        if result.status == "success":
            print("✅ Success")
            print(f"Symbol: {result.data['symbol']}")
            print(f"Bias: {result.data['bias']}")
            print(f"\nFormatted Context:")
            print(result.formatted_context)
        else:
            print(f"❌ Error: {result.error}")

    asyncio.run(test())
