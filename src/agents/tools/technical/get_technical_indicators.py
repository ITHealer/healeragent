"""
Technical Indicators Tool

Fetches historical price data and calculates comprehensive technical indicators.
Returns indicator values with detailed analysis, LLM-friendly explanations,
and actionable trading recommendations.

Key Features:
- 9 core indicators: RSI, MACD, SMA, EMA, BB, ATR, STOCH, ADX, VWAP
- Clear calculation periods specified for each indicator
- LLM-friendly text explanations for model understanding
- Actionable trading recommendations (short-term and long-term)
- Support/Resistance levels and chart patterns
- Economic/Macro context (Treasury rates, GDP, CPI, Unemployment)
"""

import httpx
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import pandas as pd

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    create_success_output,
    create_error_output
)
from src.agents.tools.technical.indicator_calculations import (
    add_technical_indicators,
    get_indicator_summary,
    generate_signals,
    generate_outlook,
    identify_support_levels,
    identify_resistance_levels,
    calculate_pivot_points,
    identify_chart_patterns,
    analyze_rsi,
    analyze_macd,
    analyze_bollinger_bands,
    analyze_stochastic,
    analyze_trend,
    analyze_adx,
    analyze_vwap,
    analyze_volume,
    analyze_obv,
    detect_ma_crossovers
)
from src.agents.tools.technical.technical_constants import TECHNICAL_CONFIG
from src.helpers.redis_cache import get_redis_client_llm


class GetTechnicalIndicatorsTool(BaseTool):
    """
    Comprehensive Technical Indicators Tool.

    Returns ALL indicators by default for complete analysis.
    Each indicator includes:
    - Numeric values
    - Calculation period (in days where applicable)
    - LLM-friendly explanation
    - Signal interpretation (BUY/SELL/NEUTRAL)
    """

    FMP_BASE_URL = "https://financialmodelingprep.com/api"
    FMP_STABLE_URL = "https://financialmodelingprep.com/stable"  # For macro/economic data
    DEFAULT_INDICATORS = ["RSI", "MACD", "SMA", "EMA", "BB", "ATR", "STOCH", "ADX", "VWAP", "VOLUME"]
    MACRO_CACHE_TTL = 3600  # 1 hour - macro data doesn't change frequently

    # Indicator aliases for flexible input
    INDICATOR_ALIASES = {
        "BOLLINGER": "BB",
        "BOLLINGER_BANDS": "BB",
        "BOLLINGERBANDS": "BB",
        "BOLL": "BB",
        "RSI_14": "RSI",
        "RSI14": "RSI",
        "SIMPLE_MOVING_AVERAGE": "SMA",
        "EXPONENTIAL_MOVING_AVERAGE": "EMA",
        "ATR_14": "ATR",
        "AVERAGE_TRUE_RANGE": "ATR",
        "STOCHASTIC": "STOCH",
        "STOCH_RSI": "STOCH",
        "VOL": "VOLUME"
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
        self.cfg = TECHNICAL_CONFIG

        self.schema = ToolSchema(
            name="getTechnicalIndicators",
            category="technical",
            description=(
                "Calculate comprehensive technical indicators from historical price data. "
                "Returns ALL indicators with detailed analysis, calculation periods, "
                "LLM-friendly explanations, actionable trading recommendations, "
                "AND economic/macro context (Treasury rates, GDP, CPI, unemployment)."
            ),
            capabilities=[
                "RSI (14-day) - Overbought/oversold momentum detection",
                "MACD (12/26/9) - Trend momentum and crossover signals",
                "SMA (20/50/200-day) - Trend direction and support/resistance",
                "EMA (12/26-day) - Responsive trend following",
                "Bollinger Bands (20-day, 2 std) - Volatility and price extremes",
                "Stochastic (14/3) - Momentum oscillator",
                "ADX (14-day) - Trend strength measurement",
                "ATR (14-day) - Volatility measurement",
                "VWAP - Institutional fair value indicator",
                "Volume Analysis - Buying/selling pressure",
                "Support/Resistance levels",
                "Chart pattern detection",
                "Actionable trading recommendations",
                "Economic context - Treasury rates, GDP, CPI, Unemployment (auto-included)"
            ],
            limitations=[
                "Requires minimum 50 data points for accurate calculations",
                "One symbol at a time",
                "Historical data may have 15-min delay"
            ],
            usage_hints=[
                "User asks: 'Apple technical analysis' -> symbol=AAPL",
                "User asks: 'TSLA RSI and MACD' -> symbol=TSLA (returns all indicators)",
                "User asks: 'Should I buy NVDA?' -> symbol=NVDA, analyze recommendations"
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock/ETF ticker symbol (e.g., AAPL, TSLA, SPY)",
                    required=True,
                    pattern="^[A-Z]{1,7}$"
                ),
                ToolParameter(
                    name="timeframe",
                    type="string",
                    description="Analysis timeframe: 1M (30 days), 3M (90 days), 6M (180 days), 1Y (252 days). Default 1Y for SMA200 calculation.",
                    required=False,
                    default="1Y",
                    allowed_values=["1M", "3M", "6M", "1Y"]
                ),
                ToolParameter(
                    name="indicators",
                    type="array",
                    description="Specific indicators to calculate (default: ALL indicators)",
                    required=False,
                    default=["RSI", "MACD", "SMA", "EMA", "BB", "ATR", "STOCH", "ADX", "VWAP", "VOLUME"]
                )
            ],
            returns={
                "symbol": "string",
                "timeframe": "string",
                "analysis_period_days": "number",
                "indicators": "object - All indicator values with analysis",
                "trading_recommendation": "object - Actionable trading advice",
                "signals": "array - Summary signals",
                "outlook": "object - Overall market outlook",
                "support_resistance": "object - Key price levels",
                "economic_context": "object - Macro data (Treasury, GDP, CPI, Unemployment)",
                "llm_summary": "string - Human-readable summary including macro context"
            },
            typical_execution_time_ms=2000,
            requires_symbol=True
        )

    async def execute(
        self,
        symbol: str,
        indicators: Optional[List[str]] = None,
        timeframe: str = "1Y",
        **kwargs
    ) -> Dict[str, Any]:
        """Execute comprehensive technical indicators calculation."""
        start_time = datetime.now()
        symbol = symbol.upper()

        # Map timeframe to lookback days
        timeframe_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 252}
        lookback_days = timeframe_map.get(timeframe, 90)

        try:
            # Always calculate ALL indicators for comprehensive analysis
            # The indicators param is kept for backward compatibility
            indicators = self._normalize_indicators(indicators)

            self.logger.info(
                f"[getTechnicalIndicators] {symbol} | timeframe={timeframe} ({lookback_days} days)"
            )

            # Fetch historical data
            historical_data = await self._fetch_historical_data(symbol, lookback_days)

            if not historical_data or len(historical_data) < 50:
                return create_error_output(
                    tool_name="getTechnicalIndicators",
                    error=f"Insufficient data for {symbol} (need 50+ days, got {len(historical_data) if historical_data else 0})",
                    metadata={"symbol": symbol, "data_points": len(historical_data) if historical_data else 0}
                )

            # Build DataFrame
            df = self._build_dataframe(historical_data)

            # Add all technical indicators
            df = add_technical_indicators(df)

            # Get indicator summary
            indicator_values = get_indicator_summary(df)
            current_price = indicator_values.get('price', 0)

            # Build comprehensive result
            result = self._build_comprehensive_result(
                symbol=symbol,
                timeframe=timeframe,
                lookback_days=lookback_days,
                df=df,
                indicator_values=indicator_values,
                current_price=current_price
            )

            # Fetch economic/macro data for market context (cached for 1 hour)
            try:
                economic_data = await self._fetch_economic_data()
                if economic_data:
                    result["economic_context"] = economic_data
                    # Add economic context to LLM summary
                    result["llm_summary"] += self._format_economic_context(economic_data)
            except Exception as e:
                self.logger.warning(f"[{symbol}] Economic data fetch failed: {e}")
                result["economic_context"] = {"error": str(e)}

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            self.logger.info(f"[{symbol}] SUCCESS ({int(execution_time)}ms)")

            return create_success_output(
                tool_name="getTechnicalIndicators",
                data=result,
                # CRITICAL: Pass llm_summary as formatted_context for synthesis
                # This ensures ALL indicators are included in the LLM prompt
                formatted_context=result.get("llm_summary", ""),
                metadata={
                    "source": "FMP + pandas_ta",
                    "execution_time_ms": int(execution_time),
                    "data_quality": "high" if len(df) > 100 else "medium",
                    "indicators_calculated": len(result.get('indicators', {}))
                }
            )

        except Exception as e:
            self.logger.error(f"[getTechnicalIndicators] Error: {e}", exc_info=True)
            return create_error_output(
                tool_name="getTechnicalIndicators",
                error=str(e)
            )

    def _normalize_indicators(self, indicators: Optional[List[str]]) -> List[str]:
        """Normalize indicator names using aliases."""
        if not indicators:
            return self.DEFAULT_INDICATORS.copy()

        normalized = []
        for ind in indicators:
            ind_upper = ind.upper().strip()
            normalized_name = self.INDICATOR_ALIASES.get(ind_upper, ind_upper)
            if normalized_name not in normalized:
                normalized.append(normalized_name)

        return normalized

    def _build_dataframe(self, historical_data: List[Dict]) -> pd.DataFrame:
        """Build DataFrame from historical data."""
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
        return df

    def _build_comprehensive_result(
        self,
        symbol: str,
        timeframe: str,
        lookback_days: int,
        df: pd.DataFrame,
        indicator_values: Dict[str, Any],
        current_price: float
    ) -> Dict[str, Any]:
        """Build comprehensive result with all indicators and analysis."""
        latest = df.iloc[-1]
        first_date = df.iloc[0]["timestamp"]
        last_date = df.iloc[-1]["timestamp"]

        # Calculate date range for display
        date_range = f"{first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}"

        # Calculate price context (factual data)
        price_context = self._calculate_price_context(df, current_price)

        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis_period_days": lookback_days,
            "date_range": date_range,
            "current_price": current_price,
            "price_context": price_context,
            "data_points": len(df),
            "timestamp": last_date.isoformat(),
            "indicators": {},
            "signals": [],
            "trading_recommendation": {},
            "support_resistance": {},
            "llm_summary": ""
        }

        # =====================================================================
        # RSI Analysis (14-day period)
        # =====================================================================
        rsi_value = indicator_values.get('rsi_14')
        rsi_analysis = analyze_rsi(rsi_value or 0)
        # NOTE: Column name in DataFrame is 'rsi', not 'rsi_14'
        rsi_trend = self._calculate_indicator_trend(df, 'rsi', lookback=5)
        result['indicators']['rsi'] = {
            "value": rsi_value,
            "period_days": self.cfg.RSI_PERIOD,
            "signal": rsi_analysis.get('signal'),
            "condition": rsi_analysis.get('condition'),
            "trend": rsi_trend.get('trend'),
            "trend_change_pct": rsi_trend.get('change_pct'),
            "thresholds": {
                "overbought": self.cfg.RSI_OVERBOUGHT,
                "oversold": self.cfg.RSI_OVERSOLD
            },
            "explanation": self._get_rsi_explanation(rsi_value, rsi_analysis, rsi_trend)
        }

        # =====================================================================
        # MACD Analysis (12/26/9 periods)
        # =====================================================================
        macd_line = indicator_values.get('macd_line')
        macd_signal = indicator_values.get('macd_signal')
        macd_histogram = indicator_values.get('macd_histogram')
        macd_analysis = analyze_macd(macd_line or 0, macd_signal or 0, macd_histogram or 0)
        histogram_trend = self._calculate_indicator_trend(df, 'macd_histogram', lookback=5)
        result['indicators']['macd'] = {
            "macd_line": macd_line,
            "signal_line": macd_signal,
            "histogram": macd_histogram,
            "histogram_trend": histogram_trend.get('trend'),
            "periods": {
                "fast": self.cfg.MACD_FAST_PERIOD,
                "slow": self.cfg.MACD_SLOW_PERIOD,
                "signal": self.cfg.MACD_SIGNAL_PERIOD
            },
            "signal": macd_analysis.get('signal'),
            "crossover": macd_analysis.get('crossover'),
            "explanation": self._get_macd_explanation(macd_line, macd_signal, macd_histogram, macd_analysis, histogram_trend)
        }

        # =====================================================================
        # Moving Averages Analysis (SMA 20/50/200, EMA 12/26)
        # =====================================================================
        sma_20 = indicator_values.get('sma_20')
        sma_50 = indicator_values.get('sma_50')
        sma_200 = indicator_values.get('sma_200')
        ema_12 = indicator_values.get('ema_12')
        ema_26 = indicator_values.get('ema_26')
        trend_analysis = analyze_trend(current_price, sma_20 or 0, sma_50 or 0, sma_200 or 0)

        result['indicators']['moving_averages'] = {
            "sma": {
                "sma_20": {"value": sma_20, "period_days": self.cfg.SMA_SHORT_PERIOD},
                "sma_50": {"value": sma_50, "period_days": self.cfg.SMA_MEDIUM_PERIOD},
                "sma_200": {"value": sma_200, "period_days": self.cfg.SMA_LONG_PERIOD}
            },
            "ema": {
                "ema_12": {"value": ema_12, "period_days": self.cfg.EMA_FAST_PERIOD},
                "ema_26": {"value": ema_26, "period_days": self.cfg.EMA_SLOW_PERIOD}
            },
            "price_position": {
                "above_sma_20": current_price > sma_20 if sma_20 else None,
                "above_sma_50": current_price > sma_50 if sma_50 else None,
                "above_sma_200": current_price > sma_200 if sma_200 else None
            },
            "trend": trend_analysis.get('trend'),
            "signal": trend_analysis.get('signal'),
            "explanation": self._get_ma_explanation(current_price, sma_20, sma_50, sma_200, trend_analysis)
        }

        # =====================================================================
        # Bollinger Bands Analysis (20-day, 2 std dev)
        # =====================================================================
        bb_upper = indicator_values.get('bb_upper')
        bb_middle = indicator_values.get('bb_middle')
        bb_lower = indicator_values.get('bb_lower')
        bb_analysis = analyze_bollinger_bands(current_price, bb_upper or 0, bb_middle or 0, bb_lower or 0)
        result['indicators']['bollinger_bands'] = {
            "upper": bb_upper,
            "middle": bb_middle,
            "lower": bb_lower,
            "period_days": self.cfg.BOLLINGER_PERIOD,
            "std_dev": self.cfg.BOLLINGER_STD_DEV,
            "signal": bb_analysis.get('signal'),
            "position": bb_analysis.get('position'),
            "squeeze": bb_analysis.get('squeeze', False),
            "explanation": self._get_bb_explanation(current_price, bb_upper, bb_middle, bb_lower, bb_analysis)
        }

        # =====================================================================
        # Stochastic Oscillator Analysis (14/3 periods)
        # =====================================================================
        stoch_k = indicator_values.get('stoch_k')
        stoch_d = indicator_values.get('stoch_d')
        stoch_analysis = analyze_stochastic(stoch_k or 0, stoch_d or 0)
        result['indicators']['stochastic'] = {
            "k": stoch_k,
            "d": stoch_d,
            "periods": {
                "k_period": self.cfg.STOCH_K_PERIOD,
                "d_period": self.cfg.STOCH_D_PERIOD
            },
            "thresholds": {
                "overbought": self.cfg.STOCH_OVERBOUGHT,
                "oversold": self.cfg.STOCH_OVERSOLD
            },
            "signal": stoch_analysis.get('signal'),
            "condition": stoch_analysis.get('condition'),
            "crossover": stoch_analysis.get('crossover'),
            "explanation": self._get_stoch_explanation(stoch_k, stoch_d, stoch_analysis)
        }

        # =====================================================================
        # ADX Analysis (14-day period)
        # =====================================================================
        adx = indicator_values.get('adx')
        di_plus = indicator_values.get('di_plus')
        di_minus = indicator_values.get('di_minus')
        adx_analysis = analyze_adx(adx or 0, di_plus or 0, di_minus or 0)
        result['indicators']['adx'] = {
            "adx": adx,
            "di_plus": di_plus,
            "di_minus": di_minus,
            "period_days": self.cfg.ADX_PERIOD,
            "thresholds": {
                "strong_trend": self.cfg.ADX_STRONG_TREND,
                "very_strong_trend": self.cfg.ADX_VERY_STRONG_TREND
            },
            "trend_strength": adx_analysis.get('trend_strength'),
            "direction": adx_analysis.get('direction'),
            "signal": adx_analysis.get('signal'),
            "explanation": self._get_adx_explanation(adx, di_plus, di_minus, adx_analysis)
        }

        # =====================================================================
        # ATR Analysis (14-day period)
        # =====================================================================
        atr = indicator_values.get('atr_14')
        atr_pct = (atr / current_price * 100) if atr and current_price else 0
        result['indicators']['atr'] = {
            "value": atr,
            "period_days": self.cfg.ATR_PERIOD,
            "atr_percent": round(atr_pct, 2) if atr_pct else None,
            "explanation": self._get_atr_explanation(atr, atr_pct, current_price)
        }

        # =====================================================================
        # VWAP Analysis (cumulative from period start)
        # =====================================================================
        vwap = indicator_values.get('vwap')
        vwap_analysis = analyze_vwap(current_price, vwap or 0)
        result['indicators']['vwap'] = {
            "value": vwap,
            "calculation": f"Cumulative VWAP from {first_date.strftime('%Y-%m-%d')}",
            "price_vs_vwap_pct": vwap_analysis.get('diff_pct'),
            "signal": vwap_analysis.get('signal'),
            "position": vwap_analysis.get('position'),
            "explanation": self._get_vwap_explanation(current_price, vwap, vwap_analysis)
        }

        # =====================================================================
        # Volume Analysis (20-day average)
        # =====================================================================
        volume = indicator_values.get('volume')
        volume_avg = indicator_values.get('volume_sma_20')
        volume_ratio = indicator_values.get('volume_ratio')
        volume_analysis = analyze_volume(volume or 0, volume_avg or 1)
        result['indicators']['volume'] = {
            "current": volume,
            "average_20d": volume_avg,
            "ratio": volume_ratio,
            "period_days": self.cfg.VOLUME_SMA_PERIOD,
            "signal": volume_analysis.get('signal'),
            "explanation": self._get_volume_explanation(volume, volume_avg, volume_ratio, volume_analysis)
        }

        # =====================================================================
        # OBV Analysis (On-Balance Volume)
        # =====================================================================
        obv = indicator_values.get('obv')
        obv_analysis = analyze_obv(df['obv'], df['close'], lookback=14)
        result['indicators']['obv'] = {
            "value": obv,
            "obv_trend": obv_analysis.get('obv_trend'),
            "price_trend": obv_analysis.get('price_trend'),
            "divergence": obv_analysis.get('divergence'),
            "signal": obv_analysis.get('signal'),
            "lookback_days": 14,
            "explanation": self._get_obv_explanation(obv_analysis)
        }

        # =====================================================================
        # MA Crossovers (Golden Cross / Death Cross)
        # =====================================================================
        ma_crossovers = detect_ma_crossovers(df, lookback=10)
        result['indicators']['ma_crossovers'] = {
            "golden_cross": ma_crossovers.get('golden_cross'),
            "death_cross": ma_crossovers.get('death_cross'),
            "sma_20_50_cross": ma_crossovers.get('sma_20_50_cross'),
            "current_alignment": ma_crossovers.get('current_alignment'),
            "signal": ma_crossovers.get('signal'),
            "lookback_days": 10,
            "explanation": self._get_ma_crossover_explanation(ma_crossovers)
        }

        # =====================================================================
        # Support and Resistance Levels
        # =====================================================================
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        support_levels = identify_support_levels(lows, closes, current_price)
        resistance_levels = identify_resistance_levels(highs, closes, current_price)
        pivot_points = calculate_pivot_points(
            float(df.tail(1)['high'].iloc[0]),
            float(df.tail(1)['low'].iloc[0]),
            float(df.tail(1)['close'].iloc[0])
        )

        result['support_resistance'] = {
            "support_levels": support_levels[:3] if support_levels else [],
            "resistance_levels": resistance_levels[:3] if resistance_levels else [],
            "pivot_points": pivot_points,
            "explanation": self._get_sr_explanation(current_price, support_levels, resistance_levels)
        }

        # =====================================================================
        # Chart Patterns
        # =====================================================================
        patterns = identify_chart_patterns(df)
        result['chart_patterns'] = {
            "detected": patterns,
            "explanation": self._get_pattern_explanation(patterns)
        }

        # =====================================================================
        # Generate Signals and Outlook
        # =====================================================================
        result['signals'] = generate_signals(result['indicators'])
        result['outlook'] = generate_outlook(df)

        # =====================================================================
        # Trading Recommendation (Actionable Insights)
        # =====================================================================
        result['trading_recommendation'] = self._generate_trading_recommendation(
            symbol=symbol,
            current_price=current_price,
            rsi_analysis=rsi_analysis,
            macd_analysis=macd_analysis,
            trend_analysis=trend_analysis,
            stoch_analysis=stoch_analysis,
            adx_analysis=adx_analysis,
            vwap_analysis=vwap_analysis,
            volume_analysis=volume_analysis,
            obv_analysis=obv_analysis,
            ma_crossovers=ma_crossovers,
            outlook=result['outlook'],
            support_levels=support_levels,
            resistance_levels=resistance_levels
        )

        # =====================================================================
        # LLM Summary (Human-readable comprehensive summary)
        # =====================================================================
        result['llm_summary'] = self._generate_llm_summary(
            symbol=symbol,
            current_price=current_price,
            result=result
        )

        return result

    def _calculate_indicator_trend(self, df: pd.DataFrame, column: str, lookback: int = 5) -> Dict[str, Any]:
        """
        Calculate indicator trend direction (rising/falling/flat).

        Args:
            df: DataFrame with indicator data
            column: Column name to analyze
            lookback: Number of periods to compare

        Returns:
            Dict with trend direction and change data
        """
        if column not in df.columns or len(df) < lookback:
            return {"trend": "unknown", "change": None, "reason": "insufficient_data"}

        # Strategy: Take 2x lookback, drop NaN, then take last lookback points
        # This avoids NaN gaps at the end of the series while getting enough points
        extended_lookback = lookback * 2
        raw_data = df[column].tail(extended_lookback).dropna()

        # If not enough points after dropna, try ffill as fallback
        if len(raw_data) < 2:
            filled_series = df[column].ffill().bfill()  # ffill then bfill for edge cases
            raw_data = filled_series.tail(lookback).dropna()

        if len(raw_data) < 2:
            return {"trend": "unknown", "change": None, "reason": "insufficient_valid_points"}

        # Take the last 'lookback' valid points
        recent = raw_data.tail(lookback) if len(raw_data) >= lookback else raw_data

        start_val = float(recent.iloc[0])
        end_val = float(recent.iloc[-1])

        # For RSI (0-100) or other bounded indicators, use absolute thresholds
        # Check if this looks like an RSI (values between 0-100)
        is_bounded = 0 <= start_val <= 100 and 0 <= end_val <= 100

        if is_bounded:
            # For RSI: use absolute change thresholds (more intuitive)
            abs_change = end_val - start_val
            if abs_change > 5:  # RSI moved up by more than 5 points
                trend = "rising"
            elif abs_change < -5:  # RSI moved down by more than 5 points
                trend = "falling"
            else:
                trend = "flat"
            return {
                "trend": trend,
                "change_abs": round(abs_change, 2),
                "start_value": round(start_val, 2),
                "end_value": round(end_val, 2),
                "lookback_periods": len(recent)
            }

        # For unbounded indicators (MACD histogram, etc), use percentage change
        if abs(start_val) < 0.001:
            # Near-zero: use absolute change
            abs_change = end_val - start_val
            if abs(abs_change) > 0.01:
                trend = "rising" if abs_change > 0 else "falling"
            else:
                trend = "flat"
            return {
                "trend": trend,
                "change_abs": round(abs_change, 4),
                "lookback_periods": len(recent)
            }

        change_pct = (end_val - start_val) / abs(start_val) * 100

        # Determine trend direction
        if change_pct > 3:
            trend = "rising"
        elif change_pct < -3:
            trend = "falling"
        else:
            trend = "flat"

        return {
            "trend": trend,
            "change_pct": round(change_pct, 2),
            "lookback_periods": len(recent)
        }

    def _calculate_price_context(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Calculate factual price context for LLM understanding.

        Provides:
        - Period high/low (from available data)
        - Price change percentages (1-day, 5-day, 20-day)
        - Position within range (percentile)
        """
        if df.empty or current_price == 0:
            return {}

        # Period high/low from the analysis data
        period_high = float(df['high'].max())
        period_low = float(df['low'].min())

        # Price position as percentile within the range
        price_range = period_high - period_low
        range_position_pct = ((current_price - period_low) / price_range * 100) if price_range > 0 else 50.0

        # Price changes over different periods
        price_changes = {}
        closes = df['close']

        # 1-day change
        if len(closes) >= 2:
            prev_close = closes.iloc[-2]
            price_changes['1_day'] = round((current_price - prev_close) / prev_close * 100, 2)

        # 5-day change
        if len(closes) >= 5:
            prev_close_5 = closes.iloc[-5]
            price_changes['5_day'] = round((current_price - prev_close_5) / prev_close_5 * 100, 2)

        # 20-day change
        if len(closes) >= 20:
            prev_close_20 = closes.iloc[-20]
            price_changes['20_day'] = round((current_price - prev_close_20) / prev_close_20 * 100, 2)

        return {
            "period_high": round(period_high, 2),
            "period_low": round(period_low, 2),
            "range_position_pct": round(range_position_pct, 2),
            "price_changes": price_changes,
            "explanation": (
                f"Price ${current_price:.2f} is at {range_position_pct:.0f}% of the period range "
                f"(Low: ${period_low:.2f}, High: ${period_high:.2f}). "
                + (f"1-day: {price_changes.get('1_day', 'N/A'):+.2f}%, " if '1_day' in price_changes else "")
                + (f"5-day: {price_changes.get('5_day', 'N/A'):+.2f}%, " if '5_day' in price_changes else "")
                + (f"20-day: {price_changes.get('20_day', 'N/A'):+.2f}%." if '20_day' in price_changes else "")
            ).rstrip(", ") + "."
        }

    # =========================================================================
    # Explanation Generator Methods
    # =========================================================================

    def _get_rsi_explanation(self, rsi_value: float, analysis: Dict, trend: Dict = None) -> str:
        """Generate LLM-friendly RSI explanation with trend info."""
        if rsi_value is None:
            return "RSI data not available."

        condition = analysis.get('condition', 'unknown')
        trend_info = ""
        if trend and trend.get('trend') != 'unknown':
            trend_dir = trend.get('trend', 'unknown')
            trend_info = f" RSI is {trend_dir} over the last 5 days."

        # More actionable RSI explanations
        explanations = {
            'overbought': f"RSI={rsi_value:.1f} (>70): OVERBOUGHT - Price may be stretched. Watch for pullback or take partial profits. NOT a signal to short immediately - wait for confirmation.{trend_info}",
            'oversold': f"RSI={rsi_value:.1f} (<30): OVERSOLD - Selling pressure extreme. Potential bounce setup, but confirm with volume/price action. NOT a blind buy - could stay oversold.{trend_info}",
            'strong': f"RSI={rsi_value:.1f} (60-70): STRONG MOMENTUM - Bullish but approaching overbought. Existing longs can hold with trailing stop. New longs risky without pullback.{trend_info}",
            'weak': f"RSI={rsi_value:.1f} (30-40): WEAK/NEAR-OVERSOLD - Bearish momentum, approaching oversold zone. Could continue down OR setup for reversal. Wait for RSI divergence or price support confirmation before buying.{trend_info}",
            'neutral': f"RSI={rsi_value:.1f} (40-60): NEUTRAL ZONE - No extreme momentum. Use other indicators (MACD, trend) for direction.{trend_info}"
        }
        return explanations.get(condition, f"RSI={rsi_value:.1f}: Momentum indicator based on 14-day price changes.{trend_info}")

    def _get_macd_explanation(self, macd_line: float, signal_line: float, histogram: float, analysis: Dict, histogram_trend: Dict = None) -> str:
        """Generate LLM-friendly MACD explanation with histogram trend."""
        if macd_line is None:
            return "MACD data not available."

        signal = analysis.get('signal', 'NEUTRAL')
        crossover = analysis.get('crossover', 'none')

        if histogram and histogram > 0:
            momentum = "bullish momentum (histogram positive)"
        else:
            momentum = "bearish momentum (histogram negative)"

        # Add histogram trend info
        trend_info = ""
        if histogram_trend and histogram_trend.get('trend') != 'unknown':
            trend_dir = histogram_trend.get('trend', 'unknown')
            trend_info = f" Histogram is {trend_dir} over the last 5 days."

        crossover_text = {
            'bullish': " Recent bullish crossover (MACD crossed above signal line) - potential BUY signal.",
            'bearish': " Recent bearish crossover (MACD crossed below signal line) - potential SELL signal.",
            'none': ""
        }

        return f"MACD Line={macd_line:.4f}, Signal={signal_line:.4f}, Histogram={histogram:.4f}. Shows {momentum}.{trend_info}{crossover_text.get(crossover, '')}"

    def _get_ma_explanation(self, price: float, sma_20: float, sma_50: float, sma_200: float, analysis: Dict) -> str:
        """Generate LLM-friendly Moving Averages explanation."""
        trend = analysis.get('trend', 'MIXED')
        parts = [f"Current price ${price:.2f}"]

        if sma_20:
            rel = "above" if price > sma_20 else "below"
            parts.append(f"{rel} SMA-20 (${sma_20:.2f})")
        if sma_50:
            rel = "above" if price > sma_50 else "below"
            parts.append(f"{rel} SMA-50 (${sma_50:.2f})")
        if sma_200:
            rel = "above" if price > sma_200 else "below"
            parts.append(f"{rel} SMA-200 (${sma_200:.2f})")

        trend_interpretations = {
            'STRONG_UPTREND': "STRONG UPTREND - Price above all major MAs. Bullish trend intact.",
            'UPTREND': "UPTREND - Price above most MAs. Generally bullish.",
            'MIXED': "MIXED - Price between MAs. No clear trend direction.",
            'DOWNTREND': "DOWNTREND - Price below most/all MAs. Bearish trend."
        }

        return f"{', '.join(parts)}. {trend_interpretations.get(trend, 'Trend analysis inconclusive.')}"

    def _get_bb_explanation(self, price: float, upper: float, middle: float, lower: float, analysis: Dict) -> str:
        """Generate LLM-friendly Bollinger Bands explanation."""
        if upper is None:
            return "Bollinger Bands data not available."

        position = analysis.get('position', 'unknown')
        squeeze = analysis.get('squeeze', False)

        positions = {
            'above_upper': f"Price ${price:.2f} is ABOVE upper band (${upper:.2f}). OVERBOUGHT - potential pullback expected.",
            'below_lower': f"Price ${price:.2f} is BELOW lower band (${lower:.2f}). OVERSOLD - potential bounce expected.",
            'within_bands': f"Price ${price:.2f} is within bands (${lower:.2f} - ${upper:.2f}). Trading in normal range."
        }

        explanation = positions.get(position, f"Price ${price:.2f}, Bands: ${lower:.2f} - ${middle:.2f} - ${upper:.2f}.")

        if squeeze:
            explanation += " SQUEEZE DETECTED - Low volatility, potential breakout imminent."

        return explanation

    def _get_stoch_explanation(self, k: float, d: float, analysis: Dict) -> str:
        """Generate LLM-friendly Stochastic explanation."""
        if k is None:
            return "Stochastic data not available."

        condition = analysis.get('condition', 'neutral')
        crossover = analysis.get('crossover', 'none')

        conditions = {
            'overbought': f"%K={k:.1f}, %D={d:.1f} (>80): OVERBOUGHT - Momentum stretched to upside. Watch for reversal.",
            'oversold': f"%K={k:.1f}, %D={d:.1f} (<20): OVERSOLD - Momentum stretched to downside. Potential bounce.",
            'neutral': f"%K={k:.1f}, %D={d:.1f}: NEUTRAL range. No extreme momentum."
        }

        explanation = conditions.get(condition, f"Stochastic %K={k:.1f}, %D={d:.1f}")

        if crossover == 'bullish':
            explanation += " BULLISH crossover (%K above %D)."
        elif crossover == 'bearish':
            explanation += " BEARISH crossover (%K below %D)."

        return explanation

    def _get_adx_explanation(self, adx: float, di_plus: float, di_minus: float, analysis: Dict) -> str:
        """Generate LLM-friendly ADX explanation."""
        if adx is None:
            return "ADX data not available."

        strength = analysis.get('trend_strength', 'weak')
        direction = analysis.get('direction', 'unknown')

        strengths = {
            'very_strong': f"ADX={adx:.1f} (>50): VERY STRONG TREND. High conviction in current direction.",
            'strong': f"ADX={adx:.1f} (25-50): STRONG TREND. Clear directional movement.",
            'moderate': f"ADX={adx:.1f} (20-25): MODERATE TREND. Developing trend, not fully established.",
            'weak': f"ADX={adx:.1f} (<20): WEAK/NO TREND. Market is ranging, avoid trend-following strategies."
        }

        dir_text = f" Direction: {'BULLISH' if direction == 'bullish' else 'BEARISH'} (+DI={di_plus:.1f}, -DI={di_minus:.1f})."

        return strengths.get(strength, f"ADX={adx:.1f}") + dir_text

    def _get_atr_explanation(self, atr: float, atr_pct: float, price: float) -> str:
        """Generate LLM-friendly ATR explanation."""
        if atr is None:
            return "ATR data not available."

        volatility = "HIGH" if atr_pct > 3 else "MODERATE" if atr_pct > 1.5 else "LOW"
        return f"ATR=${atr:.2f} ({atr_pct:.1f}% of price). {volatility} VOLATILITY. Average daily range over 14 days. Use for stop-loss placement (e.g., 1.5-2x ATR from entry)."

    def _get_vwap_explanation(self, price: float, vwap: float, analysis: Dict) -> str:
        """
        Generate LLM-friendly VWAP explanation.

        NOTE: This is PERIOD VWAP (calculated from daily OHLCV over the analysis window),
        NOT intraday VWAP (which requires minute/tick data). Interpretation differs:
        - Period VWAP = average cost of volume over days/weeks
        - Intraday VWAP = average cost within a single trading day
        """
        if vwap is None:
            return "VWAP data not available."

        position = analysis.get('position', 'unknown')
        diff_pct = analysis.get('diff_pct', 0)

        if diff_pct > 0:
            return f"Price ${price:.2f} is {diff_pct:.1f}% ABOVE period VWAP (${vwap:.2f}). Average cost for recent volume was lower - BULLISH bias. Price premium indicates demand."
        elif diff_pct < 0:
            return f"Price ${price:.2f} is {abs(diff_pct):.1f}% BELOW period VWAP (${vwap:.2f}). Price below volume-weighted average cost - BEARISH bias. Potential value zone if trend reverses."
        else:
            return f"Price ${price:.2f} at period VWAP (${vwap:.2f}). Fair value zone based on recent volume distribution. Watch for directional breakout."

    def _get_volume_explanation(self, volume: int, avg_volume: int, ratio: float, analysis: Dict) -> str:
        """Generate LLM-friendly Volume explanation."""
        if volume is None:
            return "Volume data not available."

        signal = analysis.get('signal', 'NORMAL')

        signals = {
            'STRONG': f"Volume {volume:,} is {ratio:.1f}x average ({avg_volume:,}). VERY HIGH VOLUME - significant activity, confirms price movement.",
            'HIGH': f"Volume {volume:,} is {ratio:.1f}x average ({avg_volume:,}). ABOVE AVERAGE - increased interest.",
            'LOW': f"Volume {volume:,} is {ratio:.1f}x average ({avg_volume:,}). LOW VOLUME - lack of conviction, breakouts may fail.",
            'NORMAL': f"Volume {volume:,} is {ratio:.1f}x average ({avg_volume:,}). NORMAL trading activity."
        }

        return signals.get(signal, f"Volume: {volume:,}, Avg: {avg_volume:,}, Ratio: {ratio:.1f}x")

    def _get_obv_explanation(self, analysis: Dict) -> str:
        """Generate LLM-friendly OBV explanation."""
        if not analysis or analysis.get('trend') == 'unknown':
            return "OBV data not available or insufficient data for analysis."

        signal = analysis.get('signal', 'NEUTRAL')
        obv_trend = analysis.get('obv_trend', 'flat')
        price_trend = analysis.get('price_trend', 'flat')
        divergence = analysis.get('divergence')
        description = analysis.get('description', '')

        base = f"OBV Trend: {obv_trend.upper()}, Price Trend: {price_trend.upper()} (14-day). "

        if divergence == 'bullish':
            return base + "BULLISH DIVERGENCE DETECTED - OBV rising while price falling. This often precedes a price reversal UPWARD. Strong buy signal."
        elif divergence == 'bearish':
            return base + "BEARISH DIVERGENCE DETECTED - OBV falling while price rising. This often precedes a price reversal DOWNWARD. Caution advised."
        elif signal == 'BULLISH':
            return base + "OBV confirms uptrend - accumulation phase. Buyers are in control, volume supports price increase."
        elif signal == 'BEARISH':
            return base + "OBV confirms downtrend - distribution phase. Sellers are in control, volume supports price decrease."
        else:
            return base + "No clear divergence. Watch for OBV to confirm price direction."

    def _get_ma_crossover_explanation(self, analysis: Dict) -> str:
        """Generate LLM-friendly MA Crossover explanation."""
        if not analysis:
            return "MA crossover data not available."

        golden = analysis.get('golden_cross')
        death = analysis.get('death_cross')
        sma_20_50 = analysis.get('sma_20_50_cross')
        alignment = analysis.get('current_alignment')
        description = analysis.get('description', 'No recent crossovers detected')

        parts = []

        if golden and golden.get('detected'):
            parts.append(f"GOLDEN CROSS detected {golden.get('days_ago', 'recently')} days ago (SMA_50 crossed above SMA_200). This is a classic BULLISH signal suggesting potential long-term uptrend.")

        if death and death.get('detected'):
            parts.append(f"DEATH CROSS detected {death.get('days_ago', 'recently')} days ago (SMA_50 crossed below SMA_200). This is a classic BEARISH signal suggesting potential long-term downtrend.")

        if sma_20_50 and isinstance(sma_20_50, dict):
            cross_type = sma_20_50.get('type', '')
            days = sma_20_50.get('days_ago', 'recently')
            if cross_type == 'bullish':
                parts.append(f"Short-term bullish crossover: SMA_20 crossed above SMA_50 ({days} days ago). Near-term momentum is positive.")
            elif cross_type == 'bearish':
                parts.append(f"Short-term bearish crossover: SMA_20 crossed below SMA_50 ({days} days ago). Near-term momentum is negative.")

        if alignment:
            parts.append(f"Current MA alignment: {alignment.upper()} (SMA_50 {'above' if alignment == 'bullish' else 'below'} SMA_200).")

        if not parts:
            return description

        return " ".join(parts)

    def _get_sr_explanation(self, price: float, supports: List, resistances: List) -> str:
        """Generate LLM-friendly Support/Resistance explanation."""
        parts = [f"Current price: ${price:.2f}"]

        if supports:
            nearest_support = supports[0]['price']
            parts.append(f"Nearest support: ${nearest_support:.2f} ({supports[0].get('distance_pct', 0):.1f}% below)")

        if resistances:
            nearest_resistance = resistances[0]['price']
            parts.append(f"Nearest resistance: ${nearest_resistance:.2f} ({resistances[0].get('distance_pct', 0):.1f}% above)")

        return ". ".join(parts) + "."

    def _get_pattern_explanation(self, patterns: Dict) -> str:
        """Generate LLM-friendly chart pattern explanation."""
        detected = []
        for name, pattern in patterns.items():
            if pattern and pattern.get('detected'):
                detected.append(f"{name.replace('_', ' ').title()}: {pattern.get('description', '')}")

        if detected:
            return "PATTERNS DETECTED: " + "; ".join(detected)
        return "No significant chart patterns detected in the analysis period."

    # =========================================================================
    # Trading Recommendation Generator
    # =========================================================================

    def _generate_trading_recommendation(
        self,
        symbol: str,
        current_price: float,
        rsi_analysis: Dict,
        macd_analysis: Dict,
        trend_analysis: Dict,
        stoch_analysis: Dict,
        adx_analysis: Dict,
        vwap_analysis: Dict,
        volume_analysis: Dict,
        obv_analysis: Dict,
        ma_crossovers: Dict,
        outlook: Dict,
        support_levels: List,
        resistance_levels: List
    ) -> Dict[str, Any]:
        """Generate actionable trading recommendations."""

        # =======================================================================
        # EXPLICIT WEIGHTS MAPPING for transparency
        # Each indicator has a base weight, modified by signal strength
        # =======================================================================
        indicator_weights = {
            "RSI": 1.0,           # Standard weight
            "MACD": 1.0,          # Standard weight
            "MA_TREND": 1.0,      # Standard weight
            "STOCH": 1.0,         # Standard weight
            "ADX": 1.0,           # Standard weight
            "VWAP": 0.5,          # Half weight (less reliable without intraday data)
            "OBV": 1.0,           # Base weight (1.5x if divergence)
            "MA_CROSSOVER": 1.0,  # Base weight (1.5x if Golden/Death cross)
        }

        # Track weighted scores
        bullish_count = 0.0
        bearish_count = 0.0
        total_weighted = 0.0

        # Track actual weights used (for transparency)
        weights_used = {}

        # RSI signal
        rsi_weight = indicator_weights["RSI"]
        weights_used["RSI"] = rsi_weight
        if rsi_analysis.get('signal') == 'BUY':
            bullish_count += rsi_weight
        elif rsi_analysis.get('signal') == 'SELL':
            bearish_count += rsi_weight
        total_weighted += rsi_weight

        # MACD signal
        macd_weight = indicator_weights["MACD"]
        weights_used["MACD"] = macd_weight
        if macd_analysis.get('signal') == 'BULLISH':
            bullish_count += macd_weight
        elif macd_analysis.get('signal') == 'BEARISH':
            bearish_count += macd_weight
        total_weighted += macd_weight

        # Trend signal (MA Trend)
        trend_weight = indicator_weights["MA_TREND"]
        weights_used["MA_TREND"] = trend_weight
        if trend_analysis.get('signal') == 'BULLISH':
            bullish_count += trend_weight
        elif trend_analysis.get('signal') == 'BEARISH':
            bearish_count += trend_weight
        total_weighted += trend_weight

        # Stochastic signal
        stoch_weight = indicator_weights["STOCH"]
        weights_used["STOCH"] = stoch_weight
        if stoch_analysis.get('signal') == 'BUY':
            bullish_count += stoch_weight
        elif stoch_analysis.get('signal') == 'SELL':
            bearish_count += stoch_weight
        total_weighted += stoch_weight

        # ADX with direction
        adx_weight = indicator_weights["ADX"]
        weights_used["ADX"] = adx_weight
        adx_signal = adx_analysis.get('signal', 'NEUTRAL')
        if adx_signal == 'BULLISH':
            bullish_count += adx_weight
        elif adx_signal == 'BEARISH':
            bearish_count += adx_weight
        total_weighted += adx_weight

        # VWAP signal (reduced weight - less reliable with daily data)
        vwap_weight = indicator_weights["VWAP"]
        weights_used["VWAP"] = vwap_weight
        vwap_signal = vwap_analysis.get('signal', 'NEUTRAL')
        if 'BULLISH' in vwap_signal:
            bullish_count += vwap_weight
        elif 'BEARISH' in vwap_signal:
            bearish_count += vwap_weight
        total_weighted += vwap_weight

        # OBV signal (divergence = stronger signal)
        obv_signal = obv_analysis.get('signal', 'NEUTRAL')
        if 'DIVERGENCE' in obv_signal:
            obv_weight = 1.5  # Divergence is strong signal
        else:
            obv_weight = indicator_weights["OBV"]
        weights_used["OBV"] = obv_weight
        if obv_signal == 'BULLISH_DIVERGENCE':
            bullish_count += obv_weight
        elif obv_signal == 'BEARISH_DIVERGENCE':
            bearish_count += obv_weight
        elif obv_signal == 'BULLISH':
            bullish_count += obv_weight
        elif obv_signal == 'BEARISH':
            bearish_count += obv_weight
        total_weighted += obv_weight

        # MA Crossover signal
        # IMPORTANT: Distinguish between Golden/Death Cross (50/200) and SMA20/50 crossover
        # - Golden/Death Cross (SMA50 vs SMA200) = 1.5 weight (major signal)
        # - SMA20/50 crossover = 1.0 weight (minor signal)
        ma_signal = ma_crossovers.get('signal', 'NEUTRAL')
        sma_20_50 = ma_crossovers.get('sma_20_50_cross', {})
        golden_cross = ma_crossovers.get('golden_cross', {})
        death_cross = ma_crossovers.get('death_cross', {})

        # Check if we have a TRUE Golden/Death Cross (SMA50/200)
        has_golden_cross = isinstance(golden_cross, dict) and golden_cross.get('detected')
        has_death_cross = isinstance(death_cross, dict) and death_cross.get('detected')

        # Determine MA crossover weight and type based on what's actually detected
        if has_golden_cross or has_death_cross:
            # TRUE Golden/Death Cross (50/200) - strong signal, deserves 1.5x weight
            ma_cross_weight = 1.5
            ma_cross_type = "GOLDEN_CROSS" if has_golden_cross else "DEATH_CROSS"
        elif isinstance(sma_20_50, dict) and sma_20_50.get('type') in ['bullish', 'bearish']:
            # SMA20/50 crossover only - weaker signal, standard 1.0 weight
            ma_cross_weight = 1.0
            ma_cross_type = "SMA20_50"
        else:
            # No crossover detected
            ma_cross_weight = indicator_weights["MA_CROSSOVER"]
            ma_cross_type = "NONE"

        weights_used["MA_CROSSOVER"] = ma_cross_weight
        weights_used["MA_CROSSOVER_TYPE"] = ma_cross_type  # For transparency

        # Apply bullish/bearish count based on actual signals
        if has_golden_cross:
            bullish_count += ma_cross_weight
        elif has_death_cross:
            bearish_count += ma_cross_weight
        elif isinstance(sma_20_50, dict):
            if sma_20_50.get('type') == 'bullish':
                bullish_count += ma_cross_weight
            elif sma_20_50.get('type') == 'bearish':
                bearish_count += ma_cross_weight
        total_weighted += ma_cross_weight

        # Calculate neutral weighted score for audit completeness
        neutral_weighted = total_weighted - bullish_count - bearish_count

        # Calculate bias using weighted totals
        bullish_pct = bullish_count / total_weighted if total_weighted > 0 else 0
        bearish_pct = bearish_count / total_weighted if total_weighted > 0 else 0

        # Determine overall action
        if bullish_pct >= 0.6:
            action = "BUY"
            action_strength = "STRONG" if bullish_pct >= 0.75 else "MODERATE"
        elif bearish_pct >= 0.6:
            action = "SELL"
            action_strength = "STRONG" if bearish_pct >= 0.75 else "MODERATE"
        else:
            action = "HOLD"
            action_strength = "NEUTRAL"

        # Calculate price targets
        nearest_support = support_levels[0]['price'] if support_levels else current_price * 0.95
        nearest_resistance = resistance_levels[0]['price'] if resistance_levels else current_price * 1.05

        # Short-term recommendation (1-5 days)
        # FIX: Different logic for HOLD vs BUY/SELL
        if action == "HOLD":
            # HOLD = No new position, just watch levels
            short_term = {
                "action": "HOLD",
                "timeframe": "1-5 days",
                "reason": "Mixed signals - no clear edge for new positions",
                "watch_for_long": f"Break above ${nearest_resistance:.2f} with volume",
                "watch_for_short": f"Break below ${nearest_support:.2f} with volume",
                "risk_reward": "Wait for better setup"
            }
        elif action == "BUY":
            short_term = {
                "action": "BUY",
                "timeframe": "1-5 days",
                "entry_zone": f"${current_price * 0.99:.2f} - ${current_price * 1.01:.2f}",
                "stop_loss": f"${nearest_support * 0.98:.2f}",
                "target_1": f"${current_price * 1.03:.2f}",
                "target_2": f"${nearest_resistance:.2f}",
                "risk_reward": "1:2 minimum recommended"
            }
        else:  # SELL
            short_term = {
                "action": "SELL/SHORT",
                "timeframe": "1-5 days",
                "entry_zone": f"${current_price * 0.99:.2f} - ${current_price * 1.01:.2f}",
                "stop_loss": f"${nearest_resistance * 1.02:.2f}",
                "target_1": f"${current_price * 0.97:.2f}",
                "target_2": f"${nearest_support:.2f}",
                "risk_reward": "1:2 minimum recommended"
            }

        # Swing trade recommendation (1-4 weeks)
        if action == "HOLD":
            swing_trade = {
                "action": "WAIT",
                "timeframe": "1-4 weeks",
                "condition": "Wait for clearer trend signal (RSI extreme, MA crossover, or breakout)",
                "key_support": f"${nearest_support:.2f}",
                "key_resistance": f"${nearest_resistance:.2f}"
            }
        elif action == "BUY":
            swing_trade = {
                "action": action if adx_analysis.get('trend_strength') in ['strong', 'very_strong'] else "WAIT",
                "timeframe": "1-4 weeks",
                "condition": "Enter on pullback to moving average support",
                "stop_loss": f"Below SMA-50 or ${nearest_support * 0.97:.2f}",
                "target": f"${nearest_resistance:.2f}"
            }
        else:  # SELL
            swing_trade = {
                "action": action if adx_analysis.get('trend_strength') in ['strong', 'very_strong'] else "WAIT",
                "timeframe": "1-4 weeks",
                "condition": "Enter on rally to resistance",
                "stop_loss": f"Above SMA-50 or ${nearest_resistance * 1.03:.2f}",
                "target": f"${nearest_support:.2f}"
            }

        # Key levels to watch
        key_levels = {
            "immediate_support": f"${nearest_support:.2f}",
            "immediate_resistance": f"${nearest_resistance:.2f}",
            "breakout_level": f"Above ${nearest_resistance:.2f} with volume confirms bullish breakout",
            "breakdown_level": f"Below ${nearest_support:.2f} with volume confirms bearish breakdown"
        }

        # Risk assessment
        volume_signal = volume_analysis.get('signal', 'NORMAL')
        risk_level = "LOW" if adx_analysis.get('trend_strength') == 'strong' and volume_signal in ['HIGH', 'STRONG'] else "MODERATE" if adx_analysis.get('trend_strength') in ['moderate', 'strong'] else "HIGH"

        # Signal breakdown for LLM understanding
        signal_breakdown = {
            "bullish_indicators": [],
            "bearish_indicators": [],
            "neutral_indicators": []
        }

        # Categorize each indicator signal
        if rsi_analysis.get('signal') == 'BUY':
            signal_breakdown["bullish_indicators"].append("RSI (oversold)")
        elif rsi_analysis.get('signal') == 'SELL':
            signal_breakdown["bearish_indicators"].append("RSI (overbought)")
        else:
            signal_breakdown["neutral_indicators"].append("RSI")

        if macd_analysis.get('signal') == 'BULLISH':
            signal_breakdown["bullish_indicators"].append("MACD")
        elif macd_analysis.get('signal') == 'BEARISH':
            signal_breakdown["bearish_indicators"].append("MACD")
        else:
            signal_breakdown["neutral_indicators"].append("MACD")

        if trend_analysis.get('signal') == 'BULLISH':
            signal_breakdown["bullish_indicators"].append("MA Trend")
        elif trend_analysis.get('signal') == 'BEARISH':
            signal_breakdown["bearish_indicators"].append("MA Trend")
        else:
            signal_breakdown["neutral_indicators"].append("MA Trend")

        if stoch_analysis.get('signal') == 'BUY':
            signal_breakdown["bullish_indicators"].append("Stochastic")
        elif stoch_analysis.get('signal') == 'SELL':
            signal_breakdown["bearish_indicators"].append("Stochastic")
        else:
            signal_breakdown["neutral_indicators"].append("Stochastic")

        if adx_signal == 'BULLISH':
            signal_breakdown["bullish_indicators"].append("ADX")
        elif adx_signal == 'BEARISH':
            signal_breakdown["bearish_indicators"].append("ADX")
        else:
            signal_breakdown["neutral_indicators"].append("ADX")

        if 'BULLISH' in vwap_signal:
            signal_breakdown["bullish_indicators"].append("VWAP")
        elif 'BEARISH' in vwap_signal:
            signal_breakdown["bearish_indicators"].append("VWAP")
        else:
            signal_breakdown["neutral_indicators"].append("VWAP")

        if 'BULLISH' in obv_signal:
            signal_breakdown["bullish_indicators"].append(f"OBV {'(divergence)' if 'DIVERGENCE' in obv_signal else ''}")
        elif 'BEARISH' in obv_signal:
            signal_breakdown["bearish_indicators"].append(f"OBV {'(divergence)' if 'DIVERGENCE' in obv_signal else ''}")
        else:
            signal_breakdown["neutral_indicators"].append("OBV")

        # Use specific crossover type for transparency
        if has_golden_cross:
            signal_breakdown["bullish_indicators"].append("MA Crossover (Golden Cross)")
        elif has_death_cross:
            signal_breakdown["bearish_indicators"].append("MA Crossover (Death Cross)")
        elif isinstance(sma_20_50, dict) and sma_20_50.get('type') == 'bullish':
            signal_breakdown["bullish_indicators"].append("MA Crossover (SMA20/50)")
        elif isinstance(sma_20_50, dict) and sma_20_50.get('type') == 'bearish':
            signal_breakdown["bearish_indicators"].append("MA Crossover (SMA20/50)")
        else:
            signal_breakdown["neutral_indicators"].append("MA Crossover")

        # Use list lengths for accurate indicator counts (not weighted scores)
        # This ensures consistency between JSON counts and breakdown lists
        num_bullish = len(signal_breakdown["bullish_indicators"])
        num_bearish = len(signal_breakdown["bearish_indicators"])
        num_neutral = len(signal_breakdown["neutral_indicators"])
        total_indicator_count = num_bullish + num_bearish + num_neutral

        return {
            "overall_action": action,
            "action_strength": action_strength,
            # Use list lengths (not weighted counts) for consistency
            "bullish_signals": num_bullish,
            "bearish_signals": num_bearish,
            "neutral_signals": num_neutral,
            "total_signals": total_indicator_count,
            # Explicit weights mapping for full transparency
            "indicator_weights": weights_used,
            # Weighted scores for complete audit trail
            # weighted_bullish + weighted_bearish + weighted_neutral = weighted_total
            "weighted_bullish_score": round(bullish_count, 2),
            "weighted_bearish_score": round(bearish_count, 2),
            "weighted_neutral_score": round(neutral_weighted, 2),
            "weighted_total": round(total_weighted, 2),
            # Signal agreement % based on weighted scores (used for BUY/SELL threshold)
            "signal_agreement_pct": round(max(bullish_pct, bearish_pct) * 100, 1),
            "signal_breakdown": signal_breakdown,
            "short_term_trade": short_term,
            "swing_trade": swing_trade,
            "key_levels": key_levels,
            "risk_level": risk_level,
            "note": "weighted_bullish + weighted_bearish + weighted_neutral = weighted_total. Action threshold: 60% weighted score."
        }

    def _generate_llm_summary(self, symbol: str, current_price: float, result: Dict) -> str:
        """Generate comprehensive LLM-friendly summary with all insights."""
        outlook = result.get('outlook', {})
        rec = result.get('trading_recommendation', {})
        indicators = result.get('indicators', {})
        price_ctx = result.get('price_context', {})
        signals = result.get('signals', [])

        # Build price context string
        price_changes = price_ctx.get('price_changes', {})
        price_change_str = []
        if '1_day' in price_changes:
            price_change_str.append(f"1d: {price_changes['1_day']:+.1f}%")
        if '5_day' in price_changes:
            price_change_str.append(f"5d: {price_changes['5_day']:+.1f}%")
        if '20_day' in price_changes:
            price_change_str.append(f"20d: {price_changes['20_day']:+.1f}%")

        # Build indicator trend summary
        rsi_data = indicators.get('rsi', {})
        macd_data = indicators.get('macd', {})
        rsi_trend = rsi_data.get('trend', 'unknown')
        macd_hist_trend = macd_data.get('histogram_trend', 'unknown')

        # Use signal_breakdown from trading_recommendation for consistency
        # (Don't re-count from signals array - causes mismatch!)
        signal_breakdown = rec.get('signal_breakdown', {})
        bullish_indicators = signal_breakdown.get('bullish_indicators', [])
        bearish_indicators = signal_breakdown.get('bearish_indicators', [])
        neutral_indicators = signal_breakdown.get('neutral_indicators', [])

        # Build summary
        lines = [
            f"=== TECHNICAL ANALYSIS SUMMARY: {symbol} ===",
            f"",
            f"PRICE: ${current_price:.2f} | Period: {result.get('timeframe')} ({result.get('analysis_period_days')} days)",
            f"Data Range: {result.get('date_range')}",
        ]

        # Price context section
        if price_ctx:
            range_pos = price_ctx.get('range_position_pct')
            period_high = price_ctx.get('period_high')
            period_low = price_ctx.get('period_low')
            lines.extend([
                f"",
                f"PRICE CONTEXT:",
                f"- Period Range: ${period_low:.2f} - ${period_high:.2f}",
                f"- Current Position: {range_pos:.0f}% of range (0%=low, 100%=high)",
                f"- Price Changes: {', '.join(price_change_str) if price_change_str else 'N/A'}",
            ])

        # Extract indicator values for cleaner formatting
        rsi_val = rsi_data.get('value')
        rsi_val_str = f"{rsi_val:.1f}" if isinstance(rsi_val, (int, float)) else "N/A"
        rsi_cond = (rsi_data.get('condition') or 'N/A').upper()

        macd_hist = macd_data.get('histogram')
        macd_hist_str = f"{macd_hist:.4f}" if isinstance(macd_hist, (int, float)) else "N/A"

        adx_data = indicators.get('adx', {})
        adx_val = adx_data.get('adx')
        adx_val_str = f"{adx_val:.1f}" if isinstance(adx_val, (int, float)) else "N/A"
        adx_strength = (adx_data.get('trend_strength') or 'N/A').upper()

        stoch_cond = (indicators.get('stochastic', {}).get('condition') or 'N/A').upper()
        vwap_signal = indicators.get('vwap', {}).get('signal', 'N/A')
        vwap_pct = indicators.get('vwap', {}).get('price_vs_vwap_pct', 'N/A')

        obv_sig = indicators.get('obv', {}).get('signal', 'N/A')
        obv_div = indicators.get('obv', {}).get('divergence') or 'No divergence'

        ma_cross_sig = indicators.get('ma_crossovers', {}).get('signal', 'N/A')
        ma_alignment_raw = indicators.get('ma_crossovers', {}).get('current_alignment')
        ma_alignment = (ma_alignment_raw or 'N/A').upper()

        # Explain N/A alignment (usually due to insufficient data for SMA200)
        if ma_alignment == 'N/A':
            data_points = result.get('data_points', 0)
            if data_points < 200:
                ma_alignment = f"N/A (need 200+ days, got {data_points})"
            else:
                ma_alignment = "N/A (SMA200 not available)"

        # Use weighted signal_agreement_pct for accuracy (not simple outlook.confidence which can show 100%)
        # ADX strength shows actual trend strength (weak/moderate/strong/very_strong)
        signal_agree = rec.get('signal_agreement_pct', 50)  # Weighted agreement from all 8 indicators
        weighted_total = rec.get('weighted_total', 8)
        weighted_bull = rec.get('weighted_bullish_score', 0)
        weighted_bear = rec.get('weighted_bearish_score', 0)

        # Show the dominant score
        if weighted_bear > weighted_bull:
            score_str = f"Bearish {weighted_bear:.1f}/{weighted_total:.1f}"
        elif weighted_bull > weighted_bear:
            score_str = f"Bullish {weighted_bull:.1f}/{weighted_total:.1f}"
        else:
            score_str = f"Mixed {max(weighted_bull, weighted_bear):.1f}/{weighted_total:.1f}"

        lines.extend([
            "",
            f"OVERALL OUTLOOK: {outlook.get('outlook', 'N/A')} (Agreement: {signal_agree:.0f}%, Trend Strength: {adx_strength})",
            f"ACTION: {rec.get('overall_action', 'HOLD')} ({rec.get('action_strength', 'NEUTRAL')}) | Weighted: {score_str}",
            "",
            "KEY INDICATORS (with trends):",
            f"- RSI (14d): {rsi_val_str} - {rsi_cond} | Trend: {rsi_trend.upper()}",
            f"- MACD: {macd_data.get('signal', 'N/A')} | Histogram: {macd_hist_str} ({macd_hist_trend.upper()})",
            f"- Trend: {indicators.get('moving_averages', {}).get('trend', 'N/A')}",
            f"- ADX (14d): {adx_val_str} - {adx_strength} trend",
            f"- Stochastic: {stoch_cond}",
            f"- VWAP: {vwap_signal} (Price vs VWAP: {vwap_pct}%)",
            f"- OBV: {obv_sig} - {obv_div}",
            f"- MA Crossover: {ma_cross_sig} | Alignment: {ma_alignment}",
        ])

        # Signal confluence section - use signal_breakdown for consistency
        total_signals = len(bullish_indicators) + len(bearish_indicators) + len(neutral_indicators)
        lines.extend([
            f"",
            f"SIGNAL CONFLUENCE ({total_signals} indicators analyzed):",
            f"- Bullish ({len(bullish_indicators)}): {', '.join(bullish_indicators) if bullish_indicators else 'None'}",
            f"- Bearish ({len(bearish_indicators)}): {', '.join(bearish_indicators) if bearish_indicators else 'None'}",
            f"- Neutral ({len(neutral_indicators)}): {', '.join(neutral_indicators) if neutral_indicators else 'None'}",
            f"- Net bias: {'BULLISH' if len(bullish_indicators) > len(bearish_indicators) else 'BEARISH' if len(bearish_indicators) > len(bullish_indicators) else 'MIXED'}",
        ])

        # Support/Resistance
        sr = result.get('support_resistance', {})
        support_levels = sr.get('support_levels', [])
        resistance_levels = sr.get('resistance_levels', [])
        lines.extend([
            f"",
            f"TRADING LEVELS:",
            f"- Support: ${support_levels[0].get('price', 0):.2f} ({support_levels[0].get('distance_pct', 0):.1f}% below)" if support_levels else "- Support: N/A",
            f"- Resistance: ${resistance_levels[0].get('price', 0):.2f} ({resistance_levels[0].get('distance_pct', 0):.1f}% above)" if resistance_levels else "- Resistance: N/A",
        ])

        # Build recommendation section based on action type
        short_term = rec.get('short_term_trade', {})
        swing = rec.get('swing_trade', {})
        action = rec.get('overall_action', 'HOLD')

        if action == "HOLD":
            lines.extend([
                f"",
                f"RECOMMENDATION: {action} ({rec.get('action_strength', 'NEUTRAL')})",
                f"- Reason: {short_term.get('reason', 'Mixed signals')}",
                f"- Watch for LONG: {short_term.get('watch_for_long', 'N/A')}",
                f"- Watch for SHORT: {short_term.get('watch_for_short', 'N/A')}",
                f"- Swing outlook: {swing.get('condition', 'Wait for clearer signal')}",
            ])
        else:
            lines.extend([
                f"",
                f"RECOMMENDATION: {action} ({rec.get('action_strength', 'NEUTRAL')})",
                f"Short-term (1-5d): Entry: {short_term.get('entry_zone', 'N/A')} | Target: {short_term.get('target_1', 'N/A')} | Stop: {short_term.get('stop_loss', 'N/A')}",
                f"Swing (1-4w): {swing.get('action', 'N/A')} - {swing.get('condition', 'N/A')}",
            ])

        lines.extend([
            f"",
            f"Risk Level: {rec.get('risk_level', 'N/A')} | Signal Agreement: {signal_agree:.0f}%",
            f"",
            f"Note: Technical analysis only. {rec.get('note', 'Combine with fundamental analysis.')}"
        ])

        return "\n".join(lines)

    async def _fetch_historical_data(self, symbol: str, lookback_days: int) -> List[Dict]:
        """Fetch historical data from FMP API."""
        url = f"{self.FMP_BASE_URL}/v3/historical-price-full/{symbol}"
        params = {"apikey": self.api_key, "timeseries": lookback_days}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                return data.get("historical", [])
            return []

    # =========================================================================
    # Economic/Macro Data Methods (for market context)
    # =========================================================================

    async def _fetch_economic_data(self) -> Dict[str, Any]:
        """
        Fetch economic/macro data for market context.
        Data is cached for 1 hour since macro data doesn't change frequently.

        Returns:
            Dict with treasury rates, GDP, CPI, unemployment data
        """
        cache_key = "getTechnicalIndicators_economic_data"

        try:
            # Check Redis cache first
            redis_client = await get_redis_client_llm()
            if redis_client:
                try:
                    cached_bytes = await redis_client.get(cache_key)
                    if cached_bytes:
                        self.logger.info("[Economic Data] Cache HIT")
                        if isinstance(cached_bytes, bytes):
                            return json.loads(cached_bytes.decode('utf-8'))
                        return json.loads(cached_bytes)
                except Exception as e:
                    self.logger.warning(f"[Economic Data] Cache read error: {e}")

            # Fetch fresh data from FMP
            self.logger.info("[Economic Data] Fetching from FMP API...")
            result = {}

            async with httpx.AsyncClient(timeout=15.0) as client:
                # 1. Treasury Rates (FMP stable API)
                try:
                    treasury_url = f"{self.FMP_STABLE_URL}/treasury-rates"
                    response = await client.get(treasury_url, params={"apikey": self.api_key})
                    if response.status_code == 200 and response.text.strip():
                        treasury_data = response.json()
                        if treasury_data and isinstance(treasury_data, list):
                            latest = treasury_data[0]
                            result["treasury"] = {
                                "as_of_date": latest.get("date"),  # Date of observation (market close)
                                "date": latest.get("date"),        # Kept for backwards compat
                                "year_2": latest.get("year2"),
                                "year_5": latest.get("year5"),
                                "year_10": latest.get("year10"),
                                "year_30": latest.get("year30"),
                                "note": "Rates as of market close on as_of_date. May lag 1 business day."
                            }
                            # Yield curve analysis
                            y2 = latest.get("year2")
                            y10 = latest.get("year10")
                            if y2 and y10:
                                spread = y10 - y2
                                result["yield_curve"] = {
                                    "spread_10y_2y": round(spread, 3),
                                    "status": "inverted" if spread < 0 else "flat" if spread < 0.25 else "normal",
                                    "signal": "Recession warning" if spread < 0 else "Caution" if spread < 0.25 else "Normal"
                                }
                    else:
                        self.logger.warning(f"[Economic Data] Treasury API failed: {response.status_code} | {response.text[:200]}")
                except Exception as e:
                    self.logger.warning(f"[Economic Data] Treasury fetch error: {e}")

                # 2. GDP Growth Rate (FMP stable API)
                # Try realGDPGrowth first, then calculate from GDP values
                try:
                    econ_url = f"{self.FMP_STABLE_URL}/economic-indicators"
                    # First try realGDPGrowth which should return growth rate directly
                    response = await client.get(econ_url, params={"name": "realGDPGrowth", "apikey": self.api_key})
                    if response.status_code == 200 and response.text.strip():
                        gdp_data = response.json()
                        if gdp_data and isinstance(gdp_data, list) and len(gdp_data) > 0:
                            latest = gdp_data[0]
                            result["gdp"] = {
                                "value": latest.get("value"),  # This should be growth rate %
                                "as_of_date": latest.get("date"),  # Release date of the data
                                "date": latest.get("date"),        # Kept for backwards compat
                                "type": "growth_rate",
                                "trend": "increasing" if len(gdp_data) > 1 and gdp_data[0].get("value", 0) > gdp_data[1].get("value", 0) else "decreasing",
                                "note": "GDP growth rate (%). Released quarterly, may lag current quarter."
                            }
                        else:
                            # Fallback: fetch GDP and calculate YoY growth
                            response2 = await client.get(econ_url, params={"name": "GDP", "apikey": self.api_key})
                            if response2.status_code == 200 and response2.text.strip():
                                gdp_raw = response2.json()
                                if gdp_raw and isinstance(gdp_raw, list) and len(gdp_raw) >= 5:
                                    # Calculate YoY growth (compare to ~4 quarters ago)
                                    current = gdp_raw[0].get("value", 0)
                                    previous = gdp_raw[4].get("value", 0) if len(gdp_raw) > 4 else gdp_raw[-1].get("value", 0)
                                    if previous > 0:
                                        yoy_growth = ((current - previous) / previous) * 100
                                        result["gdp"] = {
                                            "value": round(yoy_growth, 2),
                                            "as_of_date": gdp_raw[0].get("date"),  # Date of latest GDP reading
                                            "date": gdp_raw[0].get("date"),        # Kept for backwards compat
                                            "type": "calculated_yoy",
                                            "raw_current": current,
                                            "raw_previous": previous,
                                            "trend": "increasing" if yoy_growth > 0 else "decreasing",
                                            "note": "YoY GDP growth (%) calculated from nominal GDP. Released quarterly."
                                        }
                    else:
                        self.logger.warning(f"[Economic Data] GDP API failed: {response.status_code} | {response.text[:200]}")
                except Exception as e:
                    self.logger.warning(f"[Economic Data] GDP fetch error: {e}")

                # 3. Inflation Rate (FMP stable API)
                # Try inflationRate first (returns % directly), fallback to CPI YoY calculation
                try:
                    response = await client.get(econ_url, params={"name": "inflationRate", "apikey": self.api_key})
                    if response.status_code == 200 and response.text.strip():
                        inflation_data = response.json()
                        if inflation_data and isinstance(inflation_data, list) and len(inflation_data) > 0:
                            latest = inflation_data[0]
                            result["inflation"] = {
                                "value": latest.get("value"),  # Should be inflation rate %
                                "as_of_date": latest.get("date"),  # Release date of the data
                                "date": latest.get("date"),        # Kept for backwards compat
                                "type": "inflation_rate",
                                "trend": "increasing" if len(inflation_data) > 1 and inflation_data[0].get("value", 0) > inflation_data[1].get("value", 0) else "decreasing",
                                "note": "Inflation rate (CPI YoY %). Released monthly, typically mid-month for prior month."
                            }
                        else:
                            # Fallback: fetch CPI index and calculate YoY inflation
                            response2 = await client.get(econ_url, params={"name": "CPI", "apikey": self.api_key})
                            if response2.status_code == 200 and response2.text.strip():
                                cpi_raw = response2.json()
                                if cpi_raw and isinstance(cpi_raw, list) and len(cpi_raw) >= 13:
                                    # Calculate YoY inflation (compare to 12 months ago)
                                    current_cpi = cpi_raw[0].get("value", 0)
                                    previous_cpi = cpi_raw[12].get("value", 0) if len(cpi_raw) > 12 else cpi_raw[-1].get("value", 0)
                                    if previous_cpi > 0:
                                        yoy_inflation = ((current_cpi - previous_cpi) / previous_cpi) * 100
                                        result["inflation"] = {
                                            "value": round(yoy_inflation, 2),
                                            "as_of_date": cpi_raw[0].get("date"),  # Date of latest CPI reading
                                            "date": cpi_raw[0].get("date"),        # Kept for backwards compat
                                            "type": "calculated_yoy",
                                            "cpi_current": current_cpi,
                                            "cpi_previous": previous_cpi,
                                            "trend": "increasing" if yoy_inflation > 2.5 else "decreasing",
                                            "note": "YoY inflation (%) calculated from CPI index. Released monthly."
                                        }
                    else:
                        self.logger.warning(f"[Economic Data] Inflation API failed: {response.status_code} | {response.text[:200]}")
                except Exception as e:
                    self.logger.warning(f"[Economic Data] Inflation fetch error: {e}")

                # 4. Unemployment Rate (FMP stable API)
                try:
                    response = await client.get(econ_url, params={"name": "unemploymentRate", "apikey": self.api_key})
                    if response.status_code == 200 and response.text.strip():
                        unemp_data = response.json()
                        if unemp_data and isinstance(unemp_data, list) and len(unemp_data) > 0:
                            latest = unemp_data[0]
                            result["unemployment"] = {
                                "value": latest.get("value"),
                                "as_of_date": latest.get("date"),  # Release date of the data
                                "date": latest.get("date"),        # Kept for backwards compat
                                "trend": "increasing" if len(unemp_data) > 1 and unemp_data[0].get("value", 0) > unemp_data[1].get("value", 0) else "decreasing",
                                "note": "Unemployment rate (%). Released monthly (first Friday of month)."
                            }
                    else:
                        self.logger.warning(f"[Economic Data] Unemployment API failed: {response.status_code} | {response.text[:200]}")
                except Exception as e:
                    self.logger.warning(f"[Economic Data] Unemployment fetch error: {e}")

            # Generate summary
            result["summary"] = self._generate_economic_summary(result)
            result["timestamp"] = datetime.now().isoformat()

            # Cache the result
            if redis_client:
                try:
                    json_string = json.dumps(result)
                    await redis_client.set(cache_key, json_string, ex=self.MACRO_CACHE_TTL)
                    self.logger.info(f"[Economic Data] Cached for {self.MACRO_CACHE_TTL}s")
                except Exception as e:
                    self.logger.warning(f"[Economic Data] Cache write error: {e}")

            # Close Redis connection
            if redis_client:
                try:
                    await redis_client.close()
                except Exception:
                    pass

            return result

        except Exception as e:
            self.logger.error(f"[Economic Data] Error: {e}", exc_info=True)
            return {}

    def _generate_economic_summary(self, data: Dict[str, Any]) -> str:
        """Generate a quick summary of economic conditions."""
        parts = []

        treasury = data.get("treasury", {})
        if treasury.get("year_10"):
            parts.append(f"10Y Treasury: {treasury['year_10']:.2f}%")

        yc = data.get("yield_curve", {})
        if yc.get("status"):
            parts.append(f"Yield curve: {yc['status']}")

        gdp = data.get("gdp", {})
        if gdp.get("value") is not None:
            parts.append(f"GDP Growth: {gdp['value']:.1f}%")

        # Use 'inflation' (new) or fallback to 'cpi' (legacy)
        inflation = data.get("inflation", {}) or data.get("cpi", {})
        if inflation.get("value") is not None:
            parts.append(f"Inflation: {inflation['value']:.1f}%")

        unemp = data.get("unemployment", {})
        if unemp.get("value") is not None:
            parts.append(f"Unemployment: {unemp['value']:.1f}%")

        return " | ".join(parts) if parts else "Economic data unavailable"

    def _format_economic_context(self, econ_data: Dict[str, Any]) -> str:
        """Format economic data as LLM-friendly context."""
        if not econ_data:
            return "Economic context: Data unavailable"

        lines = ["", "MACRO/ECONOMIC CONTEXT:"]

        # Treasury rates
        treasury = econ_data.get("treasury", {})
        if treasury:
            lines.append(f"- Treasury Rates: 2Y={treasury.get('year_2', 'N/A')}%, "
                        f"10Y={treasury.get('year_10', 'N/A')}%, "
                        f"30Y={treasury.get('year_30', 'N/A')}%")

        # Yield curve
        yc = econ_data.get("yield_curve", {})
        if yc:
            lines.append(f"- Yield Curve: {yc.get('status', 'N/A').upper()} "
                        f"(10Y-2Y spread: {yc.get('spread_10y_2y', 'N/A')}%) - {yc.get('signal', '')}")

        # GDP Growth Rate
        gdp = econ_data.get("gdp", {})
        if gdp.get("value") is not None:
            gdp_type = gdp.get('type', 'unknown')
            lines.append(f"- GDP Growth: {gdp['value']:.1f}% ({gdp.get('date', 'N/A')}) [{gdp_type}] - {gdp.get('trend', 'N/A')}")

        # Inflation Rate (use 'inflation' or fallback to 'cpi')
        inflation = econ_data.get("inflation", {}) or econ_data.get("cpi", {})
        if inflation.get("value") is not None:
            inflation_type = inflation.get('type', 'unknown')
            lines.append(f"- Inflation: {inflation['value']:.1f}% ({inflation.get('date', 'N/A')}) [{inflation_type}] - {inflation.get('trend', 'N/A')}")

        # Unemployment Rate
        unemp = econ_data.get("unemployment", {})
        if unemp.get("value") is not None:
            lines.append(f"- Unemployment: {unemp['value']:.1f}% ({unemp.get('date', 'N/A')}) - {unemp.get('trend', 'N/A')}")

        # Implications
        lines.append("")
        lines.append("MACRO IMPLICATIONS:")

        # High rates = pressure on growth stocks
        if treasury.get("year_10") and treasury["year_10"] > 4.0:
            lines.append("- High interest rates (10Y > 4%) → Pressure on growth/tech stocks")
        elif treasury.get("year_10") and treasury["year_10"] < 3.0:
            lines.append("- Low interest rates (10Y < 3%) → Favorable for growth stocks")

        # Yield curve inversion
        if yc.get("status") == "inverted":
            lines.append("- Inverted yield curve → Recession risk elevated, consider defensive positioning")

        # Inflation implications
        if inflation.get("value") is not None and inflation["value"] > 3.0:
            lines.append("- Elevated inflation (> 3%) → Fed likely to maintain restrictive policy")
        elif inflation.get("value") is not None and inflation["value"] < 2.0:
            lines.append("- Low inflation (< 2%) → Fed may consider easing policy")

        return "\n".join(lines)


# =============================================================================
# Standalone Test
# =============================================================================
if __name__ == "__main__":
    import asyncio
    import os
    import json

    async def test_tool():
        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            print("FMP_API_KEY not set")
            return

        tool = GetTechnicalIndicatorsTool(api_key=api_key)

        print("\n" + "=" * 70)
        print("Testing Comprehensive Technical Indicators Tool")
        print("=" * 70)

        result = await tool.safe_execute(symbol="AAPL", timeframe="3M")

        if result.is_success():
            print("\nSUCCESS")
            print("\n" + result.data.get('llm_summary', 'No summary'))
            print("\n" + "=" * 70)
            print("Trading Recommendation:")
            print(json.dumps(result.data.get('trading_recommendation', {}), indent=2))
        else:
            print(f"\nERROR: {result.error}")

    asyncio.run(test_tool())
