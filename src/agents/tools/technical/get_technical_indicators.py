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
"""

import httpx
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
    DEFAULT_INDICATORS = ["RSI", "MACD", "SMA", "EMA", "BB", "ATR", "STOCH", "ADX", "VWAP", "VOLUME"]

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
                "LLM-friendly explanations, and actionable trading recommendations."
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
                "Actionable trading recommendations"
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
                    description="Analysis timeframe: 1M (30 days), 3M (90 days), 6M (180 days), 1Y (252 days)",
                    required=False,
                    default="3M",
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
                "llm_summary": "string - Human-readable summary for LLM"
            },
            typical_execution_time_ms=2000,
            requires_symbol=True
        )

    async def execute(
        self,
        symbol: str,
        indicators: Optional[List[str]] = None,
        timeframe: str = "3M",
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

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            self.logger.info(f"[{symbol}] SUCCESS ({int(execution_time)}ms)")

            return create_success_output(
                tool_name="getTechnicalIndicators",
                data=result,
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
        rsi_trend = self._calculate_indicator_trend(df, 'rsi_14', lookback=5)
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
            return {"trend": "unknown", "change": None}

        recent = df[column].tail(lookback).dropna()
        if len(recent) < 2:
            return {"trend": "unknown", "change": None}

        start_val = recent.iloc[0]
        end_val = recent.iloc[-1]

        if start_val == 0:
            return {"trend": "unknown", "change": None}

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
            "lookback_periods": lookback
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

        explanations = {
            'overbought': f"RSI={rsi_value:.1f} (>70): OVERBOUGHT - The stock has risen significantly and may be due for a pullback. Consider taking profits or waiting for a better entry.{trend_info}",
            'oversold': f"RSI={rsi_value:.1f} (<30): OVERSOLD - The stock has fallen significantly and may be due for a bounce. Potential buying opportunity if fundamentals are sound.{trend_info}",
            'strong': f"RSI={rsi_value:.1f} (60-70): STRONG MOMENTUM - Bullish momentum is building but approaching overbought territory. Monitor for continuation or reversal.{trend_info}",
            'weak': f"RSI={rsi_value:.1f} (30-40): WEAK MOMENTUM - Bearish momentum present but approaching oversold territory. Watch for potential reversal signals.{trend_info}",
            'neutral': f"RSI={rsi_value:.1f} (40-60): NEUTRAL - No extreme conditions. The stock is trading in a balanced momentum range.{trend_info}"
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
        """Generate LLM-friendly VWAP explanation."""
        if vwap is None:
            return "VWAP data not available."

        position = analysis.get('position', 'unknown')
        diff_pct = analysis.get('diff_pct', 0)

        if diff_pct > 0:
            return f"Price ${price:.2f} is {diff_pct:.1f}% ABOVE VWAP (${vwap:.2f}). Institutional buyers likely paid more - BULLISH sentiment. Good for intraday long positions."
        elif diff_pct < 0:
            return f"Price ${price:.2f} is {abs(diff_pct):.1f}% BELOW VWAP (${vwap:.2f}). Price below institutional average cost - BEARISH sentiment. Consider waiting for VWAP reclaim."
        else:
            return f"Price ${price:.2f} at VWAP (${vwap:.2f}). Fair value zone. Watch for directional breakout."

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

        # Count bullish/bearish signals
        bullish_count = 0
        bearish_count = 0
        total_indicators = 0

        # RSI signal
        if rsi_analysis.get('signal') == 'BUY':
            bullish_count += 1
        elif rsi_analysis.get('signal') == 'SELL':
            bearish_count += 1
        total_indicators += 1

        # MACD signal
        if macd_analysis.get('signal') == 'BULLISH':
            bullish_count += 1
        elif macd_analysis.get('signal') == 'BEARISH':
            bearish_count += 1
        total_indicators += 1

        # Trend signal
        if trend_analysis.get('signal') == 'BULLISH':
            bullish_count += 1
        elif trend_analysis.get('signal') == 'BEARISH':
            bearish_count += 1
        total_indicators += 1

        # Stochastic signal
        if stoch_analysis.get('signal') == 'BUY':
            bullish_count += 1
        elif stoch_analysis.get('signal') == 'SELL':
            bearish_count += 1
        total_indicators += 1

        # ADX with direction
        adx_signal = adx_analysis.get('signal', 'NEUTRAL')
        if adx_signal == 'BULLISH':
            bullish_count += 1
        elif adx_signal == 'BEARISH':
            bearish_count += 1
        total_indicators += 1

        # VWAP signal
        vwap_signal = vwap_analysis.get('signal', 'NEUTRAL')
        if 'BULLISH' in vwap_signal:
            bullish_count += 0.5  # Half weight for VWAP
        elif 'BEARISH' in vwap_signal:
            bearish_count += 0.5
        total_indicators += 0.5

        # OBV signal (divergence is strong signal)
        obv_signal = obv_analysis.get('signal', 'NEUTRAL')
        if obv_signal == 'BULLISH_DIVERGENCE':
            bullish_count += 1.5  # Divergence is strong signal
        elif obv_signal == 'BEARISH_DIVERGENCE':
            bearish_count += 1.5
        elif obv_signal == 'BULLISH':
            bullish_count += 0.5
        elif obv_signal == 'BEARISH':
            bearish_count += 0.5
        total_indicators += 1

        # MA Crossover signal (Golden/Death Cross is strong signal)
        ma_signal = ma_crossovers.get('signal', 'NEUTRAL')
        if ma_signal == 'BULLISH':  # Golden Cross detected
            bullish_count += 1.5  # Strong signal
        elif ma_signal == 'BEARISH':  # Death Cross detected
            bearish_count += 1.5

        # Also consider short-term SMA 20/50 crossover
        sma_20_50 = ma_crossovers.get('sma_20_50_cross', {})
        if isinstance(sma_20_50, dict):
            if sma_20_50.get('type') == 'bullish':
                bullish_count += 0.5
            elif sma_20_50.get('type') == 'bearish':
                bearish_count += 0.5
        total_indicators += 1

        # Calculate bias
        bullish_pct = bullish_count / total_indicators if total_indicators > 0 else 0
        bearish_pct = bearish_count / total_indicators if total_indicators > 0 else 0

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
        short_term = {
            "action": action,
            "timeframe": "1-5 days",
            "entry_zone": f"${current_price * 0.99:.2f} - ${current_price * 1.01:.2f}",
            "stop_loss": f"${nearest_support * 0.98:.2f}",
            "target_1": f"${current_price * 1.03:.2f}" if action == "BUY" else f"${current_price * 0.97:.2f}",
            "target_2": f"${nearest_resistance:.2f}" if action == "BUY" else f"${nearest_support:.2f}",
            "risk_reward": "1:2 minimum recommended"
        }

        # Swing trade recommendation (1-4 weeks)
        swing_trade = {
            "action": action if adx_analysis.get('trend_strength') in ['strong', 'very_strong'] else "WAIT",
            "timeframe": "1-4 weeks",
            "condition": "Enter on pullback to moving average support" if action == "BUY" else "Enter on rally to resistance" if action == "SELL" else "Wait for clearer trend",
            "stop_loss": f"Below SMA-50 or ${nearest_support * 0.97:.2f}",
            "target": f"${nearest_resistance:.2f}" if action == "BUY" else f"${nearest_support:.2f}"
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

        return {
            "overall_action": action,
            "action_strength": action_strength,
            "bullish_signals": int(bullish_count),
            "bearish_signals": int(bearish_count),
            "total_signals": int(total_indicators),
            "confidence_pct": round(max(bullish_pct, bearish_pct) * 100, 1),
            "short_term_trade": short_term,
            "swing_trade": swing_trade,
            "key_levels": key_levels,
            "risk_level": risk_level,
            "caution": "This is technical analysis only. Always consider fundamentals, news, and your risk tolerance before trading."
        }

    def _generate_llm_summary(self, symbol: str, current_price: float, result: Dict) -> str:
        """Generate comprehensive LLM-friendly summary."""
        outlook = result.get('outlook', {})
        rec = result.get('trading_recommendation', {})
        indicators = result.get('indicators', {})

        # Build summary
        lines = [
            f"=== TECHNICAL ANALYSIS SUMMARY: {symbol} ===",
            f"",
            f"PRICE: ${current_price:.2f} | Period: {result.get('timeframe')} ({result.get('analysis_period_days')} days)",
            f"Data Range: {result.get('date_range')}",
            f"",
            f"OVERALL OUTLOOK: {outlook.get('outlook', 'N/A')} (Confidence: {outlook.get('confidence', 0):.0%})",
            f"ACTION: {rec.get('overall_action', 'HOLD')} ({rec.get('action_strength', 'NEUTRAL')})",
            f"",
            f"KEY INDICATORS:",
            f"- RSI (14d): {indicators.get('rsi', {}).get('value', 'N/A')} - {indicators.get('rsi', {}).get('condition', 'N/A').upper()}",
            f"- MACD: {indicators.get('macd', {}).get('signal', 'N/A')} - Histogram: {indicators.get('macd', {}).get('histogram', 'N/A')}",
            f"- Trend: {indicators.get('moving_averages', {}).get('trend', 'N/A')}",
            f"- ADX (14d): {indicators.get('adx', {}).get('adx', 'N/A')} - {indicators.get('adx', {}).get('trend_strength', 'N/A').upper()} trend",
            f"- Stochastic: {indicators.get('stochastic', {}).get('condition', 'N/A').upper()}",
            f"- VWAP: {indicators.get('vwap', {}).get('signal', 'N/A')} (Price vs VWAP: {indicators.get('vwap', {}).get('price_vs_vwap_pct', 'N/A')}%)",
            f"- OBV: {indicators.get('obv', {}).get('signal', 'N/A')} - {indicators.get('obv', {}).get('divergence', 'No divergence') or 'No divergence'}",
            f"- MA Crossover: {indicators.get('ma_crossovers', {}).get('signal', 'N/A')} - Alignment: {indicators.get('ma_crossovers', {}).get('current_alignment', 'N/A').upper() if indicators.get('ma_crossovers', {}).get('current_alignment') else 'N/A'}",
            f"",
            f"TRADING LEVELS:",
            f"- Support: {result.get('support_resistance', {}).get('support_levels', [{}])[0].get('price', 'N/A') if result.get('support_resistance', {}).get('support_levels') else 'N/A'}",
            f"- Resistance: {result.get('support_resistance', {}).get('resistance_levels', [{}])[0].get('price', 'N/A') if result.get('support_resistance', {}).get('resistance_levels') else 'N/A'}",
            f"",
            f"SIGNALS: {', '.join(result.get('signals', [])[:5])}",
            f"",
            f"RECOMMENDATION:",
            f"Short-term (1-5d): {rec.get('short_term_trade', {}).get('action', 'N/A')} | Target: {rec.get('short_term_trade', {}).get('target_1', 'N/A')} | Stop: {rec.get('short_term_trade', {}).get('stop_loss', 'N/A')}",
            f"Swing (1-4w): {rec.get('swing_trade', {}).get('action', 'N/A')} - {rec.get('swing_trade', {}).get('condition', 'N/A')}",
            f"",
            f"Risk Level: {rec.get('risk_level', 'N/A')}",
            f"",
            f"Note: Technical analysis only. Combine with fundamental analysis and risk management."
        ]

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
