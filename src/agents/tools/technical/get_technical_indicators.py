"""
Technical Indicators Tool

Fetches historical price data and calculates technical indicators.
Returns indicator values with buy/sell signals for trading analysis.
"""

import httpx
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

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
    analyze_adx
)


class GetTechnicalIndicatorsTool(BaseTool):
    """
    Technical Indicators Tool using centralized calculations.

    Supports: RSI, MACD, SMA, EMA, Bollinger Bands, ATR, Stochastic, ADX
    """

    FMP_BASE_URL = "https://financialmodelingprep.com/api"
    DEFAULT_INDICATORS = ["RSI", "MACD", "SMA", "EMA", "BB", "ATR", "STOCH", "ADX"]

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
        "STOCH_RSI": "STOCH"
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

        self.schema = ToolSchema(
            name="getTechnicalIndicators",
            category="technical",
            description=(
                "Calculate technical indicators (RSI, MACD, Moving Averages, "
                "Bollinger Bands, Stochastic, ADX) from historical price data. "
                "Returns indicator values with buy/sell signals."
            ),
            capabilities=[
                "RSI - Overbought/oversold detection",
                "MACD - Trend momentum",
                "SMA & EMA - Trend direction",
                "Bollinger Bands - Volatility analysis",
                "Stochastic - Momentum oscillator",
                "ADX - Trend strength",
                "Multiple timeframes (1M to 1Y)"
            ],
            limitations=[
                "Requires minimum 50 data points",
                "One symbol at a time",
                "Historical data may be delayed"
            ],
            usage_hints=[
                "User asks: 'Apple technical indicators' -> symbol=AAPL",
                "User asks: 'TSLA RSI' -> symbol=TSLA, indicators=['RSI']",
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol (e.g., AAPL, TSLA)",
                    required=True,
                    pattern="^[A-Z]{1,7}$"
                ),
                ToolParameter(
                    name="timeframe",
                    type="string",
                    description="Analysis timeframe (1M, 3M, 6M, 1Y)",
                    required=False,
                    default="3M",
                    allowed_values=["1M", "3M", "6M", "1Y"]
                ),
                ToolParameter(
                    name="indicators",
                    type="array",
                    description="Indicators to calculate",
                    required=False,
                    default=["RSI", "MACD", "SMA", "EMA", "BB", "ATR"]
                )
            ],
            returns={
                "symbol": "string",
                "timeframe": "string",
                "indicators": "array",
                "signals": "array",
                "current_price": "number",
                "data_points": "number",
                "timestamp": "string"
            },
            typical_execution_time_ms=1500,
            requires_symbol=True
        )

    async def execute(
        self,
        symbol: str,
        indicators: Optional[List[str]] = None,
        timeframe: str = "3M",
        **kwargs
    ) -> Dict[str, Any]:
        """Execute technical indicators calculation."""
        start_time = datetime.now()
        symbol = symbol.upper()

        # Map timeframe to lookback days
        timeframe_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 252}
        lookback_days = timeframe_map.get(timeframe, 90)

        try:
            # Normalize indicators
            indicators = self._normalize_indicators(indicators)

            self.logger.info(
                f"[getTechnicalIndicators] {symbol} | timeframe={timeframe} | "
                f"indicators={indicators}"
            )

            # Fetch historical data
            historical_data = await self._fetch_historical_data(symbol, lookback_days)

            if not historical_data or len(historical_data) < 50:
                return create_error_output(
                    tool_name="getTechnicalIndicators",
                    error=f"Insufficient data for {symbol} (need 50+ days)",
                    metadata={"symbol": symbol, "data_points": len(historical_data) if historical_data else 0}
                )

            # Build DataFrame
            df = self._build_dataframe(historical_data)

            # Add all technical indicators
            df = add_technical_indicators(df)

            # Get indicator summary
            indicator_values = get_indicator_summary(df)
            indicator_values['current_price'] = indicator_values.pop('price')

            # Build result with requested indicators only
            result = self._build_result(
                symbol=symbol,
                timeframe=timeframe,
                df=df,
                indicator_values=indicator_values,
                requested_indicators=indicators
            )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            self.logger.info(f"[{symbol}] SUCCESS ({int(execution_time)}ms)")

            return create_success_output(
                tool_name="getTechnicalIndicators",
                data=result,
                metadata={
                    "source": "FMP + pandas_ta",
                    "execution_time_ms": int(execution_time),
                    "data_quality": "high" if len(df) > 100 else "medium"
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

    def _build_result(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        indicator_values: Dict[str, Any],
        requested_indicators: List[str]
    ) -> Dict[str, Any]:
        """Build result dictionary with requested indicators."""
        latest = df.iloc[-1]
        current_price = indicator_values.get('current_price', 0)

        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": current_price,
            "data_points": len(df),
            "timestamp": latest["timestamp"].isoformat() if pd.notna(latest["timestamp"]) else datetime.now().isoformat(),
            "indicators": [],
            "signals": []
        }

        # Add requested indicators
        def has_indicator(base: str) -> bool:
            return any(ind.startswith(base) for ind in requested_indicators)

        if has_indicator('RSI'):
            result['rsi_14'] = indicator_values.get('rsi_14')
            result['rsi_analysis'] = analyze_rsi(result['rsi_14'] or 0)
            result['indicators'].append('RSI')

        if has_indicator('MACD'):
            result['macd_line'] = indicator_values.get('macd_line')
            result['macd_signal'] = indicator_values.get('macd_signal')
            result['macd_histogram'] = indicator_values.get('macd_histogram')
            result['macd_analysis'] = analyze_macd(
                result['macd_line'] or 0,
                result['macd_signal'] or 0,
                result['macd_histogram'] or 0
            )
            result['indicators'].append('MACD')

        if has_indicator('SMA'):
            result['sma_20'] = indicator_values.get('sma_20')
            result['sma_50'] = indicator_values.get('sma_50')
            result['sma_200'] = indicator_values.get('sma_200')
            result['indicators'].append('SMA')

        if has_indicator('EMA'):
            result['ema_12'] = indicator_values.get('ema_12')
            result['ema_26'] = indicator_values.get('ema_26')
            result['indicators'].append('EMA')

        if has_indicator('BB'):
            result['bb_upper'] = indicator_values.get('bb_upper')
            result['bb_middle'] = indicator_values.get('bb_middle')
            result['bb_lower'] = indicator_values.get('bb_lower')
            result['bb_analysis'] = analyze_bollinger_bands(
                current_price,
                result['bb_upper'] or 0,
                result['bb_middle'] or 0,
                result['bb_lower'] or 0
            )
            result['indicators'].append('BB')

        if has_indicator('ATR'):
            result['atr_14'] = indicator_values.get('atr_14')
            result['indicators'].append('ATR')

        if has_indicator('STOCH'):
            result['stoch_k'] = indicator_values.get('stoch_k')
            result['stoch_d'] = indicator_values.get('stoch_d')
            result['stoch_analysis'] = analyze_stochastic(
                result['stoch_k'] or 0,
                result['stoch_d'] or 0
            )
            result['indicators'].append('STOCH')

        if has_indicator('ADX'):
            result['adx'] = indicator_values.get('adx')
            result['di_plus'] = indicator_values.get('di_plus')
            result['di_minus'] = indicator_values.get('di_minus')
            result['adx_analysis'] = analyze_adx(
                result['adx'] or 0,
                result['di_plus'] or 0,
                result['di_minus'] or 0
            )
            result['indicators'].append('ADX')

        # Add trend analysis
        result['trend_analysis'] = analyze_trend(
            current_price,
            indicator_values.get('sma_20') or 0,
            indicator_values.get('sma_50') or 0,
            indicator_values.get('sma_200') or 0
        )

        # Generate signals
        result['signals'] = generate_signals(result)

        # Add outlook
        result['outlook'] = generate_outlook(df)

        return result

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

    async def test_tool():
        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            print("FMP_API_KEY not set")
            return

        tool = GetTechnicalIndicatorsTool(api_key=api_key)

        print("\n" + "=" * 60)
        print("Testing getTechnicalIndicators Tool")
        print("=" * 60)

        result = await tool.safe_execute(symbol="AAPL", timeframe="3M")

        if result.is_success():
            print("SUCCESS")
            print(f"Indicators: {result.data['indicators']}")
            print(f"Signals: {result.data['signals']}")
            print(f"Outlook: {result.data['outlook']}")
            print(f"RSI: {result.data.get('rsi_14')}")
            print(f"MACD: {result.data.get('macd_histogram')}")
            print(f"Trend: {result.data.get('trend_analysis', {}).get('trend')}")
        else:
            print(f"ERROR: {result.error}")

    asyncio.run(test_tool())
