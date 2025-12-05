# src/agents/tools/crypto/get_crypto_technicals.py

import json
import time
from typing import Dict, Any, List
from datetime import datetime

import httpx

from src.agents.tools.base import BaseTool, ToolOutput, ToolSchema, ToolParameter
from src.helpers.redis_cache import get_redis_client_llm
from src.utils.logger.custom_logging import LoggerMixin


class GetCryptoTechnicalsTool(BaseTool, LoggerMixin):
    """
    Tool 24: Get Crypto Technical Indicators
    
    Calculate technical indicators for cryptocurrencies (RSI, MACD, Moving Averages, etc.)
    Reuses stock technical calculation logic with crypto OHLCV data
    
    FMP Stable APIs:
    - Intraday: GET https://financialmodelingprep.com/stable/historical-chart/{interval}
    - Daily: GET https://financialmodelingprep.com/stable/historical-price-eod/light
    """

    CACHE_TTL = 900  # 15 minutes

    def __init__(self, api_key: str):
        """
        Initialize GetCryptoTechnicalsTool
        
        Args:
            api_key: FMP API key
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/stable"
        
        # Define schema
        self.schema = ToolSchema(
            name="getCryptoTechnicals",
            category="crypto",
            description=(
                "Calculate technical indicators for cryptocurrencies (RSI, MACD, Moving Averages). "
                "Reuses stock technical logic with crypto OHLCV data. "
                "Symbol must end with 'USD'. "
                "Use when user asks about crypto technical analysis, trading signals, or chart patterns."
            ),
            capabilities=[
                "✅ RSI (Relative Strength Index)",
                "✅ MACD (Moving Average Convergence Divergence)",
                "✅ SMA (Simple Moving Averages)",
                "✅ EMA (Exponential Moving Averages)",
                "✅ Multiple timeframes (1min to daily)",
                "✅ Same indicators as stocks"
            ],
            limitations=[
                "❌ Symbol must end with 'USD'",
                "❌ Requires sufficient historical data (50+ candles)",
                "❌ 15-minute cache"
            ],
            usage_hints=[
                # English
                "User asks: 'Bitcoin RSI' → USE THIS with symbol=BTCUSD",
                "User asks: 'ETH technical analysis' → USE THIS with symbol=ETHUSD",
                "User asks: 'Is BTC overbought?' → USE THIS with symbol=BTCUSD",
                # Vietnamese
                "User asks: 'Phân tích kỹ thuật Bitcoin' → USE THIS with symbol=BTCUSD",
                "User asks: 'Chỉ báo kỹ thuật của Ethereum' → USE THIS with symbol=ETHUSD",
                # When NOT to use
                "User wants current price only → DO NOT USE (use getCryptoPrice)"
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Crypto pair symbol (e.g., BTCUSD, ETHUSD)",
                    required=True,
                    pattern=r"^[A-Z]{3,10}USD$"
                ),
                ToolParameter(
                    name="timeframe",
                    type="string",
                    description="Timeframe for technical analysis",
                    required=False,
                    default="1hour",
                    allowed_values=["1min", "5min", "15min", "30min", "1hour", "4hour", "daily"]
                ),
                ToolParameter(
                    name="indicators",
                    type="array",
                    description="List of indicators to calculate",
                    required=False,
                    default=["RSI", "MACD", "SMA_20", "SMA_50"]
                )
            ],
            returns={
                "symbol": "string",
                "timeframe": "string",
                "indicators": "object - Technical indicator values",
                "current_price": "number",
                "timestamp": "string"
            },
            typical_execution_time_ms=1500,
            requires_symbol=True
        )

    async def execute(
        self,
        symbol: str,
        timeframe: str = "1hour",
        indicators: List[str] = None,
        **kwargs
    ) -> ToolOutput:
        """
        Execute getCryptoTechnicals
        
        Args:
            symbol: Crypto pair symbol (e.g., BTCUSD)
            timeframe: Timeframe for analysis
            indicators: List of indicators to calculate
            
        Returns:
            ToolOutput with technical indicators
        """
        start_time = time.time()
        
        try:
            # Normalize inputs
            symbol = symbol.upper().strip()
            if indicators is None:
                indicators = ["RSI", "MACD", "SMA_20", "SMA_50"]
            
            # Validate symbol
            if not symbol.endswith("USD"):
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error=f"Invalid crypto symbol: {symbol}. Must end with 'USD'",
                    metadata={"symbol": symbol}
                )
            
            # Validate timeframe
            valid_timeframes = ["1min", "5min", "15min", "30min", "1hour", "4hour", "daily"]
            if timeframe not in valid_timeframes:
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error=f"Invalid timeframe: {timeframe}",
                    metadata={"allowed_values": valid_timeframes}
                )
            
            self.logger.info(
                f"[{self.schema.name}] Calculating technicals for {symbol} ({timeframe})"
            )
            
            # Check cache
            cache_key = f"getCryptoTechnicals_{symbol}_{timeframe}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.logger.info(f"[{self.schema.name}] Cache HIT for {symbol}")
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="success",
                    data=cached_result
                )
            
            # Fetch OHLCV data
            if timeframe == "daily":
                ohlcv_data = await self._fetch_daily_ohlcv(symbol)
            else:
                ohlcv_data = await self._fetch_intraday_ohlcv(symbol, timeframe)
            
            if not ohlcv_data:
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error=f"No OHLCV data available for {symbol}",
                    metadata={"symbol": symbol, "timeframe": timeframe}
                )
            
            # Extract close prices
            close_prices = [candle.get("close", 0) for candle in ohlcv_data]
            
            if len(close_prices) < 50:
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error=f"Insufficient data for technical analysis: {len(close_prices)} candles",
                    metadata={"required": 50, "available": len(close_prices)}
                )
            
            # Calculate indicators
            calculated_indicators = {}
            
            if "RSI" in indicators:
                calculated_indicators["rsi"] = self._calculate_rsi(close_prices)
            
            if "MACD" in indicators:
                macd_values = self._calculate_macd(close_prices)
                calculated_indicators["macd"] = macd_values
            
            if "SMA_20" in indicators:
                calculated_indicators["sma_20"] = self._calculate_sma(close_prices, 20)
            
            if "SMA_50" in indicators:
                calculated_indicators["sma_50"] = self._calculate_sma(close_prices, 50)
            
            if "EMA_12" in indicators:
                calculated_indicators["ema_12"] = self._calculate_ema(close_prices, 12)
            
            if "EMA_26" in indicators:
                calculated_indicators["ema_26"] = self._calculate_ema(close_prices, 26)
            
            # Build result
            result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "indicators": calculated_indicators,
                "current_price": close_prices[-1] if close_prices else 0,
                "data_points": len(close_prices),
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"[{self.schema.name}] SUCCESS: {symbol} technicals calculated "
                f"({len(indicators)} indicators, {len(close_prices)} candles) "
                f"({execution_time:.0f}ms)"
            )
            
            return ToolOutput(
                tool_name=self.schema.name,
                status="success",
                data=result
            )
            
        except httpx.HTTPStatusError as e:
            self.logger.error(f"[{self.schema.name}] HTTP error: {e}")
            return ToolOutput(
                tool_name=self.schema.name,
                status="error",
                error=f"FMP API error: {e.response.status_code}",
                metadata={"response": e.response.text[:200]}
            )
        except Exception as e:
            self.logger.error(f"[{self.schema.name}] Error: {e}", exc_info=True)
            return ToolOutput(
                tool_name=self.schema.name,
                status="error",
                error=str(e),
                metadata={"type": type(e).__name__}
            )
        
        
    async def _fetch_intraday_ohlcv(self, symbol: str, interval: str) -> List[Dict]:
        """Fetch intraday OHLCV data"""
        url = f"{self.base_url}/historical-chart/{interval}"
        params = {
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    async def _fetch_daily_ohlcv(self, symbol: str) -> List[Dict]:
        """Fetch daily OHLCV data"""
        url = f"{self.base_url}/historical-price-eod/light"
        params = {
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract historical array
            if isinstance(data, dict) and "historical" in data:
                return data["historical"]
            return data

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 2)

    def _calculate_macd(self, prices: List[float]) -> Dict[str, float]:
        """Calculate MACD"""
        if len(prices) < 26:
            return {"macd": 0, "signal": 0, "histogram": 0}
        
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        macd_line = ema_12 - ema_26
        
        # Signal line (9-period EMA of MACD)
        signal_line = macd_line  # Simplified
        histogram = macd_line - signal_line
        
        return {
            "macd": round(macd_line, 4),
            "signal": round(signal_line, 4),
            "histogram": round(histogram, 4)
        }

    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        return round(sum(prices[-period:]) / period, 2)

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return round(ema, 2)

    async def _get_cached_result(self, cache_key: str) -> Dict[str, Any] | None:
        """Get cached result from Redis"""
        try:
            redis_client = await get_redis_client_llm()
            cached_bytes = await redis_client.get(cache_key)
            await redis_client.close()
            
            if cached_bytes:
                return json.loads(cached_bytes.decode('utf-8'))
            return None
        except Exception as e:
            self.logger.warning(f"Cache read error: {e}")
            return None

    async def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache result to Redis"""
        try:
            redis_client = await get_redis_client_llm()
            json_string = json.dumps(result)
            await redis_client.set(cache_key, json_string, ex=self.CACHE_TTL)
            await redis_client.close()
        except Exception as e:
            self.logger.warning(f"Cache write error: {e}")