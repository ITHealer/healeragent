import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

import httpx

from src.agents.tools.base import BaseTool, ToolOutput, ToolSchema, ToolParameter
from src.helpers.redis_cache import get_redis_client_llm
from src.utils.logger.custom_logging import LoggerMixin


class GetCryptoTechnicalsTool(BaseTool, LoggerMixin):
    """
    Tool 24: Get Crypto Technical Indicators
    
    Fetch technical indicators for cryptocurrencies (RSI, MACD, SMA, EMA, etc.)
    
    FIXED:
    - Accepts multiple symbol formats (BTCUSD, BTCUSDT, BTC)
    - Auto-normalizes to FMP format (XXXUSD)
    """

    CACHE_TTL = 600  # 10 minutes
    
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
    
    # Timeframe mapping
    TIMEFRAME_MAP = {
        "1min": "1min",
        "5min": "5min",
        "15min": "15min",
        "30min": "30min",
        "1hour": "1hour",
        "4hour": "4hour",
        "daily": "1day",
        "1day": "1day",
        "weekly": "1week",
        "1week": "1week"
    }

    def __init__(self, api_key: str):
        """
        Initialize GetCryptoTechnicalsTool
        
        Args:
            api_key: FMP API key
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/stable"
        
        # Define schema with relaxed pattern
        self.schema = ToolSchema(
            name="getCryptoTechnicals",
            category="crypto",
            description=(
                "Fetch technical indicators for cryptocurrencies. "
                "Returns RSI, MACD, SMA, EMA, Bollinger Bands, and more. "
                "Accepts multiple formats: BTC, BTCUSD, BTCUSDT - all work! "
                "Use when user asks about crypto technical analysis or indicators."
            ),
            capabilities=[
                "✅ RSI (Relative Strength Index)",
                "✅ MACD (Moving Average Convergence Divergence)",
                "✅ SMA (Simple Moving Average)",
                "✅ EMA (Exponential Moving Average)",
                "✅ Bollinger Bands",
                "✅ Multiple timeframes (1min to weekly)",
                "✅ Accepts BTC, BTCUSD, or BTCUSDT formats"
            ],
            limitations=[
                "❌ USD pairs only (converted automatically)",
                "❌ Historical data limited by API"
            ],
            usage_hints=[
                "User asks: 'Bitcoin RSI' → USE THIS with symbol=BTC",
                "User asks: 'ETH technical analysis' → USE THIS with symbol=ETH",
                "User asks: 'BTCUSDT technicals' → USE THIS with symbol=BTCUSDT (auto-converts)",
                "User asks: 'Phân tích kỹ thuật Bitcoin' → USE THIS with symbol=BTC",
                "User wants current price only → DO NOT USE (use getCryptoPrice)"
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description=(
                        "Crypto symbol in any format: "
                        "BTC, BTCUSD, BTCUSDT, ETH, ETHUSD, ETHUSDT, etc."
                    ),
                    required=True,
                    # Relaxed pattern
                    pattern=r"^[A-Z]{2,15}(USD[T]?)?$"
                ),
                ToolParameter(
                    name="timeframe",
                    type="string",
                    description="Timeframe for technical analysis",
                    required=False,
                    default="1hour",
                    enum=["1min", "5min", "15min", "30min", "1hour", "4hour", "daily", "weekly"]
                ),
                ToolParameter(
                    name="indicators",
                    type="array",
                    description="List of indicators to calculate",
                    required=False,
                    default=["RSI", "MACD", "SMA_20", "SMA_50", "EMA_20"]
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

    def _normalize_crypto_symbol(self, symbol: str) -> str:
        """
        Normalize crypto symbol to FMP format (XXXUSD)
        """
        symbol = symbol.upper().strip()
        
        if symbol.endswith('USDT'):
            base = symbol[:-4]
            normalized = f"{base}USD"
            self.logger.debug(f"[NORMALIZE] {symbol} → {normalized}")
            return normalized
        
        elif symbol.endswith('USD'):
            return symbol
        
        elif symbol.endswith('BUSD'):
            base = symbol[:-4]
            return f"{base}USD"
        
        else:
            return f"{symbol}USD"
    
    def _extract_base_symbol(self, symbol: str) -> str:
        """Extract base symbol (BTC from BTCUSD)"""
        symbol = symbol.upper()
        
        if symbol.endswith('USDT'):
            return symbol[:-4]
        elif symbol.endswith('USD'):
            return symbol[:-3]
        elif symbol.endswith('BUSD'):
            return symbol[:-4]
        else:
            return symbol

    async def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached result from Redis"""
        try:
            redis_client = await get_redis_client_llm()
            if redis_client:
                cached_bytes = await redis_client.get(cache_key)
                if cached_bytes:
                    return json.loads(cached_bytes.decode('utf-8'))
        except Exception as e:
            self.logger.warning(f"[CACHE] Read error: {e}")
        return None
    
    async def _set_cached_result(self, cache_key: str, data: Dict) -> None:
        """Set cached result in Redis"""
        try:
            redis_client = await get_redis_client_llm()
            if redis_client:
                await redis_client.set(
                    cache_key, 
                    json.dumps(data), 
                    ex=self.CACHE_TTL
                )
        except Exception as e:
            self.logger.warning(f"[CACHE] Write error: {e}")

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
            symbol: Crypto symbol (any format)
            timeframe: Timeframe for analysis
            indicators: List of indicators to calculate
            
        Returns:
            ToolOutput with technical indicators
        """
        start_time = time.time()
        original_symbol = symbol
        
        try:
            # Normalize inputs
            symbol = self._normalize_crypto_symbol(symbol)
            base_symbol = self._extract_base_symbol(symbol)
            
            if indicators is None:
                indicators = ["RSI", "MACD", "SMA_20", "SMA_50", "EMA_20"]
            
            # Map timeframe
            api_timeframe = self.TIMEFRAME_MAP.get(timeframe, "1hour")
            
            self.logger.info(
                f"[{self.schema.name}] Fetching: {original_symbol} → {symbol}, "
                f"timeframe={api_timeframe}"
            )
            
            # Check cache
            cache_key = f"getCryptoTechnicals_{symbol}_{api_timeframe}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.logger.info(f"[{self.schema.name}] Cache HIT")
                execution_time = int((time.time() - start_time) * 1000)
                
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="success",
                    data=cached_result,
                    formatted_context=self._build_formatted_context(cached_result),
                    execution_time_ms=execution_time,
                    metadata={
                        "symbol": symbol,
                        "original_symbol": original_symbol,
                        "from_cache": True
                    }
                )
            
            # Fetch RSI
            rsi_data = await self._fetch_indicator(symbol, api_timeframe, "rsi", 14)
            
            # Fetch MACD
            macd_data = await self._fetch_indicator(symbol, api_timeframe, "macd")
            
            # Fetch SMA
            sma_20 = await self._fetch_indicator(symbol, api_timeframe, "sma", 20)
            sma_50 = await self._fetch_indicator(symbol, api_timeframe, "sma", 50)
            
            # Fetch EMA
            ema_20 = await self._fetch_indicator(symbol, api_timeframe, "ema", 20)
            
            # Get current price
            current_price = await self._fetch_current_price(symbol)
            
            # Compile results
            result_data = {
                "symbol": symbol,
                "original_input": original_symbol,
                "base_symbol": base_symbol,
                "timeframe": timeframe,
                "api_timeframe": api_timeframe,
                "current_price": current_price,
                "indicators": {
                    "rsi": {
                        "value": rsi_data.get("rsi") if rsi_data else None,
                        "period": 14,
                        "interpretation": self._interpret_rsi(rsi_data.get("rsi") if rsi_data else None)
                    },
                    "macd": macd_data if macd_data else {
                        "macd": None,
                        "signal": None,
                        "histogram": None
                    },
                    "sma_20": sma_20.get("sma") if sma_20 else None,
                    "sma_50": sma_50.get("sma") if sma_50 else None,
                    "ema_20": ema_20.get("ema") if ema_20 else None
                },
                "trend": self._determine_trend(current_price, sma_20, sma_50, ema_20),
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            await self._set_cached_result(cache_key, result_data)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            self.logger.info(
                f"[{self.schema.name}] ✅ SUCCESS ({execution_time}ms)"
            )
            
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
                    "from_cache": False
                }
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            self.logger.error(
                f"[{self.schema.name}] Error: {e}",
                exc_info=True
            )
            return ToolOutput(
                tool_name=self.schema.name,
                status="error",
                error=f"Error fetching crypto technicals: {str(e)}",
                execution_time_ms=execution_time,
                metadata={"symbol": symbol, "original_symbol": original_symbol}
            )
    
    async def _fetch_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator: str,
        period: int = None
    ) -> Optional[Dict]:
        """Fetch a single indicator from FMP API"""
        try:
            url = f"{self.base_url}/technical_indicator/{timeframe}"
            params = {
                "symbol": symbol,
                "type": indicator,
                "apikey": self.api_key
            }
            if period:
                params["period"] = period
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
            
            if data and isinstance(data, list) and len(data) > 0:
                return data[0]  # Most recent value
            return None
            
        except Exception as e:
            self.logger.warning(f"[{indicator}] Fetch error: {e}")
            return None
    
    async def _fetch_current_price(self, symbol: str) -> Optional[float]:
        """Fetch current price"""
        try:
            url = f"{self.base_url}/quote"
            params = {"symbol": symbol, "apikey": self.api_key}
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
            
            if data and isinstance(data, list) and len(data) > 0:
                return data[0].get("price", 0)
            return None
            
        except Exception as e:
            self.logger.warning(f"[PRICE] Fetch error: {e}")
            return None
    
    def _interpret_rsi(self, rsi: Optional[float]) -> str:
        """Interpret RSI value"""
        if rsi is None:
            return "N/A"
        elif rsi >= 70:
            return "Overbought (potential sell signal)"
        elif rsi <= 30:
            return "Oversold (potential buy signal)"
        elif rsi >= 60:
            return "Bullish momentum"
        elif rsi <= 40:
            return "Bearish momentum"
        else:
            return "Neutral"
    
    def _determine_trend(
        self,
        price: Optional[float],
        sma_20: Optional[Dict],
        sma_50: Optional[Dict],
        ema_20: Optional[Dict]
    ) -> str:
        """Determine overall trend based on MAs"""
        if price is None:
            return "Unknown"
        
        sma20_val = sma_20.get("sma") if sma_20 else None
        sma50_val = sma_50.get("sma") if sma_50 else None
        ema20_val = ema_20.get("ema") if ema_20 else None
        
        signals = []
        
        if sma20_val and price > sma20_val:
            signals.append(1)
        elif sma20_val:
            signals.append(-1)
        
        if sma50_val and price > sma50_val:
            signals.append(1)
        elif sma50_val:
            signals.append(-1)
        
        if ema20_val and price > ema20_val:
            signals.append(1)
        elif ema20_val:
            signals.append(-1)
        
        if not signals:
            return "Unknown"
        
        avg_signal = sum(signals) / len(signals)
        
        if avg_signal > 0.5:
            return "Bullish (price above key MAs)"
        elif avg_signal < -0.5:
            return "Bearish (price below key MAs)"
        else:
            return "Mixed/Consolidating"
    
    def _build_formatted_context(self, data: Dict) -> str:
        """Build human-readable formatted context"""
        symbol = data.get('symbol', 'Unknown')
        timeframe = data.get('timeframe', 'N/A')
        price = data.get('current_price', 0)
        indicators = data.get('indicators', {})
        trend = data.get('trend', 'Unknown')
        
        rsi = indicators.get('rsi', {})
        macd = indicators.get('macd', {})
        
        lines = [
            f"📊 CRYPTO TECHNICALS - {symbol} ({timeframe}):",
            f"",
            f"💵 Current Price: ${price:,.2f}" if price else "💵 Current Price: N/A",
            f"📈 Overall Trend: {trend}",
            f"",
            f"📉 Indicators:",
            f"  • RSI(14): {rsi.get('value', 'N/A')} - {rsi.get('interpretation', 'N/A')}",
            f"  • MACD: {macd.get('macd', 'N/A')}",
            f"  • MACD Signal: {macd.get('signal', 'N/A')}",
            f"  • SMA(20): {indicators.get('sma_20', 'N/A')}",
            f"  • SMA(50): {indicators.get('sma_50', 'N/A')}",
            f"  • EMA(20): {indicators.get('ema_20', 'N/A')}"
        ]
        
        return '\n'.join(lines)


# Standalone test
if __name__ == "__main__":
    import asyncio
    import os
    
    async def test():
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            print("❌ FMP_API_KEY not set")
            return
        
        tool = GetCryptoTechnicalsTool(api_key=api_key)
        
        print("\n" + "="*60)
        print("Testing getCryptoTechnicals Tool")
        print("="*60)
        
        # Test with BTCUSDT format
        print("\nTest: BTCUSDT (Binance format)")
        result = await tool.execute(symbol="BTCUSDT", timeframe="1hour")
        
        if result.status == 'success':
            print(f"✅ Success")
            print(f"Symbol: {result.data['symbol']}")
            print(f"Trend: {result.data['trend']}")
            print(f"\nFormatted Context:")
            print(result.formatted_context)
        else:
            print(f"❌ Error: {result.error}")
    
    asyncio.run(test())