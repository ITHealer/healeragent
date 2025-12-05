# src/agents/tools/crypto/get_crypto_price.py

import json
import time
from typing import Dict, Any
from datetime import datetime

import httpx

from src.agents.tools.base import BaseTool, ToolOutput, ToolSchema, ToolParameter
from src.helpers.redis_cache import get_redis_client_llm
from src.utils.logger.custom_logging import LoggerMixin


class GetCryptoPriceTool(BaseTool, LoggerMixin):
    """
    Tool 23: Get Crypto Price
    
    Fetch real-time price and statistics for cryptocurrencies
    
    FMP Stable API:
    GET https://financialmodelingprep.com/stable/quote?symbol=BTCUSD
    """

    CACHE_TTL = 300  # 5 minutes - crypto moves fast

    def __init__(self, api_key: str):
        """
        Initialize GetCryptoPriceTool
        
        Args:
            api_key: FMP API key
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/stable"
        
        # Define schema
        self.schema = ToolSchema(
            name="getCryptoPrice",
            category="crypto",
            description=(
                "Fetch real-time cryptocurrency price and statistics. "
                "Returns current price, 24h change, volume, market cap, and 52-week range. "
                "Symbol must end with 'USD' (e.g., BTCUSD, ETHUSD). "
                "Use when user asks about crypto prices, Bitcoin, Ethereum, or other cryptocurrencies."
            ),
            capabilities=[
                "✅ Real-time crypto prices",
                "✅ 24-hour price change",
                "✅ Volume and market cap",
                "✅ Daily high/low",
                "✅ 52-week high/low",
                "✅ Major cryptocurrencies (BTC, ETH, SOL, etc.)"
            ],
            limitations=[
                "❌ Symbol must end with 'USD' (e.g., BTCUSD)",
                "❌ Limited to USD pairs",
                "❌ 5-minute cache for free tier"
            ],
            usage_hints=[
                # English
                "User asks: 'Bitcoin price' → USE THIS with symbol=BTCUSD",
                "User asks: 'How much is ETH?' → USE THIS with symbol=ETHUSD",
                "User asks: 'Solana current price' → USE THIS with symbol=SOLUSD",
                # Vietnamese
                "User asks: 'Giá Bitcoin' → USE THIS with symbol=BTCUSD",
                "User asks: 'Ethereum bao nhiêu?' → USE THIS with symbol=ETHUSD",
                # When NOT to use
                "User wants technical analysis → DO NOT USE (use getCryptoTechnicals)"
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Crypto pair symbol (must end with USD, e.g., BTCUSD, ETHUSD, SOLUSD)",
                    required=True,
                    pattern=r"^[A-Z]{3,10}USD$"
                )
            ],
            returns={
                "symbol": "string",
                "name": "string",
                "price": "number",
                "change": "number",
                "changes_percentage": "number",
                "volume": "number",
                "market_cap": "number",
                "timestamp": "string"
            },
            typical_execution_time_ms=800,
            requires_symbol=True
        )

    async def execute(self, symbol: str, **kwargs) -> ToolOutput:
        """
        Execute getCryptoPrice
        
        Args:
            symbol: Crypto pair symbol (e.g., BTCUSD)
            
        Returns:
            ToolOutput with crypto quote
        """
        start_time = time.time()
        
        try:
            # Normalize symbol
            symbol = symbol.upper().strip()
            
            # Validate symbol format
            if not symbol.endswith("USD"):
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error=f"Invalid crypto symbol: {symbol}. Must end with 'USD' (e.g., BTCUSD)",
                    metadata={"symbol": symbol}
                )
            
            self.logger.info(f"[{self.schema.name}] Fetching crypto price for {symbol}")
            
            # Check cache
            cache_key = f"getCryptoPrice_{symbol}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.logger.info(f"[{self.schema.name}] Cache HIT for {symbol}")
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="success",
                    data=cached_result
                )
                
            # Fetch from FMP API
            url = f"{self.base_url}/quote"
            params = {
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                quote_data = response.json()
            
            # Validate response
            if not quote_data or not isinstance(quote_data, list) or len(quote_data) == 0:
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error=f"No data found for crypto symbol: {symbol}",
                    metadata={"symbol": symbol, "response": quote_data}
                )
                
            quote = quote_data[0]
            
            # Validate crypto exchange
            exchange = quote.get("exchange", "").upper()
            if exchange not in ["CRYPTO", "CRYPTOCURRENCY"]:
                self.logger.warning(
                    f"[{self.schema.name}] Unexpected exchange: {exchange} for {symbol}"
                )
            
            # Validate price
            price = quote.get("price", 0)
            if price <= 0:
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error=f"Invalid price for {symbol}: {price}",
                    metadata={"symbol": symbol, "price": price}
                )
            
            # Build result
            result = {
                "symbol": quote.get("symbol"),
                "name": quote.get("name"),
                "price": price,
                "change": quote.get("change", 0),
                "changes_percentage": quote.get("changesPercentage", 0),
                "day_low": quote.get("dayLow", 0),
                "day_high": quote.get("dayHigh", 0),
                "year_low": quote.get("yearLow", 0),
                "year_high": quote.get("yearHigh", 0),
                "market_cap": quote.get("marketCap"),
                "volume": quote.get("volume", 0),
                "avg_volume": quote.get("avgVolume"),
                "open": quote.get("open", 0),
                "previous_close": quote.get("previousClose", 0),
                "exchange": quote.get("exchange"),
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"[{self.schema.name}] SUCCESS: {symbol} @ ${price:,.2f} "
                f"({quote.get('changesPercentage', 0):+.2f}%) ({execution_time:.0f}ms)"
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