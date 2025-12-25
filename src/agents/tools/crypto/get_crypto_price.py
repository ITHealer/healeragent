import json
import time
from typing import Dict, Any, Optional
from datetime import datetime

import httpx

from src.agents.tools.base import BaseTool, ToolOutput, ToolSchema, ToolParameter
from src.helpers.redis_cache import get_redis_client_llm
from src.utils.logger.custom_logging import LoggerMixin


class GetCryptoPriceTool(BaseTool, LoggerMixin):
    """
    Tool 23: Get Crypto Price
    
    Fetch real-time price and statistics for cryptocurrencies
    
    FIXED:
    - Accepts multiple symbol formats (BTCUSD, BTCUSDT, BTC)
    - Auto-normalizes to FMP format (XXXUSD)
    - Better error messages
    """

    CACHE_TTL = 300  # 5 minutes - crypto moves fast
    
    # Known crypto base symbols for validation
    KNOWN_CRYPTO = {
        'BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI', 'AAVE',
        'XRP', 'LTC', 'BCH', 'EOS', 'TRX', 'XLM', 'VET', 'ALGO', 'ATOM', 'LUNA',
        'NEAR', 'FTM', 'CRO', 'SAND', 'MANA', 'AXS', 'GALA', 'ENJ', 'CHZ', 'BAT',
        'ZEC', 'DASH', 'XMR', 'DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BNB', 'TON', 'ICP',
        'HBAR', 'THETA', 'FIL', 'ETC', 'MKR', 'APT', 'LDO', 'OP', 'ARB', 'SUI',
        'IMX', 'GRT', 'RUNE', 'FLOW', 'EGLD', 'XTZ', 'MINA', 'ROSE', 'KAVA',
        'INJ', 'SEI', 'TIA', 'JUP', 'BONK', 'WIF', 'ORDI', 'STX', 'RENDER'
    }

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
                "Accepts multiple formats: BTC, BTCUSD, BTCUSDT - all work! "
                "Use when user asks about crypto prices, Bitcoin, Ethereum, or other cryptocurrencies."
            ),
            capabilities=[
                "✅ Real-time crypto prices",
                "✅ 24-hour price change",
                "✅ Volume and market cap",
                "✅ Daily high/low",
                "✅ 52-week high/low",
                "✅ Accepts BTC, BTCUSD, or BTCUSDT formats"
            ],
            limitations=[
                "❌ USD pairs only (converted automatically)",
                "❌ 5-minute cache for real-time data"
            ],
            usage_hints=[
                # English - multiple formats
                "User asks: 'Bitcoin price' → USE THIS with symbol=BTC or BTCUSD or BTCUSDT",
                "User asks: 'How much is ETH?' → USE THIS with symbol=ETH or ETHUSD or ETHUSDT",
                "User asks: 'Solana current price' → USE THIS with symbol=SOL or SOLUSD",
                "User asks: 'BTCUSDT price' → USE THIS with symbol=BTCUSDT (auto-converts)",
                # Vietnamese
                "User asks: 'Giá Bitcoin' → USE THIS with symbol=BTC",
                "User asks: 'Ethereum bao nhiêu?' → USE THIS with symbol=ETH",
                "User asks: 'Giá BTCUSDT' → USE THIS with symbol=BTCUSDT",
                # When NOT to use
                "User wants technical analysis → DO NOT USE (use getCryptoTechnicals)"
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description=(
                        "Crypto symbol in any format: "
                        "BTC, BTCUSD, BTCUSDT, ETH, ETHUSD, ETHUSDT, etc. "
                        "Will be automatically normalized to FMP format (XXXUSD)."
                    ),
                    required=True,
                    # Relaxed pattern - accepts more formats
                    pattern=r"^[A-Z]{2,15}(USD[T]?)?$"
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

    def _normalize_crypto_symbol(self, symbol: str) -> str:
        """
        Normalize crypto symbol to FMP format (XXXUSD)
        
        Handles:
        - BTC → BTCUSD
        - BTCUSD → BTCUSD (no change)
        - BTCUSDT → BTCUSD (remove T)
        - btc → BTCUSD (uppercase)
        
        Args:
            symbol: Input symbol in any format
            
        Returns:
            Normalized symbol (e.g., BTCUSD)
        """
        # Uppercase and strip
        symbol = symbol.upper().strip()
        
        # Remove common suffixes and normalize
        if symbol.endswith('USDT'):
            # BTCUSDT → BTCUSD
            base = symbol[:-4]  # Remove 'USDT'
            normalized = f"{base}USD"
            self.logger.debug(f"[NORMALIZE] {symbol} → {normalized} (removed T)")
            return normalized
        
        elif symbol.endswith('USD'):
            # Already correct format
            self.logger.debug(f"[NORMALIZE] {symbol} → {symbol} (no change)")
            return symbol
        
        elif symbol.endswith('BUSD'):
            # BTCBUSD → BTCUSD
            base = symbol[:-4]
            normalized = f"{base}USD"
            self.logger.debug(f"[NORMALIZE] {symbol} → {normalized} (BUSD→USD)")
            return normalized
        
        else:
            # Short format: BTC → BTCUSD
            normalized = f"{symbol}USD"
            self.logger.debug(f"[NORMALIZE] {symbol} → {normalized} (added USD)")
            return normalized
    
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
                self.logger.debug(f"[CACHE SET] {cache_key}")
        except Exception as e:
            self.logger.warning(f"[CACHE] Write error: {e}")

    async def execute(self, symbol: str, **kwargs) -> ToolOutput:
        """
        Execute getCryptoPrice
        
        Args:
            symbol: Crypto symbol (any format: BTC, BTCUSD, BTCUSDT)
            
        Returns:
            ToolOutput with crypto quote
        """
        start_time = time.time()
        original_symbol = symbol
        
        try:
            # Normalize symbol to FMP format
            symbol = self._normalize_crypto_symbol(symbol)
            base_symbol = self._extract_base_symbol(symbol)
            
            self.logger.info(
                f"[{self.schema.name}] Fetching: {original_symbol} → {symbol}"
            )
            
            # Validate base symbol is a known crypto
            if base_symbol not in self.KNOWN_CRYPTO:
                self.logger.warning(
                    f"[{self.schema.name}] Unknown crypto: {base_symbol}. "
                    f"Proceeding anyway..."
                )
            
            # Check cache
            cache_key = f"getCryptoPrice_{symbol}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.logger.info(f"[{self.schema.name}] Cache HIT for {symbol}")
                execution_time = int((time.time() - start_time) * 1000)
                
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="success",
                    data=cached_result,
                    formatted_context=self._build_formatted_context(cached_result),
                    metadata={
                        "symbol": symbol,
                        "original_symbol": original_symbol,
                        "from_cache": True,
                        "execution_time_ms": execution_time
                    }
                )
                
            # Fetch from FMP API
            url = f"{self.base_url}/quote"
            params = {
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            self.logger.debug(f"[FMP] GET {url}?symbol={symbol}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                quote_data = response.json()
            
            # Validate response
            if not quote_data or not isinstance(quote_data, list) or len(quote_data) == 0:
                self.logger.warning(
                    f"[{self.schema.name}] No data for {symbol}. "
                    f"Response: {quote_data}"
                )
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error=f"No data found for crypto symbol: {symbol}. "
                          f"Make sure it's a valid crypto pair (e.g., BTC, ETH, SOL).",
                    metadata={
                        "symbol": symbol,
                        "original_symbol": original_symbol
                    }
                )
                
            quote = quote_data[0]
            
            # Validate crypto exchange
            exchange = quote.get("exchange", "").upper()
            if exchange not in ["CRYPTO", "CRYPTOCURRENCY", "CC", ""]:
                self.logger.warning(
                    f"[{self.schema.name}] Symbol {symbol} may not be crypto. "
                    f"Exchange: {exchange}"
                )
            
            # Format result
            result_data = {
                "symbol": symbol,
                "original_input": original_symbol,
                "name": quote.get("name", base_symbol),
                "price": quote.get("price", 0),
                "change": quote.get("change", 0),
                "changes_percentage": quote.get("changesPercentage", 0),
                "day_low": quote.get("dayLow", 0),
                "day_high": quote.get("dayHigh", 0),
                "year_low": quote.get("yearLow", 0),
                "year_high": quote.get("yearHigh", 0),
                "volume": quote.get("volume", 0),
                "avg_volume": quote.get("avgVolume", 0),
                "market_cap": quote.get("marketCap", 0),
                "open": quote.get("open", 0),
                "previous_close": quote.get("previousClose", 0),
                "exchange": exchange,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            await self._set_cached_result(cache_key, result_data)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            self.logger.info(
                f"[{self.schema.name}] ✅ SUCCESS ({execution_time}ms) - "
                f"{symbol} @ ${result_data['price']:,.2f}"
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
                    "from_cache": False
                }
            )
            
        except httpx.HTTPStatusError as e:
            execution_time = int((time.time() - start_time) * 1000)
            self.logger.error(
                f"[{self.schema.name}] HTTP error for {symbol}: {e}"
            )
            return ToolOutput(
                tool_name=self.schema.name,
                status="error",
                error=f"API error fetching {symbol}: {str(e)}",
                execution_time_ms=execution_time,
                metadata={"symbol": symbol, "original_symbol": original_symbol}
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            self.logger.error(
                f"[{self.schema.name}] Error for {symbol}: {e}",
                exc_info=True
            )
            return ToolOutput(
                tool_name=self.schema.name,
                status="error",
                error=f"Error fetching crypto price: {str(e)}",
                execution_time_ms=execution_time,
                metadata={"symbol": symbol, "original_symbol": original_symbol}
            )
    
    def _build_formatted_context(self, data: Dict) -> str:
        """Build human-readable formatted context for LLM"""
        symbol = data.get('symbol', 'Unknown')
        name = data.get('name', symbol)
        price = data.get('price', 0)
        change = data.get('change', 0)
        change_pct = data.get('changes_percentage', 0)
        volume = data.get('volume', 0)
        market_cap = data.get('market_cap', 0)
        
        # Emoji for direction
        emoji = "📈" if change >= 0 else "📉"
        sign = "+" if change >= 0 else ""
        
        lines = [
            f"💰 CRYPTO PRICE - {name} ({symbol}):",
            f"",
            f"💵 Price: ${price:,.2f}",
            f"{emoji} Change: {sign}{change:,.2f} ({sign}{change_pct:.2f}%)",
            f"",
            f"📊 Day Range: ${data.get('day_low', 0):,.2f} - ${data.get('day_high', 0):,.2f}",
            f"📅 52-Week: ${data.get('year_low', 0):,.2f} - ${data.get('year_high', 0):,.2f}",
            f"",
            f"📦 Volume: {volume:,.0f}",
            f"💎 Market Cap: ${market_cap:,.0f}"
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
        
        tool = GetCryptoPriceTool(api_key=api_key)
        
        print("\n" + "="*60)
        print("Testing getCryptoPrice Tool - Symbol Normalization")
        print("="*60)
        
        # Test different formats
        test_symbols = ["BTC", "BTCUSD", "BTCUSDT", "ETH", "ETHUSDT", "SOL"]
        
        for sym in test_symbols:
            print(f"\n--- Testing: {sym} ---")
            result = await tool.execute(symbol=sym)
            
            if result.status == 'success':
                data = result.data
                print(f"✅ {sym} → {data['symbol']}")
                print(f"   Price: ${data['price']:,.2f}")
                print(f"   Change: {data['changes_percentage']:.2f}%")
            else:
                print(f"❌ Error: {result.error}")
    
    asyncio.run(test())