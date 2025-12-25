import httpx
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    create_success_output,
    create_error_output
)


class GetStockPriceTool(BaseTool):
    """
    Atomic tool để lấy giá cổ phiếu hiện tại
    
    Data source: FMP API /v3/quote/{symbol}
    
    Usage:
        tool = GetStockPriceTool()
        result = await tool.safe_execute(symbol="AAPL")
        
        if result.is_success():
            price = result.data['price']
            change_percent = result.data['change_percent']
    """
    
    # FMP API Configuration
    FMP_BASE_URL = "https://financialmodelingprep.com/api"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize tool
        
        Args:
            api_key: FMP API key (fallback to env var if not provided)
        """
        super().__init__()
        
        # Get API key from env if not provided
        if api_key is None:
            import os
            api_key = os.environ.get("FMP_API_KEY")
        
        if not api_key:
            raise ValueError("FMP_API_KEY not provided and not found in environment")
        
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Define schema
        self.schema = ToolSchema(
            name="getStockPrice",
            category="price",
            description=(
                "Get real-time stock quote with current price, change, volume, and market metrics. "
                "Returns comprehensive price data for a single stock symbol. "
                "Use when user asks about current stock price or basic quote information."
            ),
            capabilities=[
                "✅ Real-time stock price",
                "✅ Intraday price change ($ and %)",
                "✅ Daily high/low",
                "✅ 52-week high/low",
                "✅ Volume and market cap",
                "✅ Previous close price"
            ],
            limitations=[
                "❌ 15-minute delay on free tier",
                "❌ One symbol at a time",
                "❌ No historical price data"
            ],
            usage_hints=[
                # English
                "User asks: 'Apple stock price' → USE THIS with symbol=AAPL",
                "User asks: 'What is TSLA trading at?' → USE THIS with symbol=TSLA",
                "User asks: 'Current price of Microsoft' → USE THIS with symbol=MSFT",
                # Vietnamese
                "User asks: 'Giá cổ phiếu Apple' → USE THIS with symbol=AAPL",
                "User asks: 'Amazon đang giá bao nhiêu?' → USE THIS with symbol=AMZN",
                "User asks: 'Cho biết giá NVDA' → USE THIS with symbol=NVDA",
                # When NOT to use
                "User asks about MARKET overview → DO NOT USE (use getMarketIndices)",
                "User asks for price HISTORY → DO NOT USE (use getStockPerformance)",
                "User asks about multiple stocks → USE MULTIPLE TIMES (one per symbol)"
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol (e.g., AAPL, TSLA, NVDA)",
                    required=True,
                    pattern="^[A-Z]{1,7}$"
                )
            ],
            returns={
                "symbol": "string",
                "price": "number",
                "change": "number",
                "change_percent": "number",
                "volume": "number",
                "market_cap": "number",
                "timestamp": "string"
            },
            typical_execution_time_ms=800,
            requires_symbol=True
        )
    
    async def execute(self, symbol: str) -> ToolOutput:
        """
        Execute tool - Fetch stock price from FMP
        
        Args:
            symbol: Stock symbol
            
        Returns:
            ToolOutput with price data
        """
        symbol_upper = symbol.upper()
        
        try:
            # Fetch from FMP API
            price_data = await self._fetch_from_fmp(symbol_upper)
            
            if not price_data:
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"No price data found for symbol: {symbol_upper}"
                )
            
            # Extract and format data
            formatted_data = self._format_price_data(price_data, symbol_upper)
            
            return create_success_output(
                tool_name=self.schema.name,
                data=formatted_data,
                metadata={
                    "source": "FMP",
                    "endpoint": "quote",
                    "symbol_queried": symbol_upper
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching price for {symbol_upper}: {e}", exc_info=True)
            return create_error_output(
                tool_name=self.schema.name,
                error=f"Failed to fetch price data: {str(e)}"
            )
    
    async def _fetch_from_fmp(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch quote data from FMP API
        
        Endpoint: GET /v3/quote/{symbol}
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Quote data dict or None if not found
        """
        url = f"{self.FMP_BASE_URL}/v3/quote/{symbol}"
        params = {"apikey": self.api_key}
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # FMP returns list for quote endpoint
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                elif isinstance(data, dict):
                    return data
                else:
                    self.logger.warning(f"Unexpected FMP response format for {symbol}: {type(data)}")
                    return None
                    
        except httpx.HTTPError as e:
            self.logger.error(f"HTTP error fetching {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def _format_price_data(self, fmp_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """
        Format FMP response to standardized output
        
        Args:
            fmp_data: Raw FMP quote data
            symbol: Symbol being queried
            
        Returns:
            Formatted data dict
        """
        return {
            "symbol": fmp_data.get("symbol", symbol),
            "name": fmp_data.get("name", ""),
            "price": fmp_data.get("price", 0.0),
            "change": fmp_data.get("change", 0.0),
            "change_percent": fmp_data.get("changesPercentage", 0.0),
            "volume": fmp_data.get("volume", 0),
            "day_high": fmp_data.get("dayHigh", 0.0),
            "day_low": fmp_data.get("dayLow", 0.0),
            "previous_close": fmp_data.get("previousClose", 0.0),
            "market_cap": fmp_data.get("marketCap", 0),
            "timestamp": fmp_data.get("timestamp", datetime.now().timestamp())
        }


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import json
    import os
    
    async def test_tool():
        """Test GetStockPriceTool standalone"""
        
        # Check API key
        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            print("ERROR: FMP_API_KEY not found in environment")
            return
        
        print("=" * 80)
        print("TESTING GetStockPriceTool")
        print("=" * 80)
        
        tool = GetStockPriceTool(api_key=api_key)
        
        # Test 1: Valid symbol (AAPL)
        print("\nTest 1: Valid symbol (AAPL)")
        print("-" * 40)
        result = await tool.safe_execute(symbol="AAPL")
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_ms}ms")
        if result.is_success():
            print(json.dumps(result.data, indent=2))
        else:
            print(f"Error: {result.error}")
        
        # Test 2: Valid symbol (NVDA)
        print("\nTest 2: Valid symbol (NVDA)")
        print("-" * 40)
        result = await tool.safe_execute(symbol="NVDA")
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_ms}ms")
        if result.is_success():
            print(json.dumps(result.data, indent=2))
        else:
            print(f"Error: {result.error}")
        
        # Test 3: Invalid symbol
        print("\nTest 3: Invalid symbol (INVALID)")
        print("-" * 40)
        result = await tool.safe_execute(symbol="INVALID")
        print(f"Status: {result.status}")
        print(f"Error: {result.error}")
        
        # Test 4: Crypto (BTC)
        print("\nTest 4: Crypto symbol (BTCUSD)")
        print("-" * 40)
        result = await tool.safe_execute(symbol="BTCUSD")
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_ms}ms")
        if result.is_success():
            print(json.dumps(result.data, indent=2))
        else:
            print(f"Error: {result.error}")
        
        # Test 5: Invalid input (validation should fail)
        print("\nTest 5: Invalid input (lowercase)")
        print("-" * 40)
        result = await tool.safe_execute(symbol="aapl")
        print(f"Status: {result.status}")
        print(f"Error: {result.error}")
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETE")
        print("=" * 80)
    
    # Run tests
    asyncio.run(test_tool())