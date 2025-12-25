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


class GetStockPerformanceTool(BaseTool):
    """
    Atomic tool để lấy stock performance qua nhiều timeframes
    
    Data source: FMP /v3/stock-price-change/{symbol}
    
    Usage:
        tool = GetStockPerformanceTool()
        result = await tool.safe_execute(symbol="AAPL")
        
        if result.is_success():
            one_day = result.data['timeframes']['1_day']['return_percent']
            ytd = result.data['timeframes']['ytd']['return_percent']
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
            name="getStockPerformance",
            category="price",
            description=(
                "Get stock performance across multiple timeframes (1D, 1W, 1M, 3M, 6M, 1Y, YTD). "
                "Returns percentage returns and identifies best/worst performing periods. "
                "Use when user asks about stock performance history or returns over time."
            ),
            capabilities=[
                "✅ Performance across 7 timeframes",
                "✅ Percentage returns for each period",
                "✅ Best and worst performing periods",
                "✅ Momentum trend analysis",
                "✅ YTD performance"
            ],
            limitations=[
                "❌ No intraday performance (use technical indicators)",
                "❌ Historical data may be delayed",
                "❌ One symbol at a time"
            ],
            usage_hints=[
                # English
                "User asks: 'How has AAPL performed?' → USE THIS",
                "User asks: 'Tesla YTD returns' → USE THIS",
                "User asks: 'Show me Microsoft performance' → USE THIS",
                # Vietnamese
                "User asks: 'Hiệu suất của Apple' → USE THIS",
                "User asks: 'Amazon tăng/giảm bao nhiêu?' → USE THIS",
                "User asks: 'Đánh giá performance NVDA' → USE THIS",
                # When NOT to use
                "User asks for CURRENT price only → DO NOT USE (use getStockPrice)",
                "User asks for technical analysis → DO NOT USE (use getTechnicalIndicators)"
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol",
                    required=True,
                    pattern="^[A-Z]{1,7}$"
                )
            ],
            returns={
                "symbol": "string",
                "timeframes": "object",
                "best_timeframe": "string",
                "worst_timeframe": "string",
                "momentum_trend": "string",
                "timestamp": "string"
            },
            typical_execution_time_ms=1200,
            requires_symbol=True
        )

    
    async def execute(self, symbol: str) -> ToolOutput:
        """
        Execute performance retrieval
        
        Args:
            symbol: Stock symbol
            
        Returns:
            ToolOutput with performance data
        """
        symbol_upper = symbol.upper()
        
        self.logger.info(f"[getStockPerformance] Executing for symbol={symbol_upper}")
        
        try:
            # Fetch from FMP
            raw_data = await self._fetch_from_fmp(symbol_upper)
            
            if not raw_data:
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"No performance data available for {symbol_upper}"
                )
            
            # Format to schema
            formatted_data = self._format_performance_data(raw_data, symbol_upper)
            
            return create_success_output(
                tool_name=self.schema.name,
                data=formatted_data,
                metadata={
                    "source": "FMP /v3/stock-price-change",
                    "symbol_queried": symbol_upper,
                    "update_frequency": "daily_eod",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"[getStockPerformance] Error for {symbol_upper}: {e}",
                exc_info=True
            )
            return create_error_output(
                tool_name=self.schema.name,
                error=f"Failed to fetch performance: {str(e)}"
            )
    
    async def _fetch_from_fmp(self, symbol: str) -> Optional[Dict]:
        """Fetch performance data from FMP API"""
        url = f"{self.FMP_BASE_URL}/v3/stock-price-change/{symbol}"
        
        params = {
            "apikey": self.api_key
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # FMP returns list with single item or dict
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                elif isinstance(data, dict):
                    return data
                
                return None
                
        except httpx.HTTPStatusError as e:
            self.logger.error(f"FMP HTTP error {e.response.status_code}: {e.response.text}")
            return None
        except Exception as e:
            self.logger.error(f"FMP request error: {e}")
            return None
    
    def _format_performance_data(self, raw_data: Dict, symbol: str) -> Dict[str, Any]:
        """
        Format raw FMP response to tool schema
        
        Args:
            raw_data: FMP API response
            symbol: Stock symbol
            
        Returns:
            Formatted data matching schema
        """
        # Extract performance values
        # FMP returns fields like: 1D, 5D, 1M, 3M, 6M, ytd, 1Y, 3Y, 5Y, 10Y, max
        timeframes = {
            "1_day": {
                "return_percent": self._safe_float(raw_data.get("1D", 0.0)),
                "label": "1 Day"
            },
            "1_week": {
                "return_percent": self._safe_float(raw_data.get("5D", 0.0)),
                "label": "1 Week"
            },
            "1_month": {
                "return_percent": self._safe_float(raw_data.get("1M", 0.0)),
                "label": "1 Month"
            },
            "3_months": {
                "return_percent": self._safe_float(raw_data.get("3M", 0.0)),
                "label": "3 Months"
            },
            "6_months": {
                "return_percent": self._safe_float(raw_data.get("6M", 0.0)),
                "label": "6 Months"
            },
            "ytd": {
                "return_percent": self._safe_float(raw_data.get("ytd", 0.0)),
                "label": "Year to Date"
            },
            "1_year": {
                "return_percent": self._safe_float(raw_data.get("1Y", 0.0)),
                "label": "1 Year"
            }
        }
        
        # Find best and worst performing periods
        best_period = None
        worst_period = None
        best_return = float('-inf')
        worst_return = float('inf')
        
        for period, data in timeframes.items():
            ret = data['return_percent']
            if ret > best_return:
                best_return = ret
                best_period = period
            if ret < worst_return:
                worst_return = ret
                worst_period = period
        
        # Determine momentum trend
        momentum_trend = self._determine_momentum(timeframes)
        
        return {
            "symbol": symbol,
            "timeframes": timeframes,
            "best_timeframe": best_period,
            "best_return": round(best_return, 2),
            "worst_timeframe": worst_period,
            "worst_return": round(worst_return, 2),
            "momentum_trend": momentum_trend,
            "timestamp": datetime.now().isoformat()
        }
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _determine_momentum(self, timeframes: Dict) -> str:
        """Determine overall momentum trend"""
        # Count positive vs negative periods
        positive_count = sum(
            1 for tf in timeframes.values() 
            if tf['return_percent'] > 0
        )
        total_count = len(timeframes)
        
        if positive_count >= total_count * 0.7:
            return "strong_uptrend"
        elif positive_count >= total_count * 0.5:
            return "uptrend"
        elif positive_count >= total_count * 0.3:
            return "mixed"
        else:
            return "downtrend"


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import json
    import os
    
    async def test_tool():
        """Standalone test for GetStockPerformanceTool"""
        
        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            print("❌ ERROR: FMP_API_KEY not found in environment")
            return
        
        print("=" * 80)
        print("TESTING [GetStockPerformanceTool]")
        print("=" * 80)
        
        tool = GetStockPerformanceTool(api_key=api_key)
        
        # Test 1: Valid symbol
        print("\n📊 Test 1: Valid symbol (AAPL)")
        print("-" * 40)
        result = await tool.safe_execute(symbol="AAPL")
        
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_ms}ms")
        
        if result.is_success():
            print("✅ SUCCESS")
            print(f"\nSymbol: {result.data['symbol']}")
            print(f"Momentum: {result.data['momentum_trend']}")
            print(f"\nPerformance by timeframe:")
            for tf, data in result.data['timeframes'].items():
                pct = data['return_percent']
                emoji = "📈" if pct > 0 else "📉"
                print(f"  {emoji} {data['label']}: {pct:+.2f}%")
            
            print(f"\nBest: {result.data['best_timeframe']} ({result.data['best_return']:+.2f}%)")
            print(f"Worst: {result.data['worst_timeframe']} ({result.data['worst_return']:+.2f}%)")
        else:
            print(f"❌ ERROR: {result.error}")
        
        # Test 2: Another symbol
        print("\n📊 Test 2: NVDA performance")
        print("-" * 40)
        result = await tool.safe_execute(symbol="NVDA")
        
        if result.is_success():
            print(f"✅ YTD: {result.data['timeframes']['ytd']['return_percent']:+.2f}%")
            print(f"✅ 1Y: {result.data['timeframes']['1_year']['return_percent']:+.2f}%")
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETE")
        print("=" * 80)
    
    asyncio.run(test_tool())