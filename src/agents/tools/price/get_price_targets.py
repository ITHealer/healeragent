# File: src/agents/tools/price/get_price_targets.py

"""
GetPriceTargetsTool - Atomic Tool for Analyst Price Targets

Responsibility: Lấy analyst consensus price targets
- Target high (highest analyst target)
- Target low (lowest analyst target)
- Target median
- Target consensus (mean)
- Number of analysts

KHÔNG BAO GỒM:
- ❌ Current price (use getStockPrice)
- ❌ Stock recommendations (different endpoint)
- ❌ Earnings estimates (different endpoint)

Data Source: FMP /v4/price-target-consensus?symbol={symbol}
Cache TTL: 1-24 hours (data doesn't change frequently)
"""

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


class GetPriceTargetsTool(BaseTool):
    """
    Atomic tool để lấy analyst price targets
    
    Data source: FMP /v4/price-target-consensus
    
    Usage:
        tool = GetPriceTargetsTool()
        result = await tool.safe_execute(symbol="AAPL")
        
        if result.is_success():
            target_high = result.data['target_high']
            consensus = result.data['target_consensus']
            upside = result.data['upside_potential_pct']
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
            name="getPriceTargets",
            category="price",
            description=(
                "Get analyst price targets and recommendations from major financial institutions. "
                "Returns consensus price targets, analyst ratings, and price projections. "
                "Use when user asks about analyst opinions, price targets, or buy/sell ratings."
            ),
            capabilities=[
                "✅ Analyst consensus price target",
                "✅ High/low price target range",
                "✅ Number of analysts covering",
                "✅ Average rating (Buy/Hold/Sell)",
                "✅ Target vs current price comparison"
            ],
            limitations=[
                "❌ Targets may be outdated (updated quarterly)",
                "❌ Not all stocks have analyst coverage",
                "❌ No real-time target updates"
            ],
            usage_hints=[
                # English
                "User asks: 'Apple price target' → USE THIS",
                "User asks: 'What do analysts say about TSLA?' → USE THIS",
                "User asks: 'Should I buy Microsoft?' → USE THIS",
                # Vietnamese
                "User asks: 'Mục tiêu giá của Amazon' → USE THIS",
                "User asks: 'Các nhà phân tích đánh giá NVDA như thế nào?' → USE THIS",
                # When NOT to use
                "User wants technical analysis targets → DO NOT USE (use getTechnicalIndicators)"
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
                "consensus_target": "number",
                "high_target": "number",
                "low_target": "number",
                "analyst_count": "number",
                "average_rating": "string",
                "timestamp": "string"
            },
            typical_execution_time_ms=1000,
            requires_symbol=True
        )
    
    async def execute(self, symbol: str) -> ToolOutput:
        """
        Execute price targets retrieval
        
        Args:
            symbol: Stock symbol
            
        Returns:
            ToolOutput with price target data
        """
        symbol_upper = symbol.upper()
        
        self.logger.info(f"[getPriceTargets] Executing for symbol={symbol_upper}")
        
        try:
            # Fetch from FMP
            raw_data = await self._fetch_from_fmp(symbol_upper)
            
            if not raw_data:
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"No price target data available for {symbol_upper}"
                )
            
            # Format to schema
            formatted_data = self._format_targets_data(raw_data, symbol_upper)
            
            return create_success_output(
                tool_name=self.schema.name,
                data=formatted_data,
                metadata={
                    "source": "FMP /v4/price-target-consensus",
                    "symbol_queried": symbol_upper,
                    "cache_ttl": "1-24 hours",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"[getPriceTargets] Error for {symbol_upper}: {e}",
                exc_info=True
            )
            return create_error_output(
                tool_name=self.schema.name,
                error=f"Failed to fetch price targets: {str(e)}"
            )
    
    async def _fetch_from_fmp(self, symbol: str) -> Optional[Dict]:
        """Fetch price targets from FMP API"""
        url = f"{self.FMP_BASE_URL}/v4/price-target-consensus"
        
        params = {
            "symbol": symbol,
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
                elif isinstance(data, dict) and data:
                    return data
                
                return None
                
        except httpx.HTTPStatusError as e:
            self.logger.error(f"FMP HTTP error {e.response.status_code}: {e.response.text}")
            return None
        except Exception as e:
            self.logger.error(f"FMP request error: {e}")
            return None
    
    def _format_targets_data(self, raw_data: Dict, symbol: str) -> Dict[str, Any]:
        """
        Format raw FMP response to tool schema
        
        Args:
            raw_data: FMP API response
            symbol: Stock symbol
            
        Returns:
            Formatted data matching schema
        """
        # Extract target values
        target_high = self._safe_float(raw_data.get("targetHigh", 0.0))
        target_low = self._safe_float(raw_data.get("targetLow", 0.0))
        target_median = self._safe_float(raw_data.get("targetMedian", 0.0))
        target_consensus = self._safe_float(raw_data.get("targetConsensus", 0.0))
        
        # Number of analysts
        num_analysts = int(raw_data.get("numberOfAnalysts", 0))
        
        # Current price for comparison (if available)
        current_price = self._safe_float(raw_data.get("currentPrice", 0.0))
        
        # Calculate upside potential
        upside_pct = 0.0
        if current_price > 0 and target_consensus > 0:
            upside_pct = ((target_consensus - current_price) / current_price) * 100
        
        # Calculate target range
        target_range_pct = 0.0
        if target_low > 0 and target_high > 0:
            target_range_pct = ((target_high - target_low) / target_low) * 100
        
        return {
            "symbol": symbol,
            "target_high": round(target_high, 2),
            "target_low": round(target_low, 2),
            "target_median": round(target_median, 2),
            "target_consensus": round(target_consensus, 2),
            "number_of_analysts": num_analysts,
            "current_price": round(current_price, 2) if current_price > 0 else None,
            "upside_potential_pct": round(upside_pct, 2),
            "target_range_pct": round(target_range_pct, 2),
            "timestamp": datetime.now().isoformat()
        }
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import json
    import os
    
    async def test_tool():
        """Standalone test for GetPriceTargetsTool"""
        
        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            print("❌ ERROR: FMP_API_KEY not found in environment")
            return
        
        print("=" * 80)
        print("TESTING [GetPriceTargetsTool]")
        print("=" * 80)
        
        tool = GetPriceTargetsTool(api_key=api_key)
        
        # Test 1: Valid symbol
        print("\n📊 Test 1: Valid symbol (AAPL)")
        print("-" * 40)
        result = await tool.safe_execute(symbol="AAPL")
        
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_ms}ms")
        
        if result.is_success():
            print("✅ SUCCESS")
            print(f"\nSymbol: {result.data['symbol']}")
            print(f"Analysts: {result.data['number_of_analysts']}")
            print(f"\nPrice Targets:")
            print(f"  High: ${result.data['target_high']:,.2f}")
            print(f"  Consensus: ${result.data['target_consensus']:,.2f}")
            print(f"  Median: ${result.data['target_median']:,.2f}")
            print(f"  Low: ${result.data['target_low']:,.2f}")
            
            if result.data['current_price']:
                print(f"\nCurrent Price: ${result.data['current_price']:,.2f}")
                upside = result.data['upside_potential_pct']
                emoji = "🎯" if upside > 0 else "⚠️"
                print(f"{emoji} Upside to Consensus: {upside:+.2f}%")
            
            print(f"Target Range: {result.data['target_range_pct']:.2f}%")
        else:
            print(f"❌ ERROR: {result.error}")
        
        # Test 2: Another symbol
        print("\n📊 Test 2: TSLA targets")
        print("-" * 40)
        result = await tool.safe_execute(symbol="TSLA")
        
        if result.is_success():
            print(f"✅ Consensus: ${result.data['target_consensus']:,.2f}")
            print(f"✅ Upside: {result.data['upside_potential_pct']:+.2f}%")
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETE")
        print("=" * 80)
    
    asyncio.run(test_tool())