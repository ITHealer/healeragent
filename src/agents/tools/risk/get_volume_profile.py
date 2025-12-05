# File: src/agents/tools/risk/get_volume_profile.py

"""
GetVolumeProfileTool - Atomic Tool for Volume Profile Analysis

Responsibility: Phân tích phân phối khối lượng giao dịch
- Point of Control (POC) - giá có volume cao nhất
- Value Area - vùng tập trung 70% volume
- High/Low Volume Nodes
- Volume distribution analysis

KHÔNG BAO GỒM:
- ❌ Price data (use getStockPrice)
- ❌ Technical indicators (use getTechnicalIndicators)
- ❌ Chart patterns (use detectChartPatterns)

This tool WRAPS existing VolumeProfileHandler
"""

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

# Import existing handler
try:
    from src.handlers.volume_profile_handler import VolumeProfileHandler
except ImportError:
    VolumeProfileHandler = None


class GetVolumeProfileTool(BaseTool):
    """
    Atomic tool để phân tích volume profile
    
    Wraps: VolumeProfileHandler.get_volume_profile()
    
    Usage:
        tool = GetVolumeProfileTool()
        result = await tool.safe_execute(symbol="AAPL", lookback_days=60)
        
        if result.is_success():
            poc = result.data['poc']
            value_area = result.data['value_area']
    """
    
    def __init__(self):
        """Initialize tool"""
        super().__init__()
        
        if VolumeProfileHandler is None:
            raise ImportError(
                "VolumeProfileHandler not found. "
                "Make sure src.handlers.volume_profile_handler is available"
            )
        
        # Initialize handler
        self.volume_handler = VolumeProfileHandler()
        self.logger = logging.getLogger(__name__)
        
        # Define schema
        self.schema = ToolSchema(
            name="getVolumeProfile",
            category="risk",
            description=(
                "Analyze trading volume distribution across price levels. "
                "Identifies high-volume nodes (support/resistance) and volume trends. "
                "Use when user asks about volume analysis, liquidity, or institutional activity."
            ),
            capabilities=[
                "✅ Volume-by-price distribution",
                "✅ Point of Control (POC) - highest volume price",
                "✅ Value Area (70% of volume)",
                "✅ Volume trend analysis",
                "✅ Liquidity zones",
                "✅ Institutional activity indicators"
            ],
            limitations=[
                "❌ Requires detailed intraday volume data",
                "❌ Volume profile changes with time",
                "❌ Historical volume doesn't predict future",
                "❌ One symbol at a time"
            ],
            usage_hints=[
                # English
                "User asks: 'Apple volume profile' → USE THIS with symbol=AAPL",
                "User asks: 'Show me TSLA volume distribution' → USE THIS with symbol=TSLA",
                "User asks: 'Where is NVDA high volume?' → USE THIS with symbol=NVDA",
                "User asks: 'Microsoft volume analysis' → USE THIS with symbol=MSFT",
                
                # Vietnamese
                "User asks: 'Phân tích khối lượng Apple' → USE THIS with symbol=AAPL",
                "User asks: 'Volume profile của Tesla' → USE THIS with symbol=TSLA",
                "User asks: 'Khối lượng giao dịch NVDA' → USE THIS with symbol=NVDA",
                
                # When NOT to use
                "User asks about PRICE only → DO NOT USE (use getStockPrice)",
                "User asks about RISK → DO NOT USE (use assessRisk)"
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol",
                    required=True,
                    pattern="^[A-Z]{1,7}$"
                ),
                ToolParameter(
                    name="timeframe",
                    type="string",
                    description="Timeframe for volume analysis",
                    required=False,
                    default="1M",
                    allowed_values=["1W", "1M", "3M", "6M"]
                )
            ],
            returns={
                "symbol": "string",
                "volume_profile": "array - Volume distribution by price",
                "poc": "number - Point of Control price",
                "value_area_high": "number",
                "value_area_low": "number",
                "volume_trend": "string - Increasing/Decreasing/Stable",
                "avg_daily_volume": "number",
                "timestamp": "string"
            },
            typical_execution_time_ms=1400,
            requires_symbol=True
        )
            
    async def execute(
        self,
        symbol: str,
        lookback_days: int = 60,
        num_bins: int = 10
    ) -> ToolOutput:
        """
        Execute volume profile calculation with adaptive lookback
        
        ENHANCED: Automatically increase lookback if data insufficient
        """
        symbol_upper = symbol.upper()
        
        # Validate and constrain parameters
        lookback_days = max(30, min(252, lookback_days))
        num_bins = max(5, min(20, num_bins))
        
        self.logger.info(
            f"[getVolumeProfile] Executing for symbol={symbol_upper}, "
            f"lookback_days={lookback_days}, num_bins={num_bins}"
        )
        
        try:
            # ════════════════════════════════════════════════════════
            # Try with requested lookback first
            # ════════════════════════════════════════════════════════
            raw_data = await self.volume_handler.get_volume_profile(
                symbol=symbol_upper,
                lookback_days=lookback_days,
                num_bins=num_bins
            )
            
            # ════════════════════════════════════════════════════════
            # Check if data is insufficient
            # ════════════════════════════════════════════════════════
            if raw_data:
                poc = raw_data.get("poc")
                va_high = raw_data.get("value_area_high")
                
                # If POC is None and lookback < 90, try with more data
                if (poc is None or poc == 0) and lookback_days < 90:
                    self.logger.warning(
                        f"[{symbol_upper}] Insufficient data with {lookback_days} days. "
                        f"Retrying with 90 days..."
                    )
                    
                    # Retry with 90 days
                    raw_data = await self.volume_handler.get_volume_profile(
                        symbol=symbol_upper,
                        lookback_days=90,
                        num_bins=num_bins
                    )
                    
                    lookback_days = 90  # Update for metadata
            
            # ════════════════════════════════════════════════════════
            # Final check
            # ════════════════════════════════════════════════════════
            if not raw_data:
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"No volume data available for {symbol_upper}. "
                        f"Symbol may have insufficient trading history.",
                    metadata={
                        "symbol": symbol_upper,
                        "lookback_days_tried": [lookback_days, 90] if lookback_days < 90 else [lookback_days]
                    }
                )
            
            # Format data (with safe null handling)
            formatted_data = self._format_volume_profile(raw_data, symbol_upper)
            
            return create_success_output(
                tool_name=self.schema.name,
                data=formatted_data,
                metadata={
                    "source": "VolumeProfileHandler",
                    "symbol_queried": symbol_upper,
                    "lookback_days_used": lookback_days,
                    "num_bins": num_bins,
                    "data_quality": formatted_data.get("data_quality", "unknown"),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"[getVolumeProfile] Error for {symbol_upper}: {e}",
                exc_info=True
            )
            return create_error_output(
                tool_name=self.schema.name,
                error=f"Volume profile calculation failed: {str(e)}",
                metadata={"symbol": symbol_upper}
            )
    
    def _format_volume_data(self, raw_data: Dict, symbol: str) -> Dict[str, Any]:
        """
        Format raw handler output to tool schema
        
        Args:
            raw_data: Output from VolumeProfileHandler
            symbol: Stock symbol
            
        Returns:
            Formatted data matching schema
        """
        # Extract key metrics
        poc = raw_data.get("poc", {})
        value_area = raw_data.get("value_area", {})
        
        return {
            "symbol": symbol,
            "poc": poc.get("price", 0.0),
            "poc_volume": poc.get("volume", 0),
            "value_area_high": value_area.get("high", 0.0),
            "value_area_low": value_area.get("low", 0.0),
            "value_area_volume_pct": 70.0,  # Standard 70% definition
            "volume_distribution": raw_data.get("volume_distribution", []),
            "high_volume_nodes": raw_data.get("high_volume_nodes", []),
            "low_volume_nodes": raw_data.get("low_volume_nodes", []),
            "summary": raw_data.get("summary", "Volume profile analysis completed"),
            "timestamp": datetime.now().isoformat()
        }


    def _format_volume_profile(
        self,
        raw_data: Dict,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Format volume profile data with safe null handling
        
        FIXED: Return 0.0 instead of None for numeric fields
        """
        
        # ════════════════════════════════════════════════════════════
        # Extract POC with safe fallback
        # ════════════════════════════════════════════════════════════
        poc = raw_data.get("poc")
        poc_volume = raw_data.get("poc_volume")
        
        # If POC is None, calculate from value area middle
        if poc is None or poc == 0:
            value_area_high = raw_data.get("value_area_high", 0)
            value_area_low = raw_data.get("value_area_low", 0)
            
            if value_area_high > 0 and value_area_low > 0:
                # Estimate POC as middle of value area
                poc = (value_area_high + value_area_low) / 2
                self.logger.warning(
                    f"[{symbol}] POC not calculated - estimated from value area: ${poc:.2f}"
                )
            else:
                poc = 0.0
                self.logger.warning(
                    f"[{symbol}] Insufficient data to calculate POC - returning 0.0"
                )
        
        # Safe extraction with defaults
        value_area_high = raw_data.get("value_area_high") or 0.0
        value_area_low = raw_data.get("value_area_low") or 0.0
        value_area_volume_pct = raw_data.get("value_area_volume_pct") or 70.0
        
        # Volume distribution
        volume_distribution = raw_data.get("volume_distribution", [])
        
        # High/Low volume nodes
        high_volume_nodes = raw_data.get("high_volume_nodes", [])
        low_volume_nodes = raw_data.get("low_volume_nodes", [])
        
        # Summary
        summary = raw_data.get("summary", f"Volume profile analysis for {symbol}")
        
        return {
            "symbol": symbol,
            "poc": round(float(poc), 2) if poc else 0.0,
            "poc_volume": int(poc_volume) if poc_volume else 0,
            "value_area_high": round(float(value_area_high), 2) if value_area_high else 0.0,
            "value_area_low": round(float(value_area_low), 2) if value_area_low else 0.0,
            "value_area_volume_pct": round(float(value_area_volume_pct), 2),
            "volume_distribution": volume_distribution,
            "high_volume_nodes": high_volume_nodes[:5],  # Top 5
            "low_volume_nodes": low_volume_nodes[:5],  # Top 5
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
            "data_quality": self._assess_data_quality(poc, value_area_high, value_area_low)
        }

    def _assess_data_quality(
        self,
        poc: float,
        value_area_high: float,
        value_area_low: float
    ) -> str:
        """Assess quality of volume profile data"""
        if poc > 0 and value_area_high > 0 and value_area_low > 0:
            return "complete"
        elif poc > 0 or (value_area_high > 0 and value_area_low > 0):
            return "partial"
        else:
            return "insufficient"
        

# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import json
    
    async def test_tool():
        """Standalone test for GetVolumeProfileTool"""
        
        print("=" * 80)
        print("TESTING [GetVolumeProfileTool]")
        print("=" * 80)
        
        tool = GetVolumeProfileTool()
        
        # Test 1: Valid symbol with default params
        print("\n📊 Test 1: Valid symbol (AAPL) with defaults")
        print("-" * 40)
        result = await tool.safe_execute(symbol="AAPL")
        
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_ms}ms")
        
        if result.is_success():
            print("✅ SUCCESS")
            print(json.dumps(result.data, indent=2, default=str))
        else:
            print(f"❌ ERROR: {result.error}")
        
        # Test 2: Custom parameters
        print("\n📊 Test 2: NVDA with custom lookback")
        print("-" * 40)
        result = await tool.safe_execute(
            symbol="NVDA",
            lookback_days=90,
            num_bins=15
        )
        
        print(f"Status: {result.status}")
        if result.is_success():
            print(f"✅ POC: ${result.data['poc']:.2f}")
            print(f"✅ Value Area: ${result.data['value_area_low']:.2f} - ${result.data['value_area_high']:.2f}")
        
        # Test 3: Invalid symbol
        print("\n📊 Test 3: Invalid symbol")
        print("-" * 40)
        result = await tool.safe_execute(symbol="INVALID")
        
        if not result.is_success():
            print(f"✅ Expected error: {result.error}")
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETE")
        print("=" * 80)
    
    asyncio.run(test_tool())