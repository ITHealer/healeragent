import logging
from typing import Dict, Any, Optional, List
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
    Atomic tool for volume profile analysis
    
    Wraps: VolumeProfileHandler.get_volume_profile()
    
    Returns ALL required fields including volume distribution
    """
    
    def __init__(self):
        """Initialize tool"""
        super().__init__()
        
        if VolumeProfileHandler is None:
            raise ImportError(
                "VolumeProfileHandler not found. "
                "Make sure src.handlers.volume_profile_handler is available"
            )
        
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
                "User asks: 'Apple volume profile' → USE THIS with symbol=AAPL",
                "User asks: 'Show me TSLA volume distribution' → USE THIS with symbol=TSLA",
                "User asks: 'Phân tích khối lượng Apple' → USE THIS with symbol=AAPL",
                "User asks about PRICE only → DO NOT USE (use getStockPrice)"
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
                "volume_profile": "object",
                "poc": "number",
                "value_area_high": "number",
                "value_area_low": "number",
                "volume_trend": "string",
                "avg_daily_volume": "number",
                "timestamp": "string"
            },
            typical_execution_time_ms=1400,
            requires_symbol=True
        )
    
    async def execute(
        self,
        symbol: str,
        timeframe: str = "1M",
        lookback_days: int = 60,
        num_bins: int = 10
    ) -> ToolOutput:
        """
        Execute volume profile calculation
        
        Args:
            symbol: Stock ticker
            timeframe: Timeframe string (1W, 1M, 3M, 6M)
            lookback_days: Manual override for days
            num_bins: Number of price bins
            
        Returns:
            ToolOutput with ALL required fields
        """
        start_time = datetime.now()
        symbol_upper = symbol.upper()
        
        # Map timeframe to lookback_days
        timeframe_map = {
            "1W": 7,
            "1M": 30,
            "3M": 90,
            "6M": 180
        }
        
        if timeframe and timeframe in timeframe_map:
            calculated_lookback = timeframe_map[timeframe]
            self.logger.info(f"Mapped timeframe '{timeframe}' to {calculated_lookback} days")
            lookback_days = calculated_lookback
        
        # Validate parameters - ensure int for slicing
        lookback_days = int(max(7, min(365, lookback_days)))
        num_bins = int(max(5, min(50, num_bins)))
        
        self.logger.info(
            f"[getVolumeProfile] Executing: symbol={symbol_upper}, "
            f"timeframe={timeframe}, lookback={lookback_days}, bins={num_bins}"
        )
        
        try:
            # ═══════════════════════════════════════════════════════════
            # STEP 1: Try with requested lookback
            # ═══════════════════════════════════════════════════════════
            raw_data = await self.volume_handler.get_volume_profile(
                symbol=symbol_upper,
                lookback_days=lookback_days,
                num_bins=num_bins
            )
            
            # ═══════════════════════════════════════════════════════════
            # STEP 2: Check if data is insufficient
            # ═══════════════════════════════════════════════════════════
            if raw_data:
                poc = raw_data.get("poc")
                
                # If POC is None and lookback < 90, try with more data
                if (poc is None or poc == 0) and lookback_days < 90:
                    self.logger.warning(
                        f"[{symbol_upper}] Insufficient data with {lookback_days} days. "
                        f"Retrying with 90 days..."
                    )
                    
                    raw_data = await self.volume_handler.get_volume_profile(
                        symbol=symbol_upper,
                        lookback_days=90,
                        num_bins=num_bins
                    )
                    
                    lookback_days = 90
            
            # ═══════════════════════════════════════════════════════════
            # STEP 3: Final check
            # ═══════════════════════════════════════════════════════════
            if not raw_data:
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"No volume data available for {symbol_upper}",
                    metadata={
                        "symbol": symbol_upper,
                        "lookback_days_tried": [lookback_days, 90] if lookback_days < 90 else [lookback_days]
                    }
                )
            
            # ═══════════════════════════════════════════════════════════
            # STEP 4: Format with ALL required fields
            # ═══════════════════════════════════════════════════════════
            formatted_data = self._format_volume_profile(raw_data, symbol_upper)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            self.logger.info(
                f"[getVolumeProfile] ✅ SUCCESS ({int(execution_time)}ms) - "
                f"POC: ${formatted_data['poc']:.2f}, "
                f"Trend: {formatted_data['volume_trend']}"
            )
            
            return create_success_output(
                tool_name=self.schema.name,
                data=formatted_data,
                metadata={
                    "source": "VolumeProfileHandler",
                    "symbol_queried": symbol_upper,
                    "timeframe_used": timeframe,
                    "lookback_days_used": lookback_days,
                    "num_bins": num_bins,
                    "execution_time_ms": int(execution_time),
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
    
    def _format_volume_profile(
        self,
        raw_data: Dict,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Format volume profile data with ALL required fields
        
        ✅ Returns:
        - volume_profile (object/dict - REQUIRED)
        - volume_trend (string - REQUIRED)
        - avg_daily_volume (number - REQUIRED)
        - poc, value_area_high, value_area_low
        """
        # ═══════════════════════════════════════════════════════════
        # Extract POC with safe fallback
        # ═══════════════════════════════════════════════════════════
        poc = raw_data.get("poc", 0.0)
        poc_volume = raw_data.get("poc_volume", 0)
        
        # If POC is None/0, estimate from value area
        if not poc or poc == 0:
            value_area_high = raw_data.get("value_area_high", 0)
            value_area_low = raw_data.get("value_area_low", 0)
            
            if value_area_high > 0 and value_area_low > 0:
                poc = (value_area_high + value_area_low) / 2
                self.logger.warning(
                    f"[{symbol}] POC estimated from value area: ${poc:.2f}"
                )
        
        # ═══════════════════════════════════════════════════════════
        # Extract value area
        # ═══════════════════════════════════════════════════════════
        value_area_high = raw_data.get("value_area_high", 0.0)
        value_area_low = raw_data.get("value_area_low", 0.0)
        value_area_volume_pct = raw_data.get("value_area_volume_pct", 70.0)
        
        # ═══════════════════════════════════════════════════════════
        # ✅ REQUIRED: volume_profile (object/dict)
        # ═══════════════════════════════════════════════════════════
        volume_distribution = raw_data.get("volume_distribution", [])
        high_volume_nodes = raw_data.get("high_volume_nodes", [])
        low_volume_nodes = raw_data.get("low_volume_nodes", [])
        
        # Build volume_profile object
        volume_profile = {
            "distribution": volume_distribution,
            "high_volume_nodes": high_volume_nodes[:5],
            "low_volume_nodes": low_volume_nodes[:5],
            "distribution_count": len(volume_distribution),
            "poc_price": round(float(poc), 2) if poc else 0.0,
            "poc_volume": int(poc_volume) if poc_volume else 0
        }
        
        # ═══════════════════════════════════════════════════════════
        # ✅ REQUIRED: volume_trend (string)
        # ═══════════════════════════════════════════════════════════
        volume_trend = self._calculate_volume_trend(raw_data)
        
        # ═══════════════════════════════════════════════════════════
        # ✅ REQUIRED: avg_daily_volume (number)
        # ═══════════════════════════════════════════════════════════
        avg_daily_volume = self._calculate_avg_daily_volume(raw_data)
        
        # ═══════════════════════════════════════════════════════════
        # Return ALL required fields
        # ═══════════════════════════════════════════════════════════
        return {
            # ✅ Required fields from schema
            "symbol": symbol,
            "volume_profile": volume_profile,              # ← REQUIRED
            "poc": round(float(poc), 2) if poc else 0.0,
            "value_area_high": round(float(value_area_high), 2) if value_area_high else 0.0,
            "value_area_low": round(float(value_area_low), 2) if value_area_low else 0.0,
            "volume_trend": volume_trend,                  # ← REQUIRED
            "avg_daily_volume": avg_daily_volume,          # ← REQUIRED
            "timestamp": datetime.now().isoformat(),
            
            # Additional fields (not in schema but useful)
            "poc_volume": int(poc_volume) if poc_volume else 0,
            "value_area_volume_pct": round(float(value_area_volume_pct), 2),
            "summary": raw_data.get("summary", f"Volume profile for {symbol}"),
            "data_quality": self._assess_data_quality(poc, value_area_high, value_area_low)
        }
    
    def _calculate_volume_trend(self, raw_data: Dict) -> str:
        """
        Calculate volume trend from distribution data
        
        Returns: "increasing", "decreasing", or "stable"
        """
        volume_distribution = raw_data.get("volume_distribution", [])
        
        if not volume_distribution or len(volume_distribution) < 10:
            return "insufficient_data"
        
        # Split into recent vs older
        mid_point = len(volume_distribution) // 2
        recent_volumes = volume_distribution[:mid_point]
        older_volumes = volume_distribution[mid_point:]
        
        # Calculate averages
        recent_avg = sum(v.get('volume', 0) for v in recent_volumes) / len(recent_volumes) if recent_volumes else 0
        older_avg = sum(v.get('volume', 0) for v in older_volumes) / len(older_volumes) if older_volumes else 0
        
        if older_avg == 0:
            return "stable"
        
        # Compare
        change_pct = ((recent_avg - older_avg) / older_avg) * 100
        
        if change_pct > 20:
            return "increasing"
        elif change_pct < -20:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_avg_daily_volume(self, raw_data: Dict) -> int:
        """
        Calculate average daily volume
        
        Returns: Integer volume (required field)
        """
        volume_distribution = raw_data.get("volume_distribution", [])
        
        if not volume_distribution:
            # Try to get from summary or other fields
            poc_volume = raw_data.get("poc_volume", 0)
            if poc_volume:
                # Estimate based on POC (rough approximation)
                return int(poc_volume * 1.5)
            return 0
        
        # Calculate from distribution
        total_volume = sum(v.get('volume', 0) for v in volume_distribution)
        
        if total_volume == 0:
            return 0
        
        # Average across distribution
        avg_volume = total_volume / len(volume_distribution)
        
        return int(avg_volume)
    
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
    
    async def test_tool():
        """Test GetVolumeProfileTool with validation"""
        
        print("=" * 80)
        print("TESTING [GetVolumeProfileTool] - FIXED VERSION")
        print("=" * 80)
        
        tool = GetVolumeProfileTool()
        
        # Test 1: TSLA with 1M timeframe
        print("\n📊 Test 1: TSLA (timeframe=1M)")
        print("-" * 40)
        result = await tool.safe_execute(symbol="TSLA", timeframe="1M")
        
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_ms}ms")
        
        if result.is_success():
            # Verify ALL required fields
            required_fields = tool.schema.get_required_fields()
            missing = [f for f in required_fields if f not in result.data]
            
            if missing:
                print(f"⚠️  PARTIAL - Missing: {missing}")
            else:
                print(f"✅ ALL REQUIRED FIELDS PRESENT")
            
            print(f"\nData:")
            print(f"  POC: ${result.data['poc']:.2f}")
            print(f"  Value Area: ${result.data['value_area_low']:.2f} - ${result.data['value_area_high']:.2f}")
            print(f"  Volume Trend: {result.data['volume_trend']}")  # ← Required
            print(f"  Avg Daily Vol: {result.data['avg_daily_volume']:,}")  # ← Required
            
            # Check volume_profile object
            vp = result.data.get('volume_profile', {})
            print(f"  Volume Profile Keys: {list(vp.keys())}")  # ← Required
            print(f"  Distribution Count: {vp.get('distribution_count', 0)}")
        else:
            print(f"❌ ERROR: {result.error}")
        
        # Test 2: NVDA with 3M
        print("\n📊 Test 2: NVDA (timeframe=3M)")
        print("-" * 40)
        result2 = await tool.safe_execute(symbol="NVDA", timeframe="3M")
        
        if result2.is_success():
            print(f"✅ POC: ${result2.data['poc']:.2f}")
            print(f"✅ Trend: {result2.data['volume_trend']}")
            print(f"✅ Avg Vol: {result2.data['avg_daily_volume']:,}")
        
        print("\n" + "=" * 80)
    
    asyncio.run(test_tool())