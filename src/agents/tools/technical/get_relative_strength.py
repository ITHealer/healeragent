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
    from src.handlers.relative_strength_handler import RelativeStrengthHandler
    from src.stock.crawlers.market_data_provider import MarketData
except ImportError:
    RelativeStrengthHandler = None
    MarketData = None


class GetRelativeStrengthTool(BaseTool):
    """
    Atomic tool để tính relative strength
    
    Wraps: RelativeStrengthHandler.get_relative_strength()
    
    Usage:
        tool = GetRelativeStrengthTool()
        result = await tool.safe_execute(symbol="AAPL", benchmark="SPY")
        
        if result.is_success():
            outperformance = result.data['relative_performance']
            trend = result.data['trend']
    """
    
    def __init__(self):
        """Initialize tool"""
        super().__init__()
        
        if RelativeStrengthHandler is None or MarketData is None:
            raise ImportError(
                "RelativeStrengthHandler or MarketData not found. "
                "Make sure dependencies are available"
            )
        
        # Initialize handler with market data
        self.market_data = MarketData()
        self.rs_handler = RelativeStrengthHandler(self.market_data)
        self.logger = logging.getLogger(__name__)
        
        # Define schema
        self.schema = ToolSchema(
            name="getRelativeStrength",
            category="technical",
            description=(
                "Calculate relative strength of a stock compared to market benchmark (S&P 500) "
                "or peer stocks. Returns RS rating and performance comparison. "
                "Use when user asks about stock strength, relative performance, or market outperformance."
            ),
            capabilities=[
                "✅ Relative strength vs S&P 500",
                "✅ Relative strength vs sector peers",
                "✅ RS rating (0-100 scale)",
                "✅ Outperformance/underperformance metrics",
                "✅ Momentum comparison",
                "✅ Multiple timeframe analysis"
            ],
            limitations=[
                "❌ Requires historical data for benchmark",
                "❌ RS rating updated daily (not real-time)",
                "❌ Peer comparison limited to same sector",
                "❌ One symbol at a time"
            ],
            usage_hints=[
                # English
                "User asks: 'Apple relative strength' → USE THIS with symbol=AAPL",
                "User asks: 'How strong is TSLA vs market?' → USE THIS with symbol=TSLA",
                "User asks: 'Is NVDA outperforming?' → USE THIS with symbol=NVDA",
                "User asks: 'Microsoft RS rating' → USE THIS with symbol=MSFT",
                
                # Vietnamese
                "User asks: 'Sức mạnh tương đối của Apple' → USE THIS with symbol=AAPL",
                "User asks: 'Tesla mạnh hơn thị trường không?' → USE THIS with symbol=TSLA",
                "User asks: 'NVDA có outperform không?' → USE THIS with symbol=NVDA",
                
                # When NOT to use
                "User asks about absolute PRICE → DO NOT USE (use getStockPrice)",
                "User asks about technical INDICATORS → DO NOT USE (use getTechnicalIndicators)"
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
                    name="benchmark",
                    type="string",
                    description="Benchmark for comparison (default: SPY for S&P 500)",
                    required=False,
                    default="SPY"
                ),
                ToolParameter(
                    name="timeframe",
                    type="string",
                    description="Timeframe for RS calculation (default: 3M)",
                    required=False,
                    default="3M",
                    allowed_values=["1M", "3M", "6M", "1Y"]
                )
            ],
            returns={
                "symbol": "string",
                "benchmark": "string",
                "timeframe": "string",
                "relative_performance": "object",
                "is_outperforming": "boolean",
                "trend": "string",
                "percentile_rank": "number",
                "strength_score": "number",
                "summary": "string"
            },
            typical_execution_time_ms=1200,
            requires_symbol=True
        )
    
    async def execute(
        self, 
        symbol: str, 
        benchmark: str = "SPY",
        timeframe: str = "3M" 
    ) -> ToolOutput:
        """
        Execute relative strength analysis
        
        ✅ FIX: Added timeframe parameter
        
        Args:
            symbol: Stock symbol
            benchmark: Benchmark symbol
            timeframe: Analysis timeframe (logged, handler may not use yet)
        """
        symbol_upper = symbol.upper()
        benchmark_upper = benchmark.upper()
        
        self.logger.info(
            f"[getRelativeStrength] symbol={symbol_upper}, "
            f"benchmark={benchmark_upper}, timeframe={timeframe}"
        )
        
        try:
            rs_results = await self.rs_handler.get_relative_strength(
                symbol=symbol_upper,
                benchmark=benchmark_upper
            )
            
            if not rs_results or "error" in rs_results:
                error_msg = rs_results.get("error", "No data available")
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"Relative strength failed: {error_msg}"
                )
            
            formatted_data = self._format_rs_data(
                rs_results,
                symbol_upper,
                benchmark_upper,
                timeframe
            )

            # Generate LLM-friendly summary
            llm_summary = self._generate_llm_summary(formatted_data)

            return create_success_output(
                tool_name=self.schema.name,
                data=formatted_data,
                formatted_context=llm_summary,
                metadata={
                    "source": "RelativeStrengthHandler",
                    "symbol_queried": symbol_upper,
                    "benchmark": benchmark_upper,
                    "timeframe_requested": timeframe,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"[getRelativeStrength] Error: {e}", exc_info=True)
            return create_error_output(
                tool_name=self.schema.name,
                error=str(e)
            )
    
    def _format_rs_data(
        self,
        raw_data: Dict,
        symbol: str,
        benchmark: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Format raw handler output"""
        relative_perf = raw_data.get("relative_performance", {})
        recent_performance = relative_perf.get("1M", {}).get("relative_strength", 0)
        
        return {
            "symbol": symbol,
            "benchmark": benchmark,
            "timeframe": timeframe,  # ✅ ADDED
            "relative_performance": relative_perf,
            "is_outperforming": recent_performance > 0,
            "trend": self._calculate_trend(relative_perf),
            "percentile_rank": raw_data.get("percentile_rank", 50),
            "strength_score": recent_performance,
            "summary": raw_data.get("relative_strength_summary", "Analysis completed"),
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_trend(self, relative_perf: Dict) -> str:
        """Calculate trend"""
        try:
            one_m = relative_perf.get("1M", {}).get("relative_strength", 0)
            three_m = relative_perf.get("3M", {}).get("relative_strength", 0)

            if one_m > three_m + 2:
                return "improving"
            elif one_m < three_m - 2:
                return "declining"
            return "stable"
        except:
            return "unknown"

    def _generate_llm_summary(self, data: Dict[str, Any]) -> str:
        """Generate LLM-friendly summary for relative strength data."""
        symbol = data.get("symbol", "N/A")
        benchmark = data.get("benchmark", "SPY")
        timeframe = data.get("timeframe", "3M")
        is_outperforming = data.get("is_outperforming", False)
        trend = data.get("trend", "unknown")
        percentile = data.get("percentile_rank", 50)
        strength_score = data.get("strength_score", 0)
        summary = data.get("summary", "")
        relative_perf = data.get("relative_performance", {})

        lines = [
            f"=== RELATIVE STRENGTH: {symbol} vs {benchmark} ===",
            f"Timeframe: {timeframe}",
            "",
            f"OUTPERFORMING: {'YES ✅' if is_outperforming else 'NO ❌'}",
            f"Trend: {trend.upper()}",
            f"Percentile Rank: {percentile}",
            f"Strength Score: {strength_score:+.2f}%",
            "",
            "PERFORMANCE BY PERIOD:",
        ]

        # Add relative performance by period
        for period, perf_data in relative_perf.items():
            if isinstance(perf_data, dict):
                rs = perf_data.get("relative_strength", 0)
                sign = "+" if rs >= 0 else ""
                lines.append(f"- {period}: {sign}{rs:.2f}%")

        if summary:
            lines.extend(["", f"Summary: {summary}"])

        return "\n".join(lines)


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import json
    
    async def test_tool():
        """Standalone test for GetRelativeStrengthTool"""
        
        print("=" * 80)
        print("TESTING [GetRelativeStrengthTool]")
        print("=" * 80)
        
        tool = GetRelativeStrengthTool()
        
        # Test 1: AAPL vs SPY
        print("\n📊 Test 1: AAPL vs SPY")
        print("-" * 40)
        result = await tool.safe_execute(symbol="AAPL", benchmark="SPY")
        
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_ms}ms")
        
        if result.is_success():
            print("✅ SUCCESS")
            print(f"Outperforming: {result.data['is_outperforming']}")
            print(f"Trend: {result.data['trend']}")
            print(f"Strength score: {result.data['strength_score']:.2f}%")
        else:
            print(f"❌ ERROR: {result.error}")
        
        # Test 2: NVDA vs QQQ
        print("\n📊 Test 2: NVDA vs QQQ")
        print("-" * 40)
        result = await tool.safe_execute(symbol="NVDA", benchmark="QQQ")
        
        if result.is_success():
            print(f"✅ Trend: {result.data['trend']}")
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETE")
        print("=" * 80)
    
    asyncio.run(test_tool())