# File: src/agents/tools/technical/detect_chart_patterns.py

"""
DetectChartPatternsTool - Atomic Tool for Chart Pattern Recognition

Responsibility: Nhận diện các chart patterns phổ biến
- Double Top/Bottom patterns
- Head & Shoulders patterns
- Breakout patterns
- Consolidation patterns
- Pattern confidence scores

KHÔNG BAO GỒM:
- ❌ Price data (use getStockPrice)
- ❌ Technical indicators (use getTechnicalIndicators)
- ❌ Volume analysis (use getVolumeProfile)
- ❌ Trading signals (use assessRisk)

This tool WRAPS existing PatternRecognitionHandler
"""

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
    from src.handlers.pattern_recognition_handler import PatternRecognitionHandler
    from src.stock.crawlers.market_data_provider import MarketData
except ImportError:
    PatternRecognitionHandler = None
    MarketData = None


class DetectChartPatternsTool(BaseTool):
    """
    Atomic tool để nhận diện chart patterns
    
    Wraps: PatternRecognitionHandler.analyze_patterns()
    
    Usage:
        tool = DetectChartPatternsTool()
        result = await tool.safe_execute(symbol="AAPL", lookback_days=90)
        
        if result.is_success():
            patterns = result.data['patterns_detected']
            bullish = result.data['bullish_patterns']
    """
    
    def __init__(self):
        """Initialize tool"""
        super().__init__()
        
        if PatternRecognitionHandler is None or MarketData is None:
            raise ImportError(
                "PatternRecognitionHandler or MarketData not found. "
                "Make sure dependencies are available"
            )
        
        # Initialize handler with market data
        self.market_data = MarketData()
        self.pattern_handler = PatternRecognitionHandler(self.market_data)
        self.logger = logging.getLogger(__name__)
        
        # Define schema
        self.schema = ToolSchema(
            name="detectChartPatterns",
            category="technical",
            description=(
                "Detect chart patterns in stock price data (Head & Shoulders, Double Top/Bottom, "
                "Triangles, Flags, Wedges, etc.). Returns identified patterns with confidence scores. "
                "Use when user asks about chart patterns, price patterns, or pattern recognition."
            ),
            capabilities=[
                "✅ Classic reversal patterns (Head & Shoulders, Double Top/Bottom)",
                "✅ Continuation patterns (Flags, Pennants, Triangles)",
                "✅ Candlestick patterns (Doji, Hammer, Engulfing)",
                "✅ Pattern confidence scoring",
                "✅ Price target projections",
                "✅ Pattern completion status"
            ],
            limitations=[
                "❌ Requires minimum 100 data points for pattern detection",
                "❌ Pattern recognition is probabilistic (not 100% accurate)",
                "❌ Historical patterns only (no real-time detection)",
                "❌ One symbol at a time"
            ],
            usage_hints=[
                # English
                "User asks: 'Chart patterns for Apple' → USE THIS with symbol=AAPL",
                "User asks: 'Detect patterns in TSLA' → USE THIS with symbol=TSLA",
                "User asks: 'Is there a head and shoulders in Microsoft?' → USE THIS with symbol=MSFT",
                "User asks: 'Show me NVDA chart patterns' → USE THIS with symbol=NVDA",
                
                # Vietnamese
                "User asks: 'Các mô hình biểu đồ của Apple' → USE THIS with symbol=AAPL",
                "User asks: 'Phát hiện pattern Tesla' → USE THIS with symbol=TSLA",
                "User asks: 'NVDA có pattern gì không?' → USE THIS with symbol=NVDA",
                "User asks: 'Mô hình nến Amazon' → USE THIS with symbol=AMZN",
                
                # When NOT to use
                "User asks for technical INDICATORS → DO NOT USE (use getTechnicalIndicators)",
                "User asks for current PRICE → DO NOT USE (use getStockPrice)"
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
                    description="Timeframe for pattern detection",
                    required=False,
                    default="6M",
                    allowed_values=["1M", "3M", "6M", "1Y"]
                ),
                ToolParameter(
                    name="pattern_types",
                    type="array",
                    description="Types of patterns to detect",
                    required=False,
                    default=["all"]
                )
            ],
            returns={
                "symbol": "string",
                "patterns_detected": "array - List of detected patterns",
                "pattern_count": "number",
                "confidence_scores": "object",
                "price_targets": "object",
                "timestamp": "string"
            },
            typical_execution_time_ms=2000,
            requires_symbol=True
        )
        
    async def execute(
        self,
        symbol: str,
        lookback_days: int = 90,
        timeframe: str = "6M",           # ← THÊM
        pattern_types: List[str] = None, # ← THÊM
        **kwargs                          # ← THÊM để accept extra params
    ) -> ToolOutput:
        """
        Execute chart pattern detection
        
        Args:
            symbol: Stock symbol
            lookback_days: Historical days to analyze
            
        Returns:
            ToolOutput with detected patterns
        """
        symbol_upper = symbol.upper()
    
        # Handle default pattern_types
        if pattern_types is None:
            pattern_types = ["all"]
        
        self.logger.info(
            f"[detectChartPatterns] Executing for symbol={symbol_upper}, "
            f"lookback_days={lookback_days}, timeframe={timeframe}, "  # ← CẬP NHẬT LOG
            f"pattern_types={pattern_types}"
        )
        
        try:
            # Call existing handler method
            pattern_results = await self.pattern_handler.analyze_patterns(
                symbol=symbol_upper,
                lookback_days=lookback_days,
                df=None
            )
            
            if not pattern_results or "error" in pattern_results:
                error_msg = pattern_results.get("error", "No pattern data available")
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"Pattern detection failed: {error_msg}"
                )
            
            # Format to schema
            formatted_data = self._format_pattern_data(pattern_results, symbol_upper)
            
            return create_success_output(
                tool_name=self.schema.name,
                data=formatted_data,
                metadata={
                    "source": "PatternRecognitionHandler",
                    "symbol_queried": symbol_upper,
                    "lookback_days": lookback_days,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"[detectChartPatterns] Error for {symbol_upper}: {e}",
                exc_info=True
            )
            return create_error_output(
                tool_name=self.schema.name,
                error=f"Pattern detection failed: {str(e)}"
            )
    
    def _format_pattern_data(self, raw_data: Dict, symbol: str) -> Dict[str, Any]:
        """
        Format raw handler output to tool schema
        
        Args:
            raw_data: Output from PatternRecognitionHandler
            symbol: Stock symbol
            
        Returns:
            Formatted data matching schema
        """
        # Extract patterns
        patterns_detected = raw_data.get("patterns", [])
        
        # Separate bullish/bearish
        bullish_patterns = []
        bearish_patterns = []
        
        for pattern in patterns_detected:
            pattern_type = pattern.get("type", "").lower()
            if any(keyword in pattern_type for keyword in ["bullish", "inverse head", "double bottom"]):
                bullish_patterns.append(pattern)
            elif any(keyword in pattern_type for keyword in ["bearish", "head and shoulders", "double top"]):
                bearish_patterns.append(pattern)
        
        # Find highest confidence pattern
        highest_confidence = None
        if patterns_detected:
            highest_confidence = max(
                patterns_detected,
                key=lambda p: p.get("confidence", 0)
            )
        
        return {
            "symbol": symbol,
            "patterns_detected": patterns_detected,
            "total_patterns": len(patterns_detected),
            "bullish_patterns": bullish_patterns,
            "bearish_patterns": bearish_patterns,
            "highest_confidence_pattern": highest_confidence,
            "summary": raw_data.get("summary", "Pattern analysis completed"),
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import json
    
    async def test_tool():
        """Standalone test for DetectChartPatternsTool"""
        
        print("=" * 80)
        print("TESTING [DetectChartPatternsTool]")
        print("=" * 80)
        
        tool = DetectChartPatternsTool()
        
        # Test 1: Valid symbol
        print("\n📊 Test 1: Valid symbol (AAPL)")
        print("-" * 40)
        result = await tool.safe_execute(symbol="AAPL", lookback_days=90)
        
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_ms}ms")
        
        if result.is_success():
            print("✅ SUCCESS")
            print(f"Patterns detected: {result.data['total_patterns']}")
            print(f"Bullish: {len(result.data['bullish_patterns'])}")
            print(f"Bearish: {len(result.data['bearish_patterns'])}")
            
            if result.data['highest_confidence_pattern']:
                pattern = result.data['highest_confidence_pattern']
                print(f"\nHighest confidence: {pattern.get('type')} ({pattern.get('confidence')}%)")
        else:
            print(f"❌ ERROR: {result.error}")
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETE")
        print("=" * 80)
    
    asyncio.run(test_tool())