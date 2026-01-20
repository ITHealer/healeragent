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
    
    # Standard lookback periods for multi-timeframe RS analysis
    # 21d (~1 month), 63d (~3 months), 126d (~6 months), 252d (~1 year)
    DEFAULT_LOOKBACK_PERIODS = [21, 63, 126, 252]

    async def execute(
        self,
        symbol: str,
        benchmark: str = "SPY",
        timeframe: str = "multi"  # "multi" = all 4 timeframes, or specific like "3M"
    ) -> ToolOutput:
        """
        Execute relative strength analysis with multi-timeframe support.

        Args:
            symbol: Stock symbol
            benchmark: Benchmark symbol (default: SPY for S&P 500)
            timeframe: "multi" for all timeframes [21d, 63d, 126d, 252d],
                      or specific like "1M", "3M", "6M", "1Y"

        Returns:
            ToolOutput with RS metrics for multiple timeframes:
            - 21d RS: Short-term momentum (swing traders)
            - 63d RS: Medium-term trend (position traders)
            - 126d RS: Long-term trend (investors)
            - 252d RS: Very long-term (long-term investors)
        """
        symbol_upper = symbol.upper()
        benchmark_upper = benchmark.upper()

        # Determine lookback periods based on timeframe
        if timeframe == "multi" or timeframe is None:
            lookback_periods = self.DEFAULT_LOOKBACK_PERIODS
        else:
            # Map single timeframe to lookback days
            timeframe_map = {
                "1M": [21],
                "3M": [63],
                "6M": [126],
                "1Y": [252],
            }
            lookback_periods = timeframe_map.get(timeframe.upper(), self.DEFAULT_LOOKBACK_PERIODS)

        self.logger.info(
            f"[getRelativeStrength] symbol={symbol_upper}, "
            f"benchmark={benchmark_upper}, timeframe={timeframe}, "
            f"lookback_periods={lookback_periods}"
        )

        try:
            rs_results = await self.rs_handler.get_relative_strength(
                symbol=symbol_upper,
                benchmark=benchmark_upper,
                lookback_periods=lookback_periods
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
        """
        Format raw handler output for LLM consumption.

        Handler returns:
        {
            "symbol": ...,
            "benchmark": ...,
            "relative_strength": {RS_21d, Return_21d, Benchmark_21d, Excess_21d, ...},
            "relative_strength_summary": "text"
        }
        """
        # Get RS metrics from handler (flat format with RS_21d, Excess_21d, etc.)
        rs_metrics = raw_data.get("relative_strength", {})

        # Get the 21d excess return for recent performance
        recent_excess = rs_metrics.get("Excess_21d", 0) or 0
        rs_score_21d = rs_metrics.get("RS_21d", 50) or 50

        # Calculate trend based on 21d vs 63d
        trend = self._calculate_trend(rs_metrics)

        # Calculate percentile (RS score is already on 1-99 scale)
        percentile = rs_score_21d  # RS_21d is the percentile-like score

        return {
            "symbol": symbol,
            "benchmark": benchmark,
            "timeframe": timeframe,
            "relative_performance": rs_metrics,  # Pass the full metrics dict
            "is_outperforming": recent_excess > 1,  # >1% excess = outperforming
            "trend": trend,
            "percentile_rank": percentile,
            "strength_score": recent_excess,  # Excess return as strength score
            "summary": raw_data.get("relative_strength_summary", "Analysis completed"),
            "timestamp": datetime.now().isoformat()
        }

    def _calculate_trend(self, rs_metrics: Dict) -> str:
        """
        Calculate RS trend based on multi-timeframe data.

        Uses 21d vs 63d excess return comparison.
        """
        try:
            excess_21d = rs_metrics.get("Excess_21d", 0) or 0
            excess_63d = rs_metrics.get("Excess_63d", 0) or 0

            diff = excess_21d - excess_63d

            if diff > 3:
                return "improving"
            elif diff < -3:
                return "declining"
            return "stable"
        except:
            return "unknown"

    def _generate_llm_summary(self, data: Dict[str, Any]) -> str:
        """
        Generate LLM-friendly summary for relative strength data.

        Includes:
        - Methodology explanation for transparency
        - Multi-timeframe breakdown with actual returns
        - Rule-based classification (leader/laggard/neutral)
        - Actionable interpretation guidelines
        """
        symbol = data.get("symbol", "N/A")
        benchmark = data.get("benchmark", "SPY")
        relative_perf = data.get("relative_performance", {})
        summary_text = data.get("summary", "")

        # Extract raw metrics from relative_performance
        # Format: {RS_21d, Return_21d, Benchmark_21d, Excess_21d, ...}
        metrics = self._extract_metrics(relative_perf)

        # Calculate multi-timeframe analysis
        multi_tf_analysis = self._analyze_multi_timeframe(metrics)

        lines = [
            f"=== RELATIVE STRENGTH ANALYSIS: {symbol} vs {benchmark} ===",
            "",
            "## METHODOLOGY",
            f"- Benchmark: {benchmark} (S&P 500 ETF - represents overall US market)",
            "- RS Score = 50 + Excess Return (capped 1-99)",
            "- Excess Return = Stock Return - Benchmark Return",
            "- Timeframes: 21d (1M), 63d (3M), 126d (6M), 252d (1Y) trading days",
            "",
            "## MULTI-TIMEFRAME RETURNS",
        ]

        # Add detailed returns for each timeframe
        for period in ["21d", "63d", "126d", "252d"]:
            stock_ret = metrics.get(f"Return_{period}")
            bench_ret = metrics.get(f"Benchmark_{period}")
            excess = metrics.get(f"Excess_{period}")
            rs_score = metrics.get(f"RS_{period}")

            if stock_ret is not None and bench_ret is not None:
                status = "OUTPERFORM" if excess > 0 else "UNDERPERFORM" if excess < 0 else "IN-LINE"
                lines.append(f"- {period}: {symbol} {stock_ret:+.2f}% vs {benchmark} {bench_ret:+.2f}% = {excess:+.2f}% ({status})")
                if rs_score is not None:
                    lines.append(f"  └─ RS Score: {rs_score:.0f}/100")

        lines.extend([
            "",
            "## CLASSIFICATION (Rule-based)",
            f"- Short-term (21d): {multi_tf_analysis['short_term_status']}",
            f"- Medium-term (63d): {multi_tf_analysis['medium_term_status']}",
            f"- Long-term (126d+): {multi_tf_analysis['long_term_status']}",
            "",
            f"### OVERALL CLASSIFICATION: {multi_tf_analysis['classification']}",
            f"### RS TREND: {multi_tf_analysis['trend']}",
            "",
            "## INTERPRETATION RULES",
        ])

        # Add rule-based interpretation
        lines.extend(self._get_interpretation_rules(multi_tf_analysis))

        if summary_text:
            lines.extend(["", f"## RAW SUMMARY", summary_text])

        return "\n".join(lines)

    def _extract_metrics(self, relative_perf: Dict) -> Dict[str, float]:
        """Extract metrics from relative_performance dict."""
        metrics = {}

        # Handle both nested format and flat format
        if isinstance(relative_perf, dict):
            for key, value in relative_perf.items():
                if isinstance(value, dict):
                    # Nested format: {"1M": {"relative_strength": X}}
                    rs = value.get("relative_strength", 0)
                    metrics[f"Excess_{key}"] = rs
                else:
                    # Flat format: {"RS_21d": X, "Return_21d": Y, ...}
                    metrics[key] = value

        return metrics

    def _analyze_multi_timeframe(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Analyze multi-timeframe RS data to classify stock.

        Classification rules:
        - LEADER: Outperform in 3+ timeframes OR strong outperform (>5%) in all
        - EMERGING LEADER: Short-term outperform + improving trend
        - NEUTRAL: Mixed signals, near market performance
        - LAGGARD: Underperform in 3+ timeframes
        - EMERGING LAGGARD: Short-term underperform + declining trend
        """
        excess_21d = metrics.get("Excess_21d", 0) or 0
        excess_63d = metrics.get("Excess_63d", 0) or 0
        excess_126d = metrics.get("Excess_126d", 0) or 0
        excess_252d = metrics.get("Excess_252d", 0) or 0

        # Count outperform/underperform
        outperform_count = sum([
            1 if excess_21d > 1 else 0,
            1 if excess_63d > 1 else 0,
            1 if excess_126d > 1 else 0,
            1 if excess_252d > 1 else 0
        ])
        underperform_count = sum([
            1 if excess_21d < -1 else 0,
            1 if excess_63d < -1 else 0,
            1 if excess_126d < -1 else 0,
            1 if excess_252d < -1 else 0
        ])

        # Short-term status
        if excess_21d > 5:
            short_term = "STRONG OUTPERFORM ⭐⭐⭐"
        elif excess_21d > 1:
            short_term = "OUTPERFORM ⭐"
        elif excess_21d < -5:
            short_term = "STRONG UNDERPERFORM ⚠️⚠️⚠️"
        elif excess_21d < -1:
            short_term = "UNDERPERFORM ⚠️"
        else:
            short_term = "NEUTRAL"

        # Medium-term status
        if excess_63d > 5:
            medium_term = "STRONG OUTPERFORM ⭐⭐⭐"
        elif excess_63d > 1:
            medium_term = "OUTPERFORM ⭐"
        elif excess_63d < -5:
            medium_term = "STRONG UNDERPERFORM ⚠️⚠️⚠️"
        elif excess_63d < -1:
            medium_term = "UNDERPERFORM ⚠️"
        else:
            medium_term = "NEUTRAL"

        # Long-term status (average of 126d and 252d)
        long_term_excess = (excess_126d + excess_252d) / 2
        if long_term_excess > 5:
            long_term = "STRONG OUTPERFORM ⭐⭐⭐"
        elif long_term_excess > 1:
            long_term = "OUTPERFORM ⭐"
        elif long_term_excess < -5:
            long_term = "STRONG UNDERPERFORM ⚠️⚠️⚠️"
        elif long_term_excess < -1:
            long_term = "UNDERPERFORM ⚠️"
        else:
            long_term = "NEUTRAL"

        # RS Trend (short vs medium term)
        trend_diff = excess_21d - excess_63d
        if trend_diff > 3:
            trend = "IMPROVING (short-term gaining vs medium-term)"
        elif trend_diff < -3:
            trend = "DECLINING (short-term weakening vs medium-term)"
        else:
            trend = "STABLE (no significant change)"

        # Overall classification
        if outperform_count >= 3:
            classification = "LEADER ⭐ (outperforming in 3+ timeframes)"
        elif outperform_count >= 2 and excess_21d > 1:
            classification = "EMERGING LEADER (short-term strong, building momentum)"
        elif underperform_count >= 3:
            classification = "LAGGARD ⚠️ (underperforming in 3+ timeframes)"
        elif underperform_count >= 2 and excess_21d < -1:
            classification = "EMERGING LAGGARD (short-term weak, losing momentum)"
        elif excess_21d > 1 and (excess_63d < 0 or excess_126d < 0):
            classification = "ROTATION CANDIDATE (short-term improving, longer-term mixed)"
        else:
            classification = "NEUTRAL (near market performance)"

        return {
            "short_term_status": short_term,
            "medium_term_status": medium_term,
            "long_term_status": long_term,
            "trend": trend,
            "classification": classification,
            "outperform_count": outperform_count,
            "underperform_count": underperform_count
        }

    def _get_interpretation_rules(self, analysis: Dict) -> List[str]:
        """Get rule-based interpretation guidelines."""
        rules = []
        classification = analysis.get("classification", "")

        if "LEADER" in classification and "EMERGING" not in classification:
            rules.extend([
                "✅ LEADER STATUS: Confirmed relative strength leader",
                "   - Good for momentum/trend-following strategies",
                "   - Consider on pullbacks to support with volume confirmation",
                "   - Watch for: RS breakdown (losing outperformance in 2+ timeframes)"
            ])
        elif "EMERGING LEADER" in classification:
            rules.extend([
                "⚡ EMERGING LEADER: Short-term improving but not confirmed",
                "   - Add to watchlist, NOT confirmed for position yet",
                "   - Needs: 63d & 126d to also show outperformance for confirmation",
                "   - Watch for: Continued 21d outperformance + volume increase"
            ])
        elif "LAGGARD" in classification and "EMERGING" not in classification:
            rules.extend([
                "⚠️ LAGGARD STATUS: Confirmed relative strength laggard",
                "   - Avoid for new long positions",
                "   - If holding: Consider reducing on rallies",
                "   - Watch for: RS improvement in 21d first (early reversal signal)"
            ])
        elif "EMERGING LAGGARD" in classification:
            rules.extend([
                "⚠️ EMERGING LAGGARD: Short-term weakening, caution",
                "   - Tighten stops if holding",
                "   - Avoid adding to position",
                "   - Watch for: 21d RS recovery or 63d confirmation of weakness"
            ])
        elif "ROTATION" in classification:
            rules.extend([
                "🔄 ROTATION CANDIDATE: Mixed timeframe signals",
                "   - Short-term improving but longer-term not confirmed",
                "   - Watchlist candidate, not a buy signal yet",
                "   - Confirmation needed: 63d must also turn positive",
                "   - Risk: Could be a dead cat bounce if 63d stays negative"
            ])
        else:
            rules.extend([
                "📊 NEUTRAL: Near market performance",
                "   - No strong RS edge in either direction",
                "   - Use other factors (fundamentals, technicals) for decision",
                "   - Watch for: RS breakout (21d + 63d outperform) or breakdown"
            ])

        return rules


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