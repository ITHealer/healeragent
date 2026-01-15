# File: src/agents/tools/fundamentals/get_growth_metrics.py

"""
GetGrowthMetricsTool - Atomic Tool for Growth Metrics

Fetches year-over-year or quarter-over-quarter growth rates for:
- Revenue growth
- Gross profit growth
- Operating income growth
- Net income growth
- EPS growth
- Asset/equity growth
- Cash flow growth

Supports both annual and quarterly data.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import math

# Import base classes and helpers
from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    create_success_output,
    create_error_output
)

from src.agents.tools.fundamentals._financial_base import FinancialDataFetcher


class GetGrowthMetricsTool(BaseTool):
    """
    Atomic tool to retrieve Growth Metrics.
    
    Data Source: FMP /v3/financial-growth/{symbol}
    
    Usage:
        tool = GetGrowthMetricsTool()
        result = await tool.safe_execute(
            symbol="AAPL",
            lookback_years=5
        )
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize tool with API key and logger."""
        super().__init__()
        
        # Get API key
        if api_key is None:
            import os
            api_key = os.environ.get("FMP_API_KEY")
        
        if not api_key:
            raise ValueError("FMP_API_KEY required")
        
        self.api_key = api_key
        # Ensure logger matches standard pattern
        self.logger = logging.getLogger(__name__)
        
        # Initialize data fetcher
        self.fetcher = FinancialDataFetcher(api_key, self.logger)
        
        # Define schema
        self.schema = ToolSchema(
            name="getGrowthMetrics",
            category="fundamentals",
            description=(
                "Calculate company growth metrics including revenue growth, earnings growth, "
                "and historical growth rates. Returns growth trends and projections. "
                "Use when user asks about company growth, growth rate, or growth potential. "
                "\n\nIMPORTANT - FISCAL YEAR vs CALENDAR YEAR:\n"
                "- NVDA fiscal year ends in JANUARY (FY2025 = Feb 2024 - Jan 2025)\n"
                "- AAPL fiscal year ends in SEPTEMBER (FY2025 = Oct 2024 - Sep 2025)\n"
                "- Most US companies: fiscal year = calendar year\n"
                "- Use lookback_years=5+ for comprehensive growth analysis."
            ),
            capabilities=[
                "✅ Revenue growth (YoY, QoQ)",
                "✅ Earnings growth (YoY, QoQ)",
                "✅ EPS growth rates",
                "✅ Historical growth trends (1Y, 3Y, 5Y)",
                "✅ Growth consistency analysis (Stable vs Volatile)",
                "✅ Trend analysis (Accelerating vs Decelerating)"
            ],
            limitations=[
                "❌ Past growth doesn't guarantee future growth",
                "❌ Growth rates can be volatile",
                "❌ No forward guidance included",
                "❌ One symbol at a time"
            ],
            usage_hints=[
                "User asks: 'Apple growth rate' → USE THIS with symbol=AAPL",
                "User asks: 'Is TSLA growing?' → USE THIS with symbol=TSLA",
                "User asks: 'Tốc độ tăng trưởng của Apple' → USE THIS with symbol=AAPL",
                "User asks: 'Tesla có đang phát triển không?' → USE THIS with symbol=TSLA"
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
                    name="lookback_years",
                    type="integer",
                    description="Number of years to analyze",
                    required=False,
                    default=5
                )
            ],
            returns={
                "symbol": "string",
                "revenue_growth": "object - YoY and Average",
                "earnings_growth": "object - YoY and Average",
                "eps_growth": "object",
                "growth_analysis": "object - Trends and Consistency",
                "timestamp": "string"
            },
            typical_execution_time_ms=1600,
            requires_symbol=True
        )

    async def execute(
        self,
        symbol: str,
        lookback_years: int = 5
    ) -> ToolOutput:
        """
        Execute growth metrics retrieval.
        
        Args:
            symbol: Stock symbol (e.g., AAPL)
            lookback_years: Number of years to analyze (default: 5)
            
        Returns:
            ToolOutput with growth metrics data
        """
        symbol_upper = symbol.upper()
        # Clamp lookback_years between 1 and 10 to prevent API overload
        # Also ensure it's an integer (can be passed as float from LLM)
        lookback_years = int(max(1, min(10, lookback_years)))
        
        self.logger.info(
            f"[getGrowthMetrics] Executing with params: "
            f"{{'symbol': '{symbol_upper}', 'lookback_years': {lookback_years}}}"
        )
        
        try:
            # Fetch data from FMP API
            raw_data = await self.fetcher.fetch_from_fmp(
                endpoint=f"financial-growth/{symbol_upper}",
                params={
                    "period": "annual",
                    "limit": lookback_years
                }
            )
            
            if not raw_data:
                # FIX: Pass tool_name as first argument, error message as second
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"No growth metrics data available for {symbol_upper}",
                    metadata={"symbol": symbol_upper}
                )
            
            # Format data
            formatted_data = self._format_growth_metrics_data(
                raw_data,
                symbol_upper,
                lookback_years
            )
            
            self.logger.info(
                f"[getGrowthMetrics] ✅ SUCCESS - Retrieved {formatted_data.get('period_count')} periods"
            )
            
            # FIX: Pass tool_name as first argument, data as second
            return create_success_output(
                tool_name=self.schema.name,
                data=formatted_data,
                metadata={
                    "symbol_queried": symbol_upper,
                    "records_count": len(raw_data)
                }
            )
            
        except Exception as e:
            self.logger.error(f"[getGrowthMetrics] ❌ Execution error: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # FIX: Pass tool_name as first argument, error string as second
            return create_error_output(
                tool_name=self.schema.name,
                error=str(e),
                metadata={"symbol": symbol_upper}
            )

    def _calculate_average_growth(
        self,
        periods: List[Dict],
        metric_key: str
    ) -> Optional[float]:
        """
        Calculate arithmetic mean of a specific growth metric.
        """
        values = [
            p.get(metric_key)
            for p in periods
            if p.get(metric_key) is not None
        ]
        
        if not values:
            return None
            
        return round(sum(values) / len(values), 4)

    def _analyze_trend(self, periods: List[Dict], metric_key: str) -> str:
        """
        Analyze if the growth is accelerating, decelerating, or stable.
        """
        if len(periods) < 2:
            return "Insufficient Data"
            
        latest_val = periods[0].get(metric_key)
        previous_val = periods[1].get(metric_key)
        
        if latest_val is None or previous_val is None:
            return "Unknown"
            
        # If latest growth is > 5% higher than previous -> Accelerating
        if latest_val > previous_val + 0.05:
            return "Accelerating 🚀"
        # If latest growth is < 5% lower than previous -> Decelerating
        elif latest_val < previous_val - 0.05:
            return "Decelerating 📉"
        else:
            return "Stable ⚖️"

    def _format_growth_metrics_data(
        self,
        raw_data: List[Dict],
        symbol: str,
        limit: int
    ) -> Dict[str, Any]:
        """
        Format raw FMP data into a clean, LLM-friendly structure.
        """
        
        # Process each period
        formatted_periods = []
        
        # Ensure we don't exceed the limit
        data_slice = raw_data[:limit]
        
        for item in data_slice:
            formatted_period = {
                "date": item.get("date"),
                "calendar_year": item.get("calendarYear"),
                "period": item.get("period"),
                
                # Revenue
                "revenue_growth": item.get("revenueGrowth"),
                "gross_profit_growth": item.get("grossProfitGrowth"),
                
                # Profitability
                "operating_income_growth": item.get("operatingIncomeGrowth"),
                "net_income_growth": item.get("netIncomeGrowth"),
                "eps_growth": item.get("epsgrowth"),
                
                # Cash Flow
                "free_cash_flow_growth": item.get("freeCashFlowGrowth"),
                "operating_cash_flow_growth": item.get("operatingCashFlowGrowth"),
                
                # Balance Sheet
                "total_assets_growth": item.get("assetGrowth"),
                "debt_growth": item.get("debtGrowth"),
            }
            formatted_periods.append(formatted_period)
        
        # Sort by date descending
        formatted_periods.sort(key=lambda x: x.get("date", ""), reverse=True)
        
        # Get latest period
        latest = formatted_periods[0] if formatted_periods else {}
        
        # Calculate Averages
        avg_revenue = self._calculate_average_growth(formatted_periods, "revenue_growth")
        avg_net_income = self._calculate_average_growth(formatted_periods, "net_income_growth")
        avg_eps = self._calculate_average_growth(formatted_periods, "eps_growth")
        avg_fcf = self._calculate_average_growth(formatted_periods, "free_cash_flow_growth")
        
        # Analyze Trends
        revenue_trend = self._analyze_trend(formatted_periods, "revenue_growth")
        earnings_trend = self._analyze_trend(formatted_periods, "net_income_growth")
        
        # Check Consistency
        rev_consistency = all(p.get("revenue_growth", 0) > 0 for p in formatted_periods)
        
        return {
            "symbol": symbol,
            "period_type": "annual",
            "period_count": len(formatted_periods),
            "latest_period_date": latest.get("date"),
            
            "growth_summary": {
                "revenue_growth_latest": latest.get("revenue_growth"),
                "revenue_growth_avg": avg_revenue,
                "revenue_trend": revenue_trend,
                "is_revenue_consistent": "Consistent Growth ✅" if rev_consistency else "Volatile ⚠️",
                
                "net_income_growth_latest": latest.get("net_income_growth"),
                "net_income_growth_avg": avg_net_income,
                "earnings_trend": earnings_trend,
                
                "eps_growth_latest": latest.get("eps_growth"),
                "eps_growth_avg": avg_eps,
                
                "fcf_growth_latest": latest.get("free_cash_flow_growth"),
                "fcf_growth_avg": avg_fcf
            },
            
            "historical_periods": formatted_periods,
            "timestamp": datetime.now().isoformat()
        }


# Testing block
if __name__ == "__main__":
    import asyncio
    import os
    
    async def test_tool():
        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            print("❌ FMP_API_KEY not found")
            return
        
        print("=" * 80)
        print("TESTING [GetGrowthMetricsTool]")
        print("=" * 80)
        
        tool = GetGrowthMetricsTool(api_key=api_key)
        
        # Test 1: Annual growth
        print("\n📊 Test 1: NVDA Annual Growth (5 years)")
        result = await tool.safe_execute(symbol="NVDA", lookback_years=5)
        
        if result.is_success():
            print("✅ SUCCESS")
            data = result.data
            summary = data['growth_summary']
            
            print(f"\n📈 Trend Analysis for {data['symbol']}:")
            print(f"  Revenue Trend: {summary['revenue_trend']}")
            print(f"  Earnings Trend: {summary['earnings_trend']}")
            
            print(f"\n📊 Metrics (Latest vs Avg):")
            # Handle potential None values safely
            rev_latest = summary.get('revenue_growth_latest') or 0
            rev_avg = summary.get('revenue_growth_avg') or 0
            print(f"  Rev Growth: {rev_latest*100:.1f}% (Avg: {rev_avg*100:.1f}%)")
            
        else:
            print(f"❌ ERROR: {result.error}")
            # print details if available
            if result.metadata:
                print(f"Metadata: {result.metadata}")
        
        print("\n" + "=" * 80)
    
    asyncio.run(test_tool())