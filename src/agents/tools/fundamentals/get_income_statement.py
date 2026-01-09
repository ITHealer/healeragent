# File: src/agents/tools/fundamentals/get_income_statement.py

"""
GetIncomeStatementTool - Atomic Tool for Income Statements

Fetches comprehensive income statement data (P&L) including:
- Revenue, Cost of Revenue, Gross Profit
- Operating Income, EBITDA
- Net Income, EPS (basic & diluted)

Supports both annual and quarterly data.
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

from src.agents.tools.fundamentals._financial_base import FinancialDataFetcher


class GetIncomeStatementTool(BaseTool):
    """
    Atomic tool để lấy Income Statement (Báo cáo thu nhập)
    
    Data Source: FMP /v3/income-statement/{symbol}
    
    Usage:
        tool = GetIncomeStatementTool()
        result = await tool.safe_execute(
            symbol="AAPL",
            period="annual",
            limit=4
        )
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize tool"""
        super().__init__()
        
        # Get API key
        if api_key is None:
            import os
            api_key = os.environ.get("FMP_API_KEY")
        
        if not api_key:
            raise ValueError("FMP_API_KEY required")
        
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Initialize data fetcher
        self.fetcher = FinancialDataFetcher(api_key, self.logger)
        
        # Define schema
        self.schema = ToolSchema(
            name="getIncomeStatement",
            category="fundamentals",
            description=(
                "Fetch income statement (profit & loss) with revenue, expenses, net income, "
                "EPS, and profit margins. Supports quarterly and annual reports. "
                "Use when user asks about revenue, earnings, profitability, or income statement. "
                "\n\nPERIOD SELECTION GUIDE (IMPORTANT):\n"
                "- Use period='quarter' for LATEST/RECENT data (quarterly reports are more current)\n"
                "- Use period='annual' for HISTORICAL TRENDS or YEARLY COMPARISONS\n"
                "- Annual reports have 60-90 day delay after fiscal year end\n"
                "- Example: In Jan 2026, latest quarterly=Q3 2025, latest annual=FY2024\n"
                "\n\nFISCAL YEAR vs CALENDAR YEAR:\n"
                "- NVDA fiscal year ends in JANUARY (Q4 FY2025 = Oct 2024 - Jan 2025)\n"
                "- AAPL fiscal year ends in SEPTEMBER (Q1 FY2025 = Oct-Dec 2024)\n"
                "- Most US companies: fiscal year = calendar year\n"
                "- The API returns data labeled by FISCAL period (e.g., 'Q4' = fiscal Q4)\n"
                "- Use limit=8+ when user asks for a specific quarter to ensure you get enough data."
            ),
            capabilities=[
                "✅ Total revenue and revenue growth",
                "✅ Operating expenses breakdown (COGS, R&D, SG&A)",
                "✅ Operating income and net income",
                "✅ Earnings per share (EPS)",
                "✅ Profit margins (gross, operating, net)",
                "✅ Multi-period comparison (quarterly/annual)",
                "✅ Year-over-year growth rates"
            ],
            limitations=[
                "❌ Quarterly reports delayed 45-60 days",
                "❌ No real-time earnings data",
                "❌ Non-GAAP adjustments may differ",
                "❌ One symbol at a time"
            ],
            usage_hints=[
                # English - Revenue/Earnings queries
                "User asks: 'Apple revenue' → USE THIS with symbol=AAPL",
                "User asks: 'TSLA earnings' → USE THIS with symbol=TSLA",
                "User asks: 'Show me Microsoft income statement' → USE THIS with symbol=MSFT",
                "User asks: 'NVDA profitability' → USE THIS with symbol=NVDA",
                "User asks: 'How much does Amazon make?' → USE THIS with symbol=AMZN",
                
                # Vietnamese - Doanh thu/Lợi nhuận
                "User asks: 'Doanh thu của Apple' → USE THIS with symbol=AAPL",
                "User asks: 'Lợi nhuận Tesla' → USE THIS with symbol=TSLA",
                "User asks: 'Báo cáo thu nhập Microsoft' → USE THIS with symbol=MSFT",
                "User asks: 'NVDA kiếm được bao nhiêu?' → USE THIS with symbol=NVDA",
                
                # When NOT to use
                "User asks about BALANCE SHEET → DO NOT USE (use getBalanceSheet)",
                "User asks about CASH FLOW → DO NOT USE (use getCashFlow)",
                "User asks about RATIOS → DO NOT USE (use getFinancialRatios)",
                "User asks for CURRENT PRICE → DO NOT USE (use getStockPrice)"
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
                    name="period",
                    type="string",
                    description="Reporting period (quarterly or annual)",
                    required=False,
                    default="annual",
                    allowed_values=["quarter", "annual"]
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description=(
                        "Number of periods to return. IMPORTANT: When user asks for a SPECIFIC "
                        "quarter (e.g., Q4 2025, Q1 FY2025), use limit=8 or higher to ensure "
                        "you get enough historical data to find the exact period requested. "
                        "The API returns the MOST RECENT periods first."
                    ),
                    required=False,
                    default=8
                )
            ],
            returns={
                "symbol": "string",
                "period": "string",
                "statements": "array - Income statement data",
                "revenue": "number",
                "revenue_growth": "number - YoY %",
                "net_income": "number",
                "eps": "number",
                "profit_margins": "object - Gross/operating/net margins",
                "timestamp": "string"
            },
            typical_execution_time_ms=1500,
            requires_symbol=True
        )
    
    async def execute(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 4
    ) -> ToolOutput:
        """
        Execute income statement retrieval
        
        Args:
            symbol: Stock symbol
            period: "annual" or "quarter"
            limit: Number of periods (1-10)
            
        Returns:
            ToolOutput with income statement data
        """
        symbol_upper = symbol.upper()
        period = period.lower()

        # Enforce minimum limit for quarterly data to ensure enough history
        # When user asks for specific quarter, we need enough data to find it
        if period in ["quarter", "quarterly"]:
            limit = max(8, min(20, limit))  # At least 8 quarters (2 years)
        else:
            limit = max(4, min(10, limit))  # At least 4 years for annual
        
        self.logger.info(
            f"[getIncomeStatement] Executing: symbol={symbol_upper}, "
            f"period={period}, limit={limit}"
        )
        
        try:
            # Fetch from FMP
            raw_data = await self.fetcher.fetch_from_fmp(
                endpoint=f"income-statement/{symbol_upper}",
                params={
                    "period": period,
                    "limit": limit
                }
            )

            # ============================================================
            # DEBUG: Print raw data from FMP API
            # ============================================================
            if raw_data:
                self.logger.info(f"[getIncomeStatement] 📊 RAW DATA from FMP for {symbol_upper}:")
                for i, item in enumerate(raw_data[:limit]):
                    self.logger.info(
                        f"  [{i}] date={item.get('date')} | "
                        f"period={item.get('period')} | "
                        f"calendarYear={item.get('calendarYear')} | "
                        f"revenue={item.get('revenue'):,.0f} | "
                        f"netIncome={item.get('netIncome'):,.0f} | "
                        f"eps={item.get('eps')}"
                    )
            else:
                self.logger.warning(f"[getIncomeStatement] ⚠️ No raw data returned from FMP for {symbol_upper}")

            if not raw_data:
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"No income statement data for {symbol_upper}"
                )
            
            # Format data
            formatted_data = self._format_income_statement_data(
                raw_data,
                symbol_upper,
                period,
                limit
            )
            
            return create_success_output(
                tool_name=self.schema.name,
                data=formatted_data,
                metadata={
                    "source": "FMP /v3/income-statement",
                    "symbol_queried": symbol_upper,
                    "period_type": period,
                    "periods_requested": limit,
                    "periods_returned": len(raw_data),
                    "cache_ttl": self.fetcher.get_cache_ttl(period),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"[getIncomeStatement] Error for {symbol_upper}: {e}",
                exc_info=True
            )
            return create_error_output(
                tool_name=self.schema.name,
                error=f"Failed to fetch income statement: {str(e)}"
            )
    
    def _format_income_statement_data(
        self,
        raw_data: List[Dict],
        symbol: str,
        period: str,
        limit: int
    ) -> Dict[str, Any]:
        """Format raw FMP data to tool schema"""
        
        # Process each period
        formatted_periods = []
        
        for item in raw_data[:limit]:
            formatted_period = {
                "date": item.get("date"),
                "calendar_year": item.get("calendarYear"),
                "period": item.get("period"),
                
                # Revenue
                "revenue": item.get("revenue"),
                "cost_of_revenue": item.get("costOfRevenue"),
                "gross_profit": item.get("grossProfit"),
                "gross_profit_ratio": item.get("grossProfitRatio"),
                
                # Operating
                "operating_expenses": item.get("operatingExpenses"),
                "rd_expenses": item.get("researchAndDevelopmentExpenses"),
                "sg_and_a_expenses": item.get("sellingGeneralAndAdministrativeExpenses"),
                "operating_income": item.get("operatingIncome"),
                "operating_income_ratio": item.get("operatingIncomeRatio"),
                
                # EBITDA
                "ebitda": item.get("ebitda"),
                "ebitda_ratio": item.get("ebitdaratio"),
                
                # Net Income
                "income_before_tax": item.get("incomeBeforeTax"),
                "income_tax_expense": item.get("incomeTaxExpense"),
                "net_income": item.get("netIncome"),
                "net_income_ratio": item.get("netIncomeRatio"),
                
                # EPS
                "eps": item.get("eps"),
                "eps_diluted": item.get("epsdiluted"),
                
                # Shares
                "weighted_average_shares": item.get("weightedAverageShsOut"),
                "weighted_average_shares_diluted": item.get("weightedAverageShsOutDil")
            }
            
            formatted_periods.append(formatted_period)
        
        # Get latest period
        latest = formatted_periods[0] if formatted_periods else {}
        previous = formatted_periods[1] if len(formatted_periods) > 1 else {}
        
        # Calculate approximate growth if not provided
        revenue_growth = 0.0
        if latest.get("revenue") and previous.get("revenue"):
            try:
                rev_now = float(latest["revenue"])
                rev_prev = float(previous["revenue"])
                if rev_prev != 0:
                    revenue_growth = ((rev_now - rev_prev) / abs(rev_prev)) * 100
            except (ValueError, TypeError):
                pass

        # ✅ MATCH SCHEMA STRUCTURE EXACTLY
        return {
            "symbol": symbol,
            "period": period, # Matches schema key "period"
            
            # Key metrics at root level (from latest period)
            "revenue": latest.get("revenue", 0),
            "revenue_growth": round(revenue_growth, 2),
            "net_income": latest.get("net_income", 0),
            "eps": latest.get("eps", 0.0),
            
            "profit_margins": {
                "gross": latest.get("gross_profit_ratio", 0),
                "operating": latest.get("operating_income_ratio", 0),
                "net": latest.get("net_income_ratio", 0)
            },
            
            # Full history
            "statements": formatted_periods, # Matches schema key "statements"
            "period_count": len(formatted_periods),
            "timestamp": datetime.now().isoformat()
        }


# Testing
if __name__ == "__main__":
    import asyncio
    import os
    
    async def test_tool():
        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            print("❌ FMP_API_KEY not found")
            return
        
        print("=" * 80)
        print("TESTING [GetIncomeStatementTool]")
        print("=" * 80)
        
        tool = GetIncomeStatementTool(api_key=api_key)
        
        # Test 1: Annual
        print("\n📊 Test 1: AAPL Annual (4 years)")
        result = await tool.safe_execute(symbol="AAPL", period="annual", limit=4)
        
        if result.is_success():
            print("✅ SUCCESS")
            latest = result.data['latest_period']
            print(f"Latest: {latest['date']}")
            print(f"Revenue: ${latest['revenue']:,.0f}")
            print(f"Net Income: ${latest['net_income']:,.0f}")
            print(f"EPS: ${latest['eps']:.2f}")
        else:
            print(f"❌ ERROR: {result.error}")
        
        print("\n" + "=" * 80)
    
    asyncio.run(test_tool())