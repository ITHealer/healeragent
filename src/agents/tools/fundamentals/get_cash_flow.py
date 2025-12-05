# File: src/agents/tools/fundamentals/get_cash_flow.py

"""
GetCashFlowTool - Atomic Tool for Cash Flow Statements

Fetches comprehensive cash flow data including:
- Operating cash flow
- Investing cash flow
- Financing cash flow
- Free cash flow
- Capital expenditures

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


class GetCashFlowTool(BaseTool):
    """
    Atomic tool để lấy Cash Flow Statement (Báo cáo lưu chuyển tiền tệ)
    
    Data Source: FMP /v3/cash-flow-statement/{symbol}
    
    Usage:
        tool = GetCashFlowTool()
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
    name="getCashFlow",
    category="fundamentals",
    description=(
        "Fetch cash flow statement with operating, investing, and financing activities. "
        "Returns free cash flow and cash flow metrics. "
        "Use when user asks about cash flow, free cash flow, or liquidity."
    ),
    capabilities=[
        "✅ Operating cash flow (OCF)",
        "✅ Investing cash flow (capital expenditures)",
        "✅ Financing cash flow (dividends, buybacks)",
        "✅ Free cash flow (FCF)",
        "✅ Cash flow margins",
        "✅ Cash flow growth rates",
        "✅ Multi-period comparison"
    ],
    limitations=[
        "❌ Quarterly reports delayed 45-60 days",
        "❌ No real-time cash flow data",
        "❌ One symbol at a time"
    ],
    usage_hints=[
        # English
        "User asks: 'Apple cash flow' → USE THIS with symbol=AAPL",
        "User asks: 'TSLA free cash flow' → USE THIS with symbol=TSLA",
        "User asks: 'Show me Microsoft cash flow statement' → USE THIS with symbol=MSFT",
        "User asks: 'NVDA operating cash flow' → USE THIS with symbol=NVDA",
        
        # Vietnamese
        "User asks: 'Dòng tiền của Apple' → USE THIS with symbol=AAPL",
        "User asks: 'Free cash flow Tesla' → USE THIS with symbol=TSLA",
        "User asks: 'Báo cáo dòng tiền NVDA' → USE THIS with symbol=NVDA",
        
        # When NOT to use
        "User asks about INCOME → DO NOT USE (use getIncomeStatement)",
        "User asks about BALANCE SHEET → DO NOT USE (use getBalanceSheet)"
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
            description="Reporting period",
            required=False,
            default="annual",
            allowed_values=["quarter", "annual"]
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Number of periods",
            required=False,
            default=4
        )
    ],
    returns={
        "symbol": "string",
        "period": "string",
        "cash_flows": "array",
        "operating_cash_flow": "number",
        "investing_cash_flow": "number",
        "financing_cash_flow": "number",
        "free_cash_flow": "number",
        "fcf_margin": "number",
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
        Execute cash flow retrieval
        
        Args:
            symbol: Stock symbol
            period: "annual" or "quarter"
            limit: Number of periods (1-10)
            
        Returns:
            ToolOutput with cash flow data
        """
        symbol_upper = symbol.upper()
        period = period.lower()
        limit = max(1, min(10, limit))
        
        self.logger.info(
            f"[getCashFlow] Executing: symbol={symbol_upper}, "
            f"period={period}, limit={limit}"
        )
        
        try:
            # Fetch from FMP
            raw_data = await self.fetcher.fetch_from_fmp(
                endpoint=f"cash-flow-statement/{symbol_upper}",
                params={
                    "period": period,
                    "limit": limit
                }
            )
            
            if not raw_data:
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"No cash flow data for {symbol_upper}"
                )
            
            # Format data
            formatted_data = self._format_cash_flow_data(
                raw_data,
                symbol_upper,
                period,
                limit
            )
            
            return create_success_output(
                tool_name=self.schema.name,
                data=formatted_data,
                metadata={
                    "source": "FMP /v3/cash-flow-statement",
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
                f"[getCashFlow] Error for {symbol_upper}: {e}",
                exc_info=True
            )
            return create_error_output(
                tool_name=self.schema.name,
                error=f"Failed to fetch cash flow: {str(e)}"
            )
    
    def _format_cash_flow_data(
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
                
                # Operating Activities
                "operating_cash_flow": item.get("netCashProvidedByOperatingActivities"),
                "depreciation_and_amortization": item.get("depreciationAndAmortization"),
                "deferred_income_tax": item.get("deferredIncomeTax"),
                "stock_based_compensation": item.get("stockBasedCompensation"),
                "change_in_working_capital": item.get("changeInWorkingCapital"),
                "accounts_receivables": item.get("accountsReceivables"),
                "inventory": item.get("inventory"),
                "accounts_payables": item.get("accountsPayables"),
                
                # Investing Activities
                "investing_cash_flow": item.get("netCashUsedForInvestingActivites"),
                "capital_expenditure": item.get("capitalExpenditure"),
                "acquisitions": item.get("acquisitionsNet"),
                "purchases_of_investments": item.get("purchasesOfInvestments"),
                "sales_of_investments": item.get("salesMaturitiesOfInvestments"),
                
                # Financing Activities
                "financing_cash_flow": item.get("netCashUsedProvidedByFinancingActivities"),
                "debt_repayment": item.get("debtRepayment"),
                "common_stock_issued": item.get("commonStockIssued"),
                "common_stock_repurchased": item.get("commonStockRepurchased"),
                "dividends_paid": item.get("dividendsPaid"),
                
                # Summary
                "free_cash_flow": item.get("freeCashFlow"),
                "net_change_in_cash": item.get("netChangeInCash"),
                "cash_at_beginning": item.get("cashAtBeginningOfPeriod"),
                "cash_at_end": item.get("cashAtEndOfPeriod"),
                
                # Calculated metrics
                "fcf_margin": self._calculate_fcf_margin(
                    item.get("freeCashFlow"),
                    item.get("netCashProvidedByOperatingActivities")
                )
            }
            
            formatted_periods.append(formatted_period)
        
        # Get latest period
        latest = formatted_periods[0] if formatted_periods else {}
        
        return {
            "symbol": symbol,
            "period_type": period,
            "periods": formatted_periods,
            "latest_period": latest,
            "period_count": len(formatted_periods),
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_fcf_margin(
        self,
        fcf: Optional[float],
        operating_cf: Optional[float]
    ) -> Optional[float]:
        """Calculate FCF as % of operating cash flow"""
        if fcf is not None and operating_cf is not None and operating_cf != 0:
            return round((fcf / operating_cf) * 100, 2)
        return None


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
        print("TESTING [GetCashFlowTool]")
        print("=" * 80)
        
        tool = GetCashFlowTool(api_key=api_key)
        
        # Test 1: Annual
        print("\n📊 Test 1: AAPL Annual (4 years)")
        result = await tool.safe_execute(symbol="AAPL", period="annual", limit=4)
        
        if result.is_success():
            print("✅ SUCCESS")
            latest = result.data['latest_period']
            print(f"Latest: {latest['date']}")
            print(f"Operating CF: ${latest['operating_cash_flow']:,.0f}")
            print(f"Free CF: ${latest['free_cash_flow']:,.0f}")
            print(f"CapEx: ${latest['capital_expenditure']:,.0f}")
        else:
            print(f"❌ ERROR: {result.error}")
        
        print("\n" + "=" * 80)
    
    asyncio.run(test_tool())