# File: src/agents/tools/fundamentals/get_balance_sheet.py

"""
GetBalanceSheetTool - Atomic Tool for Balance Sheet Statements

Fetches comprehensive balance sheet data including:
- Assets (current, non-current, total)
- Liabilities (current, non-current, total debt)
- Shareholders' Equity
- Key balance sheet metrics

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


class GetBalanceSheetTool(BaseTool):
    """
    Atomic tool để lấy Balance Sheet (Bảng cân đối kế toán)
    
    Data Source: FMP /v3/balance-sheet-statement/{symbol}
    
    Usage:
        tool = GetBalanceSheetTool()
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
            name="getBalanceSheet",
            category="fundamentals",
            description=(
                "Fetch balance sheet with assets, liabilities, equity, and financial health metrics. "
                "Returns current/quick ratios, debt levels, and book value. "
                "Use when user asks about assets, debt, equity, or balance sheet. "
                "\n\nPERIOD SELECTION GUIDE (IMPORTANT):\n"
                "- Use period='quarter' for LATEST/RECENT data (quarterly reports are more current)\n"
                "- Use period='annual' for HISTORICAL TRENDS or YEARLY COMPARISONS\n"
                "- Annual reports have 60-90 day delay after fiscal year end\n"
                "- Example: In Jan 2026, latest quarterly=Q3 2025, latest annual=FY2024\n"
                "\n\nFISCAL YEAR vs CALENDAR YEAR:\n"
                "- NVDA fiscal year ends in JANUARY (Q4 FY2025 = Oct 2024 - Jan 2025)\n"
                "- AAPL fiscal year ends in SEPTEMBER (Q1 FY2025 = Oct-Dec 2024)\n"
                "- Most US companies: fiscal year = calendar year\n"
                "- Use limit=8+ when user asks for a specific quarter."
            ),
            capabilities=[
                "✅ Total assets and asset breakdown",
                "✅ Total liabilities and debt levels",
                "✅ Shareholders' equity",
                "✅ Current and quick ratios",
                "✅ Debt-to-equity ratio",
                "✅ Book value per share",
                "✅ Working capital",
                "✅ Multi-period comparison"
            ],
            limitations=[
                "❌ Quarterly reports delayed 45-60 days",
                "❌ Book values may differ from market values",
                "❌ No real-time balance sheet data",
                "❌ One symbol at a time"
            ],
            usage_hints=[
                # English
                "User asks: 'Apple balance sheet' → USE THIS with symbol=AAPL",
                "User asks: 'How much debt does TSLA have?' → USE THIS with symbol=TSLA",
                "User asks: 'NVDA assets' → USE THIS with symbol=NVDA",
                "User asks: 'Microsoft equity' → USE THIS with symbol=MSFT",
                "User asks: 'Amazon financial health' → USE THIS with symbol=AMZN",
                
                # Vietnamese
                "User asks: 'Bảng cân đối kế toán Apple' → USE THIS with symbol=AAPL",
                "User asks: 'Tesla có bao nhiêu nợ?' → USE THIS with symbol=TSLA",
                "User asks: 'Tài sản của NVDA' → USE THIS with symbol=NVDA",
                
                # When NOT to use
                "User asks about INCOME → DO NOT USE (use getIncomeStatement)",
                "User asks about CASH FLOW → DO NOT USE (use getCashFlow)",
                "User asks about RATIOS → DO NOT USE (use getFinancialRatios)"
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
                    description=(
                        "Number of periods to return. IMPORTANT: When user asks for a SPECIFIC "
                        "quarter (e.g., Q4 2025), use limit=8 or higher to ensure you get enough "
                        "historical data. The API returns the MOST RECENT periods first."
                    ),
                    required=False,
                    default=8
                )
            ],
            returns={
                "symbol": "string",
                "period": "string",
                "balance_sheets": "array",
                "total_assets": "number",
                "total_liabilities": "number",
                "total_equity": "number",
                "debt_to_equity": "number",
                "current_ratio": "number",
                "book_value_per_share": "number",
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
        Execute balance sheet retrieval
        
        Args:
            symbol: Stock symbol
            period: "annual" or "quarter"
            limit: Number of periods (1-10)
            
        Returns:
            ToolOutput with balance sheet data
        """
        symbol_upper = symbol.upper()
        period = period.lower()

        # Enforce minimum limit for quarterly data to ensure enough history
        # Ensure int for slicing (LLM may pass float)
        if period in ["quarter", "quarterly"]:
            limit = int(max(8, min(20, limit)))  # At least 8 quarters (2 years)
        else:
            limit = int(max(4, min(10, limit)))  # At least 4 years for annual

        self.logger.info(
            f"[getBalanceSheet] Executing: symbol={symbol_upper}, "
            f"period={period}, limit={limit}"
        )
        
        try:
            # Fetch from FMP
            raw_data = await self.fetcher.fetch_from_fmp(
                endpoint=f"balance-sheet-statement/{symbol_upper}",
                params={
                    "period": period,
                    "limit": limit
                }
            )
            
            if not raw_data:
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"No balance sheet data for {symbol_upper}"
                )
            
            # Format data
            formatted_data = self._format_balance_sheet_data(
                raw_data,
                symbol_upper,
                period,
                limit
            )
            
            return create_success_output(
                tool_name=self.schema.name,
                data=formatted_data,
                metadata={
                    "source": "FMP /v3/balance-sheet-statement",
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
                f"[getBalanceSheet] Error for {symbol_upper}: {e}",
                exc_info=True
            )
            return create_error_output(
                tool_name=self.schema.name,
                error=f"Failed to fetch balance sheet: {str(e)}"
            )
    
    def _format_balance_sheet_data(
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
                
                # Assets
                "total_assets": item.get("totalAssets"),
                "total_current_assets": item.get("totalCurrentAssets"),
                "total_non_current_assets": item.get("totalNonCurrentAssets"),
                "cash_and_cash_equivalents": item.get("cashAndCashEquivalents"),
                "short_term_investments": item.get("shortTermInvestments"),
                "net_receivables": item.get("netReceivables"),
                "inventory": item.get("inventory"),
                "property_plant_equipment": item.get("propertyPlantEquipmentNet"),
                "goodwill": item.get("goodwill"),
                "intangible_assets": item.get("intangibleAssets"),
                
                # Liabilities
                "total_liabilities": item.get("totalLiabilities"),
                "total_current_liabilities": item.get("totalCurrentLiabilities"),
                "total_non_current_liabilities": item.get("totalNonCurrentLiabilities"),
                "accounts_payable": item.get("accountPayables"),
                "short_term_debt": item.get("shortTermDebt"),
                "long_term_debt": item.get("longTermDebt"),
                "total_debt": item.get("totalDebt"),
                "net_debt": item.get("netDebt"),
                
                # Equity
                "total_equity": item.get("totalStockholdersEquity"),
                "retained_earnings": item.get("retainedEarnings"),
                "common_stock": item.get("commonStock"),
                
                # Calculated metrics
                "working_capital": self._safe_subtract(
                    item.get("totalCurrentAssets"),
                    item.get("totalCurrentLiabilities")
                ),
                "debt_to_equity_ratio": self._safe_divide(
                    item.get("totalDebt"),
                    item.get("totalStockholdersEquity")
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
    
    def _safe_subtract(self, a: Optional[float], b: Optional[float]) -> Optional[float]:
        """Safely subtract two numbers"""
        if a is not None and b is not None:
            return round(a - b, 2)
        return None
    
    def _safe_divide(self, a: Optional[float], b: Optional[float]) -> Optional[float]:
        """Safely divide two numbers"""
        if a is not None and b is not None and b != 0:
            return round(a / b, 4)
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
        print("TESTING [GetBalanceSheetTool]")
        print("=" * 80)
        
        tool = GetBalanceSheetTool(api_key=api_key)
        
        # Test 1: Annual
        print("\n📊 Test 1: AAPL Annual (4 years)")
        result = await tool.safe_execute(symbol="AAPL", period="annual", limit=4)
        
        if result.is_success():
            print("✅ SUCCESS")
            latest = result.data['latest_period']
            print(f"Latest: {latest['date']}")
            print(f"Total Assets: ${latest['total_assets']:,.0f}")
            print(f"Total Liabilities: ${latest['total_liabilities']:,.0f}")
            print(f"Total Equity: ${latest['total_equity']:,.0f}")
            print(f"Cash: ${latest['cash_and_cash_equivalents']:,.0f}")
            print(f"Total Debt: ${latest['total_debt']:,.0f}")
        else:
            print(f"❌ ERROR: {result.error}")
        
        print("\n" + "=" * 80)
    
    asyncio.run(test_tool())