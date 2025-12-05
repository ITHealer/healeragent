# File: src/agents/tools/fundamentals/get_financial_ratios.py

"""
GetFinancialRatiosTool - Atomic Tool for Financial Ratios

Fetches comprehensive financial ratios including:
- Liquidity ratios (current, quick, cash)
- Profitability ratios (margins, ROE, ROA)
- Leverage ratios (debt/equity, debt/assets)
- Valuation ratios (P/E, P/B, P/S)
- Efficiency ratios (asset turnover, inventory turnover)

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


class GetFinancialRatiosTool(BaseTool):
    """
    Atomic tool để lấy Financial Ratios (Các tỷ số tài chính)
    
    Data Source: FMP /v3/ratios/{symbol}
    
    Usage:
        tool = GetFinancialRatiosTool()
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
            name="getFinancialRatios",
            category="fundamentals",
            description=(
                "Calculate comprehensive financial ratios (P/E, P/B, ROE, ROA, debt ratios, etc.). "
                "Returns valuation, profitability, liquidity, and efficiency ratios. "
                "Use when user asks about valuation, ratios, or financial metrics."
            ),
            capabilities=[
                "✅ Valuation ratios (P/E, P/B, P/S, PEG)",
                "✅ Profitability ratios (ROE, ROA, ROI, margins)",
                "✅ Liquidity ratios (current, quick, cash)",
                "✅ Efficiency ratios (asset turnover, inventory turnover)",
                "✅ Leverage ratios (debt-to-equity, debt-to-assets)",
                "✅ Dividend metrics (yield, payout ratio)",
                "✅ Industry comparison"
            ],
            limitations=[
                "❌ Ratios based on historical data",
                "❌ Industry averages may be outdated",
                "❌ Ratios alone don't tell full story",
                "❌ One symbol at a time"
            ],
            usage_hints=[
                # English
                "User asks: 'Apple P/E ratio' → USE THIS with symbol=AAPL",
                "User asks: 'TSLA financial ratios' → USE THIS with symbol=TSLA",
                "User asks: 'Is NVDA overvalued?' → USE THIS with symbol=NVDA",
                "User asks: 'Microsoft ROE' → USE THIS with symbol=MSFT",
                "User asks: 'Amazon valuation metrics' → USE THIS with symbol=AMZN",
                
                # Vietnamese
                "User asks: 'P/E của Apple' → USE THIS with symbol=AAPL",
                "User asks: 'Các chỉ số tài chính Tesla' → USE THIS with symbol=TSLA",
                "User asks: 'NVDA có đắt không?' → USE THIS with symbol=NVDA",
                
                # When NOT to use
                "User asks about INCOME → DO NOT USE (use getIncomeStatement)",
                "User asks about CASH FLOW → DO NOT USE (use getCashFlow)",
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
                    description="Reporting period",
                    required=False,
                    default="annual",
                    allowed_values=["quarter", "annual", "ttm"]
                )
            ],
            returns={
                "symbol": "string",
                "period": "string",
                "valuation_ratios": "object - P/E, P/B, P/S, PEG",
                "profitability_ratios": "object - ROE, ROA, margins",
                "liquidity_ratios": "object - Current, quick ratios",
                "leverage_ratios": "object - Debt ratios",
                "efficiency_ratios": "object - Turnover ratios",
                "dividend_metrics": "object - Yield, payout",
                "timestamp": "string"
            },
            typical_execution_time_ms=1400,
            requires_symbol=True
        )
    
    async def execute(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 4
    ) -> ToolOutput:
        """
        Execute financial ratios retrieval
        
        Args:
            symbol: Stock symbol
            period: "annual" or "quarter"
            limit: Number of periods (1-10)
            
        Returns:
            ToolOutput with financial ratios data
        """
        symbol_upper = symbol.upper()
        period = period.lower()
        limit = max(1, min(10, limit))
        
        self.logger.info(
            f"[getFinancialRatios] Executing: symbol={symbol_upper}, "
            f"period={period}, limit={limit}"
        )
        
        try:
            # Fetch from FMP
            raw_data = await self.fetcher.fetch_from_fmp(
                endpoint=f"ratios/{symbol_upper}",
                params={
                    "period": period,
                    "limit": limit
                }
            )
            
            if not raw_data:
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"No financial ratios data for {symbol_upper}"
                )
            
            # Format data
            formatted_data = self._format_financial_ratios_data(
                raw_data,
                symbol_upper,
                period,
                limit
            )
            
            return create_success_output(
                tool_name=self.schema.name,
                data=formatted_data,
                metadata={
                    "source": "FMP /v3/ratios",
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
                f"[getFinancialRatios] Error for {symbol_upper}: {e}",
                exc_info=True
            )
            return create_error_output(
                tool_name=self.schema.name,
                error=f"Failed to fetch financial ratios: {str(e)}"
            )
    
    def _format_financial_ratios_data(
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
                "period": item.get("period"),
                
                # ═══════════════════════════════════════════════════════
                # LIQUIDITY RATIOS
                # ═══════════════════════════════════════════════════════
                "current_ratio": item.get("currentRatio"),
                "quick_ratio": item.get("quickRatio"),
                "cash_ratio": item.get("cashRatio"),
                "operating_cash_flow_ratio": item.get("operatingCashFlowRatio"),
                
                # ═══════════════════════════════════════════════════════
                # PROFITABILITY RATIOS
                # ═══════════════════════════════════════════════════════
                "gross_profit_margin": item.get("grossProfitMargin"),
                "operating_profit_margin": item.get("operatingProfitMargin"),
                "pretax_profit_margin": item.get("pretaxProfitMargin"),
                "net_profit_margin": item.get("netProfitMargin"),
                "ebitda_margin": item.get("ebitdaratio"),
                
                # Return metrics
                "return_on_assets": item.get("returnOnAssets"),
                "return_on_equity": item.get("returnOnEquity"),
                "return_on_capital_employed": item.get("returnOnCapitalEmployed"),
                
                # ═══════════════════════════════════════════════════════
                # LEVERAGE RATIOS
                # ═══════════════════════════════════════════════════════
                "debt_equity_ratio": item.get("debtEquityRatio"),
                "debt_ratio": item.get("debtRatio"),
                "long_term_debt_to_capitalization": item.get("longTermDebtToCapitalization"),
                "total_debt_to_capitalization": item.get("totalDebtToCapitalization"),
                "interest_coverage": item.get("interestCoverage"),
                "cash_flow_to_debt_ratio": item.get("cashFlowToDebtRatio"),
                
                # ═══════════════════════════════════════════════════════
                # VALUATION RATIOS
                # ═══════════════════════════════════════════════════════
                "price_earnings_ratio": item.get("priceEarningsRatio"),
                "price_to_book_ratio": item.get("priceToBookRatio"),
                "price_to_sales_ratio": item.get("priceToSalesRatio"),
                "price_cash_flow_ratio": item.get("priceCashFlowRatio"),
                "price_earnings_to_growth_ratio": item.get("priceEarningsToGrowthRatio"),
                "enterprise_value_multiple": item.get("enterpriseValueMultiple"),
                "price_fair_value": item.get("priceFairValue"),
                
                # ═══════════════════════════════════════════════════════
                # EFFICIENCY RATIOS
                # ═══════════════════════════════════════════════════════
                "asset_turnover": item.get("assetTurnover"),
                "inventory_turnover": item.get("inventoryTurnover"),
                "receivables_turnover": item.get("receivablesTurnover"),
                "payables_turnover": item.get("payablesTurnover"),
                "days_sales_outstanding": item.get("daysSalesOutstanding"),
                "days_inventory_outstanding": item.get("daysOfInventoryOutstanding"),
                "days_payables_outstanding": item.get("daysPayablesOutstanding"),
                "cash_conversion_cycle": item.get("cashConversionCycle"),
                
                # ═══════════════════════════════════════════════════════
                # DIVIDEND METRICS
                # ═══════════════════════════════════════════════════════
                "dividend_yield": item.get("dividendYield"),
                "payout_ratio": item.get("payoutRatio"),
                
                # ═══════════════════════════════════════════════════════
                # PER SHARE METRICS
                # ═══════════════════════════════════════════════════════
                "earnings_per_share": item.get("netIncomePerShare"),
                "book_value_per_share": item.get("bookValuePerShare"),
                "tangible_book_value_per_share": item.get("tangibleBookValuePerShare"),
                "shareholders_equity_per_share": item.get("shareholdersEquityPerShare"),
                "operating_cash_flow_per_share": item.get("operatingCashFlowPerShare"),
                "free_cash_flow_per_share": item.get("freeCashFlowPerShare")
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
        print("TESTING [GetFinancialRatiosTool]")
        print("=" * 80)
        
        tool = GetFinancialRatiosTool(api_key=api_key)
        
        # Test 1: Annual
        print("\n📊 Test 1: AAPL Annual Ratios (4 years)")
        result = await tool.safe_execute(symbol="AAPL", period="annual", limit=4)
        
        if result.is_success():
            print("✅ SUCCESS")
            latest = result.data['latest_period']
            print(f"\nLatest: {latest['date']}")
            print(f"\n💧 Liquidity:")
            print(f"  Current Ratio: {latest['current_ratio']:.2f}")
            print(f"  Quick Ratio: {latest['quick_ratio']:.2f}")
            print(f"\n💰 Profitability:")
            print(f"  Gross Margin: {latest['gross_profit_margin']*100:.1f}%")
            print(f"  Net Margin: {latest['net_profit_margin']*100:.1f}%")
            print(f"  ROE: {latest['return_on_equity']*100:.1f}%")
            print(f"\n⚖️ Leverage:")
            print(f"  Debt/Equity: {latest['debt_equity_ratio']:.2f}")
            print(f"\n📈 Valuation:")
            print(f"  P/E: {latest['price_earnings_ratio']:.2f}")
            print(f"  P/B: {latest['price_to_book_ratio']:.2f}")
        else:
            print(f"❌ ERROR: {result.error}")
        
        print("\n" + "=" * 80)
    
    asyncio.run(test_tool())