from typing import Dict, Any, List
from src.services.tool_call_service import ToolCallService
from src.services.financial_report_service import FinancialStatementsService
from src.utils.config import settings
import logging

logger = logging.getLogger(__name__)

class WarrenBuffettFMPAgent:
    """
    Warren Buffett agent adapted cho FMP data
    Sử dụng services có sẵn trong XYZ
    """
    
    def __init__(self):
        self.tool_call_service = ToolCallService()
        self.fs_service = FinancialStatementsService()
    
    async def analyze(self, ticker: str, end_date: str) -> Dict[str, Any]:
        """
        Phân tích theo phong cách Warren Buffett với FMP data
        """
        # 1. Get Key Metrics TTM (thay cho get_financial_metrics)
        key_metrics = await self.tool_call_service.get_key_metrics_ttm(ticker)
        
        # 2. Get Financial Statements (thay cho search_line_items)
        income_stmt = await self.fs_service.get_income_statement(
            symbol=ticker, 
            period="annual", 
            limit=5
        )
        balance_sheet = await self.fs_service.get_balance_sheet(
            symbol=ticker,
            period="annual", 
            limit=5
        )
        cashflow = await self.fs_service.get_cashflow_statement(
            symbol=ticker,
            period="annual",
            limit=5
        )
        
        # 3. Get Growth metrics
        growth_data = await self.tool_call_service.get_financial_statement_growth(
            symbol=ticker,
            period="annual",
            limit=5
        )
        
        # 4. Analyze fundamentals
        fundamental_score = self._analyze_fundamentals(key_metrics)
        consistency_score = self._analyze_consistency(income_stmt, growth_data)
        moat_score = self._analyze_moat(key_metrics)
        mgmt_score = self._analyze_management(balance_sheet, cashflow)
        
        # 5. Calculate intrinsic value
        intrinsic_value = self._calculate_intrinsic_value(cashflow, income_stmt)
        market_cap = key_metrics.marketCapTTM if key_metrics else None
        
        margin_of_safety = None
        if intrinsic_value and market_cap:
            margin_of_safety = ((intrinsic_value - market_cap) / intrinsic_value) * 100
        
        # 6. Generate signal
        signal = self._determine_signal(margin_of_safety, fundamental_score)
        
        return {
            "ticker": ticker,
            "signal": signal,
            "scores": {
                "fundamentals": fundamental_score,
                "consistency": consistency_score,
                "moat": moat_score,
                "management": mgmt_score
            },
            "valuation": {
                "intrinsic_value": intrinsic_value,
                "market_cap": market_cap,
                "margin_of_safety": margin_of_safety
            }
        }
    
    def _analyze_fundamentals(self, metrics):
        # Logic từ ai-hedge-fund nhưng dùng FMP fields
        score = 0
        if metrics:
            if metrics.peRatioTTM and 0 < metrics.peRatioTTM < 15:
                score += 2
            if metrics.roeTTM and metrics.roeTTM > 0.15:
                score += 2
            if metrics.debtToEquityTTM and metrics.debtToEquityTTM < 0.5:
                score += 1
        return min(score, 5)
    
    # ... các methods analyze khác tương tự