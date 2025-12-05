# File: src/agents/tools/risk/suggest_stop_loss.py

"""
SuggestStopLossTool - Atomic Tool for Stop Loss Recommendations

Responsibility: Đề xuất các mức stop loss dựa trên nhiều phương pháp
- ATR-based stop loss (2x, 3x ATR)
- Percentage-based stop loss (3%, 5%, 7%)
- SMA-based stop loss (below 20-day, 50-day SMA)
- Recent swing low stop loss
- Conservative vs Aggressive recommendations

KHÔNG BAO GỒM:
- ❌ Price data (use getStockPrice)
- ❌ Technical indicators calculation (use getTechnicalIndicators)
- ❌ Risk assessment (use assessRisk for full analysis)
- ❌ Position sizing (use assessRisk)

This tool WRAPS existing RiskAnalysisHandler.suggest_stop_loss_levels()
"""

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
    from src.handlers.risk_analysis_handler import RiskAnalysisHandler
    from src.stock.crawlers.market_data_provider import MarketData
except ImportError:
    RiskAnalysisHandler = None
    MarketData = None


class SuggestStopLossTool(BaseTool):
    """
    Atomic tool để đề xuất stop loss levels
    
    Wraps: RiskAnalysisHandler.suggest_stop_loss_levels()
    
    Usage:
        tool = SuggestStopLossTool()
        result = await tool.safe_execute(symbol="AAPL")
        
        if result.is_success():
            conservative = result.data['recommendations']['conservative']
            aggressive = result.data['recommendations']['aggressive']
            atr_stops = result.data['stop_levels']['atr_based']
    """
    
    def __init__(self):
        """Initialize tool"""
        super().__init__()
        
        if RiskAnalysisHandler is None or MarketData is None:
            raise ImportError(
                "RiskAnalysisHandler or MarketData not found. "
                "Make sure dependencies are available"
            )
        
        # Initialize handler with market data
        self.market_data = MarketData()
        self.risk_handler = RiskAnalysisHandler(self.market_data)
        self.logger = logging.getLogger(__name__)
        
        # Define schema
        self.schema = ToolSchema(
            name="suggestStopLoss",
            category="risk",
            description=(
                "Calculate recommended stop-loss levels based on volatility, support levels, "
                "and ATR (Average True Range). Returns multiple stop-loss strategies. "
                "Use when user asks about stop-loss, risk management, or exit strategy."
            ),
            capabilities=[
                "✅ ATR-based stop-loss (dynamic)",
                "✅ Support-based stop-loss (technical)",
                "✅ Percentage-based stop-loss (fixed risk)",
                "✅ Trailing stop recommendations",
                "✅ Risk/reward ratio calculations",
                "✅ Multiple risk levels (conservative/moderate/aggressive)"
            ],
            limitations=[
                "❌ Stop-loss suggestions are not guarantees",
                "❌ Market gaps can bypass stop-loss levels",
                "❌ Requires recent price data",
                "❌ One symbol at a time"
            ],
            usage_hints=[
                # English
                "User asks: 'Apple stop loss' → USE THIS with symbol=AAPL",
                "User asks: 'Where should I set stop for TSLA?' → USE THIS with symbol=TSLA",
                "User asks: 'NVDA stop loss recommendation' → USE THIS with symbol=NVDA",
                "User asks: 'Exit strategy for Microsoft' → USE THIS with symbol=MSFT",
                
                # Vietnamese
                "User asks: 'Stop loss cho Apple' → USE THIS with symbol=AAPL",
                "User asks: 'Nên đặt stop loss Tesla ở đâu?' → USE THIS with symbol=TSLA",
                "User asks: 'Cắt lỗ NVDA' → USE THIS with symbol=NVDA",
                
                # When NOT to use
                "User asks about ENTRY price → DO NOT USE (use getStockPrice or getTechnicalIndicators)",
                "User asks about RISK assessment → DO NOT USE (use assessRisk)"
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
                    name="entry_price",
                    type="number",
                    description="Your entry price (optional, uses current price if not provided)",
                    required=False
                ),
                ToolParameter(
                    name="risk_tolerance",
                    type="string",
                    description="Risk tolerance level",
                    required=False,
                    default="moderate",
                    allowed_values=["conservative", "moderate", "aggressive"]
                )
            ],
            returns={
                "symbol": "string",
                "current_price": "number",
                "entry_price": "number",
                "stop_loss_levels": "object - Multiple strategies",
                "atr_stop": "number",
                "support_stop": "number",
                "percentage_stop": "number",
                "risk_amount": "number",
                "risk_reward_ratio": "number",
                "timestamp": "string"
            },
            typical_execution_time_ms=1200,
            requires_symbol=True
        )
            
    async def execute(
        self,
        symbol: str,
        lookback_days: int = 60,
        risk_tolerance: str = "moderate",  # ← THÊM DÒNG NÀY
        **kwargs  # ← THÊM **kwargs để accept extra params
    ) -> ToolOutput:
        """
        Execute stop loss suggestion
        
        Args:
            symbol: Stock symbol
            lookback_days: Historical days for calculations
            risk_tolerance: Risk level - "conservative", "moderate", or "aggressive"
            
        Returns:
            ToolOutput with stop loss recommendations
        """
        symbol_upper = symbol.upper()
        
        self.logger.info(
            f"[suggestStopLoss] Executing for symbol={symbol_upper}, "
            f"lookback_days={lookback_days}, risk_tolerance={risk_tolerance}"  # ← CẬP NHẬT LOG
        )
        
        try:
            # Call existing handler method
            stop_loss_data = await self.risk_handler.suggest_stop_loss_levels(
                symbol=symbol_upper,
                lookback_days=lookback_days,
                df=None
            )
            
            if not stop_loss_data or "error" in stop_loss_data:
                error_msg = stop_loss_data.get("error", "No stop loss data available")
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"Stop loss suggestion failed: {error_msg}"
                )
            
            # Format to schema
            formatted_data = self._format_stop_loss_data(stop_loss_data, symbol_upper)
            
            return create_success_output(
                tool_name=self.schema.name,
                data=formatted_data,
                metadata={
                    "source": "RiskAnalysisHandler",
                    "symbol_queried": symbol_upper,
                    "lookback_days": lookback_days,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"[suggestStopLoss] Error for {symbol_upper}: {e}",
                exc_info=True
            )
            return create_error_output(
                tool_name=self.schema.name,
                error=f"Stop loss suggestion failed: {str(e)}"
            )
    
    def _format_stop_loss_data(self, raw_data: Dict, symbol: str) -> Dict[str, Any]:
        """
        Format raw handler output to tool schema
        
        Args:
            raw_data: Output from RiskAnalysisHandler
            symbol: Stock symbol
            
        Returns:
            Formatted data matching schema
        """
        # Extract stop levels
        stop_levels = raw_data.get("stop_levels", {})
        current_price = raw_data.get("current_price", 0.0)
        
        # Calculate risk per share for each method
        risk_per_share = {}
        for method, level in stop_levels.items():
            if isinstance(level, (int, float)) and current_price > 0:
                risk_per_share[method] = round(current_price - level, 2)
        
        # Extract recommendations
        recommendations = {
            "conservative": {
                "stop_level": stop_levels.get("atr_2x", 0.0),
                "method": "ATR 2x",
                "risk_percent": self._calculate_risk_percent(
                    current_price,
                    stop_levels.get("atr_2x", 0.0)
                )
            },
            "moderate": {
                "stop_level": stop_levels.get("percent_5", 0.0),
                "method": "5% below price",
                "risk_percent": 5.0
            },
            "aggressive": {
                "stop_level": stop_levels.get("percent_7", 0.0),
                "method": "7% below price",
                "risk_percent": 7.0
            }
        }
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "stop_levels": {
                "atr_based": {
                    "atr_2x": stop_levels.get("atr_2x", 0.0),
                    "atr_3x": stop_levels.get("atr_3x", 0.0)
                },
                "percentage_based": {
                    "percent_3": stop_levels.get("percent_3", 0.0),
                    "percent_5": stop_levels.get("percent_5", 0.0),
                    "percent_7": stop_levels.get("percent_7", 0.0)
                },
                "sma_based": {
                    "sma_20": stop_levels.get("sma_20", 0.0),
                    "sma_50": stop_levels.get("sma_50", 0.0)
                },
                "technical": {
                    "recent_swing": stop_levels.get("recent_swing", 0.0)
                }
            },
            "recommendations": recommendations,
            "risk_per_share": risk_per_share,
            "summary": raw_data.get("suggested_stop_levels", "Stop loss analysis completed"),
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_risk_percent(self, current_price: float, stop_level: float) -> float:
        """Calculate risk as percentage"""
        if current_price <= 0 or stop_level <= 0:
            return 0.0
        
        risk_pct = ((current_price - stop_level) / current_price) * 100
        return round(risk_pct, 2)


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import json
    
    async def test_tool():
        """Standalone test for SuggestStopLossTool"""
        
        print("=" * 80)
        print("TESTING [SuggestStopLossTool]")
        print("=" * 80)
        
        tool = SuggestStopLossTool()
        
        # Test 1: Valid symbol
        print("\n📊 Test 1: Valid symbol (AAPL)")
        print("-" * 40)
        result = await tool.safe_execute(symbol="AAPL", lookback_days=60)
        
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_ms}ms")
        
        if result.is_success():
            print("✅ SUCCESS")
            print(f"\nCurrent price: ${result.data['current_price']:.2f}")
            print("\n📍 Stop Levels:")
            
            atr = result.data['stop_levels']['atr_based']
            print(f"  ATR 2x: ${atr['atr_2x']:.2f}")
            print(f"  ATR 3x: ${atr['atr_3x']:.2f}")
            
            pct = result.data['stop_levels']['percentage_based']
            print(f"  5% stop: ${pct['percent_5']:.2f}")
            
            print("\n💡 Recommendations:")
            for key, rec in result.data['recommendations'].items():
                print(f"  {key.capitalize()}: ${rec['stop_level']:.2f} ({rec['method']}) - Risk: {rec['risk_percent']:.2f}%")
        else:
            print(f"❌ ERROR: {result.error}")
        
        # Test 2: Custom lookback
        print("\n📊 Test 2: NVDA with 90-day lookback")
        print("-" * 40)
        result = await tool.safe_execute(symbol="NVDA", lookback_days=90)
        
        if result.is_success():
            conservative = result.data['recommendations']['conservative']
            print(f"✅ Conservative stop: ${conservative['stop_level']:.2f}")
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETE")
        print("=" * 80)
    
    asyncio.run(test_tool())