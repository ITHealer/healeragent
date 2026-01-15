import logging
from typing import Dict, Any, Optional

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
except ImportError:
    RiskAnalysisHandler = None
from src.stock.crawlers.market_data_provider import MarketData

class AssessRiskTool(BaseTool):
    """
    Atomic tool để đánh giá rủi ro
    
    Wraps: RiskAnalysisHandler.suggest_stop_loss_levels()
    
    Usage:
        tool = AssessRiskTool()
        result = await tool.safe_execute(symbol="AAPL")
        
        if result.is_success():
            atr_stops = result.data['stop_levels']['atr_based']
            recommended = result.data['recommendation']
    """
    
    def __init__(self):
        """
        Initialize tool
        
        Args:
            market_data_service: Optional market data service for handler
        """
        super().__init__()
        
        if RiskAnalysisHandler is None:
            raise ImportError(
                "RiskAnalysisHandler not found. "
                "Make sure src.handlers.risk_analysis_handler is available"
            )
        
        # Initialize handler
        self.market_data = MarketData()
        self.risk_handler = RiskAnalysisHandler(market_data=self.market_data)
        self.logger = logging.getLogger(__name__)
        
        # Define schema
        self.schema = ToolSchema(
            name="assessRisk",
            category="risk",
            description=(
                "Comprehensive risk assessment including volatility, beta, drawdown, Sharpe ratio, "
                "and risk-adjusted returns. Returns risk metrics and risk rating (Low/Medium/High). "
                "Use when user asks about stock risk, volatility, safety, or risk analysis."
            ),
            capabilities=[
                "✅ Volatility metrics (standard deviation, ATR)",
                "✅ Beta (market correlation)",
                "✅ Maximum drawdown",
                "✅ Sharpe ratio (risk-adjusted returns)",
                "✅ Value at Risk (VaR)",
                "✅ Overall risk rating (Low/Medium/High)",
                "✅ Risk vs return analysis"
            ],
            limitations=[
                "❌ Requires 1+ year historical data for accurate assessment",
                "❌ Past volatility doesn't guarantee future risk",
                "❌ Market conditions change rapidly",
                "❌ One symbol at a time"
            ],
            usage_hints=[
                # English - Risk assessment
                "User asks: 'How risky is Apple?' → USE THIS with symbol=AAPL",
                "User asks: 'TSLA volatility' → USE THIS with symbol=TSLA",
                "User asks: 'Is Microsoft safe?' → USE THIS with symbol=MSFT",
                "User asks: 'Risk analysis for NVDA' → USE THIS with symbol=NVDA",
                "User asks: 'What is Amazon beta?' → USE THIS with symbol=AMZN",
                
                # Vietnamese - Đánh giá rủi ro
                "User asks: 'Apple có rủi ro không?' → USE THIS with symbol=AAPL",
                "User asks: 'Tesla có an toàn không?' → USE THIS with symbol=TSLA",
                "User asks: 'Độ biến động của NVDA' → USE THIS with symbol=NVDA",
                "User asks: 'Phân tích rủi ro Amazon' → USE THIS with symbol=AMZN",
                "User asks: 'Microsoft có ổn định không?' → USE THIS with symbol=MSFT",
                
                # Chinese - 风险评估
                "User asks: 'Apple风险评估' → USE THIS with symbol=AAPL",
                "User asks: '特斯拉安全吗' → USE THIS with symbol=TSLA",
                
                # When NOT to use
                "User asks for PRICE → DO NOT USE (use getStockPrice)",
                "User asks about FUNDAMENTALS → DO NOT USE (use getFinancialRatios)",
                "User asks about NEWS → DO NOT USE (use getStockNews)"
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock symbol (e.g., AAPL, NVDA)",
                    required=True,
                    pattern="^[A-Z]{1,10}$",
                    examples=["AAPL", "NVDA", "TSLA"]
                ),
                ToolParameter(
                    name="lookback_days",
                    type="number",
                    description="Days of historical data (30-252, default 60). Max 252 = 1 trading year.",
                    required=False,
                    default=60,
                    min_value=30,
                    max_value=252
                )
            ],
            
            returns={
                "symbol": "string - Stock symbol",
                "current_price": "number - Current stock price",
                "atr_value": "number - Current ATR value",
                "stop_levels": {
                    "atr_based": {
                        "conservative_1x": "number - 1x ATR stop",
                        "moderate_2x": "number - 2x ATR stop",
                        "aggressive_3x": "number - 3x ATR stop"
                    },
                    "percentage_based": {
                        "tight_2pct": "number - 2% stop",
                        "medium_5pct": "number - 5% stop",
                        "wide_8pct": "number - 8% stop"
                    },
                    "technical_support": {
                        "sma_20": "number - 20-day SMA",
                        "sma_50": "number - 50-day SMA",
                        "recent_swing_low": "number - Recent swing low"
                    }
                },
                "recommendation": "object - Recommended stop loss strategy",
                "risk_reward_ratios": "object - Risk/reward analysis",
                "timestamp": "string - Analysis timestamp"
            },
            
            examples=[
                {
                    "input": {"symbol": "AAPL"},
                    "description": "Assess risk for Apple stock"
                },
                {
                    "input": {"symbol": "NVDA", "lookback_days": 90},
                    "description": "Assess risk for Nvidia with 90-day lookback"
                }
            ],
            typical_execution_time_ms=1800,
            requires_symbol=True
        )
    
    async def execute(
        self,
        symbol: str,
        lookback_days: int = 60
    ) -> ToolOutput:
        """
        Execute tool - Assess risk using existing handler
        
        Args:
            symbol: Stock symbol
            lookback_days: Days of historical data
            
        Returns:
            ToolOutput with risk analysis
        """
        symbol_upper = symbol.upper()

        original_lookback = lookback_days
        lookback_days = int(max(30, min(252, lookback_days)))  # Ensure int
        
        if original_lookback != lookback_days:
            self.logger.warning(
                f"[ASSESS_RISK] lookback_days clamped: {original_lookback} → {lookback_days}"
            )

        try:
            # Call existing handler
            risk_results = await self.risk_handler.suggest_stop_loss_levels(
                symbol=symbol_upper,
                lookback_days=lookback_days,
                df=None  # Handler will fetch data
            )
            
            if not risk_results:
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"No risk analysis data for {symbol_upper}"
                )
            
            # Format results to match schema
            formatted_data = self._format_risk_data(risk_results)
            
            return create_success_output(
                tool_name=self.schema.name,
                data=formatted_data,
                metadata={
                    "source": "RiskAnalysisHandler",
                    "lookback_days": lookback_days,
                    "handler_version": "v1"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error assessing risk for {symbol_upper}: {e}", exc_info=True)
            return create_error_output(
                tool_name=self.schema.name,
                error=f"Failed to assess risk: {str(e)}"
            )
    
    def _format_risk_data(self, handler_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format handler results to match tool schema
        
        Args:
            handler_results: Raw results from RiskAnalysisHandler
            
        Returns:
            Formatted data dict matching schema
        """
        stop_levels = handler_results.get('stop_levels', {})
        
        # Extract and organize stop levels
        formatted = {
            "symbol": handler_results.get('symbol', ''),
            "current_price": handler_results.get('current_price', 0.0),
            "atr_value": handler_results.get('atr', 0.0),
            
            "stop_levels": {
                "atr_based": {
                    "conservative_1x": stop_levels.get('atr_1x', 0.0),
                    "moderate_2x": stop_levels.get('atr_2x', 0.0),
                    "aggressive_3x": stop_levels.get('atr_3x', 0.0)
                },
                "percentage_based": {
                    "tight_2pct": stop_levels.get('percent_2', 0.0),
                    "medium_5pct": stop_levels.get('percent_5', 0.0),
                    "wide_8pct": stop_levels.get('percent_8', 0.0)
                },
                "technical_support": {
                    "sma_20": stop_levels.get('sma_20', 0.0),
                    "sma_50": stop_levels.get('sma_50', 0.0),
                    "recent_swing_low": stop_levels.get('recent_swing', 0.0)
                }
            },
            
            "recommendation": self._create_recommendation(handler_results),
            
            "risk_reward_ratios": handler_results.get('risk_reward_ratios', {}),
            
            "timestamp": handler_results.get('timestamp', '')
        }
        
        return formatted
    
    def _create_recommendation(self, handler_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create risk management recommendation
        
        Args:
            handler_results: Handler results
            
        Returns:
            Recommendation dict
        """
        current_price = handler_results.get('current_price', 0.0)
        atr = handler_results.get('atr', 0.0)
        stop_levels = handler_results.get('stop_levels', {})
        
        # Determine recommended stop based on risk profile
        recommended_stop = stop_levels.get('atr_2x', 0.0)  # Default: Moderate
        
        risk_percent = ((current_price - recommended_stop) / current_price) * 100 if current_price > 0 else 0
        
        return {
            "recommended_stop_loss": recommended_stop,
            "risk_percent": round(risk_percent, 2),
            "risk_profile": "Moderate",
            "rationale": (
                f"Using 2x ATR stop loss at ${recommended_stop:.2f} "
                f"provides balanced risk management with {risk_percent:.1f}% downside protection"
            ),
            "alternative_conservative": stop_levels.get('atr_1x', 0.0),
            "alternative_aggressive": stop_levels.get('atr_3x', 0.0)
        }


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import json
    
    async def test_tool():
        """Test AssessRiskTool standalone"""
        
        print("=" * 80)
        print("TESTING AssessRiskTool")
        print("=" * 80)
        
        try:
            tool = AssessRiskTool()
            
            # Test 1: Basic risk assessment
            print("\nTest 1: Basic risk assessment (AAPL)")
            print("-" * 40)
            result = await tool.safe_execute(symbol="AAPL")
            print(f"Status: {result.status}")
            print(f"Execution time: {result.execution_time_ms}ms")
            
            if result.is_success():
                print(json.dumps(result.data, indent=2))
            else:
                print(f"Error: {result.error}")
            
            # Test 2: With custom lookback
            print("\nTest 2: Custom lookback (NVDA - 90 days)")
            print("-" * 40)
            result = await tool.safe_execute(
                symbol="NVDA",
                lookback_days=90
            )
            print(f"Status: {result.status}")
            print(f"Execution time: {result.execution_time_ms}ms")
            
            if result.is_success():
                # Print key metrics
                print(f"Current Price: ${result.data['current_price']:.2f}")
                print(f"ATR: ${result.data['atr_value']:.2f}")
                print(f"\nRecommended Stop Loss:")
                print(json.dumps(result.data['recommendation'], indent=2))
            
            print("\n" + "=" * 80)
            print("TESTING COMPLETE")
            print("=" * 80)
            
        except ImportError as e:
            print(f"ERROR: Cannot test without handler: {e}")
            print("This tool wraps src.handlers.risk_analysis_handler.RiskAnalysisHandler")
    
    asyncio.run(test_tool())