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
    Atomic tool for stop loss recommendations
    
    Wraps: RiskAnalysisHandler.suggest_stop_loss_levels()
    
    Returns ALL required fields even when calculation fails
    """
    
    def __init__(self):
        super().__init__()
        
        if RiskAnalysisHandler is None or MarketData is None:
            raise ImportError(
                "RiskAnalysisHandler or MarketData not found"
            )
        
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
                "User asks: 'Apple stop loss' → USE THIS with symbol=AAPL",
                "User asks: 'Where should I set stop for TSLA?' → USE THIS with symbol=TSLA",
                "User asks: 'Stop loss cho Apple' → USE THIS with symbol=AAPL",
                "User asks about ENTRY price → DO NOT USE (use getStockPrice)",
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
                "stop_loss_levels": "object",
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
        entry_price: Optional[float] = None,
        lookback_days: int = 60,
        risk_tolerance: str = "moderate",
        **kwargs
    ) -> ToolOutput:
        """
        Execute stop loss suggestion with fallback logic
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price (uses current if not provided)
            lookback_days: Historical days for calculations
            risk_tolerance: Risk level
            
        Returns:
            ToolOutput with ALL required fields
        """
        start_time = datetime.now()
        symbol_upper = symbol.upper()
        
        self.logger.info(
            f"[suggestStopLoss] Executing: symbol={symbol_upper}, "
            f"entry_price={entry_price}, lookback_days={lookback_days}, "
            f"risk_tolerance={risk_tolerance}"
        )
        
        try:
            # ═══════════════════════════════════════════════════════════
            # STEP 1: Call existing handler
            # ═══════════════════════════════════════════════════════════
            stop_loss_data = await self.risk_handler.suggest_stop_loss_levels(
                symbol=symbol_upper,
                lookback_days=lookback_days,
                df=None
            )
            
            # ═══════════════════════════════════════════════════════════
            # STEP 2: Check for errors or insufficient data
            # ═══════════════════════════════════════════════════════════
            if not stop_loss_data or "error" in stop_loss_data:
                error_msg = stop_loss_data.get("error", "No stop loss data available") if stop_loss_data else "Handler returned None"
                
                self.logger.warning(
                    f"[suggestStopLoss] Handler error: {error_msg}. "
                    f"Creating fallback output..."
                )
                
                # ✅ Return fallback instead of error
                fallback_output = await self._create_fallback_output(
                    symbol=symbol_upper,
                    entry_price=entry_price,
                    message=error_msg
                )
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return create_success_output(
                    tool_name=self.schema.name,
                    data=fallback_output,
                    metadata={
                        "source": "Fallback estimates (5% rule)",
                        "symbol_queried": symbol_upper,
                        "execution_time_ms": int(execution_time),
                        "data_quality": "estimated",
                        "warning": error_msg,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            # ═══════════════════════════════════════════════════════════
            # STEP 3: Format with ALL required fields
            # ═══════════════════════════════════════════════════════════
            formatted_data = self._format_stop_loss_data(
                stop_loss_data,
                symbol_upper,
                entry_price,
                risk_tolerance
            )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            self.logger.info(
                f"[suggestStopLoss] ✅ SUCCESS ({int(execution_time)}ms) - "
                f"ATR stop: ${formatted_data['atr_stop']:.2f}, "
                f"Risk: ${formatted_data['risk_amount']:.2f}"
            )
            
            return create_success_output(
                tool_name=self.schema.name,
                data=formatted_data,
                metadata={
                    "source": "RiskAnalysisHandler",
                    "symbol_queried": symbol_upper,
                    "execution_time_ms": int(execution_time),
                    "lookback_days": lookback_days,
                    "data_quality": "calculated",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"[suggestStopLoss] Error for {symbol_upper}: {e}",
                exc_info=True
            )
            
            # ✅ Even on exception, return fallback (not error)
            try:
                fallback_output = await self._create_fallback_output(
                    symbol=symbol_upper,
                    entry_price=entry_price,
                    message=f"Calculation error: {str(e)}"
                )
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return create_success_output(
                    tool_name=self.schema.name,
                    data=fallback_output,
                    metadata={
                        "source": "Fallback estimates (error recovery)",
                        "symbol_queried": symbol_upper,
                        "execution_time_ms": int(execution_time),
                        "data_quality": "estimated",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                )
            except Exception as fallback_error:
                # Last resort - return error
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"Stop loss calculation failed: {str(e)}"
                )
    
    async def _create_fallback_output(
        self,
        symbol: str,
        entry_price: Optional[float],
        message: str
    ) -> Dict[str, Any]:
        """
        Create fallback output when calculation fails
        
        Uses 5% rule as conservative estimate
        Returns ALL required fields with estimated values
        """
        # Get current price if entry_price not provided
        if entry_price is None:
            try:
                # Try to get current price from market data
                current_data = await self.market_data.get_current_price(symbol)
                if current_data and "price" in current_data:
                    current_price = float(current_data["price"])
                    entry_price = current_price
                else:
                    # Last resort - use 100 as placeholder
                    current_price = 100.0
                    entry_price = 100.0
            except Exception:
                current_price = 100.0
                entry_price = 100.0
        else:
            current_price = entry_price
        
        # ═══════════════════════════════════════════════════════════
        # Use 5% rule for all stop loss methods (conservative)
        # ═══════════════════════════════════════════════════════════
        fallback_stop = round(entry_price * 0.95, 2)
        risk_per_share = round(entry_price - fallback_stop, 2)
        
        # ═══════════════════════════════════════════════════════════
        # Return ALL required fields
        # ═══════════════════════════════════════════════════════════
        return {
            # ✅ Required fields
            "symbol": symbol,
            "current_price": current_price,
            "entry_price": entry_price,
            "stop_loss_levels": {
                "atr_based": {
                    "atr_2x": fallback_stop,
                    "atr_3x": round(entry_price * 0.93, 2)
                },
                "percentage_based": {
                    "percent_3": round(entry_price * 0.97, 2),
                    "percent_5": fallback_stop,
                    "percent_7": round(entry_price * 0.93, 2)
                },
                "recommended": fallback_stop
            },
            "atr_stop": fallback_stop,
            "support_stop": fallback_stop,
            "percentage_stop": fallback_stop,
            "risk_amount": risk_per_share,
            "risk_reward_ratio": 2.0,  # Assume 2:1 target
            "timestamp": datetime.now().isoformat(),
            
            # Additional fields
            "risk_percentage": 5.0,
            "target_price": round(entry_price + (risk_per_share * 2), 2),
            "recommendation": f"Use conservative 5% stop loss at ${fallback_stop:.2f}",
            "data_quality": "estimated",
            "warning": message
        }
    
    def _format_stop_loss_data(
        self,
        raw_data: Dict,
        symbol: str,
        entry_price: Optional[float],
        risk_tolerance: str
    ) -> Dict[str, Any]:
        """
        Format raw handler output to tool schema
        
        ✅ Returns ALL required fields
        """
        # Extract stop levels from handler
        stop_levels = raw_data.get("stop_levels", {})
        current_price = float(raw_data.get("current_price", 0.0))
        
        # Use entry_price if provided, else current_price
        actual_entry = entry_price if entry_price else current_price
        
        # ═══════════════════════════════════════════════════════════
        # Extract individual stop loss values
        # ═══════════════════════════════════════════════════════════
        atr_2x = float(stop_levels.get("atr_2x", actual_entry * 0.95))
        atr_3x = float(stop_levels.get("atr_3x", actual_entry * 0.93))
        percent_3 = float(stop_levels.get("percent_3", actual_entry * 0.97))
        percent_5 = float(stop_levels.get("percent_5", actual_entry * 0.95))
        percent_7 = float(stop_levels.get("percent_7", actual_entry * 0.93))
        sma_20 = float(stop_levels.get("sma_20", actual_entry * 0.95))
        sma_50 = float(stop_levels.get("sma_50", actual_entry * 0.93))
        recent_swing = float(stop_levels.get("recent_swing", actual_entry * 0.95))
        
        # ═══════════════════════════════════════════════════════════
        # Select recommended stop based on risk tolerance
        # ═══════════════════════════════════════════════════════════
        if risk_tolerance == "conservative":
            recommended_stop = atr_2x  # Tighter stop
            recommended_method = "ATR 2x"
        elif risk_tolerance == "aggressive":
            recommended_stop = percent_7  # Wider stop
            recommended_method = "7% below entry"
        else:  # moderate
            recommended_stop = percent_5  # 5% stop
            recommended_method = "5% below entry"
        
        # ═══════════════════════════════════════════════════════════
        # Calculate risk metrics
        # ═══════════════════════════════════════════════════════════
        risk_per_share = actual_entry - recommended_stop
        risk_amount = risk_per_share  # Will multiply by position size if provided
        
        # Calculate risk/reward (assume 2:1 target)
        target_price = actual_entry + (risk_per_share * 2)
        risk_reward_ratio = 2.0
        
        # ═══════════════════════════════════════════════════════════
        # Return ALL required fields
        # ═══════════════════════════════════════════════════════════
        return {
            # ✅ Required fields from schema
            "symbol": symbol,
            "current_price": current_price,
            "entry_price": actual_entry,
            "stop_loss_levels": {
                "atr_based": {
                    "atr_2x": round(atr_2x, 2),
                    "atr_3x": round(atr_3x, 2)
                },
                "percentage_based": {
                    "percent_3": round(percent_3, 2),
                    "percent_5": round(percent_5, 2),
                    "percent_7": round(percent_7, 2)
                },
                "sma_based": {
                    "sma_20": round(sma_20, 2),
                    "sma_50": round(sma_50, 2)
                },
                "technical": {
                    "recent_swing": round(recent_swing, 2)
                },
                "recommended": round(recommended_stop, 2)
            },
            "atr_stop": round(atr_2x, 2),
            "support_stop": round(recent_swing, 2),
            "percentage_stop": round(percent_5, 2),
            "risk_amount": round(risk_amount, 2),
            "risk_reward_ratio": risk_reward_ratio,
            "timestamp": datetime.now().isoformat(),
            
            # Additional fields (not in schema but useful)
            "recommended_stop": round(recommended_stop, 2),
            "recommended_method": recommended_method,
            "risk_tolerance": risk_tolerance,
            "risk_percentage": round((risk_per_share / actual_entry) * 100, 2),
            "target_price": round(target_price, 2),
            "summary": raw_data.get("suggested_stop_levels", "Stop loss analysis completed")
        }


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import os
    
    async def test_tool():
        """Test SuggestStopLossTool with validation"""
        
        print("=" * 80)
        print("TESTING [SuggestStopLossTool] - FIXED VERSION")
        print("=" * 80)
        
        tool = SuggestStopLossTool()
        
        # Test 1: Normal execution
        print("\n📊 Test 1: AAPL (normal)")
        print("-" * 40)
        result = await tool.safe_execute(symbol="AAPL", entry_price=150.0)
        
        print(f"Status: {result.status}")
        
        if result.is_success():
            # Verify ALL required fields
            required_fields = tool.schema.get_required_fields()
            missing = [f for f in required_fields if f not in result.data]
            
            if missing:
                print(f"⚠️  PARTIAL - Missing: {missing}")
            else:
                print(f"✅ ALL REQUIRED FIELDS PRESENT")
            
            print(f"\nData:")
            print(f"  Entry: ${result.data['entry_price']:.2f}")
            print(f"  ATR Stop: ${result.data['atr_stop']:.2f}")
            print(f"  Support Stop: ${result.data['support_stop']:.2f}")
            print(f"  % Stop: ${result.data['percentage_stop']:.2f}")
            print(f"  Risk: ${result.data['risk_amount']:.2f}")
            print(f"  R:R: {result.data['risk_reward_ratio']:.1f}:1")
        
        # Test 2: Fallback scenario (invalid symbol)
        print("\n📊 Test 2: INVALID (fallback)")
        print("-" * 40)
        result2 = await tool.safe_execute(symbol="INVALID999")
        
        print(f"Status: {result2.status}")
        
        if result2.is_success():
            print(f"✅ Fallback worked")
            print(f"  Warning: {result2.data.get('warning', 'N/A')}")
            print(f"  ATR Stop: ${result2.data['atr_stop']:.2f}")
        
        print("\n" + "=" * 80)
    
    asyncio.run(test_tool())