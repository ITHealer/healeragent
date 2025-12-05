from typing import Dict, Any, Optional
import pandas as pd
import logging

from src.stock.analysis.risk_analysis import RiskAnalysis

class RiskAnalysisHandler:
    """
    Handler for risk analysis and position sizing operations.
    Coordinates data retrieval and processing flow.
    """
    
    def __init__(self, market_data=None):
        """
        Initialize the handler with an optional market data provider.
        
        Args:
            market_data: Optional service to fetch market data
        """
        self.market_data = market_data
        self.logger = logging.getLogger(__name__)
    
    async def calculate_position_sizing(
        self,
        symbol: str,
        price: float,
        stop_price: float,
        risk_amount: float,
        account_size: float,
        max_risk_percent: float = 2.0
    ) -> Dict[str, Any]:
        """
        Calculate position sizing for a trade based on risk parameters.
        
        Args:
            symbol: Stock symbol
            price: Current stock price
            stop_price: Stop loss price
            risk_amount: Dollar amount willing to risk
            account_size: Total trading account size
            max_risk_percent: Maximum percentage of account to risk
            
        Returns:
            Dict[str, Any]: Position sizing recommendations
        """
        try:
            # Validate inputs
            if price <= 0:
                raise ValueError("Price must be positive")
            
            if account_size <= 0:
                raise ValueError("Account size must be positive")
            
            if risk_amount <= 0 or risk_amount > account_size:
                raise ValueError("Risk amount must be positive and less than account size")
            
            if price <= stop_price:
                raise ValueError("For long positions, stop price must be below entry price")
                
            # Calculate position size
            position_results = RiskAnalysis.calculate_position_size(
                price=price,
                stop_price=stop_price,
                risk_amount=risk_amount,
                account_size=account_size,
                max_risk_percent=max_risk_percent
            )
            
            # Add metadata
            position_results["symbol"] = symbol
            position_results["entry_price"] = price
            position_results["stop_price"] = stop_price
            
            return position_results
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {str(e)}")
            raise Exception(f"Error calculating position size for {symbol}: {str(e)}")
    
    async def suggest_stop_loss_levels(
        self,
        symbol: str,
        lookback_days: int = 60,
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Suggest stop loss levels for a stock based on technical analysis.
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of historical days to analyze
            df: Optional pre-loaded dataframe (if None, will use market_data)
            
        Returns:
            Dict[str, Any]: Suggested stop loss levels
        """
        try:
            # Get data if not provided
            if df is None:
                if self.market_data is None:
                    raise ValueError("No data provided and no market data service configured")
                
                data = await self.market_data.get_historical_data_lookback_ver2(ticker=symbol, lookback_days=lookback_days)
                
                # Convert to DataFrame if needed
                if isinstance(data, pd.DataFrame):
                    df = data
                else:
                    df = pd.DataFrame(data)
            
            # Ensure lowercase column names
            df.columns = [col.lower() for col in df.columns]
            
            # Validate required columns
            required_columns = ["close", "high", "low"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Required columns not found: {missing_columns}")
            
            # Add technical indicators if needed
            if "tech_analysis" in dir(self) and hasattr(self.tech_analysis, "add_core_indicators"):
                df = self.tech_analysis.add_core_indicators(df)
            else:
                # Add simple SMA calculations if tech_analysis is not available
                if "sma_20" not in df.columns:
                    df["sma_20"] = df["close"].rolling(window=20).mean()
                if "sma_50" not in df.columns:
                    df["sma_50"] = df["close"].rolling(window=50).mean()
                if "sma_200" not in df.columns:
                    df["sma_200"] = df["close"].rolling(window=200).mean()
                if "atr" not in df.columns:
                    # Simple ATR calculation
                    high_low = df["high"] - df["low"]
                    high_close = abs(df["high"] - df["close"].shift())
                    low_close = abs(df["low"] - df["close"].shift())
                    ranges = pd.concat([high_low, high_close, low_close], axis=1)
                    true_range = ranges.max(axis=1)
                    df["atr"] = true_range.rolling(window=14).mean()
            
            # Get suggested stop levels
            stop_levels = RiskAnalysis.suggest_stop_levels(df)
            
            # Add metadata
            current_price = df["close"].iloc[-1]
            
            # Add explanation text
            suggested_stop_levels = self.format_stop_levels({
                "symbol": symbol,
                "current_price": current_price,
                "stop_levels": stop_levels
            })

            result = {
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "stop_levels": stop_levels,
                "suggested_stop_levels": suggested_stop_levels
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error suggesting stop levels for {symbol}: {str(e)}")
            raise Exception(f"Error suggesting stop levels for {symbol}: {str(e)}")
    
    
    def format_position_results(self, results: Dict[str, Any]) -> str:
        """
        Format position sizing results as human-readable text.
        
        Args:
            results: Position sizing results
            
        Returns:
            str: Formatted text output
        """
        symbol = results.get("symbol", "Unknown")
        price = results.get("entry_price", 0.0)
        
        formatted_text = f"""
Position Sizing for {symbol} at ${price:.2f}:

📊 Recommended Position:
- {results["recommended_shares"]} shares (${results["position_cost"]:.2f})
- Risk: ${results["dollar_risk"]:.2f} ({results["account_percent_risked"]:.2f}% of account)
- Risk per share: ${results["risk_per_share"]:.2f}

🎯 Potential Targets (R-Multiples):
- R1 (1:1): ${results["r_multiples"]["r1"]:.2f}
- R2 (2:1): ${results["r_multiples"]["r2"]:.2f}
- R3 (3:1): ${results["r_multiples"]["r3"]:.2f}
"""
        return formatted_text
    
    def format_stop_levels(self, results: Dict[str, Any]) -> str:
        """
        Format stop level suggestions as human-readable text.
        
        Args:
            results: Stop level analysis results
            
        Returns:
            str: Formatted text output
        """
        symbol = results.get("symbol", "Unknown")
        current_price = results.get("current_price", 0.0)
        stop_levels = results.get("stop_levels", {})
        
        formatted_text = f"""
Suggested Stop Levels for {symbol} (Current Price: ${current_price:.2f}):

ATR-Based Stops:
- Conservative (1x ATR): ${stop_levels.get("atr_1x", 0.0):.2f}
- Moderate (2x ATR): ${stop_levels.get("atr_2x", 0.0):.2f}
- Aggressive (3x ATR): ${stop_levels.get("atr_3x", 0.0):.2f}

Percentage-Based Stops:
- Tight (2%): ${stop_levels.get("percent_2", 0.0):.2f}
- Medium (5%): ${stop_levels.get("percent_5", 0.0):.2f}
- Wide (8%): ${stop_levels.get("percent_8", 0.0):.2f}
"""
        
        # Add SMA-based stops if available
        sma_section = ""
        if "sma_20" in stop_levels or "sma_50" in stop_levels or "sma_200" in stop_levels:
            sma_section = "\nMoving Average Stops:"
            
        if "sma_20" in stop_levels:
            sma_section += f"\n- 20-day SMA: ${stop_levels['sma_20']:.2f}"
        if "sma_50" in stop_levels:
            sma_section += f"\n- 50-day SMA: ${stop_levels['sma_50']:.2f}"
        if "sma_200" in stop_levels:
            sma_section += f"\n- 200-day SMA: ${stop_levels['sma_200']:.2f}"
        
        formatted_text += sma_section
        
        # Add recent swing low if available
        if "recent_swing" in stop_levels:
            formatted_text += f"\n\nTechnical Support Level:\n- Recent Swing Low: ${stop_levels['recent_swing']:.2f}"
        
        return formatted_text
    
    def _get_recommended_stop(self, stop_levels, current_price):
        """Choose a recommended stop level based on risk profile"""
        
        # Calculate risk percentage for each stop level
        stops_with_risk = {}
        for name, price in stop_levels.items():
            risk_percent = ((current_price - price) / current_price) * 100
            stops_with_risk[name] = {
                "price": price,
                "risk_percent": round(risk_percent, 2)
            }
        
        # Find a stop with moderate risk (around 3-7%)
        preferred_stops = {k: v for k, v in stops_with_risk.items() 
                          if 3 <= v["risk_percent"] <= 7}
        
        if preferred_stops:
            # Get the one with lowest risk in the preferred range
            best_stop_name = min(preferred_stops.items(), 
                                key=lambda x: x[1]["risk_percent"])[0]
            recommended = {
                "name": best_stop_name,
                "price": stops_with_risk[best_stop_name]["price"],
                "risk_percent": stops_with_risk[best_stop_name]["risk_percent"]
            }
        else:
            # If no preferred stops, take the one closest to 5% risk
            closest_name = min(stops_with_risk.items(), 
                              key=lambda x: abs(x[1]["risk_percent"] - 5))[0]
            recommended = {
                "name": closest_name,
                "price": stops_with_risk[closest_name]["price"],
                "risk_percent": stops_with_risk[closest_name]["risk_percent"]
            }
        
        return recommended