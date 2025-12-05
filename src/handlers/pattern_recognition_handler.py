from typing import Dict, Any, Optional
import pandas as pd
import logging

from src.stock.analysis.pattern_recognition import PatternRecognition

class PatternRecognitionHandler:
    """
    Handler for chart pattern recognition operations.
    Coordinates the data retrieval and processing flow.
    """
    
    def __init__(self, market_data=None):
        """
        Initialize the handler with a market data provider.
        
        Args:
            market_data: Service to fetch market data
        """
        self.market_data = market_data
        self.logger = logging.getLogger(__name__)
        
    async def analyze_patterns(
        self, 
        symbol: str, 
        lookback_days: int = 90,
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Analyze chart patterns for a given symbol.
        
        Args:
            symbol: Stock symbol to analyze
            lookback_days: Number of days of historical data to analyze
            df: Optional pre-loaded dataframe (if None, will fetch from market_data)
            
        Returns:
            Dict[str, Any]: Analysis results with detected patterns
        """
        try:
            # Get data if not provided
            if df is None:
                if self.market_data is None:
                    raise ValueError("No data provided and no market data service configured")
                
                # Explicitly use keyword arguments to avoid parameter confusion
                data = await self.market_data.get_historical_data_lookback_ver2(ticker=symbol, lookback_days=lookback_days)
                
                # Convert to DataFrame if necessary
                if isinstance(data, pd.DataFrame):
                    df = data
                else:
                    df = pd.DataFrame(data)
            
            # Ensure lowercase column names
            df.columns = [col.lower() for col in df.columns]
            
            # Validate required columns
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Required columns not found: {missing_columns}")
            
            # Get the recent data for analysis
            df_recent = df.tail(lookback_days)
            
            # Create instance and analyze patterns
            pattern_recognition = PatternRecognition()
            pattern_results = pattern_recognition.detect_patterns(df_recent)
            
            # Add metadata
            pattern_results["symbol"] = symbol
            pattern_results["lookback_days"] = lookback_days
            pattern_results["data_points"] = len(df_recent)
            
            return pattern_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing patterns for {symbol}: {str(e)}")
            raise Exception(f"Error analyzing patterns for {symbol}: {str(e)}")
    
    def format_pattern_results(self, results: Dict[str, Any]) -> str:
        """
        Format pattern analysis results as human-readable text.
        
        Args:
            results: Pattern analysis results
            
        Returns:
            str: Formatted text output
        """
        symbol = results.get("symbol", "Unknown")
        patterns = results.get("patterns", [])
        
        if not patterns:
            pattern_text = f"No significant chart patterns detected for {symbol} in the recent data."
        else:
            pattern_text = f"Chart Patterns Detected for {symbol}:\n\n"

            for pattern in patterns:
                pattern_text += f"- {pattern['type']}"

                if "start_date" in pattern and "end_date" in pattern:
                    pattern_text += f" ({pattern['start_date']} to {pattern['end_date']})"

                pattern_text += f": Price level ${pattern['price_level']}"

                if "confidence" in pattern:
                    pattern_text += f" (Confidence: {pattern['confidence']})"

                pattern_text += "\n"

            pattern_text += "\nNote: Pattern recognition is not 100% reliable and should be confirmed with other forms of analysis."
        
        return pattern_text