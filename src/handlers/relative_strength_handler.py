from typing import Dict, List, Any
from src.stock.analysis.relative_strength import RelativeStrength
from src.utils.logger.custom_logging import LoggerMixin
from src.stock.crawlers.market_data_provider import MarketData 
import logging

class RelativeStrengthHandler(LoggerMixin):
    """Handler for relative strength analysis."""
    
    def __init__(self, market_data=None):
        
        self.market_data = market_data
        self.logger = logging.getLogger(__name__)
    
    async def get_relative_strength(
        self, 
        symbol: str, 
        benchmark: str = "SPY", 
        lookback_periods: List[int] = [21, 63, 126, 252]
    ) -> Dict[str, Any]:
        """
        Get relative strength analysis for a symbol compared to a benchmark.
        
        Args:
            symbol (str): The stock symbol to analyze
            benchmark (str): The benchmark symbol (default: SPY for S&P 500 ETF)
            lookback_periods (List[int]): Periods in trading days to calculate RS 
            
        Returns:
            Dict[str, Any]: Relative strength metrics
        """
        try:
            if self.market_data is None:
                self.market_data = MarketData()
                
            self.logger.info(f"Calculating relative strength for {symbol} vs {benchmark}")
            
            # Use the RelativeStrength class to calculate RS
            rs_metrics, rs_text = await RelativeStrength.calculate_rs(
                self.market_data,
                symbol=symbol,
                benchmark=benchmark,
                lookback_periods=lookback_periods
            )
            
            # Format the result
            result = {
                "symbol": symbol,
                "benchmark": benchmark,
                "relative_strength": rs_metrics,
                "relative_strength_summary": rs_text
            }

            # Convert numpy values to Python types for JSON serialization
            result = self._convert_numpy_types(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting relative strength for {symbol}: {str(e)}")
            raise ValueError(f"Error calculating relative strength: {str(e)}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # For numpy types that have .item() method
            return obj.item()
        else:
            return obj