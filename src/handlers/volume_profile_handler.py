import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from src.stock.analysis.volume_profile import VolumeProfile
from src.stock.crawlers.market_data_provider import MarketData
from src.utils.logger.custom_logging import LoggerMixin

class VolumeProfileHandler:
    """Handler for volume profile analysis operations."""
    
    def __init__(self, market_data=None):
        """
        Initialize the handler with an optional data provider.
        
        Args:
            data_provider: Optional service to fetch market data
        """
        self.market_data = market_data
        self.logger = logging.getLogger(__name__)
    
    async def get_volume_profile(
        self, 
        symbol: str, 
        lookback_days: int = 60,
        num_bins: int = 10,
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Get volume profile analysis for a symbol.
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to analyze
            num_bins: Number of price bins to create
            df: Optional pre-loaded dataframe (if None, will use market_data)
            
        Returns:
            Dict[str, Any]: Volume profile analysis results
        """
        try:
            # Get data if not provided
            if df is None:
                if self.market_data is None:
                    self.market_data = MarketData()
                    # raise ValueError("No data provided and no data provider configured")

                data = await self.market_data.get_historical_data_lookback_ver2(ticker=symbol, lookback_days=lookback_days)
                df = pd.DataFrame(data)

            # Ensure lowercase column names for consistency
            df.columns = df.columns.str.lower()
            
            # Validate required columns
            required_columns = ["low", "high", "volume"]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in data")
            
            # Get the tail for the lookback period
            if len(df) > lookback_days:
                df_tail = df.tail(lookback_days)
            else:
                df_tail = df

            # Run the analysis
            profile = VolumeProfile.analyze_volume_profile(df_tail, num_bins)

            # Add metadata
            profile["symbol"] = symbol
            profile["lookback_days"] = lookback_days
            profile["data_points"] = len(df_tail)

            # Add explanation text
            profile["summary"] = self.format_profile_summary(profile)
            
            return profile
            
        except Exception as e:
            raise Exception(f"Error getting volume profile for {symbol}: {str(e)}")
    
    def format_profile_summary(self, profile: Dict[str, Any]) -> str:
        """
        Format volume profile results as a human-readable text.
        
        Args:
            profile: Volume profile analysis results
            
        Returns:
            str: Formatted text summary
        """
        symbol = profile.get("symbol", "Unknown")
        lookback_days = profile.get("lookback_days", 0)
        
        profile_text = f"""
Volume Profile Analysis for {symbol} (last {lookback_days} days):

Point of Control (POC): ${profile["point_of_control"]} (Price level with highest volume)
Value Area: ${profile["value_area_low"]} - ${profile["value_area_high"]} (70% of volume)

Volume by Price Level (High to Low):
"""

        sorted_bins = sorted(profile["bins"], key=lambda x: x["volume"], reverse=True)
        for i, b in enumerate(sorted_bins[:5]):
            profile_text += f"{i+1}. ${b['price_low']} - ${b['price_high']}: {b['volume_percent']:.2f}%\n"
            
        return profile_text