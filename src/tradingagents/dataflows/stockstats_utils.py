import pandas as pd
from stockstats import wrap
from typing import Annotated
import os
import asyncio
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from src.stock.crawlers.market_data_provider import MarketData
from src.utils.logger.custom_logging import LoggerMixin
from .config import get_config


class StockstatsUtils(LoggerMixin):
    """Technical indicators calculation using FMP data"""
    
    def __init__(self):
        super().__init__()
        self.market_data = MarketData()
    
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
        data_dir: Annotated[
            str,
            "directory where the stock data is stored.",
        ],
        online: Annotated[
            bool,
            "whether to use online tools to fetch data or offline tools. If True, will use online tools.",
        ] = False,
    ):
        """Calculate technical indicator for a specific date using FMP data"""
        
        # Initialize instance for async operations
        utils = StockstatsUtils()
        
        df = None
        data = None
        
        print(f"\n>>> StockstatsUtils.get_stock_stats <<<")
        print(f"Symbol: {symbol}, Indicator: {indicator}")
        print(f"Date: {curr_date}, Online: {online}")

        
        if not online:
            # Try to read from cached FMP data first
            try:
                # Look for FMP cached file
                fmp_file = os.path.join(
                    data_dir,
                    f"{symbol}-FMP-data-cached.csv",
                )
                
                if os.path.exists(fmp_file):
                    data = pd.read_csv(fmp_file)
                    utils.logger.info(f"Using cached FMP data from {fmp_file}")
                else:
                    # Fallback to YFin cache if exists
                    yfin_file = os.path.join(
                        data_dir,
                        f"{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
                    )
                    if os.path.exists(yfin_file):
                        data = pd.read_csv(yfin_file)
                        utils.logger.info(f"Fallback to YFin cache from {yfin_file}")
                    else:
                        raise FileNotFoundError("No cached data found!")
                
                # Ensure Date column is properly formatted
                data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")
                df = wrap(data)
                
            except FileNotFoundError:
                raise Exception("Stockstats fail: No cached market data found! Use online mode.")
                
        else:
            # Fetch data from FMP online
            utils.logger.info(f"Fetching online FMP data for {symbol}")
            
            # Calculate date range - get more data for indicator calculation
            curr_date_obj = pd.to_datetime(curr_date)
            lookback_days = 365  # Get 1 year of data for indicators
            
            # Run async function to get FMP data
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                fmp_data = loop.run_until_complete(
                    utils.market_data.get_historical_data_lookback_ver2(
                        ticker=symbol,
                        lookback_days=lookback_days
                    )
                )
            finally:
                loop.close()
            
            if fmp_data.empty:
                raise Exception(f"No data available from FMP for {symbol}")
            
            # Format FMP data for stockstats
            # FMP returns with Date as index, we need it as column
            if fmp_data.index.name == 'Date':
                fmp_data = fmp_data.reset_index()
            
            # Ensure we have required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in fmp_data.columns:
                    utils.logger.warning(f"Missing column {col} in FMP data")
            
            # Format Date column
            fmp_data['Date'] = pd.to_datetime(fmp_data['Date']).dt.strftime("%Y-%m-%d")
            
            # Cache the data for future use
            config = get_config()
            cache_dir = config.get("data_cache_dir", "data_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_file = os.path.join(
                cache_dir,
                f"{symbol}-FMP-data-{datetime.now().strftime('%Y%m%d')}.csv"
            )
            fmp_data.to_csv(cache_file, index=False)
            utils.logger.info(f"Cached FMP data to {cache_file}")
            
            print(f"Cache file: {cache_file}")
        
            df = wrap(fmp_data)
        
        # Calculate the indicator
        try:
            df[indicator]  # This triggers stockstats to calculate
            print(f"Indicator calculated successfully")
        except Exception as e:
            utils.logger.error(f"Error calculating {indicator}: {str(e)}")
            raise ValueError(f"Cannot calculate indicator {indicator}: {str(e)}")
        
        # Find the value for the specific date
        curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")
        matching_rows = df[df["Date"] == curr_date_str]
        
        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: Not a trading day (weekend or holiday)"
        

# import pandas as pd
# import yfinance as yf
# from stockstats import wrap
# from typing import Annotated
# import os
# from .config import get_config


# class StockstatsUtils:
#     @staticmethod
#     def get_stock_stats(
#         symbol: Annotated[str, "ticker symbol for the company"],
#         indicator: Annotated[
#             str, "quantitative indicators based off of the stock data for the company"
#         ],
#         curr_date: Annotated[
#             str, "curr date for retrieving stock price data, YYYY-mm-dd"
#         ],
#         data_dir: Annotated[
#             str,
#             "directory where the stock data is stored.",
#         ],
#         online: Annotated[
#             bool,
#             "whether to use online tools to fetch data or offline tools. If True, will use online tools.",
#         ] = False,
#     ):
#         df = None
#         data = None

#         if not online:
#             try:
#                 data = pd.read_csv(
#                     os.path.join(
#                         data_dir,
#                         f"{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
#                     )
#                 )
#                 df = wrap(data)
#             except FileNotFoundError:
#                 raise Exception("Stockstats fail: Yahoo Finance data not fetched yet!")
#         else:
#             # Get today's date as YYYY-mm-dd to add to cache
#             today_date = pd.Timestamp.today()
#             curr_date = pd.to_datetime(curr_date)

#             end_date = today_date
#             start_date = today_date - pd.DateOffset(years=15)
#             start_date = start_date.strftime("%Y-%m-%d")
#             end_date = end_date.strftime("%Y-%m-%d")

#             # Get config and ensure cache directory exists
#             config = get_config()
#             os.makedirs(config["data_cache_dir"], exist_ok=True)

#             data_file = os.path.join(
#                 config["data_cache_dir"],
#                 f"{symbol}-YFin-data-{start_date}-{end_date}.csv",
#             )

#             if os.path.exists(data_file):
#                 data = pd.read_csv(data_file)
#                 data["Date"] = pd.to_datetime(data["Date"])
#             else:
#                 data = yf.download(
#                     symbol,
#                     start=start_date,
#                     end=end_date,
#                     multi_level_index=False,
#                     progress=False,
#                     auto_adjust=True,
#                 )
#                 data = data.reset_index()
#                 data.to_csv(data_file, index=False)

#             df = wrap(data)
#             df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
#             curr_date = curr_date.strftime("%Y-%m-%d")

#         df[indicator]  # trigger stockstats to calculate the indicator
#         matching_rows = df[df["Date"].str.startswith(curr_date)]

#         if not matching_rows.empty:
#             indicator_value = matching_rows[indicator].values[0]
#             return indicator_value
#         else:
#             return "N/A: Not a trading day (weekend or holiday)"
