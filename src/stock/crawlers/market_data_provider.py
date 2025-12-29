import datetime
import pandas as pd
import yfinance as yf
from typing import Dict, Any, List

from src.database import get_postgres_db
from src.database.models.stock_schemas import StockPrice
from src.utils.logger.custom_logging import LoggerMixin
import aiohttp
from src.utils.config import settings
from src.utils.http_client_pool import get_http_client_manager
from src.utils.async_wrappers import run_in_thread, get_yfinance_history
from dotenv import load_dotenv

load_dotenv()
FMP_API_KEY = settings.FMP_API_KEY

class MarketData(LoggerMixin):
    """Handles all market data fetching operations."""
    def __init__(self):
        super().__init__()
        self.db = get_postgres_db()
        self._http_manager = None

    async def _get_http_manager(self):
        """Lazy init HTTP client manager"""
        if self._http_manager is None:
            self._http_manager = await get_http_client_manager()
        return self._http_manager
    

    async def get_historical_data_lookback(self, ticker: str, lookback_days: int = 365) -> Dict[str, Any]:
        """
        Fetch historical daily data for a given symbol using lookback days.

        Args:
            ticker (str): Stock symbol to fetch data for.
            lookback_days (int): Number of days to look back from today.

        Returns:
            pd.DataFrame: Historical market data as DataFrame.
        """
        try:
            # Calculate start_date and end_date based on lookback_days
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=lookback_days)

            # Format dates for yfinance
            end_date_str = end_date.strftime('%Y-%m-%d')
            start_date_str = start_date.strftime('%Y-%m-%d')

            self.logger.info(f"Fetching data for {ticker} for the last {lookback_days} days ({start_date_str} to {end_date_str})")

            # Use async wrapper to avoid blocking event loop
            data = await get_yfinance_history(ticker, start=start_date_str, end=end_date_str)

            if data.empty:
                raise ValueError(f"No data returned for {ticker}")

            data["symbol"] = ticker.upper()

            # Save to database            
            # data_dict = data.reset_index().to_dict(orient="records")
            
            # with self.db.session_scope() as session:
            #     for record in data_dict:
            #         record["Date"] = record["Date"].strftime('%Y-%m-%d %H:%M:%S')
                    
            #         stock_price = StockPrice(
            #             ticker=ticker,
            #             date=datetime.datetime.strptime(record["Date"], '%Y-%m-%d %H:%M:%S'),
            #             open=record["Open"],        
            #             high=record["High"],        
            #             low=record["Low"],           
            #             close=record["Close"],       
            #             volume=record["Volume"],
            #             dividends=record["Dividends"],
            #             stock_splits=record["Stock Splits"]
            #         )
            #         session.add(stock_price)
                
            #     session.commit()
            
            # self.logger.info(f"Successfully saved {len(data_dict)} records for {ticker}")
            return data
        
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise ValueError(f"Error getting stock data: {str(e)}")
    
    async def get_historical_data_lookback_ver2(self, ticker: str, lookback_days: int = 365) -> pd.DataFrame:
        """
        Fetch historical daily data for a given symbol using FMP API.
        Returns a DataFrame with 'Date' as a DatetimeIndex.
        """
        try:
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=lookback_days)
            
            end_date_str = end_date.strftime('%Y-%m-%d')
            start_date_str = start_date.strftime('%Y-%m-%d')
            
            self.logger.info(f"Fetching data for {ticker} from FMP for ~{lookback_days} days ({start_date_str} to {end_date_str})")
            
            api_key = FMP_API_KEY # Accessing API key via instance attribute

            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date_str}&to={end_date_str}&apikey={api_key}"

            # Use connection pool instead of creating new session each time
            http_manager = await self._get_http_manager()
            async with http_manager.rate_limited_request():
                async with http_manager.get_aiohttp_session() as session:
                    async with session.get(url) as response:
                        response.raise_for_status()
                        api_data = await response.json()

            if not isinstance(api_data, dict) or "historical" not in api_data:
                error_message = api_data.get("Error Message", f"Unexpected response structure from FMP for {ticker}.") if isinstance(api_data, dict) else f"Unexpected response type from FMP for {ticker}."
                self.logger.warning(f"{error_message} Response: {str(api_data)[:200]}")
                # For pattern detection, an empty DataFrame with DatetimeIndex might be preferable to raising error here
                # Or let it raise to be handled by the caller.
                # Let's return an empty DF with expected structure for consistency.
                return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'symbol']).set_index(pd.to_datetime([]))


            if not api_data["historical"]:
                self.logger.warning(f"No data returned from FMP for {ticker} (empty 'historical' list). Dates: {start_date_str} to {end_date_str}")
                # Return an empty DataFrame with DatetimeIndex and standard columns
                # This ensures that df.index.name or df.iloc[x].name would be a Timestamp if data existed
                empty_df_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'symbol'] # Keep 'symbol' as a column
                return pd.DataFrame(columns=empty_df_columns).set_index(pd.to_datetime([]))


            data = pd.DataFrame(api_data["historical"])

            if data.empty:
                self.logger.warning(f"No data returned for {ticker} after DataFrame conversion (FMP).")
                empty_df_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'symbol']
                return pd.DataFrame(columns=empty_df_columns).set_index(pd.to_datetime([]))

            fmp_to_standard_cols = {'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
            
            actual_fmp_cols_present = [col for col in fmp_to_standard_cols.keys() if col in data.columns]
            missing_cols = set(fmp_to_standard_cols.keys()) - set(actual_fmp_cols_present)

            critical_fmp_cols = {'date', 'close'} # Ensure 'date' is critical
            if any(mc in critical_fmp_cols for mc in missing_cols):
                raise ValueError(f"Missing critical FMP columns {missing_cols} in response for {ticker}")
            
            if missing_cols:
                self.logger.warning(f"Missing optional FMP columns {missing_cols} for {ticker}. Proceeding with available data.")
                data = data.rename(columns={k: v for k, v in fmp_to_standard_cols.items() if k in actual_fmp_cols_present})
            else:
                data = data.rename(columns=fmp_to_standard_cols)
            
            if 'Date' not in data.columns:
                 raise ValueError(f"Column 'Date' (renamed from 'date') not found in FMP response for {ticker} after renaming.")

            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values(by='Date', ascending=True)
            
            # *** CRITICAL CHANGE HERE: Set 'Date' as the index ***
            data = data.set_index('Date') 

            data["symbol"] = ticker.upper() # Add symbol column back if it was not part of FMP response or renamed

            # Ensure standard column order (Date is now index)
            final_columns_in_df = ['Open', 'High', 'Low', 'Close', 'Volume', 'symbol']
            # Filter out any columns that might not exist if renaming failed for optional ones
            existing_final_columns = [col for col in final_columns_in_df if col in data.columns]
            data = data[existing_final_columns]

            # self.logger.info(f"Successfully fetched and processed {len(data)} records for {ticker} from FMP, with DatetimeIndex.")
            return data
        
        except aiohttp.ClientResponseError as e: 
            self.logger.error(f"HTTP Error fetching data for {ticker} from FMP: {e.status} {e.message} | URL: {e.request_info.url if e.request_info else 'N/A'}")
            response_text = await e.response.text() if hasattr(e, 'response') and e.response else "No response body."
            # Propagate as a specific error or return empty DF with DatetimeIndex
            # raise ValueError(f"FMP API Error for {ticker} ({e.status}): {e.message}. Details: {response_text[:200]}") from e
            # Returning empty DF for consistency:
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'symbol']).set_index(pd.to_datetime([]))

        except aiohttp.ClientError as e:
            self.logger.error(f"AIOHTTP Client Error fetching data for {ticker} from FMP: {str(e)}")
            # raise ValueError(f"Network or client error getting stock data from FMP for {ticker}: {str(e)}") from e
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'symbol']).set_index(pd.to_datetime([]))
        except ValueError as ve: 
            self.logger.error(f"ValueError during FMP data processing for {ticker}: {str(ve)}")
            raise # Re-raise specific ValueErrors you've identified
        except Exception as e:
            self.logger.error(f"Unexpected error fetching data for {ticker} from FMP: {str(e)} (Type: {type(e).__name__})")
            # raise ValueError(f"Unexpected error getting stock data from FMP for {ticker}: {str(e)}") from e
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'symbol']).set_index(pd.to_datetime([]))

    async def get_historical_data(self, ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Fetch historical daily data for a given symbol within a date range using yfinance.

        Args:
            symbol (str): Stock symbol to fetch data for.
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str, optional): End date in YYYY-MM-DD format. Defaults to today if not provided.

        Returns:
            pd.DataFrame: Historical market data as DataFrame.
        """
        try:
            self.logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
            # Use async wrapper to avoid blocking event loop
            data = await get_yfinance_history(ticker, start=start_date, end=end_date)

            if data.empty:
                raise ValueError(f"No data returned for {ticker}")

            data["symbol"] = ticker.upper()

            # # Save to database            
            # data_dict = data.reset_index().to_dict(orient="records")
            
            # with self.db.session_scope() as session:
            #     for record in data_dict:
            #         record["Date"] = record["Date"].strftime('%Y-%m-%d %H:%M:%S')
                    
            #         stock_price = StockPrice(
            #             ticker=ticker,
            #             date=datetime.datetime.strptime(record["Date"], '%Y-%m-%d %H:%M:%S'),
            #             open=record["Open"],        
            #             high=record["High"],        
            #             low=record["Low"],           
            #             close=record["Close"],       
            #             volume=record["Volume"],
            #             dividends=record["Dividends"],
            #             stock_splits=record["Stock Splits"]
            #         )
            #         session.add(stock_price)
                
            #     session.commit()
            
            # self.logger.info(f"Successfully saved {len(data_dict)} records for {ticker}")
            return data
        
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise ValueError(f"Error getting stock data: {str(e)}")


    def get_stock_data_by_ticker(self, ticker: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get stock data by ticker from database

        Args:
            ticker (str): Stock code to get data
            limit (int): Maximum number of records to return
            offset (int): Starting position

        Returns:
            List[Dict[str, Any]]: List of stock data
        """
        try:
            with self.db.session_scope() as session:
                query = session.query(StockPrice).filter(
                    StockPrice.ticker == ticker
                ).order_by(
                    StockPrice.date.desc()
                ).offset(offset).limit(limit)
                
                results = query.all()
                
                # Convert to dictionaries
                stock_data = []
                for stock in results:
                    stock_data.append({
                        "id": str(stock.id),
                        "ticker": stock.ticker,
                        "date": stock.date.isoformat(),
                        "open": float(stock.open),
                        "high": float(stock.high),
                        "low": float(stock.low),
                        "close": float(stock.close),
                        "volume": int(stock.volume),
                        "dividends": float(stock.dividends),
                        "stock_splits": float(stock.stock_splits)
                    })
                
                return stock_data
                
        except Exception as e:
            self.logger.error(f"Error getting stock data for {ticker}: {str(e)}")
            raise ValueError(f"Error getting stock data: {str(e)}")


    def get_stock_data_by_ticker_and_lookback(self, ticker: str, lookback_days: int = 30, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get stock data by ticker and lookback days from database

        Args:
            ticker (str): Stock code to get data
            lookback_days (int): Number of days to look back from today
            limit (int): Maximum number of records to return
            offset (int): Starting position for pagination

        Returns:
            List[Dict[str, Any]]: List of stock data
        """
        try:
            # Calculate date range based on lookback_days
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=lookback_days)
            
            with self.db.session_scope() as session:
                query = session.query(StockPrice).filter(
                    StockPrice.ticker == ticker,
                    StockPrice.date >= start_date,
                    StockPrice.date <= end_date
                ).order_by(
                    StockPrice.date.desc()
                ).offset(offset).limit(limit)
                
                results = query.all()
                
                # Convert to dictionaries
                stock_data = []
                for stock in results:
                    stock_data.append({
                        "id": str(stock.id),
                        "ticker": stock.ticker,
                        "date": stock.date.isoformat(),
                        "open": float(stock.open),
                        "high": float(stock.high),
                        "low": float(stock.low),
                        "close": float(stock.close),
                        "volume": int(stock.volume),
                        "dividends": float(stock.dividends),
                        "stock_splits": float(stock.stock_splits)
                    })
                
                formatted_start = start_date.strftime('%Y-%m-%d')
                formatted_end = end_date.strftime('%Y-%m-%d')
                self.logger.info(f"Retrieved {len(stock_data)} records for {ticker} for the last {lookback_days} days ({formatted_start} to {formatted_end})")
                return stock_data
                
        except Exception as e:
            self.logger.error(f"Error getting stock data for {ticker} for the last {lookback_days} days: {str(e)}")
            raise ValueError(f"Error getting stock data: {str(e)}")
    

    def get_stock_data_by_ticker_and_date_range(self, ticker: str, start_date: str, end_date: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get stock data by ticker and time period from database

        Args:
            ticker (str): Stock code to get data
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            limit (int): Maximum number of records to return
            offset (int): Starting position for pagination

        Returns:
            List[Dict[str, Any]]: List of stock data
        """
        try:
            start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            
            with self.db.session_scope() as session:
                query = session.query(StockPrice).filter(
                    StockPrice.ticker == ticker,
                    StockPrice.date >= start,
                    StockPrice.date <= end
                ).order_by(
                    StockPrice.date.desc()
                ).offset(offset).limit(limit)
                
                results = query.all()
                
                # Convert to dictionaries
                stock_data = []
                for stock in results:
                    stock_data.append({
                        "id": str(stock.id),
                        "ticker": stock.ticker,
                        "date": stock.date.isoformat(),
                        "open": float(stock.open),
                        "high": float(stock.high),
                        "low": float(stock.low),
                        "close": float(stock.close),
                        "volume": int(stock.volume),
                        "dividends": float(stock.dividends),
                        "stock_splits": float(stock.stock_splits)
                    })
                
                self.logger.info(f"Retrieved {len(stock_data)} records for {ticker} from {start_date} to {end_date}")
                return stock_data
                
        except Exception as e:
            self.logger.error(f"Error getting stock data for {ticker} in date range: {str(e)}")
            raise ValueError(f"Error getting stock data: {str(e)}")


    def delete_stock_data_by_ticker(self, ticker: str) -> int:
        """
        Delete all stock data by ticker

        Args:
            ticker (str): Stock code to delete

        Returns:
            int: Number of records deleted
        """
        try:
            with self.db.session_scope() as session:
                deleted_count = session.query(StockPrice).filter(
                    StockPrice.ticker == ticker
                ).delete(synchronize_session=False)
                
                self.logger.info(f"Deleted {deleted_count} records for ticker {ticker}")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Error deleting stock data for {ticker}: {str(e)}")
            raise ValueError(f"Error deleting stock data: {str(e)}")
        

    def delete_stock_data_by_lookback(self, ticker: str, lookback_days: int = 30) -> int:
        """
        Delete stock data by ticker and lookback days

        Args:
            ticker (str): Stock code
            lookback_days (int): Number of days to look back from today

        Returns:
            int: Number of records deleted
        """
        try:
            # Calculate date range based on lookback_days
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=lookback_days)
            
            with self.db.session_scope() as session:
                deleted_count = session.query(StockPrice).filter(
                    StockPrice.ticker == ticker,
                    StockPrice.date >= start_date,
                    StockPrice.date <= end_date
                ).delete(synchronize_session=False)
                
                formatted_start = start_date.strftime('%Y-%m-%d')
                formatted_end = end_date.strftime('%Y-%m-%d')
                self.logger.info(f"Deleted {deleted_count} records for ticker {ticker} for the last {lookback_days} days ({formatted_start} to {formatted_end})")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Error deleting stock data for {ticker} for the last {lookback_days} days: {str(e)}")
            raise ValueError(f"Error deleting stock data: {str(e)}")
    

    def delete_stock_data_by_date_range(self, ticker: str, start_date: str, end_date: str) -> int:
        """
        Delete stock data by ticker and time period

        Args:
            ticker (str): Stock code
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)

        Returns:
            int: Number of records deleted
        """
        try:
            start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            
            with self.db.session_scope() as session:
                deleted_count = session.query(StockPrice).filter(
                    StockPrice.ticker == ticker,
                    StockPrice.date >= start,
                    StockPrice.date <= end
                ).delete(synchronize_session=False)
                
                self.logger.info(f"Deleted {deleted_count} records for ticker {ticker} from {start_date} to {end_date}")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Error deleting stock data for {ticker} in date range: {str(e)}")
            raise ValueError(f"Error deleting stock data: {str(e)}")