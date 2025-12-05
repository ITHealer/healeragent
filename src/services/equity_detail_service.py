import asyncio
import httpx
from typing import List, Optional, Dict, Any
from pydantic import ValidationError
from urllib.parse import quote
import aioredis 
# import redis.asyncio as aioredis

from src.utils.config import settings
# from src.utils.logger.set_up_log_dataFMP import setup_logger
from src.utils.logger.custom_logging import LoggerMixin
from src.services.chart_pattern_service import chart_pattern_service
from src.models.equity import HistoricalDataItem, StockDetailPayload, MACDOutput, CompanyProfile

BASE_FMP_URL = settings.BASE_FMP_URL
FMP_URL_STABLE = settings.FMP_URL_STABLE
FMP_API_KEY_FOR_SERVICE = settings.FMP_API_KEY


class EquityDetailService(LoggerMixin):
    def __init__(self):
        super().__init__()

    # ============================  Technical Indicator ============================
    @staticmethod
    def _calculate_rsi(prices: List[float], period: int = 14) -> List[Optional[float]]:
        """
        When calculating RSI, we need to know:
            - Average gain
            - Average loss
        """
        # Return all None if not enough data to compute RSI
        if len(prices) <= period:
            return [None] * len(prices)

        # Calculate daily price changes # changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        changes = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            changes.append(change)

        # Separate gains and losses
        gains = [c if c > 0 else 0 for c in changes]
        losses = [-c if c < 0 else 0 for c in changes]
        
        # Calculate initial average gain and loss
        avg_gain = sum(gains[:period-1]) / (period - 1)
        avg_loss = sum(losses[:period-1]) / (period - 1)

        # Initialize RSI list with None for periods without enough data # rsi_values = [None] * (period -1)
        rsi_values = []
        for _ in range(period - 1):
            rsi_values.append(None)

        # Compute RSI using exponential smoothing (EMA): avg_today = (avg_yesterday * (period - 1) + value_today) / period
        for i in range(period - 1, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
        
        return rsi_values
    

    @staticmethod
    def _calculate_macd(
        prices: List[float], 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> List[Optional[MACDOutput]]:
        
        # Return all None if not enough data for slow EMA
        if len(prices) < slow_period:
            return [None] * len(prices)

         # Helper function to compute Exponential Moving Average
        def _calculate_ema(data, period):
            ema = []
            if len(data) >= period:
                # First EMA is simple average of first 'period' values
                ema.append(sum(data[:period]) / period)
                multiplier = 2 / (period + 1)

                # Apply EMA formula to the rest of the data
                for i in range(period, len(data)):
                    new_ema = (data[i] - ema[-1]) * multiplier + ema[-1]
                    ema.append(new_ema)

            return [None] * (len(data) - len(ema)) + ema

        # Calculate fast and slow EMAs
        ema_fast = _calculate_ema(prices, fast_period)
        ema_slow = _calculate_ema(prices, slow_period)

        # Compute MACD line = EMA(fast) - EMA(slow)
        macd_line = []
        for i in range(len(prices)):
            if ema_fast[i] is not None and ema_slow[i] is not None:
                macd_line.append(ema_fast[i] - ema_slow[i])
            else:
                macd_line.append(None)
        
        # Filter out None to compute signal line from valid MACD points
        macd_points_for_signal = [m for m in macd_line if m is not None]
        if not macd_points_for_signal or len(macd_points_for_signal) < signal_period:
            return [None] * len(prices)
        
        # Calculate EMA of MACD line to get signal line
        signal_line_ema = _calculate_ema(macd_points_for_signal, signal_period)
        signal_line = [None] * (len(macd_line) - len(signal_line_ema)) + signal_line_ema
        
        # Combine MACD line and signal line into final output with histogram
        macd_output_list: List[Optional[MACDOutput]] = []
        for i in range(len(prices)):
            if macd_line[i] is not None and signal_line[i] is not None:
                histogram = macd_line[i] - signal_line[i]
                macd_output_list.append(MACDOutput(macd_line=macd_line[i], signal_line=signal_line[i], histogram=histogram))
            else:
                macd_output_list.append(None)
                
        return macd_output_list
    
    
    # ============================  Fetch data functions ============================
    async def _fetch_fmp_data_helper(
        self, 
        client: httpx.AsyncClient, 
        endpoint_template: str, symbol: str
    ) -> Optional[Any]:
    
        if not FMP_API_KEY_FOR_SERVICE or FMP_API_KEY_FOR_SERVICE == "YOUR_FMP_API_KEY_PLACEHOLDER":
            self.logger.error(f"FMP API Key not configured for endpoint '{endpoint_template}' symbol '{symbol}'.")
            return None
            
        # Build URL with symbol and API key
        encoded_symbol = quote(symbol, safe='')
        endpoint_path_with_symbol = endpoint_template.replace("{symbol}", encoded_symbol)
        url = f"{BASE_FMP_URL}{endpoint_path_with_symbol}?apikey={FMP_API_KEY_FOR_SERVICE}"
        
        try:
            response = await client.get(url, timeout=15.0)
            response.raise_for_status()
            data = response.json()

            # Return the first item if it's a list, or return the dict directly
            if isinstance(data, list):
                if data:
                    return data[0]
                else:
                    return None
            elif isinstance(data, dict):
                return data
            else:
                return None
            
        except Exception as e:
            self.logger.warning(f"Error calling FMP for {symbol} at {endpoint_template}: {e}", exc_info=False)
        return None


    # async def _fetch_raw_historical_ohlcv(
    #     self,
    #     symbol: str,
    #     timeframe: str,
    #     from_date_str: str,
    #     to_date_str: str,
    #     client: httpx.AsyncClient
    # ) -> List[Dict[str, Any]]:
        
    #     encoded_symbol = quote(symbol, safe='')
    #     url = f"{BASE_FMP_URL}/v3/historical-chart/{timeframe}/{encoded_symbol}?from={from_date_str}&to={to_date_str}&apikey={FMP_API_KEY_FOR_SERVICE}"
        
    #     try:
    #         response = await client.get(url, timeout=20.0)
    #         response.raise_for_status()
    #         data = response.json()

    #         if isinstance(data, list):
    #             return data
    #         else:
    #             return []
    #         # return data if isinstance(data, list) else []

    #     except Exception as e:
    #         self.logger.error(f"Unable to get raw OHLCV data for {symbol}: {e}")
    #         return []

    async def _fetch_raw_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        from_date_str: Optional[str],  
        to_date_str: Optional[str],    
        client: httpx.AsyncClient
    ) -> List[Dict[str, Any]]:
        """
        Fetch raw OHLCV data from FMP.
        Nếu from_date_str và to_date_str là None, FMP sẽ tự động lấy dữ liệu ngày gần nhất.
        """
        
        encoded_symbol = quote(symbol, safe='')
        base_url = f"{BASE_FMP_URL}/v3/historical-chart/{timeframe}/{encoded_symbol}"
        
        # ✅ Build params - chỉ thêm from/to khi có giá trị
        params = []
        if from_date_str is not None:
            params.append(f"from={from_date_str}")
        if to_date_str is not None:
            params.append(f"to={to_date_str}")
        params.append(f"apikey={FMP_API_KEY_FOR_SERVICE}")
        
        url = f"{base_url}?{'&'.join(params)}"
        
        try:
            # self.logger.debug(f"Fetching OHLCV for {symbol}: {url}")
            response = await client.get(url, timeout=20.0)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list):
                # self.logger.info(f"Successfully fetched {len(data)} OHLCV records for {symbol}")
                return data
            else:
                self.logger.warning(f"Unexpected FMP response format for {symbol}: {type(data)}")
                return []

        except Exception as e:
            self.logger.error(f"Unable to get raw OHLCV data for {symbol}: {e}")
            return []
        

    async def get_equity_detail(
        self,
        symbol: str,
        timeframe: str,
        from_date_str: Optional[str],  
        to_date_str: Optional[str],    
        client: httpx.AsyncClient,
        redis_client: Optional[aioredis.Redis] = None
    ) -> Optional[StockDetailPayload]:
        
        symbol_upper = symbol.upper()

        # Run multiple FMP API calls concurrently using asyncio.gather
        quote_task = self._fetch_fmp_data_helper(client, "/v3/quote/{symbol}", symbol_upper)
        key_metrics_task = self._fetch_fmp_data_helper(client, "/v3/key-metrics-ttm/{symbol}", symbol_upper)
        ratios_task = self._fetch_fmp_data_helper(client, "/v3/ratios-ttm/{symbol}", symbol_upper)
        raw_ohlcv_task = self._fetch_raw_historical_ohlcv(
            symbol_upper, timeframe, from_date_str, to_date_str, client
        )
        
        try:
            results = await asyncio.gather(
                quote_task, key_metrics_task, ratios_task, raw_ohlcv_task,
                return_exceptions=True
            )
        except Exception as e:
            self.logger.exception(f"Fatal error in asyncio.gather for {symbol_upper}: {e}")
            return None
        
        quote_data, key_metrics_data, ratios_data, raw_ohlcv_list = results

        # Safely unwrap each result and fallback to empty dict on failure
        def _process_result(result, name):
            if isinstance(result, Exception):
                self.logger.error(f"Error getting {name} for {symbol_upper}: {result}", exc_info=False)
                return {}
            return result or {}

        fmp_quote_data = _process_result(quote_data, "Quote")
        fmp_key_metrics_data = _process_result(key_metrics_data, "Key Metrics TTM")
        fmp_ratios_data = _process_result(ratios_data, "Ratios TTM")
        
        # ✅ FIX: Xử lý riêng raw_ohlcv_list vì nó là LIST chứ không phải DICT
        if isinstance(raw_ohlcv_list, Exception):
            self.logger.error(f"Error getting Raw OHLCV for {symbol_upper}: {raw_ohlcv_list}", exc_info=False)
            raw_ohlcv_data = []
        elif raw_ohlcv_list:
            raw_ohlcv_data = raw_ohlcv_list
        else:
            raw_ohlcv_data = []
        
        # Ensure quote data is available before proceeding
        if not fmp_quote_data:
            self.logger.error(f"No quote data for {symbol_upper}, cannot create stock details.")
            return None

        # Extract closing prices for technical indicator calculation
        close_prices = []
        for p in raw_ohlcv_data:
            if isinstance(p, dict) and p.get('close') is not None:
                close_price = float(p['close'])
                close_prices.append(close_price)

        # Compute RSI and MACD if price data is available
        rsi_values, macd_values, detected_patterns = [], [], []
        if len(close_prices) > 1:
            try: rsi_values = self._calculate_rsi(close_prices)
            except Exception as e: self.logger.warning(f"Error calculating RSI for {symbol_upper}: {e}")
            
            try: macd_values = self._calculate_macd(close_prices)
            except Exception as e: self.logger.warning(f"Error calculating MACD for {symbol_upper}: {e}")
        
        try: 
            detected_patterns = chart_pattern_service.find_all_patterns(raw_ohlcv_data)
        except Exception as e: 
            self.logger.warning(f"Error finding pattern for {symbol_upper}: {e}")

        # Build a technical indicator map keyed by timestamp
        tech_map: Dict[str, Dict[str, Any]] = {}
        for i, point in enumerate(raw_ohlcv_data):
            ts_str = point.get('date')
            if ts_str:
                tech_map[ts_str] = {
                    "rsi": rsi_values[i] if len(rsi_values) > i else None,
                    "macd": macd_values[i] if len(macd_values) > i else None,
                    "ma5": sum(close_prices[max(0, i-4):i+1]) / len(close_prices[max(0, i-4):i+1]) if i >= 4 else None,
                    "ma20": sum(close_prices[max(0, i-19):i+1]) / len(close_prices[max(0, i-19):i+1]) if i >= 19 else None,
                }

        # Convert raw OHLCV data + indicators into HistoricalDataItem objects
        historical_items: List[HistoricalDataItem] = []
        for point in raw_ohlcv_data:
            if not (isinstance(point, dict) and point.get('date')): 
                continue

            ts_str = point.get('date')
            try:
                open_val = float(point.get('open'))
                close_val = float(point.get('close'))
                
                # Compute daily change and percent change
                change_calc = None
                change_percent_calc = None

                if open_val is not None and close_val is not None:
                    change_calc = close_val - open_val
                    if open_val > 0:
                        change_percent_calc = (change_calc / open_val) * 100
                    else:
                        change_percent_calc = 0.0

                # Combine raw data with indicators and append to historical list
                indicator_data = tech_map.get(ts_str, {})
                historical_items.append(
                    HistoricalDataItem(
                        date=ts_str,
                        open=float(point.get('open')),
                        high=float(point.get('high')),
                        low=float(point.get('low')),
                        close=float(point.get('close')),
                        volume=float(point.get('volume')),
                        change=change_calc,
                        change_percent=change_percent_calc,
                        ma5=indicator_data.get('ma5'),
                        ma20=indicator_data.get('ma20'),
                        rsi=indicator_data.get('rsi'),
                        macd=indicator_data.get('macd'),
                    )
                )
            except (ValueError, TypeError, KeyError) as e:
                self.logger.warning(f"Error processing historical data point for {symbol_upper} at {ts_str}: {e}")

        # Build and return the final stock detail payload
        try:
            combined_data = {**fmp_ratios_data, **fmp_key_metrics_data, **fmp_quote_data}
            stock_detail = StockDetailPayload(
                **combined_data,
                data=historical_items,
                detected_patterns=detected_patterns
            )
            return stock_detail
        
        except ValidationError as pydantic_error:
            self.logger.exception(f"Pydantic error when generating StockDetailPayload for {symbol_upper}: {pydantic_error}")
            return None
        

    async def get_equity_profile(
        self,
        symbol: str,
        client: httpx.AsyncClient
    ) -> Optional[Dict[str, Any]]:
        
        if not FMP_API_KEY_FOR_SERVICE or FMP_API_KEY_FOR_SERVICE == "YOUR_FMP_API_KEY_PLACEHOLDER":
            self.logger.error("FMP API Key not configured for Stable Profile.")
            return []

        encoded_symbol = quote(symbol.upper(), safe='')
        url = f"{FMP_URL_STABLE}/profile?symbol={encoded_symbol}&apikey={FMP_API_KEY_FOR_SERVICE}"

        try:
            if client is None:
                async with httpx.AsyncClient() as new_client:
                    resp = await new_client.get(url, timeout=30.0)
            else:
                resp = await client.get(url, timeout=30.0)

            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, list):
                self.logger.warning(f"Stable Profile unexpected format for {symbol}: {type(data)}")
                return []

            profiles: List[CompanyProfile] = []
            for item in data:
                try:
                    profiles.append(CompanyProfile.model_validate(item))
                except ValidationError as ve:
                    self.logger.warning(f"Stable Profile parse error for {symbol}: {ve}")
            return profiles

        except Exception as e:
            self.logger.warning(f"Error calling Stable Profile for {symbol}: {e}", exc_info=False)
            return []

    # ============================  Fundamental tool ============================

    ## ============================  get_insider_trades ============================
    ### get_insider_transactions (fmp_data.py)
    async def get_insider_trades(
        self,
        symbol: str,
        limit: int = 100,
        client: Optional[httpx.AsyncClient] = None
    ) -> List[Dict[str, Any]]:
        """
        Get insider trades from FMP
        
        Args:
            symbol: Stock symbol
            limit: Number of trades to fetch
            client: Optional HTTP client
            
        Returns:
            List of insider trades
        """
        # Create client if not provided
        if client is None:
            async with httpx.AsyncClient() as new_client:
                return await self._get_insider_trades_impl(symbol, limit, new_client)
        else:
            return await self._get_insider_trades_impl(symbol, limit, client)
    
    async def _get_insider_trades_impl(
        self,
        symbol: str,
        limit: int,
        client: httpx.AsyncClient
    ) -> List[Dict[str, Any]]:
        """Implementation for get_insider_trades"""
        try:
            encoded_symbol = quote(symbol.upper(), safe='')
            url = f"{BASE_FMP_URL}/v4/insider-trading?symbol={encoded_symbol}&limit={limit}&apikey={FMP_API_KEY_FOR_SERVICE}"
            
            response = await client.get(url, timeout=15.0)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list):
                # Format the response to match expected structure
                formatted_data = []
                for item in data:
                    formatted_item = {
                        "symbol": item.get("symbol", symbol),
                        "filingDate": item.get("filingDate"),
                        "transactionDate": item.get("transactionDate"),
                        "reportingCik": item.get("reportingCik"),
                        "companyCik": item.get("companyCik"),
                        "transactionType": item.get("transactionType"),
                        "securitiesOwned": item.get("securitiesOwned"),
                        "reportingName": item.get("reportingName"),
                        "typeOfOwner": item.get("typeOfOwner"),
                        "acquisitionOrDisposition": item.get("acquisitionOrDisposition"),
                        "directOrIndirect": item.get("directOrIndirect"),
                        "formType": item.get("formType"),
                        "securitiesTransacted": item.get("securitiesTransacted"),
                        "price": item.get("price"),
                        "securityName": item.get("securityName"),
                        "url": item.get("url"),
                    }
                    formatted_data.append(formatted_item)
                # print(formatted_data)
                return formatted_data
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching insider trades for {symbol}: {e}")
            return []
        

    ## ============================  get_insider_sentiment ============================
    # get_insider_sentiment -> get_insider_trading_statistics -> _get_insider_trading_statistics_impl
    async def get_insider_trading_statistics(
        self,
        symbol: str,
        client: Optional[httpx.AsyncClient] = None
    ) -> List[Dict[str, Any]]:
        """
        Get insider trading statistics from FMP
        
        Args:
            symbol: Stock symbol
            client: Optional HTTP client
            
        Returns:
            List of insider trading statistics
        """
        # Create client if not provided
        if client is None:
            async with httpx.AsyncClient() as new_client:
                return await self._get_insider_trading_statistics_impl(symbol, new_client)
        else:
            return await self._get_insider_trading_statistics_impl(symbol, client)
        
    async def _get_insider_trading_statistics_impl(
        self,
        symbol: str,
        client: httpx.AsyncClient
    ) -> List[Dict[str, Any]]:
        """Implementation for get_insider_trading_statistics"""
        try:
            encoded_symbol = quote(symbol.upper(), safe='')
            url = f"{BASE_FMP_URL}/v4/insider-roaster-statistic?symbol={encoded_symbol}&apikey={FMP_API_KEY_FOR_SERVICE}"
            
            response = await client.get(url, timeout=15.0)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list):
                # Format the response to match expected structure
                formatted_data = []
                for item in data:
                    formatted_item = {
                        'symbol':        item.get('symbol', symbol),
                        'cik':           item.get('cik', ''),
                        'year':          item.get('year', 0),
                        'quarter':       item.get('quarter', 0),
                        'purchases':     item.get('purchases', 0),
                        'sales':         item.get('sales', 0),
                        'buySellRatio':  item.get('buyToSellRatio', item.get('buySellRatio', 0)),
                        'totalBought':   item.get('totalBought', 0),
                        'totalSold':     item.get('totalSold', 0),
                        'averageBought': item.get('averageBought', 0),
                        'averageSold':   item.get('averageSold', 0),
                        'pPurchases':    item.get('pPurchases', 0),
                        'sSales':        item.get('sSales', 0),
                    }
                    formatted_data.append(formatted_item)
                return formatted_data
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching insider trading statistics for {symbol}: {e}")
            return []
        

    # ============================ get_financial_statements ============================
    ## Get financial data flow input parameter 
    async def get_balance_sheet_statement(
        self,
        symbol: str,
        period: str = 'annual',
        limit: int = 5,
        client: Optional[httpx.AsyncClient] = None
    ) -> List[Dict[str, Any]]:
        """
        Get balance sheet statements from FMP
        
        Args:
            symbol: Stock symbol
            period: 'annual' or 'quarter'
            limit: Number of periods to fetch
            client: Optional HTTP client
            
        Returns:
            List of balance sheet statements
        """
        if client is None:
            async with httpx.AsyncClient() as new_client:
                return await self._get_balance_sheet_statement_impl(symbol, period, limit, new_client)
        else:
            return await self._get_balance_sheet_statement_impl(symbol, period, limit, client)
    
    async def _get_balance_sheet_statement_impl(
        self,
        symbol: str,
        period: str,
        limit: int,
        client: httpx.AsyncClient
    ) -> List[Dict[str, Any]]:
        """Implementation for get_balance_sheet_statement"""
        try:
            encoded_symbol = quote(symbol.upper(), safe='')
            url = f"{BASE_FMP_URL}/v3/balance-sheet-statement/{encoded_symbol}?period={period}&limit={limit}&apikey={FMP_API_KEY_FOR_SERVICE}"
            
            response = await client.get(url, timeout=15.0)
            response.raise_for_status()
            data = response.json()
            
            return data if isinstance(data, list) else []
            
        except Exception as e:
            self.logger.error(f"Error fetching balance sheet for {symbol}: {e}")
            return []
        

    # ============================ get_financial_statements ============================
    ## Get financial data flow input parameter 
    async def get_income_statement(
        self,
        symbol: str,
        period: str = 'annual',
        limit: int = 5,
        client: Optional[httpx.AsyncClient] = None
    ) -> List[Dict[str, Any]]:
        """
        Get income statements from FMP
        
        Args:
            symbol: Stock symbol
            period: 'annual' or 'quarter'
            limit: Number of periods to fetch
            client: Optional HTTP client
            
        Returns:
            List of income statements
        """
        if client is None:
            async with httpx.AsyncClient() as new_client:
                return await self._get_income_statement_impl(symbol, period, limit, new_client)
        else:
            return await self._get_income_statement_impl(symbol, period, limit, client)
    
    async def _get_income_statement_impl(
        self,
        symbol: str,
        period: str,
        limit: int,
        client: httpx.AsyncClient
    ) -> List[Dict[str, Any]]:
        """Implementation for get_income_statement"""
        try:
            encoded_symbol = quote(symbol.upper(), safe='')
            url = f"{BASE_FMP_URL}/v3/income-statement/{encoded_symbol}?period={period}&limit={limit}&apikey={FMP_API_KEY_FOR_SERVICE}"
            
            response = await client.get(url, timeout=15.0)
            response.raise_for_status()
            data = response.json()
            
            return data if isinstance(data, list) else []
            
        except Exception as e:
            self.logger.error(f"Error fetching income statement for {symbol}: {e}")
            return []
        
    
    # ============================ get_financial_statements ============================
    ## Get financial data flow input parameter 
    async def get_cash_flow_statement(
        self,
        symbol: str,
        period: str = 'annual',
        limit: int = 5,
        client: Optional[httpx.AsyncClient] = None
    ) -> List[Dict[str, Any]]:
        """
        Get cash flow statements from FMP
        
        Args:
            symbol: Stock symbol
            period: 'annual' or 'quarter'
            limit: Number of periods to fetch
            client: Optional HTTP client
            
        Returns:
            List of cash flow statements
        """
        if client is None:
            async with httpx.AsyncClient() as new_client:
                return await self._get_cash_flow_statement_impl(symbol, period, limit, new_client)
        else:
            return await self._get_cash_flow_statement_impl(symbol, period, limit, client)
    
    async def _get_cash_flow_statement_impl(
        self,
        symbol: str,
        period: str,
        limit: int,
        client: httpx.AsyncClient
    ) -> List[Dict[str, Any]]:
        """Implementation for get_cash_flow_statement"""
        try:
            encoded_symbol = quote(symbol.upper(), safe='')
            url = f"{BASE_FMP_URL}/v3/cash-flow-statement/{encoded_symbol}?period={period}&limit={limit}&apikey={FMP_API_KEY_FOR_SERVICE}"
            
            response = await client.get(url, timeout=15.0)
            response.raise_for_status()
            data = response.json()
            
            return data if isinstance(data, list) else []
            
        except Exception as e:
            self.logger.error(f"Error fetching cash flow statement for {symbol}: {e}")
            return []
        

    # ============================ get_financial_statements (fmp_data.py) ============================
    ## Get all financial data
    async def get_financial_statements(
        self,
        symbol: str,
        period: str = 'annual',
        limit: int = 5,
        client: Optional[httpx.AsyncClient] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all financial statements (balance sheet, income, cash flow) from FMP
        
        Args:
            symbol: Stock symbol
            period: 'annual' or 'quarter'
            limit: Number of periods to fetch
            client: Optional HTTP client
            
        Returns:
            List of all financial statements combined
        """
        if client is None:
            async with httpx.AsyncClient() as new_client:
                return await self._get_financial_statements_impl(symbol, period, limit, new_client)
        else:
            return await self._get_financial_statements_impl(symbol, period, limit, client)
    
    async def _get_financial_statements_impl(
        self,
        symbol: str,
        period: str,
        limit: int,
        client: httpx.AsyncClient
    ) -> List[Dict[str, Any]]:
        """Implementation for get_financial_statements"""
        try:
            # Fetch all three statement types in parallel
            balance_task = self._get_balance_sheet_statement_impl(symbol, period, limit, client)
            income_task = self._get_income_statement_impl(symbol, period, limit, client)
            cashflow_task = self._get_cash_flow_statement_impl(symbol, period, limit, client)
            
            results = await asyncio.gather(
                balance_task, income_task, cashflow_task,
                return_exceptions=True
            )
            
            balance_sheets, income_statements, cashflow_statements = results
            
            # Handle errors
            if isinstance(balance_sheets, Exception):
                self.logger.error(f"Error fetching balance sheets: {balance_sheets}")
                balance_sheets = []
            if isinstance(income_statements, Exception):
                self.logger.error(f"Error fetching income statements: {income_statements}")
                income_statements = []
            if isinstance(cashflow_statements, Exception):
                self.logger.error(f"Error fetching cash flow statements: {cashflow_statements}")
                cashflow_statements = []
            
            # Combine all statements by date
            combined_statements = []
            
            # Create a map to merge statements by date
            statements_by_date = {}
            
            # Process balance sheets
            for bs in balance_sheets:
                date = bs.get('date', '')
                if date not in statements_by_date:
                    statements_by_date[date] = {}
                statements_by_date[date].update(bs)
            
            # Process income statements
            for inc in income_statements:
                date = inc.get('date', '')
                if date not in statements_by_date:
                    statements_by_date[date] = {}
                statements_by_date[date].update(inc)
            
            # Process cash flow statements
            for cf in cashflow_statements:
                date = cf.get('date', '')
                if date not in statements_by_date:
                    statements_by_date[date] = {}
                statements_by_date[date].update(cf)
            
            # Convert to list
            for date, statement in statements_by_date.items():
                combined_statements.append(statement)
            
            # Sort by date descending
            combined_statements.sort(key=lambda x: x.get('date', ''), reverse=True)
            
            return combined_statements[:limit]
            
        except Exception as e:
            self.logger.error(f"Error fetching financial statements for {symbol}: {e}")
            return []