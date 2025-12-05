import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import pandas as pd
from dateutil.relativedelta import relativedelta
import httpx
from src.utils.config import settings
from src.utils.logger.custom_logging import LoggerMixin
from src.services.equity_detail_service import EquityDetailService
from src.services.news_service import NewsService
from src.services.profile_service import ProfileService
from src.stock.crawlers.market_data_provider import MarketData
from src.helpers.redis_cache import get_cache, set_cache, get_redis_client_llm

class FMPData(LoggerMixin):
    
    def __init__(self):
        super().__init__()
        self.api_key = settings.FMP_API_KEY
        self.base_url = settings.BASE_FMP_URL
        
        self.equity_service = EquityDetailService()
        self.news_service = NewsService()
        self.profile_service = ProfileService()

        self.market_data = MarketData()
        self._cache_dir = None
        self.http_client = httpx.AsyncClient(timeout=30.0)    
        
    # ============================ FMP: News ============================
    async def get_company_news(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str,
        max_items: int = 100
    ) -> Dict[str, List[Dict]]:
        """
        Get news from FMP provider
        
        Args:
            ticker: Stock symbol
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
            max_items: Maximum number of news items
            
        Returns:
            Dict with key is date, value is list news items
        """
        try:
            news_list = await self.news_service.get_company_news(ticker.upper(), max_items)
            
            if not news_list:
                return {}
            
            # Format data
            formatted_data = {}
            
            for news_item in news_list:
                # Parse date from news item
                publish_date = news_item.date
                if isinstance(publish_date, str):
                    try:
                        date_obj = datetime.fromisoformat(publish_date.replace('Z', '+00:00'))
                        date_key = date_obj.strftime("%Y-%m-%d")
                    except:
                        continue
                else:
                    date_key = publish_date.strftime("%Y-%m-%d")
                
                # Check if date is in range
                if start_date <= date_key <= end_date:
                    if date_key not in formatted_data:
                        formatted_data[date_key] = []
                    
                    # Format data
                    formatted_item = {
                        "category": "company news",
                        "datetime": int(date_obj.timestamp()),
                        "headline": news_item.title,
                        "id": hash(news_item.news_url) % 10000000,  # Generate pseudo ID
                        "image": news_item.image_url or "",
                        "related": ticker.upper(),
                        "source": news_item.source_site,
                        "summary": news_item.description[:200] + "..." if len(news_item.description) > 200 else news_item.description,
                        "url": news_item.news_url
                    }
                    
                    formatted_data[date_key].append(formatted_item)
            
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"Error fetching news for {ticker}: {str(e)}")
            return {}
        

    # ============================ Cache Data ============================
    @property
    def cache_dir(self):
        if self._cache_dir is None:
            from .config import get_config
            config = get_config()
            self._cache_dir = config.get("data_cache_dir", "data_cache")
            os.makedirs(self._cache_dir, exist_ok=True)
        return self._cache_dir
    
    # ============================ Format Data ============================
    def _format_dataframe_for_tradingagents(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format DataFrame từ FMP sang format của YFin
        
        FMP format: Date (index), Open, High, Low, Close, Volume, symbol
        YFin format: Date, Open, High, Low, Close, Adj Close, Volume
        """
        if df.empty:
            return df
            
        # Reset index để Date thành column
        if df.index.name == 'Date':
            df = df.reset_index()
            
        # Ensure Date column exists and is properly formatted
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Add Adj Close (copy từ Close)
        if 'Close' in df.columns and 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
            
        # Select and order columns like YFin
        columns_order = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        available_columns = [col for col in columns_order if col in df.columns]
        df = df[available_columns]
        
        # Round numerical values
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].round(2)
                
        return df
    
    
    # ============================ FMP: get_historical_data_lookback_ver2 ============================
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Lấy historical data từ FMP
        
        Args:
            symbol: Stock ticker
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
            
        Returns:
            DataFrame với format giống YFin
        """
        try:
            # Calculate lookback days
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            lookback_days = (end_dt - start_dt).days + 30  # Add buffer
            
            self.logger.info(f"Fetching FMP data for {symbol} from {start_date} to {end_date}")
            
            # Use MarketData từ xyz project
            df = await self.market_data.get_historical_data_lookback_ver2(
                ticker=symbol,
                lookback_days=lookback_days
            )
            
            if df.empty:
                self.logger.warning(f"No data returned from FMP for {symbol}")
                return pd.DataFrame()
            
            # Format DataFrame
            df = self._format_dataframe_for_tradingagents(df)
            
            # Filter by date range
            df['DateOnly'] = pd.to_datetime(df['Date']).dt.date
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
            
            df = df[(df['DateOnly'] >= start_date_obj) & (df['DateOnly'] <= end_date_obj)]
            df = df.drop('DateOnly', axis=1)
            
            self.logger.info(f"Retrieved {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching FMP data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def get_historical_data_with_cache(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Lấy data với caching mechanism
        """
        cache_file = os.path.join(
            self.cache_dir,
            f"{symbol}-FMP-data-{start_date}-{end_date}.csv"
        )
        
        # Check cache
        if not force_refresh and os.path.exists(cache_file):
            # Check if cache is not too old (e.g., 1 day)
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if cache_age.days < 1:
                self.logger.info(f"Using cached data for {symbol}")
                return pd.read_csv(cache_file)
        
        # Fetch fresh data
        df = await self.get_historical_data(symbol, start_date, end_date)
        
        # Save to cache
        if not df.empty:
            df.to_csv(cache_file, index=False)
            self.logger.info(f"Cached data saved to {cache_file}")
            
        return df
    

    # ============================ FMP: Fundamental ============================

    ## get_insider_transactions -> call func get_insider_trades to get data from FMP API (equity_detail_service.py)
    async def get_insider_transactions(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, List[Dict]]:
        """
        Get insider transactions from FMP
        
        Args:
            ticker: Stock symbol
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
            
        Returns:
            Dict with date as key and transactions as value
        """
        try:
            # Create HTTP client for EquityDetailService
            async with httpx.AsyncClient() as client:
                # Use EquityDetailService to get insider trades
                trades = await self.equity_service.get_insider_trades(
                    ticker.upper(),
                    limit=500,  # Get more trades to cover date range
                    client=client
                )
            
            if not trades:
                return {}
            
            formatted_data = {}
            
            # Parse dates
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            for trade in trades:
                # Parse filing date
                filing_date_str = trade.get('filingDate', '')
                if not filing_date_str:
                    continue
                
                try:
                    if ' ' in filing_date_str:
                        filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d %H:%M:%S")
                    else:
                        filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
                except Exception as parse_error:
                    continue
                
                filing_date_only = filing_date.date()
                start_date_only = start_dt.date()
                end_date_only = end_dt.date()
                
                # # Check if in date range
                # if start_dt <= filing_date <= end_dt:
                if start_date_only <= filing_date_only <= end_date_only:
                    print(trade)
                    date_key = filing_date_str.split()[0] 
                    
                    if date_key not in formatted_data:
                        formatted_data[date_key] = []
                    
                    # Determine transaction type
                    trans_type = trade.get('transactionType', 'P')
                    acq_or_disp = trade.get('acquisitionOrDisposition', 'A')
                    
                    # Map to simple transaction codes
                    if acq_or_disp == 'D' or trans_type.upper().startswith('S'):
                        trans_code = 'S'  # Sale
                    elif acq_or_disp == 'A' or trans_type.upper().startswith('P'):
                        trans_code = 'P'  # Purchase
                    else:
                        trans_code = 'M'  # Other
                    
                    # Calculate change (negative for sales)
                    shares = trade.get('securitiesTransacted', 0)
                    change = shares if trans_code == 'P' else -shares
                    
                    transaction_item = {
                        'name': trade.get('reportingName', 'Unknown'),
                        'filingDate': trade.get('filingDate', ''),
                        'transactionDate': trade.get('transactionDate', ''),
                        'transactionCode': trans_code,
                        'transactionType': trade.get('transactionType', ''),
                        'transactionPrice': trade.get('price', 0),
                        'share': abs(shares),  # Always positive for display
                        'change': change,
                        'securityName': trade.get('securityName', ''),
                        'typeOfOwner': trade.get('typeOfOwner', '')
                    }
                    
                    formatted_data[date_key].append(transaction_item)
 
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"Error fetching insider transactions for {ticker}: {str(e)}")
            return {}
    
    
    ## ============================  get_insider_sentiment ============================
    ### get_insider_sentiment -> get_insider_trading_statistics
    async def get_insider_sentiment(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, List[Dict]]:
        """
        Get insider sentiment data from FMP
        
        Args:
            ticker: Stock symbol
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
            
        Returns:
            Dict with date as key and sentiment data as value
        """
        try:
            # Create HTTP client for EquityDetailService
            async with httpx.AsyncClient() as client:
                # Use EquityDetailService to get insider trading statistics
                insider_stats = await self.equity_service.get_insider_trading_statistics(
                    ticker.upper(),
                    client=client
                )
            # print(f"cccccccccccccccccccccccccccc {insider_stats}")
            if not insider_stats:
                return {}
            
            formatted_data = {}
            
            # Parse dates
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # print(f"Processing {len(insider_stats)} insider statistics records")  # Debug
            
            # Process insider statistics
            for stat in insider_stats:
                # Create date from year and month
                year = stat.get('year')
                quarter = stat.get('quarter')
                if not year or not quarter:
                    continue

                # Convert quarter to month (Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct)
                month = (quarter - 1) * 3 + 1
                try:
                    quarter_start = datetime(year, month, 1)
                except ValueError:
                    continue
                
                # Check if in date range
                # if not (start_dt <= stat_date <= end_dt):
                #     continue
                quarter_end = quarter_start + relativedelta(months=3) - relativedelta(days=1)
                
                # Kiểm tra overlap: nếu quarter_end < start_dt hoặc quarter_start > end_dt thì skip
                if quarter_end < start_dt or quarter_start > end_dt:
                    continue

                # Create date key
                date_key = f"{year}-{quarter:02d}"  # Use year-quarter format
                
                if date_key not in formatted_data:
                    formatted_data[date_key] = []

                # Calculate sentiment metrics
                purchases = stat.get('purchases', 0)
                sales = stat.get('sales', 0)
                total_bought = stat.get('totalBought', 0)
                total_sold = stat.get('totalSold', 0)
                
                # Calculate MSPR (Monthly Share Purchase Ratio)
                total_transactions = purchases + sales
                mspr = purchases / total_transactions if total_transactions > 0 else 0
                
                # Calculate net buying pressure
                net_change = total_bought - total_sold

                sentiment_item = {
                    'symbol': stat.get('symbol', ticker),
                    'cik': stat.get('cik'),
                    'year': year,
                    'month': month,
                    'quarter': quarter,
                    'buy_count': purchases,
                    'sell_count': sales,
                    'buySellRatio': stat.get('buyToSellRatio', stat.get('buySellRatio', 0)),
                    'totalBought': total_bought,
                    'totalSold': total_sold,
                    'averageBought': stat.get('averageBought', 0),
                    'averageSold': stat.get('averageSold', 0),
                    'mspr': mspr,  # Monthly Share Purchase Ratio
                    'change': net_change,  # Net buying pressure
                    'pPurchases': stat.get('pPurchases', 0),
                    'sSales': stat.get('sSales', 0),
                }

                formatted_data[date_key].append(sentiment_item)
                # print(f"Added sentiment data for {date_key}")  # Debug

            # print(f"Total periods with sentiment data: {len(formatted_data)}")  # Debug
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"Error fetching insider sentiment for {ticker}: {str(e)}")
            print(f"Exception details: {e}")  # Debug
            return {}
    

    # ============================ FMP: get_financial_statements ============================
    ## get_financial_statements: get_balance_sheet_statement, get_income_statement, get_cash_flow_statement, get_financial_statements (equity_detail_service.py)
    async def get_financial_statements(
        self,
        ticker: str,
        statement_type: str,  # 'balance_sheet', 'income_statement', 'cashflow'
        period: str = 'annual',  # 'annual' or 'quarter'
        limit: int = 5
    ) -> List[Dict]:
        """
        Get financial statements from FMP
        
        Args:
            ticker: Stock symbol
            statement_type: Type of statement
            period: Annual or quarterly
            limit: Number of periods to fetch
            
        Returns:
            List of financial statement data
        """
        try:
            # Create HTTP client for EquityDetailService
            async with httpx.AsyncClient() as client:
                # Map statement types to EquityDetailService methods
                if statement_type == 'balance_sheet':
                    statements = await self.equity_service.get_balance_sheet_statement(
                        ticker.upper(),
                        period=period,
                        limit=limit,
                        client=client
                    )
                elif statement_type == 'income_statement':
                    statements = await self.equity_service.get_income_statement(
                        ticker.upper(),
                        period=period,
                        limit=limit,
                        client=client
                    )
                elif statement_type == 'cashflow':
                    statements = await self.equity_service.get_cash_flow_statement(
                        ticker.upper(),
                        period=period,
                        limit=limit,
                        client=client
                    )
                else:
                    # Fallback to general financial statements
                    statements = await self.equity_service.get_financial_statements(
                        ticker.upper(),
                        period=period,
                        limit=limit,
                        client=client
                    )
                
                return statements if statements else []
                
        except Exception as e:
            self.logger.error(f"Error fetching {statement_type} for {ticker}: {str(e)}")
            return []


    async def get_balance_sheet(
        self,
        ticker: str,
        curr_date: str,
        look_back_days: int
    ) -> Dict[str, List[Dict]]:
        """
        Get balance sheet data formatted for TradingAgents
        """
        try:
            # Determine how many periods to fetch based on look_back_days
            periods_needed = max(1, look_back_days // 365 + 1)  # Annual reports
            
            # Fetch balance sheet data
            statements = await self.get_financial_statements(
                ticker,
                'balance_sheet',
                'annual',
                periods_needed
            )
            
            if not statements:
                return {}
            
            formatted_data = {}
            curr_date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
            
            for stmt in statements:
                # Get statement date
                stmt_date = stmt.get('date', stmt.get('fillingDate', ''))
                if not stmt_date:
                    continue
                
                # Check if within date range
                stmt_date_obj = datetime.strptime(stmt_date[:10], "%Y-%m-%d")
                if stmt_date_obj > curr_date_obj:
                    continue
                    
                days_diff = (curr_date_obj - stmt_date_obj).days
                if days_diff > look_back_days:
                    continue
                
                date_key = stmt_date[:10]
                if date_key not in formatted_data:
                    formatted_data[date_key] = []
                
                # Format balance sheet items
                balance_sheet_item = {
                    'date': date_key,
                    'symbol': ticker.upper(),
                    'reportedCurrency': stmt.get('reportedCurrency', 'USD'),
                    'period': stmt.get('period', 'FY'),
                    'calendarYear': stmt.get('calendarYear', ''),
                    
                    # Assets
                    'totalAssets': stmt.get('totalAssets', 0),
                    'totalCurrentAssets': stmt.get('totalCurrentAssets', 0),
                    'cashAndCashEquivalents': stmt.get('cashAndCashEquivalents', 0),
                    'cashAndShortTermInvestments': stmt.get('cashAndShortTermInvestments', 0),
                    'inventory': stmt.get('inventory', 0),
                    'currentNetReceivables': stmt.get('netReceivables', 0),
                    'totalNonCurrentAssets': stmt.get('totalNonCurrentAssets', 0),
                    'propertyPlantEquipmentNet': stmt.get('propertyPlantEquipmentNet', 0),
                    'intangibleAssets': stmt.get('intangibleAssets', 0),
                    'goodwill': stmt.get('goodwill', 0),
                    
                    # Liabilities
                    'totalLiabilities': stmt.get('totalLiabilities', 0),
                    'totalCurrentLiabilities': stmt.get('totalCurrentLiabilities', 0),
                    'currentAccountsPayable': stmt.get('accountPayables', 0),
                    'currentDebt': stmt.get('shortTermDebt', 0),
                    'totalNonCurrentLiabilities': stmt.get('totalNonCurrentLiabilities', 0),
                    'longTermDebt': stmt.get('longTermDebt', 0),
                    
                    # Equity
                    'totalShareholderEquity': stmt.get('totalStockholdersEquity', 0),
                    'retainedEarnings': stmt.get('retainedEarnings', 0),
                    'commonStock': stmt.get('commonStock', 0),
                    'treasuryStock': stmt.get('treasuryStock', 0),
                }
                
                formatted_data[date_key].append(balance_sheet_item)
            
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"Error fetching balance sheet for {ticker}: {str(e)}")
            return {}


    async def get_income_statement(
        self,
        ticker: str,
        curr_date: str,
        look_back_days: int
    ) -> Dict[str, List[Dict]]:
        """
        Get income statement data formatted for TradingAgents
        """
        try:
            # For financial statements, we need more periods since they're quarterly/annual
            # Calculate years back instead of days for financial data
            if isinstance(look_back_days, str):
                look_back_days = int(look_back_days)

            years_back = max(1, look_back_days // 365 + 1)
            periods_needed = years_back * 4  # Get both annual and quarterly if needed
            
            print(f"Looking for {periods_needed} periods covering {years_back} years back from {curr_date}")
            
            # Fetch both annual and quarterly data for comprehensive coverage
            annual_statements = await self.get_financial_statements(
                ticker,
                'income_statement',
                'annual',
                years_back + 1  # Get extra year to ensure coverage
            )
            
            quarterly_statements = await self.get_financial_statements(
                ticker,
                'income_statement',  
                'quarter',
                years_back * 4 + 2  # Get extra quarters
            )
            
            # Combine all statements
            all_statements = []
            if annual_statements:
                all_statements.extend(annual_statements)
            if quarterly_statements:
                all_statements.extend(quarterly_statements)
                
            print(f"Retrieved {len(annual_statements)} annual and {len(quarterly_statements)} quarterly statements")
            
            if not all_statements:
                return {}
            
            formatted_data = {}
            curr_date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
            
            # Calculate the cutoff date based on look_back_days
            cutoff_date = curr_date_obj - timedelta(days=look_back_days)
            print(f"Date range: {cutoff_date.date()} to {curr_date_obj.date()}")
            
            for stmt in all_statements:
                # Get statement date - try multiple possible date fields
                stmt_date = stmt.get('date') or stmt.get('fillingDate') or stmt.get('filingDate')
                if not stmt_date:
                    print(f"No date found in statement: {list(stmt.keys())[:5]}")
                    continue
                
                # Parse date (handle both date and datetime formats)
                try:
                    if len(stmt_date) > 10:
                        stmt_date_obj = datetime.strptime(stmt_date[:10], "%Y-%m-%d")
                    else:
                        stmt_date_obj = datetime.strptime(stmt_date, "%Y-%m-%d")
                except ValueError as e:
                    print(f"Failed to parse date '{stmt_date}': {e}")
                    continue
                
                print(f"Processing statement date: {stmt_date_obj.date()}")
                
                # Check if within date range - be more flexible for financial data
                # Financial statements are typically released after the period end
                if stmt_date_obj > curr_date_obj:
                    print(f"  Skipping - future date")
                    continue
                    
                if stmt_date_obj < cutoff_date:
                    print(f"  Skipping - too old ({stmt_date_obj.date()} < {cutoff_date.date()})")
                    continue
                
                print(f"  ✓ Including statement from {stmt_date_obj.date()}")
                
                date_key = stmt_date_obj.strftime("%Y-%m-%d")
                if date_key not in formatted_data:
                    formatted_data[date_key] = []
                
                # Calculate ratios if missing
                revenue = stmt.get('revenue', 0)
                gross_profit = stmt.get('grossProfit', 0)
                operating_income = stmt.get('operatingIncome', 0)
                net_income = stmt.get('netIncome', 0)
                
                gross_profit_ratio = stmt.get('grossProfitRatio', 0)
                if not gross_profit_ratio and revenue > 0:
                    gross_profit_ratio = gross_profit / revenue
                    
                operating_income_ratio = stmt.get('operatingIncomeRatio', 0)  
                if not operating_income_ratio and revenue > 0:
                    operating_income_ratio = operating_income / revenue
                    
                net_income_ratio = stmt.get('netIncomeRatio', 0)
                if not net_income_ratio and revenue > 0:
                    net_income_ratio = net_income / revenue
                
                # Format income statement items
                income_stmt_item = {
                    'date': date_key,
                    'symbol': ticker.upper(),
                    'reportedCurrency': stmt.get('reportedCurrency', 'USD'),
                    'period': stmt.get('period', 'FY'),
                    'fiscalYear': stmt.get('fiscalYear', stmt.get('calendarYear', '')),
                    'calendarYear': stmt.get('calendarYear', ''),
                    
                    # Revenue & Costs
                    'revenue': revenue,
                    'costOfRevenue': stmt.get('costOfRevenue', 0),
                    'grossProfit': gross_profit,
                    'grossProfitRatio': gross_profit_ratio,
                    
                    # Operating Expenses
                    'researchAndDevelopmentExpenses': stmt.get('researchAndDevelopmentExpenses', 0),
                    'sellingGeneralAndAdministrativeExpenses': stmt.get('sellingGeneralAndAdministrativeExpenses', 0),
                    'operatingExpenses': stmt.get('operatingExpenses', 0),
                    'operatingIncome': operating_income,
                    'operatingIncomeRatio': operating_income_ratio,
                    
                    # Other Income/Expenses
                    'interestExpense': stmt.get('interestExpense', 0),
                    'interestIncome': stmt.get('interestIncome', 0),
                    'totalOtherIncomeExpensesNet': stmt.get('totalOtherIncomeExpensesNet', 0),
                    
                    # Net Income
                    'incomeBeforeTax': stmt.get('incomeBeforeTax', 0),
                    'incomeTaxExpense': stmt.get('incomeTaxExpense', 0),
                    'netIncome': net_income,
                    'netIncomeRatio': net_income_ratio,
                    
                    # Per Share Data
                    'eps': stmt.get('eps', 0),
                    'epsdiluted': stmt.get('epsdiluted', stmt.get('epsDiluted', 0)),
                    'weightedAverageShsOut': stmt.get('weightedAverageShsOut', 0),
                    'weightedAverageShsOutDil': stmt.get('weightedAverageShsOutDil', 0),
                }
                
                formatted_data[date_key].append(income_stmt_item)
            
            print(f"Final result: {len(formatted_data)} dates with income statements")
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"Error fetching income statement for {ticker}: {str(e)}")
            print(f"Exception: {e}")
            return {}


    async def get_cashflow_statement(
        self,
        ticker: str,
        curr_date: str,
        look_back_days: int
    ) -> Dict[str, List[Dict]]:
        """
        Get cashflow statement data formatted for TradingAgents
        """
        try:
            # Determine periods needed
            periods_needed = max(1, look_back_days // 365 + 1)
            
            # Fetch cashflow data
            statements = await self.get_financial_statements(
                ticker,
                'cashflow',
                'annual',
                periods_needed
            )
            
            if not statements:
                return {}
            
            formatted_data = {}
            curr_date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
            
            for stmt in statements:
                # Get statement date
                stmt_date = stmt.get('date', stmt.get('fillingDate', ''))
                if not stmt_date:
                    continue
                
                # Check if within date range
                stmt_date_obj = datetime.strptime(stmt_date[:10], "%Y-%m-%d")
                if stmt_date_obj > curr_date_obj:
                    continue
                    
                days_diff = (curr_date_obj - stmt_date_obj).days
                if days_diff > look_back_days:
                    continue
                
                date_key = stmt_date[:10]
                if date_key not in formatted_data:
                    formatted_data[date_key] = []
                
                # Format cashflow items
                cashflow_item = {
                    'date': date_key,
                    'symbol': ticker.upper(),
                    'reportedCurrency': stmt.get('reportedCurrency', 'USD'),
                    'period': stmt.get('period', 'FY'),
                    'calendarYear': stmt.get('calendarYear', ''),
                    
                    # Operating Activities
                    'netIncome': stmt.get('netIncome', 0),
                    'depreciationAndAmortization': stmt.get('depreciationAndAmortization', 0),
                    'stockBasedCompensation': stmt.get('stockBasedCompensation', 0),
                    'changeInWorkingCapital': stmt.get('changeInWorkingCapital', 0),
                    'accountsReceivables': stmt.get('accountsReceivables', 0),
                    'inventory': stmt.get('inventory', 0),
                    'accountsPayables': stmt.get('accountsPayables', 0),
                    'operatingCashFlow': stmt.get('operatingCashFlow', 0),
                    
                    # Investing Activities
                    'investmentsInPropertyPlantAndEquipment': stmt.get('investmentsInPropertyPlantAndEquipment', 0),
                    'acquisitionsNet': stmt.get('acquisitionsNet', 0),
                    'purchasesOfInvestments': stmt.get('purchasesOfInvestments', 0),
                    'salesMaturitiesOfInvestments': stmt.get('salesMaturitiesOfInvestments', 0),
                    'netCashUsedForInvestingActivites': stmt.get('netCashUsedForInvestingActivites', 0),
                    
                    # Financing Activities
                    'debtRepayment': stmt.get('debtRepayment', 0),
                    'commonStockIssued': stmt.get('commonStockIssued', 0),
                    'commonStockRepurchased': stmt.get('commonStockRepurchased', 0),
                    'dividendsPaid': stmt.get('dividendsPaid', 0),
                    'netCashUsedProvidedByFinancingActivities': stmt.get('netCashUsedProvidedByFinancingActivities', 0),
                    
                    # Summary
                    'netChangeInCash': stmt.get('netChangeInCash', 0),
                    'cashAtEndOfPeriod': stmt.get('cashAtEndOfPeriod', 0),
                    'cashAtBeginningOfPeriod': stmt.get('cashAtBeginningOfPeriod', 0),
                    'freeCashFlow': stmt.get('freeCashFlow', 0),
                }
                
                formatted_data[date_key].append(cashflow_item)
            
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"Error fetching cashflow for {ticker}: {str(e)}")
            return {}
        
    
    async def get_crypto_quote(
        self, 
        symbol: str,
        client: Optional[httpx.AsyncClient] = None
    ) -> Dict[str, Any]:
        """Get crypto quote data from FMP"""
        
        # Add USD suffix for crypto if not already present
        if not symbol.upper().endswith('USD'):
            crypto_symbol = f"{symbol.upper()}"
        else:
            crypto_symbol = symbol.upper()
        
        # FIXED: Include the symbol in the URL
        url = f"{self.base_url}/v3/quote/{crypto_symbol}"
        params = {"apikey": self.api_key}
        
        try:
            if client:
                response = await client.get(url, params=params)
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, params=params)
            
            # Debug logging
            print(f"Full URL: {url}?apikey=***")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # FMP returns a list for quote endpoint
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                elif isinstance(data, dict):
                    return data
                else:
                    print(f"Unexpected response format: {type(data)}")
                    
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Exception in get_crypto_quote: {e}")
            
        return {}
    

    async def get_historical_prices(self, symbol: str, start_date: str, lookback_days: int = 30) -> Dict[str, float]:
        """
        Get historical prices for a symbol
        
        Args:
            symbol: Stock/Crypto symbol
            start_date: End date in YYYY-MM-DD format (most recent date)
            lookback_days: Number of days to look back from start_date
            
        Returns:
            Dict mapping dates to closing prices
        """
        
        symbol_upper = symbol.upper()
        
        # Calculate end_date (earlier date) from start_date and lookback_days
        if start_date is None:
            start_date_obj = datetime.now()
            start_date = start_date_obj.strftime("%Y-%m-%d")
        else:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        
        end_date_obj = start_date_obj - timedelta(days=lookback_days)
        end_date = end_date_obj.strftime("%Y-%m-%d")
        
        # Cache key includes lookback_days for clarity
        cache_key = f"historical_prices_{symbol_upper}_{start_date}_{lookback_days}d"
        
        redis_client = await get_redis_client_llm()

        # Try to get from cache
        if redis_client:
            try:
                cached_json = await redis_client.get(cache_key)
                if cached_json:
                    return json.loads(cached_json)
            except Exception as e:
                self.logger.error(f"Redis GET error for {cache_key}: {e}", exc_info=True)

        # FMP API expects from (earlier) to (later) date
        url = f"{self.base_url}/v3/historical-price-full/{symbol_upper}?from={end_date}&to={start_date}&apikey={self.api_key}"
        
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data and "historical" in data and isinstance(data["historical"], list):
                price_map = {item['date']: item['close'] for item in data['historical']}
                
                # Cache the result
                if redis_client and price_map:
                    cache_ttl = getattr(settings, 'CACHE_TTL_HISTORY', 86400 * 7)
                    await redis_client.set(
                        name=cache_key, 
                        value=json.dumps(price_map), 
                        ex=int(cache_ttl)
                    )
                return price_map
                
        except Exception as e:
            self.logger.error(f"Failed to get historical prices for {symbol_upper} from FMP: {e}", exc_info=True)
        
        return {}

# Singleton instance
fmp_data = FMPData()

if __name__ == "__main__":
    async def main():
        fmp = FMPData()
        
        # ticker = "AAPL"  
        # start_date = "2025-01-01"  
        # end_date = "2025-07-25"
        
        # print(f"Testing FMPData for ticker: {ticker}")
        # print(f"Date range: {start_date} to {end_date}")
        # print("="*60)
        
        # # 1. Test get_company_news -> DONE
        # print("\n1. Testing get_company_news...")
        # try:
        #     news_data = await fmp.get_company_news(
        #         ticker=ticker,
        #         start_date=start_date,
        #         end_date=end_date,
        #         max_items=10
        #     )
            
        #     if news_data:
        #         print(f"✓ Found news for {len(news_data)} dates")
        #         for date, news_list in list(news_data.items())[:2]:
        #             print(f"\n  Date: {date}")
        #             for news in news_list[:2]: # 2 news
        #                 print(f"  - {news['headline']}")
        #                 print(f"    Source: {news['source']}, URL: {news['url']}")
        #     else:
        #         print("✗ No news found")
        # except Exception as e:
        #     print(f"✗ Error: {str(e)}")
        
        # 2. Test get_historical_data -> DONE
        # print("\n\n2. Testing get_historical_data...")
        # try:
        #     hist_data = await fmp.get_historical_data(
        #         symbol=ticker,
        #         start_date=start_date,
        #         end_date=end_date
        #     )
            
        #     if not hist_data.empty:
        #         print(f"✓ Retrieved {len(hist_data)} records")
        #         print("\nFirst 5 rows:")
        #         print(hist_data.head())
        #         print("\nLast 5 rows:")
        #         print(hist_data.tail())
        #         print(f"\nColumns: {list(hist_data.columns)}")
        #     else:
        #         print("✗ No historical data found")
        # except Exception as e:
        #     print(f"✗ Error: {str(e)}")
        
        # 3. Test get_historical_data_with_cache -> DONE
        # print("\n\n3. Testing get_historical_data_with_cache...")
        # try:
        #     # The first -> Fetch from API
        #     import time
        #     start_time = time.time()
        #     cached_data = await fmp.get_historical_data_with_cache(
        #         symbol=ticker,
        #         start_date=start_date,
        #         end_date=end_date,
        #         force_refresh=False # Force true refresh to test
        #     )
        #     first_call_time = time.time() - start_time
        #     print(f"✓ First call (from API): {first_call_time:.2f} seconds")
            
        #     # The second - Fetch from cache
        #     start_time = time.time()
        #     cached_data2 = await fmp.get_historical_data_with_cache(
        #         symbol=ticker,
        #         start_date=start_date,
        #         end_date=end_date,
        #         force_refresh=False
        #     )
        #     second_call_time = time.time() - start_time
        #     print(f"✓ Second call (from cache): {second_call_time:.2f} seconds")
        #     print(f"  Cache speedup: {first_call_time/second_call_time:.1f}x faster")
        # except Exception as e:
        #     print(f"✗ Error: {str(e)}")
        
        # 4. Test get_insider_transactions
        # print("\n\n4. Testing get_insider_transactions...")
        # try:
        #     insider_trans = await fmp.get_insider_transactions(
        #         ticker=ticker,
        #         start_date=start_date,
        #         end_date=end_date
        #     )
            
        #     if insider_trans:
        #         print(f"✓ Found insider transactions for {len(insider_trans)} dates")
        #         total_trans = sum(len(trans_list) for trans_list in insider_trans.values())
        #         print(f"  Total transactions: {total_trans}")
                
        #         # Show samples and statistics
        #         buy_count = 0
        #         sell_count = 0
                
        #         for date, trans_list in list(insider_trans.items())[:3]:
        #             print(f"\n  Date: {date}")
        #             for trans in trans_list[:2]:
        #                 print(f"  - {trans['name'][:30]}: {trans['transactionCode']} "
        #                       f"{trans['share']:,.0f} shares @ ${trans['transactionPrice']:.2f}")
                        
        #                 if trans['transactionCode'] == 'P':
        #                     buy_count += 1
        #                 elif trans['transactionCode'] == 'S':
        #                     sell_count += 1
                
        #         print(f"\n  Summary: {buy_count} buys, {sell_count} sells")
        #     else:
        #         print("✗ No insider transactions found")
        # except Exception as e:
        #     print(f"✗ Error: {str(e)}")
        
        
        # # 5. Test get_insider_sentiment
        # print("\n\n5. Testing get_insider_sentiment...")
        # try:
        #     sentiment_data = await fmp.get_insider_sentiment(
        #         ticker=ticker,
        #         start_date=start_date,
        #         end_date=end_date
        #     )
            
        #     if sentiment_data:
        #         print(f"✓ Found insider sentiment for {len(sentiment_data)} months")
        #         # Hiển thị sentiment
        #         for date, sentiment_list in sentiment_data.items():
        #             sentiment = sentiment_list[0]
        #             print(f"\n  Month: {sentiment['year']}-{sentiment['month']:02d}")
        #             print(f"  - Buy Count: {sentiment['buy_count']}, Sell Count: {sentiment['sell_count']}")
        #             print(f"  - MSPR (Monthly Share Purchase Ratio): {sentiment['mspr']:.2%}")
        #             print(f"  - Net Buying Pressure: {sentiment['change']}")
        #     else:
        #         print("✗ No insider sentiment data found")
        # except Exception as e:
        #     print(f"✗ Error: {str(e)}")

        # 7. Test get_income_statement
        
        # curr_date = "2025-07-25"
        ticker="BTCUSD"
        data = await fmp.get_crypto_quote(
                symbol=ticker
            )
        print(data)

        # look_back_days = 365

        # print("\n\n7. Testing get_income_statement...")
        # try:
        #     income_stmt = await fmp.get_income_statement(
        #         ticker=ticker,
        #         curr_date=curr_date,
        #         look_back_days=look_back_days
        #     )
            
        #     if income_stmt:
        #         print(f"✓ Found income statement data for {len(income_stmt)} dates")
                
        #         # Hiển thị sample
        #         for date, stmts in list(income_stmt.items()):
        #             stmt = stmts[0]
        #             print(f"\n  Date: {date} ({stmt['period']})")
        #             print(f"  - Revenue: ${stmt['revenue']:,.0f}")
        #             print(f"  - Gross Profit: ${stmt['grossProfit']:,.0f} "
        #                   f"({stmt['grossProfitRatio']*100:.1f}%)")
        #             print(f"  - Operating Income: ${stmt['operatingIncome']:,.0f} "
        #                   f"({stmt['operatingIncomeRatio']*100:.1f}%)")
        #             print(f"  - Net Income: ${stmt['netIncome']:,.0f} "
        #                   f"({stmt['netIncomeRatio']*100:.1f}%)")
        #             print(f"  - EPS: ${stmt['eps']:.2f}")
        #             print(f"  - Diluted EPS: ${stmt['epsdiluted']:.2f}")
        #     else:
        #         print("✗ No income statement data found")
        # except Exception as e:
        #     print(f"✗ Error: {str(e)}")
        
        # # 8. Test get_cashflow_statement
        # print("\n\n8. Testing get_cashflow_statement...")
        # try:
        #     cashflow = await fmp.get_cashflow_statement(
        #         ticker=ticker,
        #         curr_date=curr_date,
        #         look_back_days=look_back_days
        #     )
            
        #     if cashflow:
        #         print(f"✓ Found cashflow data for {len(cashflow)} dates")
                
        #         # Hiển thị sample
        #         for date, flows in list(cashflow.items())[:2]:
        #             flow = flows[0]
        #             print(f"\n  Date: {date} ({flow['period']})")
        #             print(f"  - Operating Cash Flow: ${flow['operatingCashFlow']:,.0f}")
        #             print(f"  - Investing Cash Flow: ${flow['netCashUsedForInvestingActivites']:,.0f}")
        #             print(f"  - Financing Cash Flow: ${flow['netCashUsedProvidedByFinancingActivities']:,.0f}")
        #             print(f"  - Free Cash Flow: ${flow['freeCashFlow']:,.0f}")
        #             print(f"  - Net Change in Cash: ${flow['netChangeInCash']:,.0f}")
        #             print(f"  - Dividends Paid: ${abs(flow['dividendsPaid']):,.0f}")
        #             print(f"  - Stock Repurchased: ${abs(flow['commonStockRepurchased']):,.0f}")
        #     else:
        #         print("✗ No cashflow data found")
        # except Exception as e:
        #     print(f"✗ Error: {str(e)}")
        
        # # 9. Test get_financial_statements (general)
        # print("\n\n9. Testing get_financial_statements (quarterly data)...")
        # try:
        #     quarterly_stmts = await fmp.get_financial_statements(
        #         ticker=ticker,
        #         statement_type='income_statement',
        #         period='quarter',
        #         limit=4
        #     )
            
        #     if quarterly_stmts:
        #         print(f"✓ Found {len(quarterly_stmts)} quarterly statements")
                
        #         # Hiển thị quarterly trend
        #         print("\n  Quarterly Revenue Trend:")
        #         for stmt in quarterly_stmts[:4]:
        #             date = stmt.get('date', stmt.get('fillingDate', 'N/A'))
        #             revenue = stmt.get('revenue', 0)
        #             print(f"  - {date}: ${revenue:,.0f}")
        #     else:
        #         print("✗ No quarterly data found")
        # except Exception as e:
        #     print(f"✗ Error: {str(e)}")
                
        print("\n" + "="*60)
        print("Testing completed!")
    
    # Run async main
    asyncio.run(main())