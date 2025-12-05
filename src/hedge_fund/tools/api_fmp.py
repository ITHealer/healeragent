import datetime
import os
import pandas as pd
import requests
import time
from typing import Optional, List, Dict, Any
from datetime import datetime as dt, timedelta
from src.hedge_fund.data.cache import get_cache
from src.hedge_fund.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFacts,
    CompanyFactsResponse,
)

# Global cache instance
_cache = get_cache()

# FMP Base URL - Using stable endpoints
FMP_BASE_URL = "https://financialmodelingprep.com/stable"
from src.hedge_fund.utils.api_key import get_api_key_from_state

def _make_fmp_request(
    endpoint: str, 
    params: dict = None, 
    api_key: str = None, 
    max_retries: int = 3,
    api_version: str = "stable"  # Add version parameter
) -> requests.Response:
    """
    Make request to FMP API with version support.
    """
    # Determine base URL based on version
    if api_version == "v4":
        url = f"https://financialmodelingprep.com/api/v4/{endpoint}"
    else:
        url = f"{FMP_BASE_URL}/{endpoint}"  # Default to stable
    
    if params is None:
        params = {}
    
    fmp_api_key = api_key or os.environ.get("FMP_API_KEY")
    if not fmp_api_key:
        raise Exception("FMP API key not found in environment variables")
    
    params['apikey'] = fmp_api_key
    
    for attempt in range(max_retries + 1):
        response = requests.get(url, params=params)
        
        if response.status_code == 429 and attempt < max_retries:
            # Rate limited - use linear backoff
            delay = 60 + (30 * attempt)  # 60s, 90s, 120s
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s...")
            time.sleep(delay)
            continue
            
        return response

def _map_fmp_price_to_model(fmp_data: dict, ticker: str) -> dict:
    return {
        "ticker": ticker,
        "time": fmp_data.get("date", ""),
        "open": fmp_data.get("open"),
        "high": fmp_data.get("high"),
        "low": fmp_data.get("low"),
        "close": fmp_data.get("close"),
        "volume": fmp_data.get("volume"),
        "vwap": fmp_data.get("vwap"),
        "adj_close": fmp_data.get("adjClose"),
        "change": fmp_data.get("change"),
        "change_percent": fmp_data.get("changePercent"),
    }

def _fetch_growth_metrics(ticker: str, api_key: str = None) -> Dict[str, Dict]:
    """
    Fetch all growth metrics from FMP stable endpoints.
    Returns a dictionary with growth data from different statements.
    """
    growth_data = {
        "income": None,
        "balance": None,
        "cashflow": None
    }
    
    # Fetch Income Statement Growth
    income_response = _make_fmp_request("income-statement-growth", {"symbol": ticker}, api_key)
    if income_response.status_code == 200:
        data = income_response.json()
        if isinstance(data, list) and data:
            growth_data["income"] = data[0]  # Get most recent period
    else:
        print(f"Warning: Could not fetch income statement growth for {ticker}")
    
    # Fetch Balance Sheet Growth
    balance_response = _make_fmp_request("balance-sheet-statement-growth", {"symbol": ticker}, api_key)
    if balance_response.status_code == 200:
        data = balance_response.json()
        if isinstance(data, list) and data:
            growth_data["balance"] = data[0]  # Get most recent period
    else:
        print(f"Warning: Could not fetch balance sheet growth for {ticker}")
    
    # Fetch Cash Flow Statement Growth
    cashflow_response = _make_fmp_request("cash-flow-statement-growth", {"symbol": ticker}, api_key)
    if cashflow_response.status_code == 200:
        data = cashflow_response.json()
        if isinstance(data, list) and data:
            growth_data["cashflow"] = data[0]  # Get most recent period
    else:
        print(f"Warning: Could not fetch cash flow growth for {ticker}")
    
    return growth_data

def _map_fmp_metrics_to_model(
    fmp_key_metrics: dict, 
    fmp_ratios: dict, 
    growth_data: dict,
    ticker: str
) -> dict:
    """Map FMP key metrics TTM, ratios TTM, and growth data to FinancialMetrics model format."""
    
    def safe_float_optional(value) -> float | None:
        """Convert value to float, return None if conversion fails."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def get_value(key: str, source_dict: dict) -> float | None:
        """Get value from dictionary with safe float conversion."""
        return safe_float_optional(source_dict.get(key)) if source_dict else None
    
    def convert_growth_to_percentage(value: float | None) -> float | None:
        """Convert growth value to percentage format (0.1 = 10%)"""
        if value is None:
            return None
        return value  # FMP already returns as decimal (0.1 = 10%)
    
    # Get the date from key metrics (usually has more complete data)
    # report_date = fmp_key_metrics.get("date", "") if fmp_key_metrics else fmp_ratios.get("date", "")
    
    # Extract growth data from the fetched dictionaries
    income_growth = growth_data.get("income", {})
    balance_growth = growth_data.get("balance", {})
    cashflow_growth = growth_data.get("cashflow", {})

    report_date = growth_data['income'].get('date', '')
    
    return {
        "ticker": ticker,
        "report_period": report_date,
        "period": "ttm",
        "currency": "USD",
        
        # Valuation metrics
        "market_cap": get_value("marketCap", fmp_key_metrics),
        "enterprise_value": get_value("enterpriseValueTTM", fmp_key_metrics),
        "price_to_earnings_ratio": get_value("priceToEarningsRatioTTM", fmp_ratios),
        "price_to_book_ratio": get_value("priceToBookRatioTTM", fmp_ratios),
        "price_to_sales_ratio": get_value("priceToSalesRatioTTM", fmp_ratios),
        "enterprise_value_to_ebitda_ratio": get_value("enterpriseValueMultipleTTM", fmp_ratios),
        "enterprise_value_to_revenue_ratio": get_value("evToSalesTTM", fmp_key_metrics),
        "free_cash_flow_yield": get_value("freeCashFlowYieldTTM", fmp_key_metrics),
        "peg_ratio": get_value("priceToEarningsGrowthRatioTTM", fmp_ratios),
        
        # Profitability metrics
        "gross_margin": get_value("grossProfitMarginTTM", fmp_ratios),
        "operating_margin": get_value("operatingProfitMarginTTM", fmp_ratios),
        "net_margin": get_value("netProfitMarginTTM", fmp_ratios),
        "return_on_equity": get_value("returnOnEquityTTM", fmp_key_metrics),
        "return_on_assets": get_value("returnOnAssetsTTM", fmp_key_metrics),
        "return_on_invested_capital": get_value("returnOnInvestedCapitalTTM", fmp_key_metrics),
        
        # Efficiency metrics
        "asset_turnover": get_value("assetTurnoverTTM", fmp_ratios),
        "inventory_turnover": get_value("inventoryTurnoverTTM", fmp_ratios),
        "receivables_turnover": get_value("receivablesTurnoverTTM", fmp_ratios),
        "days_sales_outstanding": get_value("daysOfSalesOutstandingTTM", fmp_key_metrics),
        "operating_cycle": get_value("operatingCycleTTM", fmp_key_metrics),
        "working_capital_turnover": get_value("workingCapitalTurnoverRatioTTM", fmp_ratios),
        
        # Liquidity metrics
        "current_ratio": get_value("currentRatioTTM", fmp_ratios),
        "quick_ratio": get_value("quickRatioTTM", fmp_ratios),
        "cash_ratio": get_value("cashRatioTTM", fmp_ratios),
        "operating_cash_flow_ratio": get_value("operatingCashFlowRatioTTM", fmp_ratios),
        
        # Leverage metrics
        "debt_to_equity": get_value("debtToEquityRatioTTM", fmp_ratios),
        "debt_to_assets": get_value("debtToAssetsRatioTTM", fmp_ratios),
        "interest_coverage": get_value("interestCoverageRatioTTM", fmp_ratios),
        
        # Growth metrics - NOW WITH ACTUAL DATA
        "revenue_growth": convert_growth_to_percentage(get_value("growthRevenue", income_growth)),
        "earnings_growth": convert_growth_to_percentage(get_value("growthNetIncome", income_growth)),
        "book_value_growth": convert_growth_to_percentage(get_value("growthTotalStockholdersEquity", balance_growth)),
        "earnings_per_share_growth": convert_growth_to_percentage(get_value("growthEPS", income_growth)),
        "free_cash_flow_growth": convert_growth_to_percentage(get_value("growthFreeCashFlow", cashflow_growth)),
        "operating_income_growth": convert_growth_to_percentage(get_value("growthOperatingIncome", income_growth)),
        "ebitda_growth": convert_growth_to_percentage(get_value("growthEBITDA", income_growth)),
        
        # Other metrics
        "payout_ratio": get_value("dividendPayoutRatioTTM", fmp_ratios),
        
        # Per share metrics
        "earnings_per_share": get_value("netIncomePerShareTTM", fmp_ratios),
        "book_value_per_share": get_value("bookValuePerShareTTM", fmp_ratios),
        "free_cash_flow_per_share": get_value("freeCashFlowPerShareTTM", fmp_ratios),
    }

def _map_fmp_line_item(fmp_data: dict, line_item_name: str, ticker: str, period: str) -> dict:
    """Map FMP financial statement data to LineItem model format."""
    
    # Map our standard line item names to FMP field names
    field_mapping = {
        "revenue": "revenue",
        "net_income": "netIncome", 
        "ebit": "ebit",
        "free_cash_flow": "freeCashFlow",
        "gross_profit": "grossProfit",
        "operating_income": "operatingIncome",
        "total_assets": "totalAssets",
        "total_liabilities": "totalLiabilities",
        "total_equity": "totalStockholdersEquity",
        "operating_cash_flow": "operatingCashFlow"
    }
    
    fmp_field = field_mapping.get(line_item_name, line_item_name)
    value = fmp_data.get(fmp_field)
    
    return {
        "ticker": ticker,
        "report_period": fmp_data.get("date", ""),
        "period": period,
        "line_item": line_item_name,
        "value": float(value) if value is not None else None,
        "currency": fmp_data.get("reportedCurrency", "USD")
    }


def _map_fmp_insider_trade(fmp_trade: dict, ticker: str) -> dict:
    """Map FMP insider trade data to InsiderTrade model format."""
    
    # Determine transaction type based on transaction shares (negative = sell)
    transaction_shares = fmp_trade.get("transactionShares", 0)
    transaction_type = "Sell" if transaction_shares < 0 else "Buy"
    
    # Calculate transaction value
    price = fmp_trade.get("transactionPrice", 0)
    shares = abs(transaction_shares)
    value = price * shares if price and shares else None
    
    return {
        "ticker": ticker,
        "filing_date": fmp_trade.get("filingDate", ""),
        "transaction_date": fmp_trade.get("transactionDate", ""),
        "name": fmp_trade.get("insiderName", ""),
        "issuer": fmp_trade.get("issuer", ""),
        "security_title": fmp_trade.get("securityTitle", ""),
        "transaction_price_per_share": price,
        "transaction_shares": shares,
        "shares_owned_before_transaction": fmp_trade.get("sharesOwnedPriorToTransaction"),
        "shares_owned_after_transaction": fmp_trade.get("postTransactionAmounts"),
        "is_board_director": fmp_trade.get("isDirector", False),
        "is_officer": fmp_trade.get("isOfficer", False),
        "transaction_type": transaction_type,
        "value": value
    }

def _map_fmp_news(fmp_news: dict, ticker: str) -> dict:
    """Map FMP news data to CompanyNews model format."""
    
    return {
        "ticker": ticker,
        "date": fmp_news.get("publishedDate", ""),
        "source": fmp_news.get("site", ""),
        "title": fmp_news.get("title", ""),
        "url": fmp_news.get("url", ""),
        "sentiment": fmp_news.get("sentiment", ""),  # FMP may provide sentiment in some endpoints
        "content": fmp_news.get("text", ""),  # Some endpoints provide text/content
        "author": fmp_news.get("author", "")  # If available
    }


# ============================================================================
# PUBLIC FUNCTIONS - Compatible with Financial Dataset API interface
# ============================================================================

def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> List[Price]:
    """Fetch price data from cache or FMP API."""
    cache_key = f"{ticker}_{start_date}_{end_date}"
    
    # Check cache first
    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]
    
    # Fetch from FMP API using stable endpoint
    params = {
        "symbol": ticker,
        "from": start_date,
        "to": end_date
    }
    response = _make_fmp_request(f"historical-price-eod/full", params, api_key)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching FMP data: {ticker} - {response.status_code} - {response.text}")
    
    data = response.json()
    if not data:
        return []
    
    # Map FMP data to our model format
    prices = []
    for item in data:
        mapped_data = _map_fmp_price_to_model(item, ticker)
        prices.append(Price(**mapped_data))
    
    # Sort by date (oldest first)
    prices.sort(key=lambda x: x.time)
    
    # Cache the results
    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    
    return prices

def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> List[FinancialMetrics]:
    """Fetch financial metrics from cache or FMP API, including growth metrics."""
    cache_key = f"{ticker}_{period}_{end_date}_{limit}_2"  # Updated version for cache
    
    # Check cache first
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]
    
    # Fetch key metrics and ratios from FMP stable endpoints
    key_metrics_params = {"symbol": ticker}
    ratios_params = {"symbol": ticker}
    
    key_metrics_response = _make_fmp_request("key-metrics-ttm", key_metrics_params, api_key)
    ratios_response = _make_fmp_request("ratios-ttm", ratios_params, api_key)
    
    print(f"key_metrics_response.status_code: {key_metrics_response.status_code}")
    print(f"ratios_response.status_code: {ratios_response.status_code}")
    
    if key_metrics_response.status_code != 200:
        print(f"Warning: Could not fetch key metrics for {ticker}")
        key_metrics_data = None
    else:
        key_metrics_data = key_metrics_response.json()
        if isinstance(key_metrics_data, list) and key_metrics_data:
            key_metrics_data = key_metrics_data[0]
    
    if ratios_response.status_code != 200:
        print(f"Warning: Could not fetch ratios for {ticker}")
        ratios_data = None
    else:
        ratios_data = ratios_response.json()
        if isinstance(ratios_data, list) and ratios_data:
            ratios_data = ratios_data[0]
    
    if not key_metrics_data and not ratios_data:
        return []
    
    # Fetch growth metrics
    print(f"Fetching growth metrics for {ticker}...")
    growth_data = _fetch_growth_metrics(ticker, api_key)
    
    # Log growth data for debugging
    if growth_data["income"]:
        print(f"Income growth data retrieved: Revenue Growth = {growth_data['income'].get('growthRevenue')}")
    if growth_data["balance"]:
        print(f"Balance sheet growth data retrieved: Book Value Growth = {growth_data['balance'].get('growthTotalStockholdersEquity')}")
    if growth_data["cashflow"]:
        print(f"Cash flow growth data retrieved: FCF Growth = {growth_data['cashflow'].get('growthFreeCashFlow')}")
    
    # Combine and map all data
    mapped_data = _map_fmp_metrics_to_model(
        key_metrics_data or {},
        ratios_data or {},
        growth_data,
        ticker
    )
    
    metrics = [FinancialMetrics(**mapped_data)]
    
    print(f"================metrics: {metrics}")
    
    # Cache the results
    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics])
    
    return metrics

def search_line_items(
    ticker: str,
    line_items: List[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> List[LineItem]:  # IMPORTANT: Return List[LineItem] NOT LineItemResponse
    """
    Fetch line items from FMP API and return List[LineItem] for compatibility.
    
    IMPORTANT: This function returns List[LineItem] directly, NOT LineItemResponse,
    to maintain compatibility with ai-hedge-fund agents that expect a list.
    """    
    cache_key = f"{ticker}_{'_'.join(sorted(line_items))}_{period}_{end_date}_{limit}_v2"
    
    # Check cache first
    if cached_data := _cache.get_line_items(cache_key):
        # Reconstruct LineItem objects from cached data
        line_items_list = []
        for item_data in cached_data:
            line_item = LineItem(
                ticker=item_data.get("ticker", ticker),
                report_period=item_data.get("report_period", ""),
                period=item_data.get("period", period),
                currency=item_data.get("currency", "USD")
            )
            # Set all other fields as attributes
            for key, value in item_data.items():
                if key not in ["ticker", "report_period", "period", "currency"]:
                    setattr(line_item, key, value)
            line_items_list.append(line_item)
        return line_items_list  # Return list directly
    
    results = []
    
    # Extended field categorization for all agents
    income_items = {
        "revenue", "net_income", "ebit", "gross_profit", "operating_income",
        "depreciation_and_amortization", "ebitda", "eps", "interest_expense",
        "earnings_per_share", "gross_margin", "operating_margin", 
        "research_and_development", "operating_expense"
    }
    
    balance_items = {
        "total_assets", "total_liabilities", "total_equity", 
        "outstanding_shares", "cash_and_cash_equivalents",
        "total_debt", "shareholders_equity", "cash_and_equivalents",
        "current_assets", "current_liabilities", "working_capital",
        "long_term_debt", "short_term_debt"
    }
    
    cashflow_items = {
        "free_cash_flow", "operating_cash_flow", "capital_expenditure",
        "dividends_and_other_cash_distributions", 
        "issuance_or_purchase_of_equity_shares"
    }
    
    # Ratio items
    ratio_items = {
        "debt_to_equity", "current_ratio", "quick_ratio",
        "return_on_equity", "return_on_assets", "profit_margin"
    }
    
    items_to_fetch = set(line_items)
    data_fetched = {}
    
    # Determine endpoint based on period
    use_ttm = (period.lower() == "ttm")
    
    # Fetch Income Statement
    if items_to_fetch & income_items:
        if use_ttm:
            endpoint = "income-statement-ttm"
            params = {"symbol": ticker}
        else:
            endpoint = "income-statement"
            params = {
                "symbol": ticker,
                "period": "quarter" if period.lower() in ["quarterly", "quarter", "q"] else "annual",
                "limit": limit
            }
        
        response = _make_fmp_request(endpoint, params, api_key)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                data_fetched["income"] = data if not use_ttm else [data[0]]
            elif isinstance(data, dict):
                data_fetched["income"] = [data]
    
    # Fetch Cash Flow Statement
    if items_to_fetch & cashflow_items:
        if use_ttm:
            endpoint = "cash-flow-statement-ttm"
            params = {"symbol": ticker}
        else:
            endpoint = "cash-flow-statement"
            params = {
                "symbol": ticker,
                "period": "quarter" if period.lower() in ["quarterly", "quarter", "q"] else "annual",
                "limit": limit
            }
        
        response = _make_fmp_request(endpoint, params, api_key)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                data_fetched["cashflow"] = data if not use_ttm else [data[0]]
            elif isinstance(data, dict):
                data_fetched["cashflow"] = [data]
    
    # Fetch Balance Sheet Statement
    if items_to_fetch & balance_items:
        if use_ttm:
            endpoint = "balance-sheet-statement-ttm"
            params = {"symbol": ticker}
        else:
            endpoint = "balance-sheet-statement"
            params = {
                "symbol": ticker,
                "period": "quarter" if period.lower() in ["quarterly", "quarter", "q"] else "annual",
                "limit": limit
            }
        
        response = _make_fmp_request(endpoint, params, api_key)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                data_fetched["balance"] = data if not use_ttm else [data[0]]
            elif isinstance(data, dict):
                data_fetched["balance"] = [data]
    
    # Fetch Ratios if needed
    if items_to_fetch & ratio_items:
        if use_ttm:
            endpoint = "ratios-ttm"
            params = {"symbol": ticker}
        else:
            endpoint = "ratios"
            params = {
                "symbol": ticker,
                "period": "quarter" if period.lower() in ["quarterly", "quarter", "q"] else "annual",
                "limit": limit
            }
        
        response = _make_fmp_request(endpoint, params, api_key)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                data_fetched["ratios"] = data if not use_ttm else [data[0]]
            elif isinstance(data, dict):
                data_fetched["ratios"] = [data]
    
    # Process periods
    periods_to_process = 1 if use_ttm else min(
        limit,
        len(data_fetched.get("income", [{}])),
        len(data_fetched.get("cashflow", [{}])),
        len(data_fetched.get("balance", [{}]))
    )
    
    for period_idx in range(periods_to_process):
        # Get data for this period
        income_data = {}
        cashflow_data = {}
        balance_data = {}
        ratios_data = {}
        
        if "income" in data_fetched and period_idx < len(data_fetched["income"]):
            income_data = data_fetched["income"][period_idx]
        if "cashflow" in data_fetched and period_idx < len(data_fetched["cashflow"]):
            cashflow_data = data_fetched["cashflow"][period_idx]
        if "balance" in data_fetched and period_idx < len(data_fetched["balance"]):
            balance_data = data_fetched["balance"][period_idx]
        if "ratios" in data_fetched and period_idx < len(data_fetched["ratios"]):
            ratios_data = data_fetched["ratios"][period_idx]
        
        # Get report date
        report_date = (
            income_data.get("date", "") or
            cashflow_data.get("date", "") or
            balance_data.get("date", "")
        )
        
        # Get currency
        currency = (
            income_data.get("reportedCurrency", "USD") or
            cashflow_data.get("reportedCurrency", "USD") or
            balance_data.get("reportedCurrency", "USD")
        )
        
        # Create LineItem object
        line_item = LineItem(
            ticker=ticker,
            report_period=report_date,
            period=period,
            currency=currency
        )
        
        # Map each requested field
        for line_item_name in line_items:
            value = None
            
            # Income Statement mappings
            if line_item_name in ["revenue", "net_income", "ebit", "ebitda", "gross_profit", 
                                 "operating_income", "eps", "earnings_per_share", "interest_expense",
                                 "gross_margin", "operating_margin", "research_and_development", 
                                 "operating_expense"]:
                
                if income_data:
                    field_mapping = {
                        "revenue": "revenue",
                        "net_income": "netIncome",
                        "ebit": "ebit",
                        "ebitda": "ebitda",
                        "gross_profit": "grossProfit",
                        "operating_income": "operatingIncome",
                        "depreciation_and_amortization": "depreciationAndAmortization",
                        "eps": "eps",
                        "earnings_per_share": "eps",
                        "interest_expense": "interestExpense",
                        "gross_margin": "grossProfitRatio",
                        "operating_margin": "operatingIncomeRatio",
                        "research_and_development": "researchAndDevelopmentExpenses",
                        "operating_expense": "operatingExpenses"
                    }
                    
                    # Handle TTM fields (have TTM suffix in FMP)
                    if use_ttm and line_item_name == "revenue":
                        # For TTM, FMP uses revenueTTM
                        value = income_data.get("revenueTTM") or income_data.get("revenue")
                    else:
                        fmp_field = field_mapping.get(line_item_name)
                        if fmp_field:
                            value = income_data.get(fmp_field)
                    
                    # Convert ratios to percentages
                    if line_item_name in ["gross_margin", "operating_margin"] and value is not None:
                        value = value * 100 if value < 1 else value
            
            # Cash Flow mappings
            elif line_item_name in ["free_cash_flow", "operating_cash_flow", "capital_expenditure",
                                   "dividends_and_other_cash_distributions", 
                                   "issuance_or_purchase_of_equity_shares"]:
                
                if cashflow_data:
                    field_mapping = {
                        "free_cash_flow": "freeCashFlow",
                        "operating_cash_flow": "operatingCashFlow",
                        "capital_expenditure": "capitalExpenditure",
                        "dividends_and_other_cash_distributions": "dividendsPaid",
                        "issuance_or_purchase_of_equity_shares": "commonStockRepurchased"
                    }
                    fmp_field = field_mapping.get(line_item_name)
                    if fmp_field:
                        value = cashflow_data.get(fmp_field)
            
            # Balance Sheet mappings
            elif line_item_name in balance_items:
                if balance_data:
                    field_mapping = {
                        "total_assets": "totalAssets",
                        "total_liabilities": "totalLiabilities",
                        "total_equity": "totalStockholdersEquity",
                        "outstanding_shares": "commonStock",
                        "cash_and_cash_equivalents": "cashAndCashEquivalents",
                        "cash_and_equivalents": "cashAndCashEquivalents",
                        "total_debt": "totalDebt",
                        "shareholders_equity": "totalStockholdersEquity",
                        "current_assets": "totalCurrentAssets",
                        "current_liabilities": "totalCurrentLiabilities",
                        "working_capital": "netWorkingCapital",
                        "long_term_debt": "longTermDebt",
                        "short_term_debt": "shortTermDebt"
                    }
                    fmp_field = field_mapping.get(line_item_name)
                    if fmp_field:
                        value = balance_data.get(fmp_field)
                    
                    # Special case for outstanding_shares
                    if line_item_name == "outstanding_shares" and not value and income_data:
                        value = income_data.get("weightedAverageShsOut") or income_data.get("weightedAverageShsOutDil")
            
            # Ratio mappings
            elif line_item_name in ratio_items:
                if ratios_data:
                    field_mapping = {
                        "debt_to_equity": "debtEquityRatio",
                        "current_ratio": "currentRatio",
                        "quick_ratio": "quickRatio",
                        "return_on_equity": "returnOnEquity",
                        "return_on_assets": "returnOnAssets",
                        "profit_margin": "netProfitMargin"
                    }
                    fmp_field = field_mapping.get(line_item_name)
                    if fmp_field:
                        value = ratios_data.get(fmp_field)
            
            # Set the value as attribute
            setattr(line_item, line_item_name, float(value) if value is not None else None)
        
        results.append(line_item)
    
    # If no data, create empty LineItem with None values
    if not results:
        line_item = LineItem(
            ticker=ticker,
            report_period="",
            period=period,
            currency="USD"
        )
        for line_item_name in line_items:
            setattr(line_item, line_item_name, None)
        results.append(line_item)
    
    # Cache the results
    if results:
        cache_data = []
        for item in results:
            item_dict = {
                "ticker": item.ticker,
                "report_period": item.report_period,
                "period": item.period,
                "currency": item.currency
            }
            for line_item_name in line_items:
                item_dict[line_item_name] = getattr(item, line_item_name, None)
            cache_data.append(item_dict)
        _cache.set_line_items(cache_key, cache_data)
    
    # Return list directly for compatibility
    return results

# def get_insider_trades(
#     ticker: str,
#     end_date: str,
#     start_date: str = None,
#     limit: int = 1000,
#     api_key: str = None,
# ) -> List[InsiderTrade]: #InsiderTradeResponse:
#     """
#     Fetch insider trades from FMP API and return InsiderTradeResponse format.
#     Updated to return InsiderTradeResponse instead of List[InsiderTrade]
#     """
    
#     cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
#     # Check cache first
#     # if cached_data := _cache.get_insider_trades(cache_key):
#     #     return InsiderTradeResponse(insider_trades=[InsiderTrade(**trade) for trade in cached_data])
#     if cached_data := _cache.get_insider_trades(cache_key):
#         return [InsiderTrade(**trade) for trade in cached_data]
    
#     # Try latest insider trading endpoint first
#     # params = {"symbol": ticker}
#     params = {
#         "symbol": ticker,
#         "limit": limit
#     }
#     # response = _make_fmp_request("latest-insider-trade", params, api_key)

#     response = _make_fmp_request("insider-trading", params, api_key)
    
#     if response.status_code != 200:
#         print(f"Warning: Could not fetch insider trades for {ticker}: {response.status_code}")
#         return []  # Return empty list on error
    
#     data = response.json()
#     if not data:
#         return []
    
#     # Map FMP data to InsiderTrade model
#     trades = []
#     for trade_data in data[:limit]:  # Limit results
#         # Parse date if needed
#         trade_date = trade_data.get("transactionDate", "")
#         if start_date and trade_date < start_date:
#             continue  # Skip trades before start_date
#         if end_date and trade_date > end_date:
#             continue  # Skip trades after end_date
            
#         trade = InsiderTrade(
#             ticker=ticker,
#             published_date=trade_data.get("filingDate", ""),
#             transaction_date=trade_data.get("transactionDate", ""),
#             owner_cik=trade_data.get("reportingCik", ""),
#             owner_name=trade_data.get("reportingName", ""),
#             transaction_shares=float(trade_data.get("securitiesTransacted", 0)),
#             transaction_price=float(trade_data.get("pricePerShare", 0)) if trade_data.get("pricePerShare") else None,
#             transaction_value=float(trade_data.get("securitiesTransacted", 0)) * float(trade_data.get("pricePerShare", 0)) 
#                             if trade_data.get("pricePerShare") else None,
#             transaction_type=trade_data.get("transactionType", ""),
#             acquisition_or_disposition=trade_data.get("acquistionOrDisposition", ""),
#             form_type=trade_data.get("formType", ""),
#             securities_owned=float(trade_data.get("securitiesOwned", 0)) if trade_data.get("securitiesOwned") else None,
#             insider_title=trade_data.get("typeOfOwner", "")
#         )
#         trades.append(trade)
    
#     # Cache the results
#     if trades:
#         _cache.set_insider_trades(cache_key, [t.model_dump() for t in trades])
    
#     # Return list directly for compatibility with agents
#     return trades

def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str = None,
    limit: int = 1000,
    api_key: str = None,
) -> List[InsiderTrade]:
    """
    Fetch insider trades from FMP API and return list of InsiderTrade objects.
    Uses legacy /v4/insider-trading endpoint with pagination to respect limit.
    """
    
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first
    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]
    
    all_trades = []
    page = 0
    max_pages = 20  # Safety cap to prevent excessive API calls
    
    while len(all_trades) < limit and page < max_pages:
        params = {
            "symbol": ticker,
            "page": page
        }
        response = _make_fmp_request("insider-trading", params, api_key)
        
        if response.status_code != 200:
            print(f"Warning: Could not fetch insider trades for {ticker} (page {page}): {response.status_code}")
            break
        
        data = response.json()
        if not data:
            break
        
        new_trades = []
        for trade_data in data:
            # Parse and filter by date range
            trade_date = trade_data.get("transactionDate", "")
            if start_date and trade_date < start_date:
                continue
            if end_date and trade_date > end_date:
                continue
                
            # Map FMP data to InsiderTrade model (fixed 'price' field)
            price = float(trade_data.get("price", 0)) if trade_data.get("price") is not None else None
            transaction_value = (
                float(trade_data.get("securitiesTransacted", 0)) * price
                if price is not None else None
            )
            
            trade = InsiderTrade(
                ticker=ticker,
                published_date=trade_data.get("filingDate", ""),
                transaction_date=trade_date,
                owner_cik=trade_data.get("reportingCik", ""),
                owner_name=trade_data.get("reportingName", ""),
                transaction_shares=float(trade_data.get("securitiesTransacted", 0)),
                transaction_price=price,
                transaction_value=transaction_value,
                transaction_type=trade_data.get("transactionType", ""),
                acquisition_or_disposition=trade_data.get("acquistionOrDisposition", ""),
                form_type=trade_data.get("formType", ""),
                securities_owned=float(trade_data.get("securitiesOwned", 0)) if trade_data.get("securitiesOwned") is not None else None,
                insider_title=trade_data.get("typeOfOwner", "")
            )
            new_trades.append(trade)
        
        all_trades.extend(new_trades)
        if len(new_trades) == 0:
            break  # No more relevant trades
        page += 1
    
    # Trim to limit if over
    all_trades = all_trades[:limit]
    
    # Cache the results
    if all_trades:
        _cache.set_insider_trades(cache_key, [t.model_dump() for t in all_trades])
    
    return all_trades
 
    # all_trades = []
    
    # if response.status_code == 200:
    #     data = response.json()
    #     if isinstance(data, list):
    #         for fmp_trade in data:
    #             # Filter by date range if specified
    #             filing_date = fmp_trade.get("filingDate", "")
    #             if start_date and filing_date < start_date:
    #                 continue
    #             if end_date and filing_date > end_date:
    #                 continue
                
    #             # Map FMP data to InsiderTrade model
    #             transaction_shares = fmp_trade.get("transactionShares", 0)
    #             transaction_type = "Sell" if transaction_shares < 0 else "Buy"
    #             price = fmp_trade.get("transactionPrice", 0)
    #             shares = abs(transaction_shares)
    #             value = price * shares if price and shares else None
                
    #             trade_data = {
    #                 "ticker": ticker,
    #                 "filing_date": fmp_trade.get("filingDate", ""),
    #                 "transaction_date": fmp_trade.get("transactionDate"),
    #                 "name": fmp_trade.get("insiderName"),
    #                 "title": fmp_trade.get("insiderTitle"),  # Add title mapping
    #                 "issuer": fmp_trade.get("issuer"),
    #                 "security_title": fmp_trade.get("securityTitle"),
    #                 "transaction_price_per_share": price,
    #                 "transaction_shares": shares,
    #                 "transaction_value": value,  # Use transaction_value instead of value
    #                 "shares_owned_before_transaction": fmp_trade.get("sharesOwnedPriorToTransaction"),
    #                 "shares_owned_after_transaction": fmp_trade.get("postTransactionAmounts"),
    #                 "is_board_director": fmp_trade.get("isDirector", False),
    #             }
                
    #             all_trades.append(InsiderTrade(**trade_data))
                
    #             if len(all_trades) >= limit:
    #                 break
    # else:
    #     print(f"Warning: Could not fetch insider trades for {ticker}")
        
    #     # Try alternative endpoint: search-insider-trades
    #     params = {"symbol": ticker}
    #     response = _make_fmp_request("search-insider-trades", params, api_key)
        
    #     if response.status_code == 200:
    #         data = response.json()
    #         if isinstance(data, list):
    #             for fmp_trade in data:
    #                 filing_date = fmp_trade.get("filingDate", "")
    #                 if start_date and filing_date < start_date:
    #                     continue
    #                 if end_date and filing_date > end_date:
    #                     continue
                    
    #                 transaction_shares = fmp_trade.get("transactionShares", 0)
    #                 price = fmp_trade.get("transactionPrice", 0)
    #                 shares = abs(transaction_shares)
    #                 value = price * shares if price and shares else None
                    
    #                 trade_data = {
    #                     "ticker": ticker,
    #                     "filing_date": fmp_trade.get("filingDate", ""),
    #                     "transaction_date": fmp_trade.get("transactionDate"),
    #                     "name": fmp_trade.get("insiderName"),
    #                     "title": fmp_trade.get("insiderTitle"),
    #                     "issuer": fmp_trade.get("issuer"),
    #                     "security_title": fmp_trade.get("securityTitle"),
    #                     "transaction_price_per_share": price,
    #                     "transaction_shares": shares,
    #                     "transaction_value": value,
    #                     "shares_owned_before_transaction": fmp_trade.get("sharesOwnedPriorToTransaction"),
    #                     "shares_owned_after_transaction": fmp_trade.get("postTransactionAmounts"),
    #                     "is_board_director": fmp_trade.get("isDirector", False),
    #                 }
                    
    #                 all_trades.append(InsiderTrade(**trade_data))
                    
    #                 if len(all_trades) >= limit:
    #                     break
    
    # # Sort by filing date (newest first)
    # all_trades.sort(key=lambda x: x.filing_date, reverse=True)
    
    # # Limit results
    # all_trades = all_trades[:limit]
    
    # # Cache the results
    # if all_trades:
    #     _cache.set_insider_trades(cache_key, [trade.model_dump() for trade in all_trades])
    
    # # Return in correct format
    # return InsiderTradeResponse(insider_trades=all_trades)


# def get_company_news(
#     ticker: str,
#     end_date: str,
#     start_date: str = None,
#     limit: int = 1000,
#     api_key: str = None,
# ) -> CompanyNewsResponse:
#     """
#     Fetch company news from FMP API and return CompanyNewsResponse format.
#     Updated to return CompanyNewsResponse instead of List[CompanyNews]
#     """
#     cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
#     # Check cache first  
#     if cached_data := _cache.get_company_news(cache_key):
#         return CompanyNewsResponse(news=[CompanyNews(**news) for news in cached_data])
    
#     all_news = []
    
#     # Try stable news endpoint
#     params = {"symbols": ticker}  # Note: FMP uses 'symbols' not 'symbol'
#     response = _make_fmp_request("news/stock", params, api_key)
    
#     if response.status_code == 200:
#         data = response.json()
#         if isinstance(data, list):
#             for fmp_news in data:
#                 # Filter by date range if specified
#                 published_date = fmp_news.get("publishedDate", "")
#                 if start_date and published_date < start_date:
#                     continue
#                 if end_date and published_date > end_date:
#                     continue
                
#                 news_data = {
#                     "ticker": ticker,
#                     "date": fmp_news.get("publishedDate", ""),
#                     "source": fmp_news.get("site", ""),
#                     "title": fmp_news.get("title", ""),
#                     "url": fmp_news.get("url", ""),
#                     "sentiment": fmp_news.get("sentiment", ""),
#                     "author": fmp_news.get("author", "")
#                 }
                
#                 all_news.append(CompanyNews(**news_data))
                
#                 if len(all_news) >= limit:
#                     break
#     else:
#         print(f"Warning: Could not fetch news for {ticker}, trying stock-latest endpoint")
        
#         # Try alternative endpoint: news/stock-latest
#         response = _make_fmp_request("news/stock-latest", {}, api_key)
        
#         if response.status_code == 200:
#             data = response.json()
#             if isinstance(data, list):
#                 # Filter for specific ticker
#                 for fmp_news in data:
#                     # Check if ticker is mentioned
#                     if ticker not in fmp_news.get("symbol", ""):
#                         continue
                        
#                     published_date = fmp_news.get("publishedDate", "")
#                     if start_date and published_date < start_date:
#                         continue
#                     if end_date and published_date > end_date:
#                         continue
                    
#                     news_data = {
#                         "ticker": ticker,
#                         "date": fmp_news.get("publishedDate", ""),
#                         "source": fmp_news.get("site", ""),
#                         "title": fmp_news.get("title", ""),
#                         "url": fmp_news.get("url", ""),
#                         "sentiment": fmp_news.get("sentiment", ""),
#                         "author": fmp_news.get("author", "")
#                     }
                    
#                     all_news.append(CompanyNews(**news_data))
                    
#                     if len(all_news) >= limit:
#                         break
    
#     # Sort by date (newest first)
#     all_news.sort(key=lambda x: x.date, reverse=True)
    
#     # Limit results
#     all_news = all_news[:limit]
    
#     # Cache the results
#     if all_news:
#         _cache.set_company_news(cache_key, [news.model_dump() for news in all_news])
    
#     # Return in correct format
#     return CompanyNewsResponse(news=all_news)

# def get_company_news(
#     ticker: str,
#     end_date: str,
#     start_date: str = None,
#     limit: int = 100,
#     api_key: str = None,
# ) -> List[CompanyNews]:  # Return List[CompanyNews] for compatibility with agents
#     """
#     Fetch company news with sentiment from FMP RSS Feed API.
#     Returns List[CompanyNews] for compatibility with ai-hedge-fund agents.
#     """
#     cache_key = f"{ticker}_news_{start_date or 'none'}_{end_date}_{limit}_v2"
    
#     # Check cache first  
#     if cached_data := _cache.get_company_news(cache_key):
#         return [CompanyNews(**news) for news in cached_data]
    
#     all_news = []
#     pages_to_fetch = min(5, (limit // 50) + 1)  # RSS feed returns ~50 items per page
    
#     # Use the RSS feed endpoint with sentiment scores
#     for page in range(pages_to_fetch):
#         params = {
#             "page": page
#         }
        
#         # Using v4 endpoint for sentiment RSS feed
#         response = _make_fmp_request("stock-news-sentiments-rss-feed", params, api_key, api_version="v4")
        
#         if response.status_code == 200:
#             data = response.json()
#             if isinstance(data, list):
#                 for fmp_news in data:
#                     # Check if ticker matches
#                     news_symbol = fmp_news.get("symbol", "")
#                     if ticker.upper() not in news_symbol.upper():
#                         continue
                    
#                     # Filter by date range if specified
#                     published_date = fmp_news.get("publishedDate", "")
                    
#                     # Convert date format if needed (2023-10-10T21:10:53.000Z -> 2023-10-10)
#                     if "T" in published_date:
#                         published_date = published_date.split("T")[0]
                    
#                     if start_date and published_date < start_date:
#                         continue
#                     if end_date and published_date > end_date:
#                         continue
                    
#                     # Map sentiment score to category
#                     sentiment_score = fmp_news.get("sentimentScore", 0)
#                     sentiment = fmp_news.get("sentiment", "")
                    
#                     # If sentiment text not provided, derive from score
#                     if not sentiment and sentiment_score:
#                         if sentiment_score > 0.6:
#                             sentiment = "Positive"
#                         elif sentiment_score < -0.6:
#                             sentiment = "Negative"
#                         else:
#                             sentiment = "Neutral"
                    
#                     news_data = {
#                         "ticker": ticker,
#                         "date": published_date,
#                         "source": fmp_news.get("site", ""),
#                         "title": fmp_news.get("title", ""),
#                         "url": fmp_news.get("url", ""),
#                         "sentiment": sentiment,
#                         "sentiment_score": sentiment_score,  # Add numeric score if model supports it
#                         "author": fmp_news.get("author", ""),
#                         "text": fmp_news.get("text", "")[:500] if fmp_news.get("text") else ""  # First 500 chars
#                     }
                    
#                     # Create CompanyNews object
#                     news_item = CompanyNews(**{k: v for k, v in news_data.items() 
#                                               if k in ["ticker", "date", "source", "title", "url", "sentiment", "author"]})
                    
#                     # Add sentiment_score as attribute if model allows extra fields
#                     if hasattr(news_item, '__dict__'):
#                         news_item.sentiment = sentiment_score
                    
#                     all_news.append(news_item)
                    
#                     if len(all_news) >= limit:
#                         break
#             else:
#                 print(f"Warning: Unexpected response format from RSS feed for {ticker}")
#         else:
#             print(f"Warning: Could not fetch news from RSS feed for {ticker}: {response.status_code}")
#             break  # Stop trying more pages if API fails
        
#         if len(all_news) >= limit:
#             break
    
#     # If RSS feed didn't return enough results, try fallback endpoint
#     if len(all_news) < min(10, limit):  # If we got less than 10 items, try fallback
#         print(f"RSS feed returned only {len(all_news)} items for {ticker}, trying fallback endpoint")
        
#         params = {"tickers": ticker, "limit": limit - len(all_news)}
#         response = _make_fmp_request("stock_news", params, api_key)
        
#         if response.status_code == 200:
#             data = response.json()
#             if isinstance(data, list):
#                 for fmp_news in data:
#                     published_date = fmp_news.get("publishedDate", "")
#                     if start_date and published_date < start_date:
#                         continue
#                     if end_date and published_date > end_date:
#                         continue
                    
#                     news_data = {
#                         "ticker": ticker,
#                         "date": published_date,
#                         "source": fmp_news.get("site", ""),
#                         "title": fmp_news.get("title", ""),
#                         "url": fmp_news.get("url", ""),
#                         "sentiment": fmp_news.get("sentiment", "Neutral"),  # Default to Neutral if not provided
#                         "author": fmp_news.get("author", "")
#                     }
                    
#                     all_news.append(CompanyNews(**news_data))
                    
#                     if len(all_news) >= limit:
#                         break
    
#     # Sort by date (newest first)
#     all_news.sort(key=lambda x: x.date, reverse=True)
    
#     # Limit results
#     all_news = all_news[:limit]
    
#     # Cache the results
#     if all_news:
#         _cache.set_company_news(cache_key, [news.model_dump() for news in all_news])
    
#     # Return list directly for compatibility
#     return all_news

def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str = None,
    limit: int = 100,
    api_key: str = None,
) -> List[CompanyNews]:  # Return List[CompanyNews] for compatibility with agents
    """
    Fetch company news with sentiment from FMP RSS Feed API.
    Returns List[CompanyNews] for compatibility with ai-hedge-fund agents.
    """
    cache_key = f"{ticker}_news_{start_date or 'none'}_{end_date}_{limit}_v3"  # Updated cache version
    
    # Check cache first - FIX: Handle sentiment that might be float in old cache
    if cached_data := _cache.get_company_news(cache_key):
        fixed_news = []
        for news in cached_data:
            # Fix sentiment field if it's not a proper string
            if "sentiment" in news:
                sentiment_val = news["sentiment"]
                # Check if sentiment is already a valid string
                if isinstance(sentiment_val, str) and sentiment_val in ["Positive", "Negative", "Neutral"]:
                    # Good, keep it as is
                    pass
                elif isinstance(sentiment_val, (int, float)):
                    # This is old cache data where sentiment was stored as float
                    # Convert numeric sentiment to string category
                    if sentiment_val > 0.6:
                        news["sentiment"] = "Positive"
                    elif sentiment_val < -0.6:
                        news["sentiment"] = "Negative"
                    else:
                        news["sentiment"] = "Neutral"
                else:
                    # Invalid or None, default to Neutral
                    news["sentiment"] = "Neutral"
            else:
                # No sentiment field, add default
                news["sentiment"] = "Neutral"
            
            fixed_news.append(CompanyNews(**news))
        return fixed_news
    
    all_news = []
    pages_to_fetch = min(5, (limit // 50) + 1)  # RSS feed returns ~50 items per page
    
    # Use the RSS feed endpoint with sentiment scores
    for page in range(pages_to_fetch):
        params = {
            "page": page
        }
        
        # Using v4 endpoint for sentiment RSS feed
        response = _make_fmp_request("stock-news-sentiments-rss-feed", params, api_key, api_version="v4")
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                for fmp_news in data:
                    # Check if ticker matches
                    news_symbol = fmp_news.get("symbol", "")
                    if ticker.upper() not in news_symbol.upper():
                        continue
                    
                    # Filter by date range if specified
                    published_date = fmp_news.get("publishedDate", "")
                    
                    # Convert date format if needed (2023-10-10T21:10:53.000Z -> 2023-10-10)
                    if "T" in published_date:
                        published_date = published_date.split("T")[0]
                    
                    if start_date and published_date < start_date:
                        continue
                    if end_date and published_date > end_date:
                        continue
                    
                    # Handle sentiment - FIX: Use sentiment string directly from API
                    sentiment = fmp_news.get("sentiment")  # This should already be "Positive"/"Negative"/"Neutral"
                    
                    # Validate and ensure it's a proper string
                    if not sentiment or not isinstance(sentiment, str):
                        # Only fallback to score conversion if sentiment string is missing
                        sentiment_score = fmp_news.get("sentimentScore")
                        if isinstance(sentiment_score, (int, float)):
                            # Convert score to category ONLY if sentiment string not provided
                            if sentiment_score > 0.6:
                                sentiment = "Positive"
                            elif sentiment_score < -0.6:
                                sentiment = "Negative"
                            else:
                                sentiment = "Neutral"
                        else:
                            # Default if both sentiment and sentimentScore are missing
                            sentiment = "Neutral"
                    
                    news_data = {
                        "ticker": ticker,
                        "date": published_date,
                        "source": fmp_news.get("site", ""),
                        "title": fmp_news.get("title", ""),
                        "url": fmp_news.get("url", ""),
                        "sentiment": sentiment,  # Now guaranteed to be a string
                        "author": fmp_news.get("author", "")
                    }
                    
                    all_news.append(CompanyNews(**news_data))
                    
                    if len(all_news) >= limit:
                        break
            else:
                print(f"Warning: Unexpected response format from RSS feed for {ticker}")
        else:
            print(f"Warning: Could not fetch news from RSS feed for {ticker}: {response.status_code}")
            break  # Stop trying more pages if API fails
        
        if len(all_news) >= limit:
            break
    
    # If RSS feed didn't return enough results, try fallback endpoint
    if len(all_news) < min(10, limit):  # If we got less than 10 items, try fallback
        print(f"RSS feed returned only {len(all_news)} items for {ticker}, trying fallback endpoint")
        
        params = {"tickers": ticker, "limit": limit - len(all_news)}
        response = _make_fmp_request("stock_news", params, api_key)
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                for fmp_news in data:
                    published_date = fmp_news.get("publishedDate", "")
                    
                    if start_date and published_date < start_date:
                        continue
                    if end_date and published_date > end_date:
                        continue
                    
                    # Handle sentiment for fallback endpoint
                    sentiment = fmp_news.get("sentiment")
                    
                    # Validate sentiment is proper string
                    if not sentiment or not isinstance(sentiment, str):
                        # Fallback endpoint might not have sentiment
                        sentiment = "Neutral"
                    elif sentiment not in ["Positive", "Negative", "Neutral"]:
                        # If sentiment exists but not in expected values, default to Neutral
                        sentiment = "Neutral"
                    
                    news_data = {
                        "ticker": ticker,
                        "date": published_date,
                        "source": fmp_news.get("site", ""),
                        "title": fmp_news.get("title", ""),
                        "url": fmp_news.get("url", ""),
                        "sentiment": sentiment,
                        "author": fmp_news.get("author", "")
                    }
                    
                    all_news.append(CompanyNews(**news_data))
                    
                    if len(all_news) >= limit:
                        break
    
    # Sort by date (newest first)
    all_news.sort(key=lambda x: x.date, reverse=True)
    
    # Limit results
    all_news = all_news[:limit]
    
    # Cache the results with proper sentiment format
    if all_news:
        _cache.set_company_news(cache_key, [news.model_dump() for news in all_news])
    
    # Return list directly for compatibility
    return all_news

def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> CompanyFactsResponse:
    """
    Fetch market cap and company facts from FMP API and return CompanyFactsResponse format.
    Updated to return CompanyFactsResponse instead of float | None
    """    
    cache_key = f"{ticker}_company_facts_{end_date}"
    
    # Check cache first
    if cached_data := _cache.get_market_cap(cache_key):
        return CompanyFactsResponse(company_facts=CompanyFacts(**cached_data))
    
    # Initialize company facts with defaults
    company_facts_data = {
        "ticker": ticker,
        "name": ticker,  # Default to ticker if name not found
        "cik": None,
        "industry": None,
        "sector": None,
        "category": None,
        "exchange": None,
        "is_active": None,
        "listing_date": None,
        "location": None,
        "market_cap": None,
        "number_of_employees": None,
        "sec_filings_url": None,
        "sic_code": None,
        "sic_industry": None,
        "sic_sector": None,
        "website_url": None,
        "weighted_average_shares": None
    }
    
    # Fetch company profile for comprehensive data
    params = {"symbol": ticker}
    profile_response = _make_fmp_request("profile", params, api_key)
    
    if profile_response.status_code == 200:
        data = profile_response.json()
        if isinstance(data, list) and data:
            profile_data = data[0]
            
            # Map FMP profile data to CompanyFacts
            company_facts_data.update({
                "name": profile_data.get("companyName", ticker),
                "cik": profile_data.get("cik"),
                "industry": profile_data.get("industry"),
                "sector": profile_data.get("sector"),
                "exchange": profile_data.get("exchange"),
                "is_active": profile_data.get("isActivelyTrading", None),
                "listing_date": profile_data.get("ipoDate"),
                "location": f"{profile_data.get('city', '')}, {profile_data.get('state', '')} {profile_data.get('country', '')}".strip(),
                "market_cap": profile_data.get("mktCap"),
                "number_of_employees": profile_data.get("fullTimeEmployees"),
                "website_url": profile_data.get("website"),
            })
    
    # If market cap not found in profile, try key-metrics-ttm
    if company_facts_data["market_cap"] is None:
        params = {"symbol": ticker}
        response = _make_fmp_request("key-metrics-ttm", params, api_key)
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and data:
                company_facts_data["market_cap"] = data[0].get("marketCap")
    
    # If still not found, try market-capitalization endpoint
    if company_facts_data["market_cap"] is None:
        params = {"symbol": ticker, "limit": 1}
        response = _make_fmp_request("market-capitalization", params, api_key)
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and data:
                company_facts_data["market_cap"] = data[0].get("marketCap")
    
    # Try to get additional data from enterprise-values endpoint for weighted average shares
    params = {"symbol": ticker, "limit": 1}
    ev_response = _make_fmp_request("enterprise-values", params, api_key)
    
    if ev_response.status_code == 200:
        data = ev_response.json()
        if isinstance(data, list) and data:
            company_facts_data["weighted_average_shares"] = data[0].get("numberOfShares")
    
    # Cache the result
    _cache.set_market_cap(cache_key, company_facts_data)
    
    # Return in correct format
    return CompanyFactsResponse(company_facts=CompanyFacts(**company_facts_data))


def prices_to_df(prices: List[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df

def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)


def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        # FMP sometimes returns "1.23%" as string for changesPercentage
        if isinstance(x, str) and x.endswith("%"):
            return float(x.replace("%", "").strip())
        return float(x)
    except (ValueError, TypeError):
        return None

def _now_iso_utc() -> str:
    return dt.utcnow().replace(microsecond=0).isoformat() + "Z"

def _derive_change_percent(price: Optional[float], prev_close: Optional[float], explicit: Optional[float]) -> Optional[float]:
    """
    Trả về %change (đơn vị: %) theo ưu tiên:
    1) explicit (changesPercentage từ API, đã normalize về float không có dấu %)
    2) tự tính từ price & previous_close
    """
    if explicit is not None:
        return explicit
    if price is not None and prev_close not in (None, 0):
        try:
            return (price / prev_close - 1.0) * 100.0
        except ZeroDivisionError:
            return None
    return None

def get_current_price(symbol: str) -> Dict[str, Any]:

    api_key = get_api_key_from_state(None, "FMP_API_KEY")

    sym = (symbol or "").upper().strip()
    if not sym:
        return {
            "current_price": None,
            "previous_close": None,
            "change": None,
            "change_percent": None,
            "as_of": None,
            "market_state": "unknown",
            "source": "none",
        }

    # -------------------------
    # 1) TRY: stable/quote
    # -------------------------
    try:
        resp = _make_fmp_request("quote", {"symbol": sym}, api_key)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                q = data[0] or {}
                price = q.get("price")
                prev = q.get("previousClose")
                change = q.get("change")
                chg_pct = q.get("changesPercentage")
                ts = q.get("timestamp")

                # FMP có thể trả % dạng "1.23%" hoặc số; chuẩn hóa về float %
                if isinstance(chg_pct, str) and chg_pct.endswith("%"):
                    try:
                        chg_pct = float(chg_pct.replace("%", "").strip())
                    except Exception:
                        chg_pct = None

                as_of = None
                if ts:
                    try:
                        # FMP timestamp là epoch seconds
                        as_of = dt.utcfromtimestamp(int(ts)).isoformat() + "Z"
                    except Exception:
                        as_of = None

                return {
                    "current_price": float(price) if price is not None else None,
                    "previous_close": float(prev) if prev is not None else None,
                    "change": float(change) if change is not None else None,
                    "change_percent": float(chg_pct) if chg_pct is not None else None,
                    "as_of": as_of,
                    "market_state": "real-time",   # quote là nguồn chính
                    "source": "quote",
                }
    except Exception:
        # fall through to next tier
        pass

    # -------------------------
    # 2) FALLBACK: stable/quote-short
    # -------------------------
    try:
        resp = _make_fmp_request("quote-short", {"symbol": sym}, api_key)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                qs = data[0] or {}
                price = qs.get("price")
                # quote-short không có previousClose/change → để None
                return {
                    "current_price": float(price) if price is not None else None,
                    "previous_close": None,
                    "change": None,
                    "change_percent": None,
                    "as_of": dt.utcnow().isoformat() + "Z",
                    "market_state": "delayed-quote",
                    "source": "quote-short",
                }
    except Exception:
        pass

    # -------------------------
    # 3) FALLBACK: stable/historical-chart/1min/{symbol}?limit=1
    # -------------------------
    try:
        resp = _make_fmp_request(f"historical-chart/1min/{sym}", {"limit": 1}, api_key)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                bar = data[0] or {}
                close_ = bar.get("close")
                date_iso = bar.get("date")
                # FMP trả "date" dạng "2025-11-09 15:59:00" (UTC). Chuẩn hóa ISO-8601
                as_of = None
                if isinstance(date_iso, str) and date_iso:
                    try:
                        # chấp nhận cả "YYYY-MM-DD HH:MM:SS" hoặc ISO có 'T'
                        if "T" in date_iso:
                            as_of = date_iso if date_iso.endswith("Z") else date_iso + "Z"
                        else:
                            # parse dạng "YYYY-MM-DD HH:MM:SS"
                            as_of = dt.strptime(date_iso, "%Y-%m-%d %H:%M:%S").isoformat() + "Z"
                    except Exception:
                        as_of = dt.utcnow().isoformat() + "Z"
                else:
                    as_of = dt.utcnow().isoformat() + "Z"

                return {
                    "current_price": float(close_) if close_ is not None else None,
                    "previous_close": None,
                    "change": None,
                    "change_percent": None,
                    "as_of": as_of,
                    "market_state": "latest-1min",
                    "source": "historical-1min",
                }
    except Exception:
        pass

    return {
        "current_price": None,
        "previous_close": None,
        "change": None,
        "change_percent": None,
        "as_of": None,
        "market_state": "unknown",
        "source": "none",
    }
