from typing import Optional, List
from pydantic import BaseModel, Field


# ============================================================================
# 1. Stock Symbol Search Models
# ============================================================================
class StockSymbolSearchItem(BaseModel):
    """
    Model for individual search result from search-symbol endpoint.
    
    FMP Endpoint: /stable/search-symbol?query=AAPL
    """
    symbol: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")
    currency: Optional[str] = Field(None, description="Trading currency (e.g., USD)")
    stock_exchange: Optional[str] = Field(None, alias="stockExchange", description="Exchange name (e.g., NASDAQ)")
    exchange_short_name: Optional[str] = Field(None, alias="exchangeShortName", description="Short exchange name")


# ============================================================================
# 2. Company Name Search Models  
# ============================================================================
class CompanyNameSearchItem(BaseModel):
    """
    Model for company name search results.
    
    FMP Endpoint: /stable/search-name?query=Apple
    """
    symbol: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Full company name")
    currency: Optional[str] = Field(None, description="Trading currency")
    stock_exchange: Optional[str] = Field(None, alias="stockExchange", description="Exchange name")
    exchange_short_name: Optional[str] = Field(None, alias="exchangeShortName", description="Short exchange name")


# ============================================================================
# 3. CIK Search Models
# ============================================================================
class CIKSearchItem(BaseModel):
    """
    Model for CIK (Central Index Key) search results.
    
    FMP Endpoint: /stable/search-cik?cik=320193
    
    CIK is a unique identifier used by the SEC for public companies.
    """
    cik: str = Field(..., description="Central Index Key (CIK) number")
    name: str = Field(..., description="Company name")
    symbol: Optional[str] = Field(None, description="Stock ticker symbol if available")


# ============================================================================
# 4. CUSIP Search Models
# ============================================================================
class CUSIPSearchItem(BaseModel):
    """
    Model for CUSIP search results.
    
    FMP Endpoint: /stable/search-cusip?cusip=037833100
    
    CUSIP (Committee on Uniform Securities Identification Procedures) 
    is the 9-character alphanumeric code to identify securities in North America.
    """
    cusip: str = Field(..., description="9-character CUSIP identifier")
    symbol: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")


# ============================================================================
# 5. ISIN Search Models
# ============================================================================
class ISINSearchItem(BaseModel):
    """
    Model for ISIN (International Securities Identification Number) search.
    
    FMP Endpoint: /stable/search-isin?isin=US0378331005
    
    ISIN is an international standard (ISO 6166) 12-character code 
    to identify securities globally.
    """
    isin: str = Field(..., description="12-character ISIN identifier")
    symbol: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")
    cusip: Optional[str] = Field(None, description="CUSIP number if available")


# ============================================================================
# 6. Stock Screener Models
# ============================================================================
class StockScreenerItem(BaseModel):
    """
    Model for stock screener results.
    
    FMP Endpoint: /stable/company-screener
    
    Screener allows to filter stocks by many criteria such as market cap,
    price, volume, beta, sector, country, etc.
    """
    symbol: str = Field(..., description="Stock ticker symbol")
    company_name: Optional[str] = Field(None, alias="companyName", description="Company name")
    market_cap: Optional[float] = Field(None, alias="marketCap", description="Market capitalization")
    sector: Optional[str] = Field(None, description="Business sector")
    industry: Optional[str] = Field(None, description="Industry classification")
    beta: Optional[float] = Field(None, description="Beta coefficient (volatility measure)")
    price: Optional[float] = Field(None, description="Current stock price")
    last_annual_dividend: Optional[float] = Field(None, alias="lastAnnualDividend", description="Last annual dividend")
    volume: Optional[int] = Field(None, description="Trading volume")
    exchange: Optional[str] = Field(None, description="Stock exchange")
    exchange_short_name: Optional[str] = Field(None, alias="exchangeShortName", description="Short exchange name")
    country: Optional[str] = Field(None, description="Company country")
    is_etf: Optional[bool] = Field(None, alias="isEtf", description="Is this an ETF?")
    is_actively_trading: Optional[bool] = Field(None, alias="isActivelyTrading", description="Is actively trading?")


# ============================================================================
# 7. Exchange Variants Models
# ============================================================================
class ExchangeVariantItem(BaseModel):
    """
    Model for exchange variants search.
    
    FMP Endpoint: /stable/search-exchange-variants?symbol=AAPL
    
    Find all exchanges where a symbol is listed (e.g., AAPL on NASDAQ, XETRA, etc.)
    """
    symbol: str = Field(..., description="Stock ticker symbol on this exchange")
    name: Optional[str] = Field(None, description="Company name")
    exchange: Optional[str] = Field(None, description="Full exchange name")
    exchange_short_name: Optional[str] = Field(None, alias="exchangeShortName", description="Short exchange name")
    price: Optional[float] = Field(None, description="Current price on this exchange")


# ============================================================================
# 8. Generic Search Result
# ============================================================================
class GenericCompanySearchItem(BaseModel):
    """
    Generic model can be used for many different types of searches.
    """
    symbol: Optional[str] = Field(None, description="Stock ticker symbol")
    name: Optional[str] = Field(None, description="Company/Security name")
    cik: Optional[str] = Field(None, description="CIK number")
    cusip: Optional[str] = Field(None, description="CUSIP identifier")
    isin: Optional[str] = Field(None, description="ISIN identifier")
    currency: Optional[str] = Field(None, description="Trading currency")
    stock_exchange: Optional[str] = Field(None, alias="stockExchange", description="Exchange name")
    exchange_short_name: Optional[str] = Field(None, alias="exchangeShortName", description="Short exchange name")
    market_cap: Optional[float] = Field(None, alias="marketCap", description="Market capitalization")
    sector: Optional[str] = Field(None, description="Business sector")
    industry: Optional[str] = Field(None, description="Industry")
    price: Optional[float] = Field(None, description="Current price")


# ============================================================================
# Request/Query Models
# ============================================================================
class StockScreenerFilters(BaseModel):
    """
    Query parameters cho Stock Screener endpoint.
    """
    market_cap_more_than: Optional[float] = Field(None, alias="marketCapMoreThan", description="Min market cap")
    market_cap_lower_than: Optional[float] = Field(None, alias="marketCapLowerThan", description="Max market cap")
    price_more_than: Optional[float] = Field(None, alias="priceMoreThan", description="Min price")
    price_lower_than: Optional[float] = Field(None, alias="priceLowerThan", description="Max price")
    beta_more_than: Optional[float] = Field(None, alias="betaMoreThan", description="Min beta")
    beta_lower_than: Optional[float] = Field(None, alias="betaLowerThan", description="Max beta")
    volume_more_than: Optional[int] = Field(None, alias="volumeMoreThan", description="Min volume")
    volume_lower_than: Optional[int] = Field(None, alias="volumeLowerThan", description="Max volume")
    dividend_more_than: Optional[float] = Field(None, alias="dividendMoreThan", description="Min dividend")
    dividend_lower_than: Optional[float] = Field(None, alias="dividendLowerThan", description="Max dividend")
    is_etf: Optional[bool] = Field(None, alias="isEtf", description="Filter ETFs")
    is_actively_trading: Optional[bool] = Field(None, alias="isActivelyTrading", description="Filter active stocks")
    sector: Optional[str] = Field(None, description="Sector filter (e.g., Technology)")
    industry: Optional[str] = Field(None, description="Industry filter")
    country: Optional[str] = Field(None, description="Country filter (e.g., US)")
    exchange: Optional[str] = Field(None, description="Exchange filter (e.g., NASDAQ)")
    limit: Optional[int] = Field(100, description="Max results to return", ge=1, le=10000)


# ============================================================================
# Example Usage (for documentation)
# ============================================================================
"""
EXAMPLE RESPONSE DATA:

1. Stock Symbol Search:
[
    {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "currency": "USD",
        "stockExchange": "NASDAQ Global Select",
        "exchangeShortName": "NASDAQ"
    }
]

2. CIK Search:
[
    {
        "cik": "0000320193",
        "name": "Apple Inc.",
        "symbol": "AAPL"
    }
]

3. CUSIP Search:
[
    {
        "cusip": "037833100",
        "symbol": "AAPL",
        "name": "Apple Inc."
    }
]

4. Stock Screener:
[
    {
        "symbol": "AAPL",
        "companyName": "Apple Inc.",
        "marketCap": 3000000000000,
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "beta": 1.2,
        "price": 180.50,
        "volume": 50000000,
        "exchange": "NASDAQ Global Select",
        "exchangeShortName": "NASDAQ",
        "country": "US",
        "isEtf": false,
        "isActivelyTrading": true
    }
]
"""