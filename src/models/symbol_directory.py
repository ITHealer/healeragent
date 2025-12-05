"""
Symbol Directory Models
=======================
Pydantic models cho Symbol Directory system.
Quản lý danh mục stock/crypto symbols, symbol changes, delisted companies.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from datetime import date, datetime
from enum import Enum
from datetime import datetime, date as Date

# ============================================================================
# ENUMS
# ============================================================================

class AssetClass(str, Enum):
    """Asset classification types."""
    STOCK = "stock"
    ETF = "etf"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"
    ADR = "adr"
    MUTUAL_FUND = "mutual_fund"


class SymbolStatus(str, Enum):
    """Symbol status classifications."""
    ACTIVE = "active"                  # Đang giao dịch bình thường
    RENAMED = "renamed"                # Đã đổi mã ticker
    DELISTED = "delisted"              # Đã bị hủy niêm yết
    INACTIVE = "inactive"              # Ngừng giao dịch tạm thời
    SUSPENDED = "suspended"            # Tạm ngừng giao dịch
    MERGED = "merged"                  # Bị merge vào công ty khác


# ============================================================================
# 1. STOCK LIST MODELS
# ============================================================================

class StockListItem(BaseModel):
    """
    Model for individual stock from FMP stock-list endpoint.
    
    FMP Endpoint: /stable/stock-list
    """
    symbol: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")
    price: Optional[float] = Field(None, description="Current price")
    exchange: Optional[str] = Field(None, description="Exchange name")
    exchange_short_name: Optional[str] = Field(None, alias="exchangeShortName", description="Short exchange code")
    type: Optional[str] = Field(None, description="Asset type (stock, etf, adr)")
    
    class Config:
        populate_by_name = True


class StockListResponse(BaseModel):
    """Response wrapper cho stock list."""
    total_count: int = Field(..., description="Total number of stocks")
    data: List[StockListItem] = Field(..., description="List of stocks")
    cached_at: datetime = Field(default_factory=datetime.utcnow, description="Cache timestamp")


# ============================================================================
# 2. CRYPTO LIST MODELS
# ============================================================================

class CryptoListItem(BaseModel):
    """
    Model for individual crypto from FMP cryptocurrency-list endpoint.
    
    FMP Endpoint: /stable/cryptocurrency-list
    """
    symbol: str = Field(..., description="Crypto symbol (e.g., BTCUSD)")
    name: str = Field(..., description="Crypto name (e.g., Bitcoin)")
    currency: Optional[str] = Field(None, description="Quote currency (usually USD)")
    stock_exchange: Optional[str] = Field(None, alias="stockExchange", description="Exchange name")
    exchange_short_name: Optional[str] = Field(None, alias="exchangeShortName", description="Short exchange code")
    
    class Config:
        populate_by_name = True


class CryptoListResponse(BaseModel):
    """Response wrapper cho crypto list."""
    total_count: int = Field(..., description="Total number of cryptos")
    data: List[CryptoListItem] = Field(..., description="List of cryptos")
    cached_at: datetime = Field(default_factory=datetime.utcnow, description="Cache timestamp")


# ============================================================================
# 3. SYMBOL CHANGES MODELS
# ============================================================================

class SymbolChangeItem(BaseModel):
    """
    Model for symbol change record from FMP.
    
    FMP Endpoint: /stable/symbol-change
    
    Tracks ticker changes due to M&A, rebranding, splits, etc.
    """
    date: Date = Field(..., description="Date of symbol change")
    name: Optional[str] = Field(None, description="Company name")
    old_symbol: str = Field(..., alias="oldSymbol", description="Old ticker symbol")
    new_symbol: str = Field(..., alias="newSymbol", description="New ticker symbol")
    
    class Config:
        populate_by_name = True


class SymbolChangesResponse(BaseModel):
    """Response wrapper cho symbol changes."""
    total_changes: int = Field(..., description="Total number of symbol changes")
    data: List[SymbolChangeItem] = Field(..., description="List of symbol changes")
    cached_at: datetime = Field(default_factory=datetime.utcnow, description="Cache timestamp")


# ============================================================================
# 4. DELISTED COMPANIES MODELS
# ============================================================================

class DelistedCompanyItem(BaseModel):
    """
    Model for delisted company from FMP.
    
    FMP Endpoint: /stable/delisted-companies
    
    Tracks companies that have been removed from exchanges.
    """
    symbol: str = Field(..., description="Stock symbol that was delisted")
    company_name: Optional[str] = Field(None, alias="companyName", description="Company name")
    exchange: Optional[str] = Field(None, description="Exchange where it was listed")
    ipo_date: Optional[date] = Field(None, alias="ipoDate", description="IPO date")
    delisted_date: Optional[date] = Field(None, alias="delistedDate", description="Date of delisting")
    
    class Config:
        populate_by_name = True


class DelistedCompaniesResponse(BaseModel):
    """Response wrapper cho delisted companies."""
    total_delisted: int = Field(..., description="Total number of delisted companies")
    data: List[DelistedCompanyItem] = Field(..., description="List of delisted companies")
    cached_at: datetime = Field(default_factory=datetime.utcnow, description="Cache timestamp")


# ============================================================================
# 5. ACTIVELY TRADING LIST MODELS
# ============================================================================

class ActivelyTradingItem(BaseModel):
    """
    Model for actively trading symbol.
    
    FMP Endpoint: /stable/actively-trading-list
    """
    symbol: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")
    price: Optional[float] = Field(None, description="Current price")
    exchange: Optional[str] = Field(None, description="Exchange name")
    exchange_short_name: Optional[str] = Field(None, alias="exchangeShortName", description="Short exchange code")
    type: Optional[str] = Field(None, description="Asset type")
    
    class Config:
        populate_by_name = True


# ============================================================================
# 6. SYMBOL STATUS MODELS (Unified Response)
# ============================================================================

class SymbolStatusDetail(BaseModel):
    """
    Unified model cho symbol status check.
    
    Kết hợp thông tin từ:
    - Stock/Crypto list (exists?)
    - Symbol changes (renamed?)
    - Delisted companies (delisted?)
    - Actively trading (active?)
    """
    input_symbol: str = Field(..., description="Symbol được query")
    status: SymbolStatus = Field(..., description="Current status of symbol")
    asset_class: Optional[AssetClass] = Field(None, description="Asset classification")
    
    # Basic info
    exists: bool = Field(..., description="Symbol có tồn tại trong database không?")
    name: Optional[str] = Field(None, description="Company/Asset name")
    exchange: Optional[str] = Field(None, description="Exchange name")
    
    # Status details
    is_actively_trading: bool = Field(True, description="Đang giao dịch không?")
    current_symbol: Optional[str] = Field(None, description="Current valid symbol (if renamed)")
    redirect_symbol: Optional[str] = Field(None, description="New symbol if renamed")
    
    # Important dates
    changed_on: Optional[date] = Field(None, description="Date of symbol change")
    delisted_on: Optional[date] = Field(None, description="Date of delisting")
    
    # Additional metadata
    message: Optional[str] = Field(None, description="Human-readable status message")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional information")


class SymbolValidationResult(BaseModel):
    """
    Simple validation result cho quick checks.
    
    Use case: Validate symbol trước khi gọi API khác.
    """
    symbol: str = Field(..., description="Symbol được validate")
    is_valid: bool = Field(..., description="Symbol có valid không?")
    asset_class: Optional[AssetClass] = Field(None, description="Asset type")
    exists_in_cache: bool = Field(..., description="Có trong cache không?")
    message: Optional[str] = Field(None, description="Validation message")


# ============================================================================
# 7. BULK VALIDATION MODELS
# ============================================================================

class BulkValidationRequest(BaseModel):
    """Request model cho bulk symbol validation."""
    symbols: List[str] = Field(..., min_items=1, max_items=100, description="List of symbols to validate")
    asset_class: Optional[AssetClass] = Field(None, description="Filter by asset class")


class BulkValidationResponse(BaseModel):
    """Response cho bulk validation."""
    total_checked: int = Field(..., description="Total symbols checked")
    valid_symbols: List[str] = Field(..., description="List of valid symbols")
    invalid_symbols: List[str] = Field(..., description="List of invalid symbols")
    details: List[SymbolValidationResult] = Field(..., description="Detailed results per symbol")


# ============================================================================
# 8. SYMBOL SEARCH MODELS
# ============================================================================

class SymbolSearchResult(BaseModel):
    """Result item for symbol search."""
    symbol: str = Field(..., description="Symbol")
    name: str = Field(..., description="Name")
    asset_class: AssetClass = Field(..., description="Asset type")
    exchange: Optional[str] = Field(None, description="Exchange")
    is_active: bool = Field(True, description="Is actively trading?")
    match_score: Optional[float] = Field(None, description="Fuzzy match score (0-1)")


class SymbolSearchRequest(BaseModel):
    """Request for symbol search."""
    query: str = Field(..., min_length=1, description="Search query")
    asset_class: Optional[AssetClass] = Field(None, description="Filter by asset class")
    limit: int = Field(20, ge=1, le=100, description="Max results")
    include_inactive: bool = Field(False, description="Include inactive symbols?")


# ============================================================================
# 9. CACHE STATISTICS MODELS
# ============================================================================

class CacheStatistics(BaseModel):
    """Statistics về symbol directory cache."""
    stocks_count: int = Field(0, description="Number of stocks in cache")
    crypto_count: int = Field(0, description="Number of cryptos in cache")
    symbol_changes_count: int = Field(0, description="Number of symbol changes")
    delisted_count: int = Field(0, description="Number of delisted companies")
    
    last_stock_sync: Optional[datetime] = Field(None, description="Last stock list sync")
    last_crypto_sync: Optional[datetime] = Field(None, description="Last crypto list sync")
    last_changes_sync: Optional[datetime] = Field(None, description="Last symbol changes sync")
    last_delisted_sync: Optional[datetime] = Field(None, description="Last delisted sync")
    
    cache_ttl_seconds: int = Field(..., description="Cache TTL in seconds")
    next_sync_in: Optional[int] = Field(None, description="Seconds until next sync")


# ============================================================================
# 10. SYNC JOB MODELS
# ============================================================================

class SyncJobResult(BaseModel):
    """Result of a sync job execution."""
    job_type: Literal["stock_list", "crypto_list", "symbol_changes", "delisted"] = Field(
        ..., description="Type of sync job"
    )
    success: bool = Field(..., description="Job succeeded?")
    records_fetched: int = Field(0, description="Number of records fetched from FMP")
    records_cached: int = Field(0, description="Number of records cached")
    duration_seconds: float = Field(..., description="Job execution time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    synced_at: datetime = Field(default_factory=datetime.utcnow, description="Sync timestamp")


class SyncJobsStatus(BaseModel):
    """Overall status of all sync jobs."""
    last_full_sync: Optional[datetime] = Field(None, description="Last complete sync")
    jobs: List[SyncJobResult] = Field(..., description="Individual job results")
    total_fmp_calls: int = Field(..., description="Total FMP API calls made")
    all_successful: bool = Field(..., description="All jobs succeeded?")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
"""
EXAMPLE DATA STRUCTURES:

1. Stock List Item:
{
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "price": 180.50,
    "exchange": "NASDAQ Global Select",
    "exchangeShortName": "NASDAQ",
    "type": "stock"
}

2. Symbol Change Item:
{
    "date": "2024-01-15",
    "name": "Facebook Inc.",
    "oldSymbol": "FB",
    "newSymbol": "META"
}

3. Delisted Company:
{
    "symbol": "TSLA_OLD",
    "companyName": "Tesla Motors",
    "exchange": "NASDAQ",
    "ipoDate": "2010-06-29",
    "delistedDate": "2023-05-15"
}

4. Symbol Status Detail:
{
    "input_symbol": "FB",
    "status": "renamed",
    "asset_class": "stock",
    "exists": true,
    "name": "Meta Platforms Inc.",
    "exchange": "NASDAQ",
    "is_actively_trading": true,
    "current_symbol": "META",
    "redirect_symbol": "META",
    "changed_on": "2024-01-15",
    "message": "Symbol renamed from FB to META on 2024-01-15"
}

5. Symbol Validation Result:
{
    "symbol": "AAPL",
    "is_valid": true,
    "asset_class": "stock",
    "exists_in_cache": true,
    "message": "Valid stock symbol"
}

6. Sync Job Result:
{
    "job_type": "stock_list",
    "success": true,
    "records_fetched": 15000,
    "records_cached": 15000,
    "duration_seconds": 2.5,
    "synced_at": "2025-01-18T10:00:00Z"
}
"""