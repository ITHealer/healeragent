from fastapi import APIRouter, Query, HTTPException, Depends, BackgroundTasks
from typing import Optional, List
from sqlalchemy.orm import Session

from src.utils.logger.custom_logging import LoggerMixin
from src.database import get_db_dependency

from src.models.symbol_directory import (
    SymbolValidationResult,
    BulkValidationRequest, BulkValidationResponse,
    CacheStatistics, SyncJobsStatus, AssetClass,
    SymbolStatusDetail
)

from src.services.data_providers.fmp.symbol_directory_service import get_symbol_directory_service
from src.models.equity import APIResponse, APIResponseData

router = APIRouter()
logger = LoggerMixin().logger


# ============================================================================
# 1. GET STOCK LIST
# ============================================================================
@router.get(
    "/stocks",
    response_model=APIResponse[dict],
    summary="Lấy danh sách stocks từ PostgreSQL",
    description="""
    Lấy danh sách stocks từ DB với pagination.
    
    **Performance:** < 50ms cho 1000 records (với proper indexing)
    **No API calls needed** - All from DB
    """
)
async def get_stocks_http(
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db_dependency)
):
    """Get stock list from DB"""
    service = get_symbol_directory_service(db)
    
    # Get from DB via repository
    stocks = service.repo.get_all_stocks(limit=limit, offset=offset)
    total_count = service.repo.count_stocks()
    
    if not stocks:
        raise HTTPException(
            status_code=404,
            detail="No stocks found in database. Please run sync job first."
        )
    
    # Build response
    response_data = {
        "total_count": total_count,
        "returned_count": len(stocks),
        "offset": offset,
        "limit": limit,
        "symbols": [
            {
                "symbol": stock.symbol,
                "name": stock.name,
                "exchange": stock.exchange,
                "price": stock.price
            }
            for stock in stocks
        ]
    }
    
    response_payload = APIResponseData[dict](data=[response_data])
    
    return APIResponse[dict](
        message=f"OK - {total_count} total stocks, returned {len(stocks)}",
        status="200",
        provider_used="database",
        data=response_payload
    )


# ============================================================================
# 2. GET CRYPTO LIST
# ============================================================================
@router.get(
    "/cryptos",
    response_model=APIResponse[dict],
    summary="Lấy danh sách cryptos từ PostgreSQL",
    description="Lấy danh sách cryptos từ DB với pagination"
)
async def get_cryptos_http(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db_dependency)
):
    """Get crypto list from DB"""
    service = get_symbol_directory_service(db)
    
    # Query from DB
    from src.database.models.symbol_directory import Cryptocurrency
    from sqlalchemy import select
    
    stmt = select(Cryptocurrency).offset(offset).limit(limit)
    cryptos = db.execute(stmt).scalars().all()
    
    total_count = service.repo.count_cryptos()
    
    if not cryptos:
        raise HTTPException(
            status_code=404,
            detail="No cryptos found in database"
        )
    
    response_data = {
        "total_count": total_count,
        "returned_count": len(cryptos),
        "offset": offset,
        "limit": limit,
        "symbols": [
            {
                "symbol": crypto.symbol,
                "name": crypto.name,
                "currency": crypto.currency
            }
            for crypto in cryptos
        ]
    }
    
    response_payload = APIResponseData[dict](data=[response_data])
    
    return APIResponse[dict](
        message=f"OK - {total_count} total cryptos, returned {len(cryptos)}",
        status="200",
        provider_used="database",
        data=response_payload
    )


# ============================================================================
# 3. VALIDATE SYMBOL
# ============================================================================
@router.get(
    "/validate/{symbol}",
    response_model=APIResponse[SymbolValidationResult],
    summary="Validate symbol existence",
    description="""
    Fast symbol validation với DB index lookup.
    
    **Performance:** < 5ms
    **No FMP API calls**
    """
)
async def validate_symbol_http(
    symbol: str,
    asset_class: Optional[AssetClass] = Query(None, description="Filter by asset class"),
    db: Session = Depends(get_db_dependency)
):
    """Validate symbol"""
    service = get_symbol_directory_service(db)
    
    result = service.validate_symbol(symbol, asset_class)
    
    response_payload = APIResponseData[SymbolValidationResult](data=[result])
    
    message = result.message
    status_code = "200" if result.is_valid else "404"
    
    return APIResponse[SymbolValidationResult](
        message=message,
        status=status_code,
        provider_used="database",
        data=response_payload
    )


# ============================================================================
# 4. GET SYMBOL STATUS
# ============================================================================
@router.get(
    "/status/{symbol}",
    response_model=APIResponse[SymbolStatusDetail],
    summary="Get symbol status detail",
    description="""
    Get comprehensive status:
    - Active/Inactive/Delisted/Renamed
    - Redirect information
    - Delisting date
    
    **Performance:** < 10ms
    """
)
async def get_symbol_status_http(
    symbol: str,
    db: Session = Depends(get_db_dependency)
):
    """Get symbol status"""
    service = get_symbol_directory_service(db)
    
    status = service.get_symbol_status(symbol)
    
    response_payload = APIResponseData[SymbolStatusDetail](data=[status])
    
    return APIResponse[SymbolStatusDetail](
        message=status.message or "OK",
        status="200",
        provider_used="database",
        data=response_payload
    )


# ============================================================================
# 5. BULK VALIDATION
# ============================================================================
@router.post(
    "/validate-bulk",
    response_model=APIResponse[BulkValidationResponse],
    summary="Bulk validate symbols",
    description="Validate up to 100 symbols at once"
)
async def bulk_validate_symbols_http(
    request: BulkValidationRequest,
    db: Session = Depends(get_db_dependency)
):
    """Bulk symbol validation"""
    service = get_symbol_directory_service(db)
    
    results = []
    valid_symbols = []
    invalid_symbols = []
    
    for symbol in request.symbols:
        result = service.validate_symbol(symbol, request.asset_class)
        results.append(result)
        
        if result.is_valid:
            valid_symbols.append(symbol)
        else:
            invalid_symbols.append(symbol)
    
    bulk_response = BulkValidationResponse(
        total_checked=len(request.symbols),
        valid_symbols=valid_symbols,
        invalid_symbols=invalid_symbols,
        details=results
    )
    
    response_payload = APIResponseData[BulkValidationResponse](data=[bulk_response])
    
    message = f"Validated {len(request.symbols)} symbols: {len(valid_symbols)} valid, {len(invalid_symbols)} invalid"
    
    return APIResponse[BulkValidationResponse](
        message=message,
        status="200",
        provider_used="database",
        data=response_payload
    )


# ============================================================================
# 6. SEARCH/AUTOCOMPLETE
# ============================================================================
@router.get(
    "/search",
    response_model=APIResponse[dict],
    summary="Search symbols (autocomplete)",
    description="""
    Search symbols by prefix for autocomplete.
    
    **Performance:** < 20ms với LIKE query + index
    **Use case:** Autocomplete dropdown
    """
)
async def search_symbols_http(
    q: str = Query(..., min_length=1, max_length=10, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Max results"),
    db: Session = Depends(get_db_dependency)
):
    """Search symbols"""
    service = get_symbol_directory_service(db)
    
    results = service.repo.search_symbols(q, limit)
    
    response_data = {
        "query": q,
        "total_results": len(results),
        "results": results
    }
    
    response_payload = APIResponseData[dict](data=[response_data])
    
    return APIResponse[dict](
        message=f"Found {len(results)} symbols matching '{q}'",
        status="200",
        provider_used="database",
        data=response_payload
    )


# ============================================================================
# 7. GET SYMBOL CHANGES
# ============================================================================
@router.get(
    "/symbol-changes",
    response_model=APIResponse[dict],
    summary="Get symbol changes from DB",
    description="Get list of symbol renames with pagination"
)
async def get_symbol_changes_http(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db_dependency)
):
    """Get symbol changes"""
    from src.database.models.symbol_directory import SymbolChange
    from sqlalchemy import select
    
    service = get_symbol_directory_service(db)
    
    # Query with pagination
    stmt = (
        select(SymbolChange)
        .order_by(SymbolChange.date.desc())
        .offset(offset)
        .limit(limit)
    )
    changes = db.execute(stmt).scalars().all()
    
    total_count = service.repo.count_symbol_changes()
    
    response_data = {
        "total_count": total_count,
        "returned_count": len(changes),
        "changes": [
            {
                "old_symbol": c.old_symbol,
                "new_symbol": c.new_symbol,
                "date": c.date.isoformat(),
                "name": c.name
            }
            for c in changes
        ]
    }
    
    response_payload = APIResponseData[dict](data=[response_data])
    
    return APIResponse[dict](
        message=f"OK - {total_count} total changes",
        status="200",
        provider_used="database",
        data=response_payload
    )


# ============================================================================
# 8. GET DELISTED COMPANIES
# ============================================================================
@router.get(
    "/delisted",
    response_model=APIResponse[dict],
    summary="Get delisted companies from DB"
)
async def get_delisted_companies_http(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db_dependency)
):
    """Get delisted companies"""
    from src.database.models.symbol_directory import DelistedCompany
    from sqlalchemy import select
    
    service = get_symbol_directory_service(db)
    
    stmt = (
        select(DelistedCompany)
        .order_by(DelistedCompany.delisted_date.desc())
        .offset(offset)
        .limit(limit)
    )
    delisted = db.execute(stmt).scalars().all()
    
    total_count = service.repo.count_delisted()
    
    response_data = {
        "total_count": total_count,
        "returned_count": len(delisted),
        "delisted": [
            {
                "symbol": d.symbol,
                "company_name": d.company_name,
                "delisted_date": d.delisted_date.isoformat() if d.delisted_date else None,
                "exchange": d.exchange
            }
            for d in delisted
        ]
    }
    
    response_payload = APIResponseData[dict](data=[response_data])
    
    return APIResponse[dict](
        message=f"OK - {total_count} delisted companies",
        status="200",
        provider_used="database",
        data=response_payload
    )


# ============================================================================
# 9. GET STATISTICS
# ============================================================================
@router.get(
    "/statistics",
    response_model=APIResponse[CacheStatistics],
    summary="Get DB statistics",
    description="Get statistics về symbol directory data trong DB"
)
async def get_statistics_http(
    db: Session = Depends(get_db_dependency)
):
    """Get statistics"""
    service = get_symbol_directory_service(db)
    
    stats = service.get_statistics()
    
    response_payload = APIResponseData[CacheStatistics](data=[stats])
    
    return APIResponse[CacheStatistics](
        message="OK",
        status="200",
        provider_used="database",
        data=response_payload
    )


# ============================================================================
# 10. MANUAL SYNC TRIGGER (Admin endpoint)
# ============================================================================
@router.post(
    "/sync",
    response_model=APIResponse[SyncJobsStatus],
    summary="Trigger manual sync from FMP (Admin)",
    description="""
    Manually trigger full sync from FMP to PostgreSQL.
    
    **Warning:** Costs 4 FMP API calls
    **Use case:** 
    - Bootstrap
    - Force refresh
    - Recovery
    """
)
async def trigger_manual_sync_http(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_dependency)
):
    """Trigger manual sync"""
    service = get_symbol_directory_service(db)
    
    logger.info("Manual sync triggered")
    
    # Run sync
    results = await service.sync_all()
    
    # Build response
    all_successful = all(r.success for r in results)
    total_calls = len(results)
    
    sync_status = SyncJobsStatus(
        last_full_sync=results[0].synced_at if results else None,
        jobs=results,
        total_fmp_calls=total_calls,
        all_successful=all_successful
    )
    
    response_payload = APIResponseData[SyncJobsStatus](data=[sync_status])
    
    message = "Sync completed successfully" if all_successful else "Sync completed with some failures"
    
    return APIResponse[SyncJobsStatus](
        message=message,
        status="200",
        provider_used="fmp",
        data=response_payload
    )


# ============================================================================
# 11. HEALTH CHECK
# ============================================================================
@router.get(
    "/health",
    summary="Health check",
    description="Check symbol directory DB health"
)
async def health_check_http(
    db: Session = Depends(get_db_dependency)
):
    """Health check"""
    service = get_symbol_directory_service(db)
    
    try:
        stats = service.get_statistics()
        
        health = {
            "status": "healthy",
            "database_connected": True,
            "stocks_count": stats.stocks_count,
            "cryptos_count": stats.crypto_count,
            "last_stock_sync": stats.last_stock_sync.isoformat() if stats.last_stock_sync else None,
            "data_available": stats.stocks_count > 0 and stats.crypto_count > 0
        }
        
        return health
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }