from fastapi import APIRouter, Query, HTTPException, Depends
from typing import Optional, List
import aioredis

from src.utils.config import settings
from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.redis_cache import get_redis_client, get_cache, set_cache

from src.models.company_search import (
    StockSymbolSearchItem,
    CompanyNameSearchItem,
    CIKSearchItem,
    CUSIPSearchItem,
    ISINSearchItem,
    StockScreenerItem,
    ExchangeVariantItem,
    StockScreenerFilters
)

from src.services.data_providers.fmp.company_search_service import get_company_search_service
from src.models.equity import APIResponse, APIResponseData


# ============================================================================
# ROUTER SETUP
# ============================================================================
router = APIRouter()

logger = LoggerMixin().logger
# Get service instance
company_search_service = get_company_search_service()


# ============================================================================
# 1. STOCK SYMBOL SEARCH
# ============================================================================
@router.get("/symbol",
    response_model=APIResponse[StockSymbolSearchItem],
    summary="Search by stock symbol",
    description="""
    Search stocks by symbol or partial symbol match.

    **Examples:**
    - `query=AAPL` → Find Apple Inc.
    - `query=AA` → Find all symbols starting with AA
    - `query=AAPL&exchange=NASDAQ` → Filter by exchange
    """
)
async def search_by_symbol_http(
    query: str = Query(
        ...,
        min_length=1,
        max_length=10,
        description="Stock symbol or partial symbol (e.g., AAPL, MSFT)"
    ),
    limit: Optional[int] = Query(
        10,
        ge=1,
        le=100,
        description="Số lượng kết quả tối đa"
    ),
    exchange: Optional[str] = Query(
        None,
        description="Filter theo exchange (e.g., NASDAQ, NYSE, AMEX)"
    ),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """Search stocks by symbol."""
    # Build cache key
    cache_key_parts = ["company_search_symbol", query.lower()]
    if limit:
        cache_key_parts.append(f"limit_{limit}")
    if exchange:
        cache_key_parts.append(f"exchange_{exchange.lower()}")
    cache_key = "_".join(cache_key_parts)
    
    logger.debug(f"Symbol Search: query='{query}', limit={limit}, exchange={exchange}")
    
    # Try cache first
    cached_data: Optional[APIResponse[StockSymbolSearchItem]] = await get_cache(
        redis_client, cache_key, APIResponse[StockSymbolSearchItem]
    )
    if cached_data and cached_data.data:
        logger.info(f"Symbol Search: Cache HIT for {cache_key}")
        cached_data.message = "OK (cached)"
        cached_data.provider_used = "fmp_cached"
        return cached_data
    
    logger.info(f"Symbol Search: Cache MISS for {cache_key}. Fetching from FMP.")
    
    # Fetch from service
    results = await company_search_service.search_by_symbol(
        query=query,
        limit=limit,
        exchange=exchange
    )
    
    if results is None:
        logger.error(f"Symbol Search: Failed to fetch data for '{query}'")
        raise HTTPException(
            status_code=502,
            detail=f"Không thể tìm kiếm symbol '{query}' từ FMP"
        )
    
    # Build response
    response_data_payload = APIResponseData[StockSymbolSearchItem](data=results)
    message = "OK" if results else f"Không tìm thấy kết quả cho '{query}'"
    
    api_response = APIResponse[StockSymbolSearchItem](
        message=message,
        status="200",
        provider_used="fmp",
        data=response_data_payload
    )
    
    # Cache the result
    await set_cache(
        redis_client,
        cache_key,
        api_response,
        expiry=settings.CACHE_TTL_LISTS  # Use lists TTL (usually 1 day)
    )
    logger.info(f"Symbol Search: Cached {len(results)} results for {cache_key}")
    
    return api_response


# ============================================================================
# 2. COMPANY NAME SEARCH
# ============================================================================
@router.get(
    "/name",
    response_model=APIResponse[CompanyNameSearchItem],
    summary="Tìm kiếm theo tên công ty",
    description="""
    Tìm kiếm companies bằng tên hoặc partial name match.
    
    **Examples:**
    - `query=Apple` → Tìm Apple Inc.
    - `query=Microsoft Corp` → Tìm Microsoft
    - `query=Bank&limit=20` → Top 20 banks
    """
)
async def search_by_name_http(
    query: str = Query(
        ...,
        min_length=2,
        description="Company name hoặc partial name"
    ),
    limit: Optional[int] = Query(
        10,
        ge=1,
        le=100,
        description="Số lượng kết quả tối đa"
    ),
    exchange: Optional[str] = Query(
        None,
        description="Filter theo exchange"
    ),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """Search companies by name."""
    cache_key = f"company_search_name_{query.lower()}_limit_{limit}_{exchange or 'all'}"
    logger.debug(f"Name Search: query='{query}', limit={limit}")
    
    # Try cache
    cached_data: Optional[APIResponse[CompanyNameSearchItem]] = await get_cache(
        redis_client, cache_key, APIResponse[CompanyNameSearchItem]
    )
    if cached_data and cached_data.data:
        logger.info(f"Name Search: Cache HIT for {cache_key}")
        cached_data.message = "OK (cached)"
        cached_data.provider_used = "fmp_cached"
        return cached_data
    
    logger.info(f"Name Search: Cache MISS. Fetching from FMP.")
    
    # Fetch from service
    results = await company_search_service.search_by_name(
        query=query,
        limit=limit,
        exchange=exchange
    )
    
    if results is None:
        logger.error(f"Name Search: Failed for '{query}'")
        raise HTTPException(
            status_code=502,
            detail=f"Không thể tìm kiếm tên '{query}' từ FMP"
        )
    
    # Build response
    response_data_payload = APIResponseData[CompanyNameSearchItem](data=results)
    api_response = APIResponse[CompanyNameSearchItem](
        message="OK" if results else f"Không tìm thấy kết quả cho '{query}'",
        status="200",
        provider_used="fmp",
        data=response_data_payload
    )
    
    # Cache
    await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_LISTS)
    logger.info(f"Name Search: Cached {len(results)} results")
    
    return api_response


# ============================================================================
# 3. CIK SEARCH
# ============================================================================
@router.get(
    "/cik",
    response_model=APIResponse[CIKSearchItem],
    summary="Tìm kiếm theo CIK number",
    description="""
    Tìm kiếm company bằng CIK (Central Index Key) từ SEC.
    
    **Example:**
    - `cik=320193` → Apple Inc.
    - `cik=0000320193` → Apple Inc. (với leading zeros)
    """
)
async def search_by_cik_http(
    cik: str = Query(
        ...,
        min_length=1,
        max_length=10,
        description="CIK number (có thể có leading zeros)"
    ),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """Search company by CIK."""
    # Normalize CIK for cache key
    cik_normalized = cik.lstrip("0")
    cache_key = f"company_search_cik_{cik_normalized}"
    
    logger.debug(f"CIK Search: cik='{cik}'")
    
    # Try cache
    cached_data: Optional[APIResponse[CIKSearchItem]] = await get_cache(
        redis_client, cache_key, APIResponse[CIKSearchItem]
    )
    if cached_data and cached_data.data:
        logger.info(f"CIK Search: Cache HIT for {cache_key}")
        cached_data.message = "OK (cached)"
        cached_data.provider_used = "fmp_cached"
        return cached_data
    
    logger.info(f"CIK Search: Cache MISS. Fetching from FMP.")
    
    # Fetch from service
    results = await company_search_service.search_by_cik(cik)
    
    # if results is None:
    #     logger.error(f"CIK Search: Failed for CIK '{cik}'")
    #     raise HTTPException(
    #         status_code=502,
    #         detail=f"Không thể tìm kiếm CIK '{cik}' từ FMP"
    #     )
    
    if not results or results is None:
        logger.info(f"CIK Search: Not found for CIK '{cik}'")
        message = f"Không tìm thấy company với CIK '{cik}'"
        api_response = APIResponse[CIKSearchItem](
            message=message,
            status="404", 
            provider_used="fmp",
            data=APIResponseData[CIKSearchItem](data=[])
        )
        await set_cache(
            redis_client,
            cache_key,
            api_response,
            expiry=settings.CACHE_TTL_LISTS * 1
        )
        logger.info(f"CIK Search: Cached result for CIK {cik} (404)")
        return api_response
    
    response_data_payload = APIResponseData[CIKSearchItem](data=results)
    api_response = APIResponse[CIKSearchItem](
        message="OK",
        status="200",
        provider_used="fmp",
        data=response_data_payload
    )
    
    await set_cache(
        redis_client,
        cache_key,
        api_response,
        expiry=settings.CACHE_TTL_LISTS * 1
    )
    logger.info(f"CIK Search: Cached result for CIK {cik}")
    
    return api_response


# ============================================================================
# 4. CUSIP SEARCH
# ============================================================================
@router.get(
    "/cusip",
    response_model=APIResponse[CUSIPSearchItem],
    summary="Tìm kiếm theo CUSIP identifier",
    description="""
    Tìm kiếm security bằng CUSIP (9-character identifier).
    
    **Example:**
    - `cusip=037833100` → Apple Inc.
    """
)
async def search_by_cusip_http(
    cusip: str = Query(
        ...,
        min_length=9,
        max_length=9,
        description="9-character CUSIP identifier"
    ),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """Search security by CUSIP."""
    cache_key = f"company_search_cusip_{cusip}"
    logger.debug(f"CUSIP Search: cusip='{cusip}'")
    
    # Validate format
    if not cusip.isalnum():
        raise HTTPException(
            status_code=400,
            detail="CUSIP phải là 9 ký tự alphanumeric"
        )
    
    # Try cache
    cached_data: Optional[APIResponse[CUSIPSearchItem]] = await get_cache(
        redis_client, cache_key, APIResponse[CUSIPSearchItem]
    )
    if cached_data and cached_data.data:
        logger.info(f"CUSIP Search: Cache HIT")
        cached_data.message = "OK (cached)"
        cached_data.provider_used = "fmp_cached"
        return cached_data
    
    logger.info(f"CUSIP Search: Cache MISS. Fetching from FMP.")
    
    # Fetch from service
    results = await company_search_service.search_by_cusip(cusip)
    
    if results is None:
        logger.error(f"CUSIP Search: Failed for '{cusip}'")
        raise HTTPException(
            status_code=502,
            detail=f"Không thể tìm kiếm CUSIP '{cusip}' từ FMP"
        )
    
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy security với CUSIP '{cusip}'"
        )
    
    # Build response
    response_data_payload = APIResponseData[CUSIPSearchItem](data=results)
    api_response = APIResponse[CUSIPSearchItem](
        message="OK",
        status="200",
        provider_used="fmp",
        data=response_data_payload
    )
    
    # Cache
    await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_LISTS * 7)
    logger.info(f"CUSIP Search: Cached result")
    
    return api_response


# ============================================================================
# 5. ISIN SEARCH
# ============================================================================
@router.get(
    "/isin",
    response_model=APIResponse[ISINSearchItem],
    summary="Tìm kiếm theo ISIN identifier",
    description="""
    Tìm kiếm security bằng ISIN (12-character international identifier).
    
    **Example:**
    - `isin=US0378331005` → Apple Inc.
    """
)
async def search_by_isin_http(
    isin: str = Query(
        ...,
        min_length=12,
        max_length=12,
        description="12-character ISIN identifier"
    ),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """Search security by ISIN."""
    cache_key = f"company_search_isin_{isin}"
    logger.debug(f"ISIN Search: isin='{isin}'")
    
    # Validate format (2 letters + 10 alphanumeric)
    if not (isin[:2].isalpha() and isin[2:].isalnum()):
        raise HTTPException(
            status_code=400,
            detail="ISIN phải bắt đầu bằng 2 chữ cái và theo sau là 10 ký tự alphanumeric"
        )
    
    # Try cache
    cached_data: Optional[APIResponse[ISINSearchItem]] = await get_cache(
        redis_client, cache_key, APIResponse[ISINSearchItem]
    )
    if cached_data and cached_data.data:
        logger.info(f"ISIN Search: Cache HIT")
        cached_data.message = "OK (cached)"
        cached_data.provider_used = "fmp_cached"
        return cached_data
    
    logger.info(f"ISIN Search: Cache MISS. Fetching from FMP.")
    
    # Fetch from service
    results = await company_search_service.search_by_isin(isin)
    
    if results is None:
        logger.error(f"ISIN Search: Failed for '{isin}'")
        raise HTTPException(
            status_code=502,
            detail=f"Không thể tìm kiếm ISIN '{isin}' từ FMP"
        )
    
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy security với ISIN '{isin}'"
        )
    
    # Build response
    response_data_payload = APIResponseData[ISINSearchItem](data=results)
    api_response = APIResponse[ISINSearchItem](
        message="OK",
        status="200",
        provider_used="fmp",
        data=response_data_payload
    )
    
    # Cache
    await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_LISTS * 7)
    logger.info(f"ISIN Search: Cached result")
    
    return api_response


# ============================================================================
# 6. STOCK SCREENER
# ============================================================================
@router.get(
    "/screener",
    response_model=APIResponse[StockScreenerItem],
    summary="Stock screener với multiple filters",
    description="""
    Filter stocks theo nhiều tiêu chí như market cap, price, volume, sector, etc.
    
    **Examples:**
    - `marketCapMoreThan=1000000000&sector=Technology` → Tech stocks > $1B
    - `priceMoreThan=50&priceLowerThan=200&isActivelyTrading=true` → Stocks $50-$200
    - `sector=Energy&country=US&limit=50` → Top 50 US energy stocks
    """
)
async def screen_stocks_http(
    market_cap_more_than: Optional[float] = Query(None, alias="marketCapMoreThan"),
    market_cap_lower_than: Optional[float] = Query(None, alias="marketCapLowerThan"),
    price_more_than: Optional[float] = Query(None, alias="priceMoreThan"),
    price_lower_than: Optional[float] = Query(None, alias="priceLowerThan"),
    beta_more_than: Optional[float] = Query(None, alias="betaMoreThan"),
    beta_lower_than: Optional[float] = Query(None, alias="betaLowerThan"),
    volume_more_than: Optional[int] = Query(None, alias="volumeMoreThan"),
    volume_lower_than: Optional[int] = Query(None, alias="volumeLowerThan"),
    dividend_more_than: Optional[float] = Query(None, alias="dividendMoreThan"),
    dividend_lower_than: Optional[float] = Query(None, alias="dividendLowerThan"),
    is_etf: Optional[bool] = Query(None, alias="isEtf"),
    is_actively_trading: Optional[bool] = Query(None, alias="isActivelyTrading"),
    sector: Optional[str] = Query(None),
    industry: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    exchange: Optional[str] = Query(None),
    limit: Optional[int] = Query(100, ge=1, le=10000),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """Screen stocks with multiple criteria."""
    # Build cache key from all non-None parameters
    cache_parts = ["company_search_screener"]
    params_dict = {
        "mcMoreThan": market_cap_more_than,
        "mcLowerThan": market_cap_lower_than,
        "priceMore": price_more_than,
        "priceLower": price_lower_than,
        "betaMore": beta_more_than,
        "betaLower": beta_lower_than,
        "volMore": volume_more_than,
        "volLower": volume_lower_than,
        "divMore": dividend_more_than,
        "divLower": dividend_lower_than,
        "isEtf": is_etf,
        "isActive": is_actively_trading,
        "sector": sector,
        "industry": industry,
        "country": country,
        "exchange": exchange,
        "limit": limit
    }
    
    for key, value in params_dict.items():
        if value is not None:
            cache_parts.append(f"{key}_{str(value).lower()}")
    
    cache_key = "_".join(cache_parts)
    logger.debug(f"Screener: {len([v for v in params_dict.values() if v is not None])} filters applied")
    
    # Try cache
    cached_data: Optional[APIResponse[StockScreenerItem]] = await get_cache(
        redis_client, cache_key, APIResponse[StockScreenerItem]
    )
    if cached_data and cached_data.data:
        logger.info(f"Screener: Cache HIT")
        cached_data.message = "OK (cached)"
        cached_data.provider_used = "fmp_cached"
        return cached_data
    
    logger.info(f"Screener: Cache MISS. Fetching from FMP.")
    
    # Build filters
    filters = StockScreenerFilters(
        market_cap_more_than=market_cap_more_than,
        market_cap_lower_than=market_cap_lower_than,
        price_more_than=price_more_than,
        price_lower_than=price_lower_than,
        beta_more_than=beta_more_than,
        beta_lower_than=beta_lower_than,
        volume_more_than=volume_more_than,
        volume_lower_than=volume_lower_than,
        dividend_more_than=dividend_more_than,
        dividend_lower_than=dividend_lower_than,
        is_etf=is_etf,
        is_actively_trading=is_actively_trading,
        sector=sector,
        industry=industry,
        country=country,
        exchange=exchange,
        limit=limit
    )
    
    # Fetch from service
    results = await company_search_service.screen_stocks(filters=filters)
    
    if results is None:
        logger.error(f"Screener: Failed to fetch data")
        raise HTTPException(
            status_code=502,
            detail="Không thể thực hiện stock screening từ FMP"
        )
    
    # Build response
    response_data_payload = APIResponseData[StockScreenerItem](data=results)
    message = "OK" if results else "Không tìm thấy stocks phù hợp với criteria"
    
    api_response = APIResponse[StockScreenerItem](
        message=message,
        status="200",
        provider_used="fmp",
        data=response_data_payload
    )
    
    # Cache (shorter TTL cho screener vì data thay đổi nhanh)
    await set_cache(
        redis_client,
        cache_key,
        api_response,
        expiry=settings.CACHE_TTL_CHART  # Use chart TTL (usually shorter)
    )
    logger.info(f"Screener: Cached {len(results)} results")
    
    return api_response


# ============================================================================
# 7. EXCHANGE VARIANTS
# ============================================================================
@router.get(
    "/exchange-variants",
    response_model=APIResponse[ExchangeVariantItem],
    summary="Tìm symbol trên tất cả exchanges",
    description="""
    Tìm tất cả exchanges mà một symbol được list.
    
    **Example:**
    - `symbol=AAPL` → AAPL trên NASDAQ, XETRA, etc.
    """
)
async def search_exchange_variants_http(
    symbol: str = Query(
        ...,
        min_length=1,
        max_length=10,
        description="Stock symbol (e.g., AAPL)"
    ),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """Search all exchanges where symbol is listed."""
    cache_key = f"company_search_exchange_variants_{symbol.upper()}"
    logger.debug(f"Exchange Variants: symbol='{symbol}'")
    
    # Try cache
    cached_data: Optional[APIResponse[ExchangeVariantItem]] = await get_cache(
        redis_client, cache_key, APIResponse[ExchangeVariantItem]
    )
    if cached_data and cached_data.data:
        logger.info(f"Exchange Variants: Cache HIT")
        cached_data.message = "OK (cached)"
        cached_data.provider_used = "fmp_cached"
        return cached_data
    
    logger.info(f"Exchange Variants: Cache MISS. Fetching from FMP.")
    
    # Fetch from service
    results = await company_search_service.search_exchange_variants(symbol)
    
    if results is None:
        logger.error(f"Exchange Variants: Failed for '{symbol}'")
        raise HTTPException(
            status_code=502,
            detail=f"Không thể tìm exchange variants cho '{symbol}' từ FMP"
        )
    
    # Build response
    response_data_payload = APIResponseData[ExchangeVariantItem](data=results)
    message = "OK" if results else f"Không tìm thấy '{symbol}' trên bất kỳ exchange nào"
    
    api_response = APIResponse[ExchangeVariantItem](
        message=message,
        status="200",
        provider_used="fmp",
        data=response_data_payload
    )
    
    # Cache
    await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_LISTS)
    logger.info(f"Exchange Variants: Cached {len(results)} variants")
    
    return api_response


# ============================================================================
# 8. UNIFIED SEARCH (Convenience Endpoint)
# ============================================================================
@router.get(
    "/unified",
    response_model=APIResponse[dict],  # Generic response type
    summary="Unified search endpoint (auto-detect type)",
    description="""
    Smart search tự động detect search type dựa vào query format.
    
    **Auto-detection:**
    - 9 alphanumeric chars → CUSIP search
    - 12 chars (2 letters + 10 alphanumeric) → ISIN search  
    - All digits → CIK search
    - 1-5 uppercase chars → Symbol search
    - Other → Name search
    
    **Examples:**
    - `query=AAPL` → Symbol search
    - `query=Apple Inc` → Name search
    - `query=320193` → CIK search
    - `query=037833100` → CUSIP search
    """
)
async def unified_search_http(
    query: str = Query(..., min_length=1, description="Search query"),
    search_type: str = Query(
        "auto",
        description="Search type: auto, symbol, name, cik, cusip, isin"
    ),
    limit: Optional[int] = Query(10, ge=1, le=100),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """Unified search with auto-detection."""
    cache_key = f"company_search_unified_{query.lower()}_{search_type}_limit_{limit}"
    logger.debug(f"Unified Search: query='{query}', type={search_type}")
    
    # Try cache
    cached_data = await get_cache(redis_client, cache_key, APIResponse[dict])
    if cached_data and cached_data.data:
        logger.info(f"Unified Search: Cache HIT")
        cached_data.message = "OK (cached)"
        cached_data.provider_used = "fmp_cached"
        return cached_data
    
    logger.info(f"Unified Search: Cache MISS. Fetching from FMP.")
    
    # Fetch from service
    results = await company_search_service.unified_search(
        query=query,
        search_type=search_type,
        limit=limit
    )
    
    if results is None:
        logger.error(f"Unified Search: Failed for '{query}'")
        raise HTTPException(
            status_code=502,
            detail=f"Không thể tìm kiếm '{query}' từ FMP"
        )
    
    # Build response (generic dict type)
    response_data_payload = APIResponseData[dict](
        data=[item.model_dump() for item in results]
    )
    
    api_response = APIResponse[dict](
        message="OK" if results else f"Không tìm thấy kết quả cho '{query}'",
        status="200",
        provider_used="fmp",
        data=response_data_payload
    )
    
    # Cache
    await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_LISTS)
    logger.info(f"Unified Search: Cached {len(results)} results")
    
    return api_response