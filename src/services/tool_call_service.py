"""
ToolCallService - FMP API calls with Redis caching

This service provides cached access to FMP API endpoints.
Caching reduces API calls, improves response time, and prevents rate limiting.

Cache TTL Strategy:
- Quote: 30s (real-time price data)
- Key Metrics: 5 min (changes with price)
- Key Metrics TTM: 5 min (rolling metrics)
- Financials (Income/Balance/CashFlow): 15 min (rarely changes)
- Growth Data: 15 min (historical)
- Analyst Estimates: 30 min (updated periodically)
- Financial Ratios: 15 min (calculated from financials)
"""

import logging
import json
import hashlib
from typing import Any, Dict, List, Optional

import httpx

from src.models.equity import (
    AnalystEstimateItem,
    FinancialStatementGrowthItem,
    KeyMetricsTTMItem,
    FinancialRatiosItem
)
from src.utils.logger.set_up_log_dataFMP import setup_logger
from src.utils.config import settings
from src.helpers.redis_cache import get_redis_client_llm

logger = setup_logger(__name__, log_level=logging.INFO)

# FMP API Configuration
FMP_API_KEY = settings.FMP_API_KEY
BASE_FMP_URL = settings.BASE_FMP_URL
FMP_URL_STABLE = settings.FMP_URL_STABLE

# HTTP Client Configuration
HTTP_TIMEOUT = 20.0


# =============================================================================
# CACHING UTILITIES
# =============================================================================

def _make_cache_key(endpoint: str, symbol: str, params: Optional[Dict] = None) -> str:
    """
    Generate a unique cache key for FMP API requests.

    Format: fmp:{endpoint}:{symbol}:{params_hash}

    Args:
        endpoint: API endpoint name (e.g., 'key-metrics', 'income-statement')
        symbol: Stock symbol
        params: Additional parameters (period, limit, etc.)

    Returns:
        Cache key string
    """
    symbol_upper = symbol.upper()

    if params:
        # Sort params for consistent hashing, exclude apikey
        filtered_params = {k: v for k, v in sorted(params.items()) if k != 'apikey'}
        params_str = json.dumps(filtered_params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return f"fmp:{endpoint}:{symbol_upper}:{params_hash}"

    return f"fmp:{endpoint}:{symbol_upper}"


async def _get_cached_data(cache_key: str) -> Optional[Any]:
    """
    Get data from Redis cache.

    Args:
        cache_key: Cache key to retrieve

    Returns:
        Cached data (parsed JSON) or None if not found
    """
    try:
        redis = await get_redis_client_llm()
        if redis is None:
            return None

        cached = await redis.get(cache_key)
        if cached:
            logger.info(f"[CACHE HIT] {cache_key}")
            # Handle both string and bytes
            if isinstance(cached, bytes):
                cached = cached.decode('utf-8')
            return json.loads(cached)

        logger.debug(f"[CACHE MISS] {cache_key}")
        return None

    except Exception as e:
        logger.warning(f"[CACHE ERROR] Get failed for {cache_key}: {e}")
        return None


async def _set_cached_data(cache_key: str, data: Any, ttl: int) -> bool:
    """
    Set data in Redis cache.

    Args:
        cache_key: Cache key
        data: Data to cache (will be JSON serialized)
        ttl: Time-to-live in seconds

    Returns:
        True if successful, False otherwise
    """
    if data is None:
        return False

    try:
        redis = await get_redis_client_llm()
        if redis is None:
            return False

        json_data = json.dumps(data, default=str)
        await redis.set(cache_key, json_data, ex=ttl)
        logger.info(f"[CACHE SET] {cache_key} (TTL: {ttl}s)")
        return True

    except Exception as e:
        logger.warning(f"[CACHE ERROR] Set failed for {cache_key}: {e}")
        return False


# =============================================================================
# FMP API HTTP CLIENT
# =============================================================================

async def _fetch_fmp_data(
    url: str,
    params: Dict[str, Any],
    timeout: float = HTTP_TIMEOUT
) -> Optional[Any]:
    """
    Make HTTP request to FMP API with error handling.

    Args:
        url: Full API URL
        params: Query parameters
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response or None on error
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(
                f"[FMP HTTP Error] {url}: {e.response.status_code} - "
                f"{e.response.text[:200]}"
            )
        except httpx.RequestError as e:
            logger.error(f"[FMP Request Error] {url}: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"[FMP JSON Error] {url}: {e}")
        except Exception as e:
            logger.exception(f"[FMP Unknown Error] {url}: {e}")

    return None


# =============================================================================
# TOOL CALL SERVICE CLASS
# =============================================================================

class ToolCallService:
    """
    Service for making FMP API calls with Redis caching.

    All methods are cached to reduce API calls and improve performance.
    Cache TTLs are configured in settings based on data update frequency.
    """

    # =========================================================================
    # FINANCIAL STATEMENT GROWTH
    # =========================================================================
    async def get_financial_statement_growth(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 20
    ) -> Optional[List[FinancialStatementGrowthItem]]:
        """
        Get financial report growth data from FMP.

        Cached for 15 minutes (CACHE_TTL_FMP_GROWTH).
        """
        if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error(f"FMP API Key not configured for {symbol}")
            return None

        # Check cache first
        cache_params = {"period": period, "limit": limit}
        cache_key = _make_cache_key("financial-growth", symbol, cache_params)

        cached_data = await _get_cached_data(cache_key)
        if cached_data is not None:
            # Parse cached data to models
            return [FinancialStatementGrowthItem(**item) for item in cached_data]

        # Fetch from API
        endpoint = f"/v3/financial-growth/{symbol.upper()}"
        params = {"period": period, "limit": limit, "apikey": FMP_API_KEY}
        url = f"{BASE_FMP_URL}{endpoint}"

        raw_data = await _fetch_fmp_data(url, params)

        if raw_data is None or not isinstance(raw_data, list):
            return None

        if not raw_data:
            logger.info(f"No growth data for {symbol}")
            return []

        # Parse to models
        parsed_list: List[FinancialStatementGrowthItem] = []
        for item in raw_data:
            try:
                parsed_list.append(FinancialStatementGrowthItem(**item))
            except Exception as e:
                logger.warning(f"Parse error for growth item {symbol}: {e}")

        # Cache the raw data (not models) for serialization
        await _set_cached_data(cache_key, raw_data, settings.CACHE_TTL_FMP_GROWTH)

        logger.info(f"Fetched {len(parsed_list)} growth records for {symbol}")
        return parsed_list

    # =========================================================================
    # FINANCIAL RATIOS
    # =========================================================================
    async def get_financial_ratios(
        self,
        symbol: str,
        limit: Optional[int] = 10,
        period: Optional[str] = "annual",
    ) -> Optional[List[FinancialRatiosItem]]:
        """
        Get Financial Ratios from FMP.

        Cached for 15 minutes (CACHE_TTL_FMP_RATIOS).
        """
        if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error(f"FMP API Key not configured for {symbol}")
            return None

        # Check cache first
        cache_params = {"limit": limit, "period": period}
        cache_key = _make_cache_key("ratios", symbol, cache_params)

        cached_data = await _get_cached_data(cache_key)
        if cached_data is not None:
            return [FinancialRatiosItem(**item) for item in cached_data]

        # Fetch from API
        endpoint = "/ratios"
        url = f"{FMP_URL_STABLE}{endpoint}"
        params: Dict[str, Any] = {"symbol": symbol.upper(), "apikey": FMP_API_KEY}

        if limit is not None:
            params["limit"] = limit
        if period:
            params["period"] = period

        raw_data = await _fetch_fmp_data(url, params)

        if raw_data is None or not isinstance(raw_data, list) or not raw_data:
            return None

        # Parse to models
        parsed_items: List[FinancialRatiosItem] = []
        for item in raw_data:
            if isinstance(item, dict):
                parsed_items.append(FinancialRatiosItem(**item))

        if not parsed_items:
            return None

        # Cache raw data
        await _set_cached_data(cache_key, raw_data, settings.CACHE_TTL_FMP_RATIOS)

        logger.info(f"Fetched {len(parsed_items)} ratio records for {symbol}")
        return parsed_items

    # =========================================================================
    # KEY METRICS TTM
    # =========================================================================
    async def get_key_metrics_ttm(
        self,
        symbol: str,
    ) -> Optional[KeyMetricsTTMItem]:
        """
        Get Key Metrics TTM from FMP.

        Cached for 5 minutes (CACHE_TTL_FMP_KEY_METRICS_TTM).
        """
        if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error(f"FMP API Key not configured for {symbol}")
            return None

        # Check cache first
        cache_key = _make_cache_key("key-metrics-ttm", symbol)

        cached_data = await _get_cached_data(cache_key)
        if cached_data is not None:
            return KeyMetricsTTMItem(**cached_data)

        # Fetch from API
        endpoint = f"/v3/key-metrics-ttm/{symbol.upper()}"
        params = {"apikey": FMP_API_KEY}
        url = f"{BASE_FMP_URL}{endpoint}"

        raw_data = await _fetch_fmp_data(url, params)

        if raw_data is None or not isinstance(raw_data, list) or not raw_data:
            return None

        # Get first item (TTM endpoint returns list with 1 item)
        item = raw_data[0]
        if not isinstance(item, dict):
            return None

        # Cache the single item
        await _set_cached_data(cache_key, item, settings.CACHE_TTL_FMP_KEY_METRICS_TTM)

        logger.info(f"Fetched key metrics TTM for {symbol}")
        return KeyMetricsTTMItem(**item)

    # =========================================================================
    # ANALYST ESTIMATES
    # =========================================================================
    async def get_analyst_estimates(
        self,
        symbol: str,
        period: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Optional[List[AnalystEstimateItem]]:
        """
        Get Analyst Estimates from FMP.

        Cached for 30 minutes (CACHE_TTL_FMP_ANALYST).
        """
        if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error(f"FMP API Key not configured for {symbol}")
            return None

        # Check cache first
        cache_params = {"period": period, "limit": limit}
        cache_key = _make_cache_key("analyst-estimates", symbol, cache_params)

        cached_data = await _get_cached_data(cache_key)
        if cached_data is not None:
            return [AnalystEstimateItem(**item) for item in cached_data]

        # Fetch from API
        endpoint = f"/v3/analyst-estimates/{symbol.upper()}"
        params: Dict[str, Any] = {"apikey": FMP_API_KEY}
        if period:
            params["period"] = period
        if limit:
            params["limit"] = limit

        url = f"{BASE_FMP_URL}{endpoint}"

        raw_data = await _fetch_fmp_data(url, params)

        if raw_data is None or not isinstance(raw_data, list):
            return None

        if not raw_data:
            return []

        # Parse to models
        parsed_list: List[AnalystEstimateItem] = []
        for item in raw_data:
            try:
                parsed_list.append(AnalystEstimateItem(**item))
            except Exception as e:
                logger.warning(f"Parse error for analyst estimate {symbol}: {e}")

        # Cache raw data
        await _set_cached_data(cache_key, raw_data, settings.CACHE_TTL_FMP_ANALYST)

        logger.info(f"Fetched {len(parsed_list)} analyst estimates for {symbol}")
        return parsed_list

    # =========================================================================
    # KEY METRICS (P/E, P/B, ROE, ROA, beta, etc.)
    # =========================================================================
    async def get_key_metrics(
        self,
        symbol: str,
        limit: int = 1
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get key metrics including P/E, P/B, ROE, ROA, beta, etc.

        Cached for 5 minutes (CACHE_TTL_FMP_KEY_METRICS).
        """
        if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error("FMP API Key not configured")
            return None

        # Check cache first
        cache_params = {"limit": limit}
        cache_key = _make_cache_key("key-metrics", symbol, cache_params)

        cached_data = await _get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        endpoint = f"/v3/key-metrics/{symbol.upper()}"
        params = {"limit": limit, "apikey": FMP_API_KEY}
        url = f"{BASE_FMP_URL}{endpoint}"

        raw_data = await _fetch_fmp_data(url, params)

        if raw_data is None or not isinstance(raw_data, list):
            return None

        # Cache raw data
        await _set_cached_data(cache_key, raw_data, settings.CACHE_TTL_FMP_KEY_METRICS)

        logger.info(f"Fetched {len(raw_data)} key metrics for {symbol}")
        return raw_data

    # =========================================================================
    # INCOME STATEMENT
    # =========================================================================
    async def get_income_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get Income Statement to calculate margins.

        Cached for 15 minutes (CACHE_TTL_FMP_FINANCIALS).
        """
        # Check cache first
        cache_params = {"period": period, "limit": limit}
        cache_key = _make_cache_key("income-statement", symbol, cache_params)

        cached_data = await _get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        endpoint = f"/v3/income-statement/{symbol.upper()}"
        params = {"period": period, "limit": limit, "apikey": FMP_API_KEY}
        url = f"{BASE_FMP_URL}{endpoint}"

        raw_data = await _fetch_fmp_data(url, params)

        if raw_data is None:
            return None

        # Cache raw data
        await _set_cached_data(cache_key, raw_data, settings.CACHE_TTL_FMP_FINANCIALS)

        logger.info(f"Fetched income statement for {symbol}")
        return raw_data

    # =========================================================================
    # BALANCE SHEET
    # =========================================================================
    async def get_balance_sheet(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get Balance Sheet to calculate D/E, current ratio.

        Cached for 15 minutes (CACHE_TTL_FMP_FINANCIALS).
        """
        # Check cache first
        cache_params = {"period": period, "limit": limit}
        cache_key = _make_cache_key("balance-sheet", symbol, cache_params)

        cached_data = await _get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        endpoint = f"/v3/balance-sheet-statement/{symbol.upper()}"
        params = {"period": period, "limit": limit, "apikey": FMP_API_KEY}
        url = f"{BASE_FMP_URL}{endpoint}"

        raw_data = await _fetch_fmp_data(url, params)

        if raw_data is None:
            return None

        # Cache raw data
        await _set_cached_data(cache_key, raw_data, settings.CACHE_TTL_FMP_FINANCIALS)

        logger.info(f"Fetched balance sheet for {symbol}")
        return raw_data

    # =========================================================================
    # CASH FLOW STATEMENT
    # =========================================================================
    async def get_cash_flow_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get Cash Flow Statement to calculate FCF.

        Cached for 15 minutes (CACHE_TTL_FMP_FINANCIALS).
        """
        # Check cache first
        cache_params = {"period": period, "limit": limit}
        cache_key = _make_cache_key("cash-flow", symbol, cache_params)

        cached_data = await _get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        endpoint = f"/v3/cash-flow-statement/{symbol.upper()}"
        params = {"period": period, "limit": limit, "apikey": FMP_API_KEY}
        url = f"{BASE_FMP_URL}{endpoint}"

        raw_data = await _fetch_fmp_data(url, params)

        if raw_data is None:
            return None

        # Cache raw data
        await _set_cached_data(cache_key, raw_data, settings.CACHE_TTL_FMP_FINANCIALS)

        logger.info(f"Fetched cash flow statement for {symbol}")
        return raw_data

    # =========================================================================
    # QUOTE (Real-time price data)
    # =========================================================================
    async def get_quote(
        self,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get quote data including current price and market cap.

        Cached for 30 seconds (CACHE_TTL_FMP_QUOTE) - real-time data.
        """
        # Check cache first
        cache_key = _make_cache_key("quote", symbol)

        cached_data = await _get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        endpoint = f"/v3/quote/{symbol.upper()}"
        params = {"apikey": FMP_API_KEY}
        url = f"{BASE_FMP_URL}{endpoint}"

        raw_data = await _fetch_fmp_data(url, params)

        if raw_data is None:
            return None

        # Quote endpoint returns list with 1 item
        if isinstance(raw_data, list) and len(raw_data) > 0:
            quote = raw_data[0]
            # Cache single quote (short TTL for real-time)
            await _set_cached_data(cache_key, quote, settings.CACHE_TTL_FMP_QUOTE)
            logger.info(f"Fetched quote for {symbol}")
            return quote

        return None
