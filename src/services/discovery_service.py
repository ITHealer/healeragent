# src/services/discovery_service.py
import json
import asyncio
import httpx
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone
import logging
import aioredis
from pydantic import ValidationError

from src.services.list_service import ListService
from src.services.history_chart_service import HistoryChartService
from src.models.equity import DiscoveryItemOutput, FMPGainerItem, ScreenerOutput, ScreenerStep1Data
from src.utils.logger.set_up_log_dataFMP import setup_logger
from src.utils.config import settings
import os 
from dotenv import load_dotenv

logger = setup_logger(__name__, log_level=logging.INFO)

load_dotenv()
FMP_API_KEY_FOR_SERVICE = os.getenv("FMP_API_KEY", "YOUR_FMP_API_KEY_PLACEHOLDER")
BASE_FMP_URL = "https://financialmodelingprep.com/api"
FMP_URL_STABLE = settings.FMP_URL_STABLE  
MAX_CONCURRENT_REQUESTS = 10 

class DiscoveryService:
    def __init__(self):
        self.list_service = ListService()


    async def _fetch_batch_quotes(
        self,
        symbols: List[str],
        client: httpx.AsyncClient,
        redis_client: Optional[aioredis.Redis] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch batch quotes from FMP for multiple symbols.
        Short cache (30s) for real-time data.
        
        Returns:
            Dict with symbol as key, value contains dayHigh, dayLow, volume, marketCap, previousClose
        """
        if not symbols:
            return {}
        
        # Check cache first (TTL 30s)
        symbols_str = "_".join(sorted(symbols))
        cache_key = f"discovery_batch_quotes_{symbols_str}"
        
        if redis_client:
            try:
                cached_json = await redis_client.get(cache_key)
                if cached_json:
                    logger.debug(f"Cache HIT for batch quotes: {cache_key}")
                    return json.loads(cached_json.decode('utf-8'))
            except Exception as e:
                logger.debug(f"Cache check failed for batch quotes: {e}")
        
        # Fetch from API
        symbols_api_str = ",".join(symbols)
        url = f"{BASE_FMP_URL}/v3/quote/{symbols_api_str}?apikey={FMP_API_KEY_FOR_SERVICE}"
        
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            quotes_data = response.json()
            
            if not isinstance(quotes_data, list):
                logger.warning(f"Unexpected quote API response format: {type(quotes_data)}")
                return {}
            
            # Create map for easy lookup
            quotes_map = {}
            for quote in quotes_data:
                if isinstance(quote, dict) and quote.get("symbol"):
                    quotes_map[quote["symbol"]] = {
                        "dayHigh": quote.get("dayHigh"),
                        "dayLow": quote.get("dayLow"),
                        "volume": quote.get("volume"),
                        "marketCap": quote.get("marketCap"),
                        "previousClose": quote.get("previousClose")
                    }
            
            # Cache results (30s TTL)
            if redis_client and quotes_map:
                try:
                    await redis_client.set(
                        cache_key,
                        json.dumps(quotes_map),
                        ex=settings.CACHE_TTL_DISCOVERY_QUOTE  # 30 seconds
                    )
                    logger.debug(f"Cached batch quotes with TTL {settings.CACHE_TTL_DISCOVERY_QUOTE}s")
                except Exception as e:
                    logger.warning(f"Failed to cache batch quotes: {e}")
            
            return quotes_map
            
        except httpx.HTTPStatusError as hse:
            logger.error(f"HTTP error fetching batch quotes: {hse.response.status_code}")
            return {}
        except Exception as e:
            logger.exception(f"Error fetching batch quotes: {e}")
            return {}
    
    async def _fetch_historical_volume_and_market_cap(
        self,
        symbols: List[str],
        client: httpx.AsyncClient,
        redis_client: Optional[aioredis.Redis] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch historical data (1 day ago) to calculate volume change and market cap change.
        Cache 24h since data only changes once per day.
        
        Returns:
            Dict with symbol as key, value contains:
            - previous_volume: volume from yesterday
            - previous_market_cap: market cap from yesterday
        """
        if not symbols:
            return {}
        
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        historical_map = {}
        
        async def fetch_one_historical(symbol: str):
            """Fetch historical data for 1 symbol with 24h caching"""
            
            # Check cache first (TTL 24h)
            cache_key_vol = f"discovery_hist_vol_{symbol}_{yesterday}"
            cache_key_mcap = f"discovery_hist_mcap_{symbol}_{yesterday}"
            
            cached_volume = None
            cached_market_cap = None
            
            if redis_client:
                try:
                    # Check volume cache
                    vol_bytes = await redis_client.get(cache_key_vol)
                    if vol_bytes:
                        cached_volume = float(vol_bytes.decode('utf-8'))
                    
                    # Check market cap cache
                    mcap_bytes = await redis_client.get(cache_key_mcap)
                    if mcap_bytes:
                        cached_market_cap = float(mcap_bytes.decode('utf-8'))
                    
                    # If both are cached, return immediately
                    if cached_volume is not None and cached_market_cap is not None:
                        logger.debug(f"Cache HIT for {symbol} historical data")
                        return symbol, {
                            "previous_volume": cached_volume,
                            "previous_market_cap": cached_market_cap
                        }
                except Exception as e:
                    logger.debug(f"Cache check failed for {symbol}: {e}")
            
            # Fetch from API
            # Call historical-price-full to get volume
            url_price = f"{BASE_FMP_URL}/v3/historical-price-full/{symbol}?limit=2&apikey={FMP_API_KEY_FOR_SERVICE}"
            
            previous_volume = None
            previous_market_cap = None
            
            try:
                response = await client.get(url_price, timeout=20.0)
                response.raise_for_status()
                price_data = response.json()
                
                if isinstance(price_data, dict) and "historical" in price_data:
                    hist_list = price_data["historical"]
                    if len(hist_list) >= 2:
                        # Index 0 = today (or latest), index 1 = yesterday
                        yesterday_data = hist_list[1]
                        previous_volume = yesterday_data.get("volume")
                
                # Call historical-market-capitalization
                url_mcap = f"{BASE_FMP_URL}/v3/historical-market-capitalization/{symbol}?limit=2&apikey={FMP_API_KEY_FOR_SERVICE}"
                response_mcap = await client.get(url_mcap, timeout=20.0)
                response_mcap.raise_for_status()
                mcap_data = response_mcap.json()
                
                if isinstance(mcap_data, list) and len(mcap_data) >= 2:
                    # Index 0 = latest, index 1 = previous day
                    yesterday_mcap = mcap_data[1]
                    previous_market_cap = yesterday_mcap.get("marketCap")
                
                # Cache results (24h TTL)
                if redis_client:
                    try:
                        if previous_volume is not None:
                            await redis_client.set(
                                cache_key_vol,
                                str(previous_volume),
                                ex=settings.CACHE_TTL_DISCOVERY_HISTORICAL  # 24 hours
                            )
                        
                        if previous_market_cap is not None:
                            await redis_client.set(
                                cache_key_mcap,
                                str(previous_market_cap),
                                ex=settings.CACHE_TTL_DISCOVERY_HISTORICAL  # 24 hours
                            )
                        
                        logger.debug(f"Cached historical data for {symbol} with 24h TTL")
                    except Exception as e:
                        logger.warning(f"Failed to cache historical data for {symbol}: {e}")
                
                return symbol, {
                    "previous_volume": previous_volume,
                    "previous_market_cap": previous_market_cap
                }
                
            except Exception as e:
                logger.debug(f"Error fetching historical for {symbol}: {e}")
                return symbol, {}
        
        # Concurrent fetching with semaphore
        semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests
        
        async def bounded_fetch(symbol: str):
            async with semaphore:
                return await fetch_one_historical(symbol)
        
        results = await asyncio.gather(*[bounded_fetch(s) for s in symbols], return_exceptions=True)
        
        for result in results:
            if isinstance(result, tuple):
                symbol, data = result
                if data:
                    historical_map[symbol] = data
        
        return historical_map
    
    
    async def _fetch_fmp_direct_stock_market_movers(
        self, 
        mover_type: str, 
        client: httpx.AsyncClient,
        max_retries: int = 3
    ) -> Optional[List[FMPGainerItem]]:
        """
        Fetch stock market movers (gainers/losers/actives) from FMP with retry logic.
        """
        if not FMP_API_KEY_FOR_SERVICE or FMP_API_KEY_FOR_SERVICE == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error(f"FMP API Key not configured for {mover_type}.")
            return None
        
        url_map = { 
            "gainers": f"{BASE_FMP_URL}/v3/stock_market/gainers", 
            "losers": f"{BASE_FMP_URL}/v3/stock_market/losers", 
            "actives": f"{BASE_FMP_URL}/v3/stock_market/actives",
        }
        
        url_base = url_map.get(mover_type)
        if not url_base:
            logger.error(f"Invalid mover_type '{mover_type}'")
            return None
        
        url = f"{url_base}?apikey={FMP_API_KEY_FOR_SERVICE}"
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                # Increase timeout with exponential backoff
                current_timeout = 45.0 * (1.5 ** attempt)  # 45s, 67.5s, 101.25s
                
                logger.debug(f"Fetching {mover_type}, attempt {attempt + 1}/{max_retries}, timeout={current_timeout}s")
                
                response = await client.get(url, timeout=current_timeout)
                response.raise_for_status()
                fmp_data = response.json()
                
                if not isinstance(fmp_data, list):
                    logger.warning(f"Unexpected data format for {mover_type}, expected list, got {type(fmp_data)}.")
                    return []
                
                return [FMPGainerItem(**item) for item in fmp_data if isinstance(item, dict) and item.get("symbol")]
            
            except (httpx.ConnectTimeout, httpx.ReadTimeout) as timeout_err:
                # Handle timeout
                if attempt < max_retries - 1:
                    wait_time = 4 ** attempt  # Exponential backoff: 1s, 4s, 16s
                    logger.warning(
                        f"Timeout for {mover_type} on attempt {attempt + 1}/{max_retries}. "
                        f"Error type: {type(timeout_err).__name__}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Final timeout for {mover_type} after {max_retries} attempts. "
                        f"Error: {timeout_err}. URL: {url}"
                    )
                    return None
            
            except httpx.HTTPStatusError as hse:
                # HTTP error (4xx, 5xx) - no retry
                logger.error(
                    f"HTTP error fetching {mover_type}: {hse.response.status_code} - "
                    f"{hse.response.text[:200]}. URL: {url}", 
                    exc_info=False
                )
                return None
            
            except Exception as e:
                # Other errors
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Error for {mover_type} on attempt {attempt + 1}/{max_retries}. "
                        f"Retrying in {wait_time}s... Error: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.exception(f"Final error fetching {mover_type} after {max_retries} attempts. URL: {url}")
                    return None
        
        return None

    async def fetch_fmp_direct_stock_market_movers(
        self, 
        mover_types: List[str], 
        client: httpx.AsyncClient,
        max_retries: int = 3  
    ) -> Optional[Dict[str, List[FMPGainerItem]]]:
        """
        Fetch stock market movers (gainers/losers/actives) from FMP with retry logic.
        
        Args:
            mover_types: List of mover types to fetch (["gainers", "losers", "actives"])
            client: httpx AsyncClient
            max_retries: Maximum retry attempts
            
        Returns:
            Dict with mover_type as key and list of FMPGainerItem as value
            Example: {"gainers": [...], "losers": [...], "actives": [...]}
        """
        if not FMP_API_KEY_FOR_SERVICE or FMP_API_KEY_FOR_SERVICE == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error(f"FMP API Key not configured for market movers.")
            return None
        
        url_map = { 
            "gainers": f"{BASE_FMP_URL}/v3/stock_market/gainers", 
            "losers": f"{BASE_FMP_URL}/v3/stock_market/losers", 
            "actives": f"{BASE_FMP_URL}/v3/stock_market/actives",
        }
        
        # Validate mover_types
        valid_types = [mt for mt in mover_types if mt in url_map]
        if not valid_types:
            logger.error(f"No valid mover_types provided. Got: {mover_types}")
            return None
        
        results = {}
        
        # Fetch each mover type
        for mover_type in valid_types:
            url_base = url_map[mover_type]
            url = f"{url_base}?apikey={FMP_API_KEY_FOR_SERVICE}"
            
            # Retry logic
            for attempt in range(max_retries):
                try:
                    # Increase timeout with exponential backoff
                    current_timeout = 45.0 * (1.5 ** attempt)  # 45s, 67.5s, 101.25s
                    
                    logger.debug(f"Fetching {mover_type}, attempt {attempt + 1}/{max_retries}, timeout={current_timeout}s")
                    
                    response = await client.get(url, timeout=current_timeout)
                    response.raise_for_status()
                    fmp_data = response.json()
                    
                    if not isinstance(fmp_data, list):
                        logger.warning(f"Unexpected data format for {mover_type}, expected list, got {type(fmp_data)}.")
                        results[mover_type] = []
                        break
                    
                    results[mover_type] = [
                        FMPGainerItem(**item) 
                        for item in fmp_data 
                        if isinstance(item, dict) and item.get("symbol")
                    ]
                    logger.info(f"Successfully fetched {len(results[mover_type])} {mover_type} items")
                    break  # Success, exit retry loop
                
                except (httpx.ConnectTimeout, httpx.ReadTimeout) as timeout_err:
                    # Handle timeout
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        logger.warning(
                            f"Timeout for {mover_type} on attempt {attempt + 1}/{max_retries}. "
                            f"Error type: {type(timeout_err).__name__}. Retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"Final timeout for {mover_type} after {max_retries} attempts. "
                            f"Error: {timeout_err}. URL: {url}"
                        )
                        results[mover_type] = []  # Assign empty list if failed
                
                except httpx.HTTPStatusError as hse:
                    # HTTP error (4xx, 5xx) - no retry
                    logger.error(
                        f"HTTP error fetching {mover_type}: {hse.response.status_code} - "
                        f"{hse.response.text[:200]}. URL: {url}", 
                        exc_info=False
                    )
                    results[mover_type] = []
                    break  # Don't retry for HTTP errors
                
                except Exception as e:
                    # Other errors
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(
                            f"Error for {mover_type} on attempt {attempt + 1}/{max_retries}. "
                            f"Retrying in {wait_time}s... Error: {e}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.exception(f"Final error fetching {mover_type} after {max_retries} attempts. URL: {url}")
                        results[mover_type] = []
        
        return results if results else None

    async def _fetch_discovery_data(
        self,
        discovery_type: str,
        limit: int = 10,
        redis_client: Optional[aioredis.Redis] = None
    ) -> Optional[List[DiscoveryItemOutput]]:
        """
        Fetch discovery data (gainers/losers/actives) with new fields added.
        Uses optimized caching strategy for each data type.
        """
        
        limits = httpx.Limits(
            max_keepalive_connections=50,
            max_connections=200,
            keepalive_expiry=30.0
        )
        
        timeout = httpx.Timeout(
            connect=15.0,
            read=40.0,
            write=10.0,
            pool=10.0
        )
        
        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            
            # Step 1: Fetch basic movers list
            basic_movers_list = await self._fetch_fmp_direct_stock_market_movers(discovery_type, client)

            if basic_movers_list is None:
                logger.error(f"Failed to fetch basic movers list for {discovery_type}.")
                return None
            if not basic_movers_list:
                logger.info(f"No movers found for {discovery_type} from FMP.")
                return []

            symbols_to_process = [item.symbol for item in basic_movers_list[:limit] if item.symbol]
            
            if not symbols_to_process:
                logger.warning(f"No valid symbols to process for {discovery_type}.")
                return []

            # Check Redis connection
            redis_safe = None
            if redis_client:
                try:
                    await asyncio.wait_for(redis_client.ping(), timeout=2.0)
                    redis_safe = redis_client
                except:
                    redis_safe = None

            # Step 2: Run ALL tasks in parallel
            enrich_task = self.list_service.process_stocks_etfs_batch(
                symbols_to_process=symbols_to_process,
                client=client,
                redis_client=redis_safe
            )

            chart_task = self.list_service._get_batch_charts(
                symbols=symbols_to_process,
                client=client,
                redis_client=redis_safe,
                asset_type="stock"
            )
            
            # New tasks: batch quotes + historical (with caching)
            quotes_task = self._fetch_batch_quotes(
                symbols_to_process, 
                client, 
                redis_safe
            )
            
            historical_task = self._fetch_historical_volume_and_market_cap(
                symbols_to_process, 
                client, 
                redis_safe
            )

            # Run all tasks in parallel
            results = await asyncio.gather(
                enrich_task, 
                chart_task, 
                quotes_task,
                historical_task,
                return_exceptions=True
            )

            enriched_data_no_charts = results[0] if isinstance(results[0], list) else []
            charts_map = results[1] if isinstance(results[1], dict) else {}
            quotes_map = results[2] if isinstance(results[2], dict) else {}
            historical_map = results[3] if isinstance(results[3], dict) else {}

            # Log exceptions if any
            if isinstance(results[0], Exception):
                logger.error(f"[{discovery_type}] Enrich task FAILED: {results[0]}", exc_info=results[0])
            if isinstance(results[1], Exception):
                logger.error(f"[{discovery_type}] Chart task FAILED: {results[1]}", exc_info=results[1])
            if isinstance(results[2], Exception):
                logger.error(f"[{discovery_type}] Quotes task FAILED: {results[2]}", exc_info=results[2])
            if isinstance(results[3], Exception):
                logger.error(f"[{discovery_type}] Historical task FAILED: {results[3]}", exc_info=results[3])

            # Log sample keys to verify symbol matching
            if enriched_data_no_charts:
                enriched_symbols = {item.symbol for item in enriched_data_no_charts}
                logger.debug(f"[{discovery_type}] Enriched symbols: {list(enriched_symbols)[:5]}")
                logger.debug(f"[{discovery_type}] Quotes symbols: {list(quotes_map.keys())[:5]}")
                logger.debug(f"[{discovery_type}] Historical symbols: {list(historical_map.keys())[:5]}")
                
                # Check symbol mismatch
                quotes_symbols = set(quotes_map.keys())
                missing_quotes = enriched_symbols - quotes_symbols
                if missing_quotes:
                    logger.warning(
                        f"[{discovery_type}] Missing quotes for {len(missing_quotes)} symbols: "
                        f"{list(missing_quotes)[:5]}"
                    )

            if not enriched_data_no_charts:
                logger.warning(f"No enriched data for {discovery_type}, returning empty list.")
                return []

            enriched_map = {item.symbol: item for item in enriched_data_no_charts}

            # Step 3: Merge all data into DiscoveryItemOutput
            successful_merges = 0
            failed_merges = 0
            
            for symbol, item in enriched_map.items():
                # Assign chartData
                chart_data_list = charts_map.get(symbol, [])
                item.chartData = chart_data_list
                
                # Merge data from quotes (real-time)
                quote_data = quotes_map.get(symbol, {})
                
                if quote_data:  # Dict has data
                    item.high_24h = quote_data.get("dayHigh")
                    item.low_24h = quote_data.get("dayLow")
                    item.volume_24h = quote_data.get("volume")
                    item.market_cap = quote_data.get("marketCap")
                    
                    # Calculate 24h change from historical data
                    hist_data = historical_map.get(symbol, {})
                    
                    # Volume change = current volume - previous volume
                    current_volume = quote_data.get("volume")
                    previous_volume = hist_data.get("previous_volume")
                    
                    if current_volume is not None and previous_volume is not None:
                        item.volume_change_24h = current_volume - previous_volume
                    else:
                        item.volume_change_24h = None
                    
                    # Market cap change = current market cap - previous market cap
                    current_market_cap = quote_data.get("marketCap")
                    previous_market_cap = hist_data.get("previous_market_cap")
                    
                    if current_market_cap is not None and previous_market_cap is not None:
                        item.market_cap_change_24h = current_market_cap - previous_market_cap
                    else:
                        item.market_cap_change_24h = None
                    
                    successful_merges += 1
                else:
                    # Log when no quote data available
                    logger.warning(f"[{discovery_type}] No quote data for {symbol}")
                    failed_merges += 1
                
                # Reset detected_patterns
                item.detected_patterns = None

            final_enriched_list = list(enriched_map.values())
            
            logger.info(
                f"[Fetched {len(final_enriched_list)} {discovery_type} symbol]:"
                f"(merged quotes: {successful_merges}, failed: {failed_merges})"
            )
            
            return final_enriched_list

    async def get_gainers(self, limit: int = 10, redis_client: Optional[aioredis.Redis] = None) -> Optional[List[DiscoveryItemOutput]]:
        return await self._fetch_discovery_data("gainers", limit=limit, redis_client=redis_client)

    async def get_losers(self, limit: int = 10, redis_client: Optional[aioredis.Redis] = None) -> Optional[List[DiscoveryItemOutput]]:
        return await self._fetch_discovery_data("losers", limit=limit, redis_client=redis_client)

    async def get_actives(self, limit: int = 10, redis_client: Optional[aioredis.Redis] = None) -> Optional[List[DiscoveryItemOutput]]:
        return await self._fetch_discovery_data("actives", limit=limit, redis_client=redis_client)
    
    async def get_step1_screener_data(
        self,
        limit: Optional[int] = 100,
        market_cap_more_than: Optional[float] = None,
        market_cap_lower_than: Optional[float] = None,
        price_more_than: Optional[float] = None,
        price_lower_than: Optional[float] = None,
        beta_more_than: Optional[float] = None,
        beta_lower_than: Optional[float] = None,
        volume_more_than: Optional[float] = None,
        volume_lower_than: Optional[float] = None,
        dividend_more_than: Optional[float] = None,
        dividend_lower_than: Optional[float] = None,
        is_etf: Optional[bool] = None,
        is_fund: Optional[bool] = None,
        is_actively_trading: Optional[bool] = True,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        country: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> Optional[List[ScreenerStep1Data]]:
        if not FMP_API_KEY_FOR_SERVICE or FMP_API_KEY_FOR_SERVICE == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error("FMP API Key not configured for get_step1_screener_data.")
            return None

        screener_params_for_fmp = {"apikey": FMP_API_KEY_FOR_SERVICE}
        if limit is not None: screener_params_for_fmp["limit"] = limit
        if market_cap_more_than is not None: screener_params_for_fmp["marketCapMoreThan"] = market_cap_more_than
        if market_cap_lower_than is not None: screener_params_for_fmp["marketCapLowerThan"] = market_cap_lower_than
        if price_more_than is not None: screener_params_for_fmp["priceMoreThan"] = price_more_than
        if price_lower_than is not None: screener_params_for_fmp["priceLowerThan"] = price_lower_than
        if beta_more_than is not None: screener_params_for_fmp["betaMoreThan"] = beta_more_than
        if beta_lower_than is not None: screener_params_for_fmp["betaLowerThan"] = beta_lower_than
        if volume_more_than is not None: screener_params_for_fmp["volumeMoreThan"] = volume_more_than
        if volume_lower_than is not None: screener_params_for_fmp["volumeLowerThan"] = volume_lower_than
        if dividend_more_than is not None: screener_params_for_fmp["dividendMoreThan"] = dividend_more_than
        if dividend_lower_than is not None: screener_params_for_fmp["dividendLowerThan"] = dividend_lower_than
        if is_etf is not None: screener_params_for_fmp["isEtf"] = is_etf
        if is_fund is not None: screener_params_for_fmp["isFund"] = is_fund
        if is_actively_trading is not None: screener_params_for_fmp["isActivelyTrading"] = is_actively_trading
        if sector is not None: screener_params_for_fmp["sector"] = sector
        if industry is not None: screener_params_for_fmp["industry"] = industry
        if country is not None: screener_params_for_fmp["country"] = country
        if exchange is not None: screener_params_for_fmp["exchange"] = exchange

        async with httpx.AsyncClient(timeout=120.0) as client:
            screener_url = f"{BASE_FMP_URL}/v3/stock-screener"
            active_params = {k: v for k,v in screener_params_for_fmp.items() if v is not None and k != "apikey"}
            try:
                response_screener = await client.get(screener_url, params=screener_params_for_fmp)
                response_screener.raise_for_status()
                fmp_screener_raw_list = response_screener.json()

                if isinstance(fmp_screener_raw_list, list):
                    parsed_data: List[ScreenerStep1Data] = []
                    for item_dict in fmp_screener_raw_list:
                        if isinstance(item_dict, dict) and item_dict.get("symbol"):
                            try:
                                parsed_data.append(ScreenerStep1Data(**item_dict))
                            except Exception as p_error:
                                logger.warning(f"Pydantic validation error for screener item symbol {item_dict.get('symbol', 'N/A')}: {p_error}. Data: {item_dict}")
                    logger.debug(f"Successfully parsed {len(parsed_data)} items from screener.")
                    return parsed_data
                else:
                    logger.warning(f"Unexpected FMP Stock Screener format, received {type(fmp_screener_raw_list)}. URL: {screener_url}, Params: {active_params}")
                    return []
            except httpx.HTTPStatusError as hse:
                logger.error(f"HTTP error fetching screener data: {hse.response.status_code} - {hse.response.text[:200]}. URL: {screener_url}, Params: {active_params}", exc_info=False)
                return None
            except Exception as e:
                logger.exception(f"General error fetching screener data. URL: {screener_url}, Params: {active_params}")
                return None
        logger.error("Exited screener client block unexpectedly.")
        return [] 

    async def get_raw_quotes_for_symbols(self, symbols: List[str]) -> Optional[List[Dict[str, Any]]]:
        if not FMP_API_KEY_FOR_SERVICE or FMP_API_KEY_FOR_SERVICE == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error("FMP API Key not configured for get_raw_quotes_for_symbols.")
            return None
        if not symbols:
            logger.warning("Empty symbol list provided to get_raw_quotes_for_symbols.")
            return []

        symbols_str = ",".join(symbols)
        quote_url = f"{BASE_FMP_URL}/v3/quote/{symbols_str}?apikey={FMP_API_KEY_FOR_SERVICE}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            logger.debug(f"Fetching raw quotes for {len(symbols)} symbols. First few: {symbols[:5]}")
            try:
                response_quote = await client.get(quote_url)
                response_quote.raise_for_status()
                fmp_quote_raw_list = response_quote.json()
                if isinstance(fmp_quote_raw_list, list):
                    return fmp_quote_raw_list
                else:
                    if isinstance(fmp_quote_raw_list, dict) and "Error Message" in fmp_quote_raw_list:
                         logger.error(f"Error from FMP API quote: {fmp_quote_raw_list['Error Message']}. Symbols: {symbols_str[:100]}")
                    else:
                        logger.warning(f"Unexpected FMP Quote format, received {type(fmp_quote_raw_list)}. Symbols: {symbols_str[:100]}")
                    return []
            except httpx.HTTPStatusError as hse:
                logger.error(f"HTTP error fetching raw quotes: {hse.response.status_code} - {hse.response.text[:200]}. Symbols: {symbols_str[:100]}", exc_info=False)
                return [] 
            except Exception as e:
                logger.exception(f"General error fetching raw quotes. Symbols: {symbols_str[:100]}")
                return None 
        logger.error("Exited raw quotes client block unexpectedly.")
        return None 

    async def get_screener_http_compatible(
        self,
        **screener_filters: Any
    ) -> Optional[List[ScreenerOutput]]:
        step1_data_list = await self.get_step1_screener_data(**screener_filters)

        if step1_data_list is None:
            logger.error("Failed to get step 1 screener data for HTTP compatible screener.")
            return None
        if not step1_data_list:
            logger.info("No data from step 1 screener for HTTP compatible screener.")
            return []

        symbols_to_quote = [item.symbol for item in step1_data_list if item.symbol] 
        if not symbols_to_quote:
            logger.info("No valid symbols from step 1 data to fetch quotes for.")

        raw_quotes_list = await self.get_raw_quotes_for_symbols(symbols_to_quote)

        if raw_quotes_list is None:
            logger.warning("Failed to get raw quotes for symbols in HTTP compatible screener. Results might be incomplete.")
            raw_quotes_list = [] 

        step1_map: Dict[str, ScreenerStep1Data] = {item.symbol: item for item in step1_data_list if item.symbol}
        final_results: List[ScreenerOutput] = []

        for quote_dict in raw_quotes_list:
            symbol = quote_dict.get("symbol")
            if not symbol:
                logger.debug("Skipping quote item without symbol in HTTP compatible screener.")
                continue
            step1_item = step1_map.get(symbol)
            if not step1_item:
                logger.debug(f"No step 1 data found for symbol '{symbol}' in quote list during HTTP screener merge.")
                continue

            combined_data = {**quote_dict}
            combined_data['beta'] = step1_item.beta
            combined_data['sector'] = step1_item.sector
            combined_data['lastAnnualDividend'] = step1_item.lastAnnualDividend
            if not combined_data.get('name') and step1_item.companyName:
                combined_data['name'] = step1_item.companyName
            try:
                final_results.append(ScreenerOutput(**combined_data))
            except Exception as p_err:
                logger.warning(f"Pydantic validation error for combined screener data for symbol {symbol}: {p_err}. Data: {combined_data}")
        return final_results