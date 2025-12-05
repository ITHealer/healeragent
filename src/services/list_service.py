import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import json
import httpx
import aioredis
import logging

from src.models.equity import ChartDataItem, DiscoveryItemOutput
from src.services.history_chart_service import HistoryChartService
from src.utils.config import settings
from src.utils.logger.set_up_log_dataFMP import setup_logger

FMP_API_KEY_FOR_SERVICE = settings.FMP_API_KEY
BASE_FMP_URL = settings.BASE_FMP_URL
FMP_URL_STABLE = settings.FMP_URL_STABLE
logger = setup_logger(__name__, log_level=logging.INFO)

MAX_CONCURRENT_REQUESTS = 15

class ListService:
    async def _get_batch_grades_consensus(
        self,
        symbols: List[str],
        client: httpx.AsyncClient,
        redis_client: Optional[aioredis.Redis]
    ) -> Dict[str, Dict[str, Any]]:
        if not symbols:
            return {}

        today_str = date.today().isoformat()
        grades_map: Dict[str, Dict[str, Any]] = {}
        symbols_to_fetch_from_api: List[str] = []

        # --- Bước 1: Kiểm tra cache cho từng symbol ---
        if redis_client:
            for symbol in symbols:
                symbol_upper = symbol.upper()
                cache_key = f"grades_consensus_{symbol_upper}_{today_str}"
                try:
                    cached_data = await redis_client.get(cache_key)
                    if cached_data:
                        logger.debug(f"Cache HIT cho grades của {symbol_upper} (key: {cache_key})")
                        grades_map[symbol_upper] = json.loads(cached_data)
                    else:
                        symbols_to_fetch_from_api.append(symbol)
                except Exception as e:
                    logger.error(f"Lỗi Redis GET cho grades của {symbol_upper}: {e}")
                    symbols_to_fetch_from_api.append(symbol)
        else:
            symbols_to_fetch_from_api = symbols

        # --- Bước 2: Gọi API cho những symbol chưa có trong cache ---
        if symbols_to_fetch_from_api:
            
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

            async def _fetch_grade(symbol_to_fetch: str) -> Optional[Dict[str, Any]]:
                async with semaphore:
                    url = f"{FMP_URL_STABLE}/grades-consensus?symbol={symbol_to_fetch.upper()}&apikey={FMP_API_KEY_FOR_SERVICE}"
                    try:
                        response = await client.get(url, timeout=120.0)
                        response.raise_for_status()
                        data = response.json()
                        if isinstance(data, list) and data:
                            return data[0]
                        return {"symbol": symbol_to_fetch.upper(), "data_found": False}
                    except Exception as e:
                        logger.warning(f"Không thể lấy grade cho {symbol_to_fetch}: {e}")
                        return {"symbol": symbol_to_fetch.upper(), "error": str(e)}

            tasks = [_fetch_grade(s) for s in symbols_to_fetch_from_api]
            api_results = await asyncio.gather(*tasks)

            # --- Bước 3: Cập nhật cache và map kết quả ---
            for res in api_results:
                if not isinstance(res, dict): continue

                symbol_to_cache = res.get("symbol", "").upper()
                if not symbol_to_cache: continue
                
                data_to_cache = {"data_found": False}
                if res.get("data_found") is False or res.get("error"):
                    grades_map[symbol_to_cache] = {} 
                else:
                    grades_map[symbol_to_cache] = res
                    data_to_cache = res 

                if redis_client:
                    try:
                        cache_key = f"grades_consensus_{symbol_to_cache}_{today_str}"
                        await redis_client.set(cache_key, json.dumps(data_to_cache), ex=86400)
                        logger.debug(f"Đã cache trạng thái grades cho {symbol_to_cache} (key: {cache_key})")
                    except Exception as e_set:
                        logger.error(f"Lỗi Redis SET cho grades của {symbol_to_cache}: {e_set}")
        
        final_grades_map = {k: v for k, v in grades_map.items() if v.get("data_found") is not False}
        return final_grades_map
        
    def _safe_float(self, value: Any) -> Optional[float]:
        if value is None: return None
        try: return float(value)
        except (ValueError, TypeError): return None

    async def _get_batch_quotes(self, symbols: List[str], client: httpx.AsyncClient) -> Dict[str, Dict[str, Any]]:
        if not symbols: return {}
        
        symbols_str = ",".join(symbols).upper()
        url = f"{BASE_FMP_URL}/v3/quote/{symbols_str}?apikey={FMP_API_KEY_FOR_SERVICE}"
        logger.debug(f"Fetching batch quotes for {len(symbols)} symbols.")
        try:
            response = await client.get(url, timeout=120)
            response.raise_for_status()
            data_list = response.json()
            if isinstance(data_list, list):
                return {item['symbol'].upper(): item for item in data_list if isinstance(item, dict) and 'symbol' in item}
            logger.warning(f"Unexpected format from batch quote API. Response: {data_list}")
            return {}
        except Exception as e:
            logger.error(f"Error fetching batch quotes: {e}", exc_info=True)
            return {}

    async def _get_batch_charts(
        self,
        symbols: List[str],
        client: httpx.AsyncClient,
        redis_client: Optional[aioredis.Redis],
        asset_type: str = "stock"
    ) -> Dict[str, List[ChartDataItem]]:
        if not symbols: return {}

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        days_to_fetch = 3
        if asset_type in ["stock", "etf"]:
            days_to_fetch = 5
        elif asset_type == "crypto":
            days_to_fetch = 3

        end_d = datetime.now(timezone.utc)
        start_d = end_d - timedelta(days=days_to_fetch)
        
        async def _fetch_chart(symbol: str):
            async with semaphore:
                return await HistoryChartService.get_historical_chart_from_fmp(
                    symbol=symbol, interval="4hour", 
                    start_date_str=start_d.strftime("%Y-%m-%d"), end_date_str=end_d.strftime("%Y-%m-%d"),
                    client=client, redis_client=redis_client
                )
        
        tasks = [_fetch_chart(symbol.upper()) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        chart_map: Dict[str, List[ChartDataItem]] = {} 
        for i, res in enumerate(results):
            symbol_upper = symbols[i].upper()
            if isinstance(res, Exception):
                logger.error(f"Error fetching chart for '{symbol_upper}': {res}", exc_info=False)
                chart_map[symbol_upper] = [] 
            elif isinstance(res, list):
                chart_map[symbol_upper] = res
            else:
                chart_map[symbol_upper] = []

        return chart_map

    async def process_stocks_etfs_batch(
        self,
        symbols_to_process: List[str],
        client: httpx.AsyncClient,
        redis_client: Optional[aioredis.Redis]
    ) -> List[DiscoveryItemOutput]:
        quote_task =  self._get_batch_quotes(symbols_to_process, client)
        grades_task = self._get_batch_grades_consensus(symbols_to_process, client, redis_client)
        results = await asyncio.gather(quote_task, grades_task, return_exceptions=True)
        # chart_task = await self._get_batch_charts(symbols_to_process, client, redis_client, asset_type="stock")
        quotes_map = results[0] if isinstance(results[0], dict) else {}
        grades_map = results[1] if isinstance(results[1], dict) else {}

        if isinstance(results[0], Exception): logger.error(f"Lỗi khi lấy batch quotes: {results[0]}")
        if isinstance(results[1], Exception): logger.error(f"Lỗi khi lấy batch grades: {results[1]}")

        # quotes_map, charts_map = await asyncio.gather(
        #     quote_task, chart_task
        # )

        final_results = []
        for symbol in symbols_to_process:
            symbol_upper = symbol.upper()
            quote_data = quotes_map.get(symbol_upper)
            if not quote_data:
                logger.warning(f"No quote data for {symbol_upper}, skipping.")
                continue
            # chart_info = chart_task.get(symbol_upper, {"chart_data": [], "patterns": []})
            grade_data = grades_map.get(symbol_upper, {})
            
            final_item = DiscoveryItemOutput(
                symbol=symbol_upper,
                name=quote_data.get("name", symbol_upper),
                url_logo=None,
                price=self._safe_float(quote_data.get("price")),
                change=self._safe_float(quote_data.get("change")),
                percent_change=self._safe_float(quote_data.get("changesPercentage")),
                volume=self._safe_float(quote_data.get("volume")),
                # chartData=chart_info.get("chart_data", []),
                strongBuy=grade_data.get("strongBuy"),
                buy=grade_data.get("buy"),
                hold=grade_data.get("hold"),
                sell=grade_data.get("sell"),
                strongSell=grade_data.get("strongSell"),
                consensus=grade_data.get("consensus")
            )
            final_results.append(final_item)
            
        return final_results

    async def process_crypto_batch(
        self,
        symbols_to_process: List[str],
        client: httpx.AsyncClient,
        redis_client: Optional[aioredis.Redis]
    ) -> List[DiscoveryItemOutput]:
        
        quote_task = self._get_batch_quotes(symbols_to_process, client)
        chart_task = self._get_batch_charts(symbols_to_process, client, redis_client, asset_type="crypto")

        results = await asyncio.gather(quote_task, chart_task, return_exceptions=True)

        quotes_map = results[0] if isinstance(results[0], dict) else {}
        charts_map = results[1] if isinstance(results[1], dict) else {}

        if isinstance(results[0], Exception): logger.error(f"Lỗi khi lấy batch quotes cho crypto: {results[0]}", exc_info=False)
        if isinstance(results[1], Exception): logger.error(f"Lỗi khi lấy batch charts cho crypto: {results[1]}", exc_info=False)
         
        final_results = []
        for symbol in symbols_to_process:
            symbol_upper = symbol.upper()
            quote_data = quotes_map.get(symbol_upper)
            if not quote_data:
                continue

            chart_data = charts_map.get(symbol_upper, [])
            
            final_item = DiscoveryItemOutput(
                symbol=symbol_upper,
                name=quote_data.get("name", symbol_upper),
                url_logo=None,
                price=self._safe_float(quote_data.get("price")),
                change=self._safe_float(quote_data.get("change")),
                percent_change=self._safe_float(quote_data.get("changesPercentage")),
                volume=self._safe_float(quote_data.get("volume")),
                chartData=chart_data,
                detected_patterns=None 
            )
            final_results.append(final_item)
            
        return final_results
    
    async def process_crypto_batch_scheduler(
        self,
        symbols_to_process: List[str],
        client: httpx.AsyncClient,
        redis_client: Optional[aioredis.Redis]
    ) -> List[DiscoveryItemOutput]:
        
        quote_task = await self._get_batch_quotes(symbols_to_process, client)
        # chart_task = await self._get_batch_charts(symbols_to_process, client, redis_client, asset_type="crypto")

        # quotes_map, charts_map = await asyncio.gather(quote_task, chart_task)
         
        final_results = []
        for symbol in symbols_to_process:
            symbol_upper = symbol.upper()
            quote_data = quote_task.get(symbol_upper)
            if not quote_data:
                continue

            # chart_info = chart_task.get(symbol_upper, {"chart_data": [], "patterns": []})
            final_item = DiscoveryItemOutput(
                symbol=symbol_upper,
                name=quote_data.get("name", symbol_upper),
                url_logo=None,
                price=self._safe_float(quote_data.get("price")),
                change=self._safe_float(quote_data.get("change")),
                percent_change=self._safe_float(quote_data.get("changesPercentage")),
                volume=self._safe_float(quote_data.get("volume")),
                # chartData=chart_info.get("chart_data", [])
            )
            final_results.append(final_item)
            
        return final_results

    
    