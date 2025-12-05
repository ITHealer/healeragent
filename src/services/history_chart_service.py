import httpx
from typing import List, Optional
import json
from dotenv import load_dotenv
import aioredis
import logging
import asyncio
import random

from src.models.equity import ChartDataItem
from src.utils.config import settings
from src.utils.logger.set_up_log_dataFMP import setup_logger

load_dotenv()

FMP_API_KEY_FOR_SERVICE = settings.FMP_API_KEY
BASE_FMP_URL = settings.BASE_FMP_URL
FMP_URL_STABLE = settings.FMP_URL_STABLE
logger = setup_logger(__name__, log_level=logging.INFO)

class HistoryChartService:
    @staticmethod
    async def get_historical_chart_from_fmp(
        symbol: str,
        interval: str,
        start_date_str: str,
        end_date_str: str,
        client: httpx.AsyncClient,
        redis_client: Optional[aioredis.Redis],
        max_retries: int = 3 
    ) -> List[ChartDataItem]:
        """
        Lấy dữ liệu biểu đồ lịch sử thô (chỉ time và value) từ FMP và cache lại.
        Không thực hiện bất kỳ tính toán chỉ số nào.
        
        Args:
            symbol: Mã cổ phiếu/crypto
            interval: Khung thời gian (1min, 5min, 15min, 30min, 1hour, 4hour)
            start_date_str: Ngày bắt đầu (YYYY-MM-DD)
            end_date_str: Ngày kết thúc (YYYY-MM-DD)
            client: httpx.AsyncClient được share
            redis_client: Redis client cho caching
            max_retries: Số lần retry tối đa khi timeout (default: 3)
        
        Returns:
            List[ChartDataItem]: Danh sách các điểm dữ liệu chart
        """
        cache_key = f"chart_simple_{symbol.upper()}_{interval}_{start_date_str}_to_{end_date_str}"
        logger.debug(f"Attempting to fetch simple chart for {cache_key}")

        # ========== CHECK CACHE ==========
        if redis_client:
            try:
                cached_chart_data_json = await redis_client.get(cache_key)
                if cached_chart_data_json:
                    data_list = json.loads(cached_chart_data_json)
                    return [ChartDataItem(**item) for item in data_list]
            except Exception as e:
                logger.warning(f"Redis GET error for chart {cache_key}: {e}")
        # ====================================

        if not FMP_API_KEY_FOR_SERVICE or FMP_API_KEY_FOR_SERVICE == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error(f"FMP API Key not configured. Cannot fetch chart for {symbol}.")
            return []

        url = f"{FMP_URL_STABLE}/historical-chart/{interval}?symbol={symbol.upper()}&from={start_date_str}&to={end_date_str}&apikey={FMP_API_KEY_FOR_SERVICE}"
        
        # ========== RETRY LOGIC START ==========
        for attempt in range(max_retries):
            try:
                connect_timeout = 60.0 * (1.5 ** attempt)
                read_timeout = 60.0 * (1.5 ** attempt)

                timeout = httpx.Timeout(
                    connect=connect_timeout,
                    read=read_timeout,
                    write=60.0, 
                    pool=60.0
                )
                
                # Add delay between retries to avoid rate limit
                if attempt > 0:
                    jitter = random.uniform(0.5, 1.5)
                    wait_time = (2 * (2 ** (attempt - 1))) + jitter  # 2s, 4s, 8s + jitter
                    logger.info(f"[{symbol}] Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                    await asyncio.sleep(wait_time)

                response = await client.get(url, timeout=timeout)
                response.raise_for_status()

                fmp_data = response.json()
                if not isinstance(fmp_data, list):
                    logger.warning(
                        f"Unexpected data format from FMP for {symbol} chart. "
                        f"Expected list, got {type(fmp_data)}. URL: {url}"
                    )
                    return []
                
                # Parse data
                chart_data_output: List[ChartDataItem] = []
                for point in fmp_data:
                    ts_str, val = point.get('date'), point.get('close')
                    if ts_str and val is not None:
                        try:
                            chart_data_output.append(ChartDataItem(time=str(ts_str), value=float(val)))
                        except (ValueError, TypeError):
                            continue
                
                # ========== CACHE RESULT ==========
                if chart_data_output and redis_client:
                    try:
                        serializable_data = [item.model_dump() for item in chart_data_output]
                        await redis_client.set(
                            cache_key, 
                            json.dumps(serializable_data), 
                            ex=int(settings.CACHE_TTL_CHART)
                        )
                    except Exception as e_set:
                        logger.warning(f"Redis SET error for chart {cache_key}: {e_set}")
                # ===================================
                
                return chart_data_output
            
            except (httpx.ConnectTimeout, httpx.ReadTimeout) as timeout_err:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"[{symbol}] Timeout on attempt {attempt + 1}/{max_retries}. "
                        f"Error: {type(timeout_err).__name__}. Will retry..."
                    )
                    continue
                else:
                    logger.error(
                        f"[{symbol}] Final timeout after {max_retries} attempts. "
                        f"Last timeout: {read_timeout:.1f}s. URL: {url}"
                    )
                    return []
            
            except httpx.HTTPStatusError as http_err:
                # ERROR429 (rate limit)
                if http_err.response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = 10 * (2 ** attempt)  # 10s, 20s, 40s
                        logger.warning(
                            f"[{symbol}] Rate limit (429). Waiting {wait_time}s before retry..."
                        )
                        await asyncio.sleep(wait_time)
                        continue

                logger.error(
                    f"HTTP error {http_err.response.status_code} fetching chart for {symbol}. "
                    f"Response: {http_err.response.text[:200]}. URL: {url}"
                )
                return []
            
            except json.JSONDecodeError as json_err:
                logger.error(
                    f"JSON decode error for chart {symbol}: {json_err}. "
                    f"Response: {response.text[:200] if 'response' in locals() else 'N/A'}. URL: {url}"
                )
                return []
            
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"[{symbol}] Unexpected error on attempt {attempt + 1}/{max_retries}: {e}. "
                        "Will retry..."
                    )
                    continue
                else:
                    logger.exception(
                        f"[{symbol}] Final error after {max_retries} attempts. URL: {url}"
                    )
                    return []
        
        return []
        # ========== END RETRY LOGIC ==========