import asyncio
from datetime import date, timedelta
import json
import httpx
from typing import Any, Dict, List, Optional
import logging

from src.services.history_chart_service import HistoryChartService
from src.mappers.equity_mapper import EquityMapper
from src.models.equity import ChartDataItem, CryptoSpotlightItem

import os
from dotenv import load_dotenv

from src.utils.logger.set_up_log_dataFMP import setup_logger

load_dotenv()
FMP_API_KEY_FOR_SERVICE = os.getenv("FMP_API_KEY", "YOUR_FMP_API_KEY_PLACEHOLDER")
BASE_FMP_URL = "https://financialmodelingprep.com/api"

logger = setup_logger(__name__, log_level=logging.INFO)


class CryptoSpotlightService:
    def __init__(self):
        self._discovery_cache: Dict[str, Dict[str, Any]] = {}
        self.base_url = BASE_FMP_URL
        self.api_key = FMP_API_KEY_FOR_SERVICE
    

    async def get_crypto_spotlight(self, limit: int = 10) -> Optional[List[CryptoSpotlightItem]]:
        logger.info(f"CryptoSpotlightService: Bắt đầu lấy Crypto Spotlight, limit={limit}")

        if not FMP_API_KEY_FOR_SERVICE or FMP_API_KEY_FOR_SERVICE == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error("CryptoSpotlightService: FMP API Key không được cấu hình.")
            return None

        all_crypto_quotes_raw: Optional[List[Dict[str, Any]]] = None
        crypto_quotes_url = f"{BASE_FMP_URL}/v3/quotes/crypto?apikey={FMP_API_KEY_FOR_SERVICE}"
        logger.debug(f"CryptoSpotlightService: Gọi API: {crypto_quotes_url}")

        async with httpx.AsyncClient(timeout=25.0) as client:
            try:
                response = await client.get(crypto_quotes_url)
                response.raise_for_status()
                all_crypto_quotes_raw = response.json()
                if not isinstance(all_crypto_quotes_raw, list):
                    logger.warning(f"CryptoSpotlightService: API /v3/quotes/crypto không trả về list như mong đợi. Dữ liệu nhận được: {type(all_crypto_quotes_raw)}")
                    all_crypto_quotes_raw = None
            except httpx.HTTPStatusError as e:
                logger.error(f"CryptoSpotlightService: Lỗi HTTP khi lấy /v3/quotes/crypto: Status {e.response.status_code} - Response: {e.response.text[:200]}...")
                return None
            except httpx.RequestError as e:
                logger.error(f"CryptoSpotlightService: Lỗi request khi lấy /v3/quotes/crypto: {type(e).__name__} - {e}")
                return None
            except json.JSONDecodeError as e:
                logger.error(f"CryptoSpotlightService: Lỗi giải mã JSON từ /v3/quotes/crypto: {e}")
                return None


            if not all_crypto_quotes_raw:
                logger.info("CryptoSpotlightService: Không nhận được dữ liệu crypto quotes hoặc dữ liệu không hợp lệ.")
                return []

            def get_market_cap_safe(item_dict: Dict[str, Any]) -> float:
                mc = item_dict.get("marketCap")
                try:
                    return float(mc) if mc is not None else 0.0
                except (ValueError, TypeError):
                    logger.warning(f"CryptoSpotlightService: Không thể chuyển đổi marketCap '{mc}' của symbol '{item_dict.get('symbol')}' thành float.")
                    return 0.0

            sorted_cryptos = sorted(all_crypto_quotes_raw, key=get_market_cap_safe, reverse=True)
            spotlight_candidates_raw = sorted_cryptos[:limit] 
            logger.debug(f"CryptoSpotlightService: Số lượng ứng viên ban đầu: {len(all_crypto_quotes_raw)}, số ứng viên spotlight: {len(spotlight_candidates_raw)}")

            chart_tasks = []
            symbols_for_charting = [item.get("symbol") for item in spotlight_candidates_raw if item.get("symbol")]

            to_date = date.today()
            from_date = to_date - timedelta(days=3)
            interval = "4hour"
            
            for symbol in symbols_for_charting:
                task = HistoryChartService.get_historical_chart_from_fmp(
                    symbol=symbol,
                    interval=interval,
                    start_date_str=from_date.isoformat(),
                    end_date_str=to_date.isoformat(),
                    client=client,
                    redis_client=None
                )
                chart_tasks.append(task)
            logger.info(f"Bắt đầu lấy dữ liệu biểu đồ cho {len(chart_tasks)} mã crypto.")
            chart_results = await asyncio.gather(*chart_tasks, return_exceptions=True)
            chart_data_map: Dict[str, List[ChartDataItem]] = {}
            for i, result_or_exc in enumerate(chart_results):
                symbol = symbols_for_charting[i]
                if isinstance(result_or_exc, Exception):
                    logger.error(f"Lỗi khi lấy dữ liệu biểu đồ cho {symbol}: {result_or_exc}")
                elif isinstance(result_or_exc, list):
                    chart_data_map[symbol] = result_or_exc

            final_spotlight_items: List[CryptoSpotlightItem] = []
            for raw_item_data in spotlight_candidates_raw:
                symbol = raw_item_data.get("symbol")
                if not symbol:
                    logger.debug("CryptoSpotlightService: Bỏ qua item không có symbol trong spotlight_candidates_raw.")
                    continue

                chart_data_for_item = chart_data_map.get(symbol, [])
                item = EquityMapper.map_fmp_crypto_quote_to_spotlight_item(
                    fmp_crypto_item_data=raw_item_data,
                    logo_url=None
                )

                if item:
                    item.chartData = chart_data_for_item
                    final_spotlight_items.append(item)
                else:
                    logger.warning(f"CryptoSpotlightService: Không thể map FMP data cho symbol '{symbol}' sang SpotlightItem.")

            logger.info(f"CryptoSpotlightService: Hoàn thành lấy Crypto Spotlight. Trả về {len(final_spotlight_items)} items.")
            return final_spotlight_items