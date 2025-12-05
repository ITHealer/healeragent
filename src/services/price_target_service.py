
import asyncio
from datetime import datetime, timedelta, timezone
import httpx
import logging
from typing import List, Optional, Dict, Any
import aioredis
from src.services.history_chart_service import HistoryChartService
from src.models.equity import PriceTargetItem, PriceTargetWithChartOutput
from src.utils.config import settings

FMP_API_KEY = settings.FMP_API_KEY
BASE_FMP_URL = settings.BASE_FMP_URL
FMP_URL_STABLE = settings.FMP_URL_STABLE
logger = logging.getLogger(__name__)

class PriceTargetService:
    async def get_price_targets_from_fmp(self, page: int, client: httpx.AsyncClient) -> Optional[List[PriceTargetItem]]:
        """Lấy danh sách mục tiêu giá thô từ API FMP."""
        url = f"{BASE_FMP_URL}/v4/price-target-rss-feed?page={page}&apikey={FMP_API_KEY}"
        logger.info(f"Fetching price targets from FMP for page {page}.")
        try:
            response = await client.get(url, timeout=20.0)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                logger.warning(f"Price target API returned non-list data: {type(data)}")
                return None
        
            return [PriceTargetItem(**item) for item in data if isinstance(item, dict)]
        except Exception as e:
            logger.error(f"Failed to fetch price targets from FMP: {e}", exc_info=True)
            return None
        
    async def get_price_target_for_symbol(
        self, 
        symbol: str,
        client: Optional[httpx.AsyncClient] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Lấy price target cho một symbol cụ thể (nếu FMP có endpoint riêng).
        
        Note: Endpoint này có thể là:
        - /v3/price-target/{symbol}
        - /v4/price-target-consensus/{symbol}
        """
        should_close = False
        if client is None:
            client = httpx.AsyncClient()
            should_close = True
        
        try:
            # Thử endpoint consensus (aggregate data)
            url = f"{FMP_URL_STABLE}/price-target-consensus?symbol={symbol}&apikey={FMP_API_KEY}"
            logger.info(f"Fetching price target consensus for {symbol}")
            
            response = await client.get(url, timeout=90.0)
            response.raise_for_status()
            data = response.json()
            
            return data if data else None
            
        except Exception as e:
            logger.error(f"Failed to fetch price target for {symbol}: {e}", exc_info=True)
            return None
        finally:
            if should_close:
                await client.aclose()

    async def get_price_targets_with_charts(
        self, 
        page: int, 
        redis_client: Optional[aioredis.Redis]
    ) -> Optional[List[PriceTargetWithChartOutput]]:
        """
        Hàm chính: Lấy mục tiêu giá và làm giàu chúng bằng dữ liệu biểu đồ.
        """
        async with httpx.AsyncClient() as client:
            # Bước 1: Lấy danh sách mục tiêu giá
            price_targets = await self.get_price_targets_from_fmp(page, client)
            if price_targets is None:
                return None
            if not price_targets:
                return [] 

            # Bước 2: Tạo các tác vụ lấy chart cho mỗi mục tiêu giá một cách đồng thời
            async def _fetch_chart_for_target(target: PriceTargetItem):
                end_d = datetime.now(timezone.utc)
                start_d = end_d - timedelta(days=7)
                chart_data = await HistoryChartService.get_historical_chart_from_fmp(
                    symbol=target.symbol,
                    interval="4hour",
                    start_date_str=start_d.strftime("%Y-%m-%d"),
                    end_date_str=end_d.strftime("%Y-%m-%d"),
                    client=client,
                    redis_client=redis_client
                )
                return PriceTargetWithChartOutput(
                    **target.model_dump(), 
                    chart_data=chart_data
                )

            tasks = [_fetch_chart_for_target(target) for target in price_targets]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            final_list = []
            for res in results:
                if isinstance(res, Exception):
                    logger.error(f"An error occurred while enriching price target with chart: {res}")
                elif res is not None:
                    final_list.append(res)
            
            return final_list

price_target_service = PriceTargetService()