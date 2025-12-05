
import httpx
import logging
from typing import List, Optional

from src.models.equity import SenateTradingItem
from src.utils.config import settings
FMP_API_KEY = settings.FMP_API_KEY
FMP_URL_STABLE = settings.FMP_URL_STABLE
logger = logging.getLogger(__name__)

class SenateTradingService:
    async def get_latest_senate_trades(self, page: int, limit: int) -> Optional[List[SenateTradingItem]]:
        """Lấy danh sách các giao dịch của Thượng viện từ FMP."""
        
        url = f"{FMP_URL_STABLE}/senate-latest?page={page}&limit={limit}&apikey={FMP_API_KEY}"
        logger.info(f"Fetching latest senate trades from FMP for page {page}, limit {limit}.")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
                if not isinstance(data, list):
                    logger.warning(f"Senate trading API returned non-list data: {type(data)}")
                    return None
                return [SenateTradingItem(**item) for item in data if isinstance(item, dict)]
            except httpx.HTTPStatusError as hse:
                logger.error(f"HTTP error fetching senate trades: {hse.response.status_code} - {hse.response.text[:200]}")
                return None
            except Exception as e:
                logger.error(f"Failed to fetch or process senate trades: {e}", exc_info=True)
                return None
            
    async def get_trades_by_senator_name(self, name: str) -> Optional[List[SenateTradingItem]]:
        """
        Lấy danh sách các giao dịch của một Thượng nghị sĩ cụ thể dựa vào tên.
        """
        from urllib.parse import quote_plus
        encoded_name = quote_plus(name)
        
        endpoint_path = f"/senate-trades-by-name?name={encoded_name}&apikey={FMP_API_KEY}"
        url = f"{FMP_URL_STABLE}{endpoint_path}"
        
        logger.info(f"Fetching senate trades for senator name: '{name}'")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                fmp_data = response.json()
                
                if not isinstance(fmp_data, list):
                    logger.warning(f"Senate trades by name API returned non-list data for '{name}'.")
                    return None

                trades = [SenateTradingItem(**item) for item in fmp_data if isinstance(item, dict)]
                return trades
                
            except httpx.HTTPStatusError as hse:
                logger.error(f"HTTP error fetching senate trades for name '{name}': {hse.response.status_code} - {hse.response.text[:200]}")
                return None
            except Exception as e:
                logger.error(f"Failed to fetch or process senate trades for name '{name}': {e}", exc_info=True)
                return None

senate_trading_service = SenateTradingService()