import json
import httpx
from typing import Optional, Any
import logging

import os
from dotenv import load_dotenv

from src.utils.logger.set_up_log_dataFMP import setup_logger

load_dotenv()

FMP_API_KEY_FOR_SERVICE = os.getenv("FMP_API_KEY", "YOUR_FMP_API_KEY_PLACEHOLDER")
BASE_FMP_URL = "https://financialmodelingprep.com/api"

logger = setup_logger(__name__, log_level=logging.INFO) 

if FMP_API_KEY_FOR_SERVICE == "YOUR_FMP_API_KEY_PLACEHOLDER" or not FMP_API_KEY_FOR_SERVICE :
    logger.warning("FMP_API_KEY chưa được thiết lập đúng trong biến môi trường hoặc bị thiếu.")

class EquityService:
    def __init__(self):
        pass

    async def get_logo(self, symbol: str) -> Optional[str]:
        """
        Lấy URL logo cho một mã cổ phiếu từ API Profile của FMP.
        """
        upper_symbol = symbol.upper()
        endpoint = f"/v3/profile/{upper_symbol}"
        logger.info(f"Đang lấy logo cho symbol: {upper_symbol}")

        async with httpx.AsyncClient(timeout=10.0) as client:
            data = await self.fetch_fmp_data_helper(client, endpoint, upper_symbol)
            logo_url = None
            if isinstance(data, dict) and data:
                logo_url = data.get('image')
            elif isinstance(data, list) and data: 
                if isinstance(data[0], dict):
                    logo_url = data[0].get('image')
                else:
                    logger.warning(f"Dữ liệu logo cho {upper_symbol} là list nhưng item đầu tiên không phải dict: {type(data[0])}")
            else:
                logger.warning(f"Không nhận được dữ liệu hoặc định dạng không mong đợi cho logo của {upper_symbol}. Dữ liệu: {data}")

            if logo_url:
                logger.info(f"Tìm thấy logo cho {upper_symbol}: {logo_url}")
            else:
                logger.info(f"Không tìm thấy logo cho {upper_symbol}.")
            return logo_url

    async def fetch_fmp_data_helper(self, client: httpx.AsyncClient, endpoint_template: str, symbol: str) -> Optional[Any]:
        """
        Hàm helper để lấy dữ liệu từ FMP cho một symbol cụ thể.
        Trả về dữ liệu JSON đã parse.
        """
        if not FMP_API_KEY_FOR_SERVICE or FMP_API_KEY_FOR_SERVICE == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error(f"FMP API Key không được cấu hình. Không thể gọi endpoint '{endpoint_template}' cho symbol '{symbol}'.")
            return None

        endpoint_path_with_symbol = endpoint_template.replace("{symbol}", symbol.upper())

        if "?" in endpoint_path_with_symbol: 
            url = f"{BASE_FMP_URL}{endpoint_path_with_symbol}&apikey={FMP_API_KEY_FOR_SERVICE}"
        else:
            url = f"{BASE_FMP_URL}{endpoint_path_with_symbol}?apikey={FMP_API_KEY_FOR_SERVICE}"

        logger.debug(f"Gọi FMP API: {url}")

        try:
            response = await client.get(url, timeout=12.0)
            response.raise_for_status() 
            data = response.json()
            logger.debug(f"Nhận được phản hồi thành công từ FMP cho {symbol} tại {endpoint_template}. Loại dữ liệu: {type(data)}")
            if isinstance(data, list):
                return data[0] if data else None 
            elif isinstance(data, dict):
                return data 
            else:
                logger.warning(f"Định dạng dữ liệu không mong muốn từ FMP cho {symbol} tại {endpoint_template}. Nhận được: {type(data)}. URL: {url}")
                return None

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Lỗi HTTPStatusError khi gọi FMP cho {symbol} (endpoint: {endpoint_template}): "
                f"Status {e.response.status_code} - Response: {e.response.text[:200]}. URL: {url}",
                exc_info=False 
            )
        except httpx.RequestError as e:
            logger.error(
                f"Lỗi RequestError khi gọi FMP cho {symbol} (endpoint: {endpoint_template}): {type(e).__name__} - {e}. URL: {url}",
                exc_info=True 
            )
        except json.JSONDecodeError as e: 
            logger.error(
                f"Lỗi JSONDecodeError khi xử lý phản hồi từ FMP cho {symbol} (endpoint: {endpoint_template}): {e}. "
                f"Response text (đầu): {response.text[:200] if 'response' in locals() else 'N/A'}. URL: {url}",
                exc_info=False
            )
        except Exception as e:
            logger.exception( 
                f"Lỗi không xác định khi gọi FMP cho {symbol} (endpoint: {endpoint_template}). URL: {url}"
            )
        return None