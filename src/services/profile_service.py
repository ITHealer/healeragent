from typing import Any, Dict, Optional
import httpx
import logging

from src.mappers.equity_mapper import EquityMapper
from src.models.equity import FMPCompanyOutlookProfile
from src.services.equity_service import EquityService
from src.utils.logger.set_up_log_dataFMP import setup_logger
from src.utils.config import settings 

logger = setup_logger(__name__, log_level=logging.INFO)

FMP_API_KEY = settings.FMP_API_KEY
BASE_FMP_URL = settings.BASE_FMP_URL

equity_service_instance = EquityService()

class ProfileService:
    def __init__(self):
        self._discovery_cache: Dict[str, Dict[str, Any]] = {}

    async def _fetch_single_fmp_object(self, client: httpx.AsyncClient, endpoint_template: str, symbol: str) -> Optional[Dict[str, Any]]:
        logger.debug(f"Fetching single FMP object for {symbol} using endpoint template: {endpoint_template}")
        data = await equity_service_instance.fetch_fmp_data_helper(client, endpoint_template, symbol)
        if isinstance(data, list) and data:
            if isinstance(data[0], dict):
                # logger.debug(f"Successfully fetched and using first item from list for {symbol} at {endpoint_template}")
                return data[0]
            else:
                logger.warning(f"Fetched list for {symbol} at {endpoint_template}, but first item is not a dict: {type(data[0])}")
                return None
        elif isinstance(data, dict):
            # logger.debug(f"Successfully fetched dict data for {symbol} at {endpoint_template}")
            return data
        else:
            logger.warning(f"No data or unexpected data type ({type(data)}) for {symbol} at {endpoint_template}")
            return None

    async def _fetch_and_map_one_symbol_profile_fmp_direct(self, symbol: str, client: httpx.AsyncClient) -> Optional[FMPCompanyOutlookProfile]:
        logger.info(f"Starting to fetch detailed profile for {symbol} (up to 3 FMP API calls)")
        upper_symbol = symbol.upper()

        fmp_outlook_data = await self._fetch_single_fmp_object(client, "/v4/company-outlook?symbol={symbol}", upper_symbol)
        if not fmp_outlook_data:
            logger.warning(f"No FMP company-outlook data found for {upper_symbol}. Cannot proceed with detailed profile mapping.")
            return None
        logger.debug(f"Fetched company-outlook data for {upper_symbol}.")

        fmp_quote_data = await self._fetch_single_fmp_object(client, "/v3/quote/{symbol}", upper_symbol)
        if not fmp_quote_data:
            logger.info(f"No supplementary quote data found for {upper_symbol}. Proceeding without it.")
        else:
            logger.debug(f"Fetched supplementary quote data for {upper_symbol}.")


        fmp_key_metrics_ttm = await self._fetch_single_fmp_object(client, "/v3/key-metrics-ttm/{symbol}", upper_symbol)
        if not fmp_key_metrics_ttm:
            logger.info(f"No key metrics TTM data found for {upper_symbol}. Proceeding without it.")
        else:
            logger.debug(f"Fetched key metrics TTM data for {upper_symbol}.")


        try:
            mapped_profile = EquityMapper.map_all_fmp_to_detailed_profile(
                symbol_upper=upper_symbol,
                outlook_data=fmp_outlook_data, 
                quote_data_supplement=fmp_quote_data, 
                key_metrics_ttm_data=fmp_key_metrics_ttm
            )
            if mapped_profile:
                logger.info(f"Successfully mapped detailed profile for {upper_symbol}.")
            else:
                logger.warning(f"EquityMapper returned None for detailed profile of {upper_symbol}.")
            return mapped_profile
        except Exception as e:
            logger.error(f"Error mapping FMP data to detailed profile for {upper_symbol}: {e}", exc_info=True)
            return None


    async def get_company_profile(self, symbol: str, provider: Optional[str] = None) -> Optional[FMPCompanyOutlookProfile]:
        if provider:
            logger.debug(f"Provider '{provider}' was passed to get_company_profile but current implementation uses FMP direct.")
        logger.info(f"Fetching company profile for symbol: {symbol}")
        async with httpx.AsyncClient(timeout=35.0) as client:
            return await self._fetch_and_map_one_symbol_profile_fmp_direct(symbol, client)
        
    async def get_company_description(self, symbol: str) -> Optional[str]:
        """
        Lấy mô tả của một công ty từ API Profile của FMP.
        """
        logger.info(f"Bắt đầu lấy mô tả cho symbol: {symbol}")

        if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error(f"FMP API Key không được cấu hình cho profile của {symbol}.")
            return None

        endpoint = f"/v3/profile/{symbol.upper()}"
        url = f"{BASE_FMP_URL}{endpoint}"
        params = {"apikey": FMP_API_KEY}

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data_list = response.json()

                if isinstance(data_list, list) and data_list:
                    profile_data = data_list[0]
                    description = profile_data.get("description")
                    if description:
                        logger.info(f"Lấy thành công mô tả cho {symbol}.")
                        return description
                    else:
                        logger.warning(f"Không tìm thấy trường 'description' trong dữ liệu profile cho {symbol}.")
                        return None
                else:
                    logger.warning(f"FMP không trả về dữ liệu profile hợp lệ cho {symbol}.")
                    return None
            except httpx.HTTPStatusError as e:
                logger.error(f"Lỗi HTTPStatusError khi lấy profile cho {symbol}: {e.response.status_code} - {e.response.text[:200]}", exc_info=False)
            except Exception as e:
                logger.exception(f"Lỗi không xác định khi lấy profile cho {symbol}")
        
        return None