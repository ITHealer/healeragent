
import logging
from typing import Any, Dict, List, Optional
from src.models.equity import AnalystEstimateItem, FinancialStatementGrowthItem, KeyMetricsTTMItem, FinancialRatiosItem
from src.utils.logger.set_up_log_dataFMP import setup_logger
from src.utils.config import settings
import httpx
import json

logger = setup_logger(__name__, log_level=logging.INFO) 
FMP_API_KEY = settings.FMP_API_KEY 
BASE_FMP_URL = settings.BASE_FMP_URL 
FMP_URL_STABLE = settings.FMP_URL_STABLE

class ToolCallService:

    async def get_financial_statement_growth(
        self,
        symbol: str,
        period: str = "annual", 
        limit: int = 20
    ) -> Optional[List[FinancialStatementGrowthItem]]:
        """
        Get financial report growth data from FMP.
        """
        # logger.info(f"Bắt đầu lấy dữ liệu tăng trưởng BCTC cho {symbol}, period: {period}, limit: {limit}")

        if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error(f"FMP API Key is not configured for financial statement growth {symbol}.")
            return None

        endpoint = f"/v3/financial-growth/{symbol.upper()}"
        params = {"period": period, "limit": limit, "apikey": FMP_API_KEY}
        # log_params_display = {k:v for k,v in params.items() if k != "apikey"}
        url = f"{BASE_FMP_URL}{endpoint}"
        # logger.debug(f"Gọi FMP API: {url} với params: {log_params_display}")

        raw_data_list: Optional[List[Dict[str, Any]]] = None
        async with httpx.AsyncClient(timeout=25.0) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                raw_data_list = response.json()
                if not isinstance(raw_data_list, list):
                    logger.warning(f"FMP financial statement growth data for {symbol} ({period}) is not a list: {type(raw_data_list)}. URL: {url}")
                    return None 
                # logger.info(f"Successfully fetched {len(raw_data_list)} records of financial statement growth for {symbol} ({period}).")
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTPStatusError error when getting financial statement growth {symbol} ({period}): {e.response.status_code} - {e.response.text[:200]}", exc_info=False)
                return None
            except httpx.RequestError as e:
                logger.error(f"RequestError error when getting financial statement growth {symbol} ({period}): {e}", exc_info=True)
                return None
            except json.JSONDecodeError as e:
                logger.error(f"JSONDecodeError error when getting financial statement growth {symbol} ({period}): {e}. Response: {response.text[:200] if 'response' in locals() else 'N/A'}", exc_info=False)
                return None
            except Exception as e:
                logger.exception(f"Unknown error when retrieving financial statement growth for {symbol} ({period})")
                return None

        if raw_data_list is None:
            return None
        if not raw_data_list:
             logger.info(f"No financial statement growth data is available for {symbol}, period {period}, limit {limit}.")
             return []

        parsed_list: List[FinancialStatementGrowthItem] = []
        for item_raw in raw_data_list:
            try:
                parsed_item = FinancialStatementGrowthItem(**item_raw)
                parsed_list.append(parsed_item)
            except Exception as e_parse:
                logger.warning(f"Error parsing item for FinancialStatementGrowthItem (symbol: {symbol}): {item_raw}. Error: {e_parse}")
        
        logger.info(f"Parse successfully {len(parsed_list)} items FinancialStatementGrowthItem for {symbol}.")
        return parsed_list
    
    async def get_financial_ratios(
        self,
        symbol: str,
        limit: Optional[int] = 10,
        period: Optional[str] = "annual",   # Q1,Q2,Q3,Q4,FY,annual,quarter
    ) -> Optional[List[FinancialRatiosItem]]:
        """
        Lấy dữ liệu Financial Ratios từ FMP (/stable/ratios?symbol=...).
        Trả về list FinancialRatiosItem (mỗi item là 1 kỳ).
        """
        logger.info(f"Bắt đầu lấy Financial Ratios cho {symbol}")

        if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error(f"FMP API Key không được cấu hình cho Financial Ratios của {symbol}.")
            return None

        # endpoint = "/stable/ratios"
        endpoint = "/ratios"
        url = f"{FMP_URL_STABLE}{endpoint}"

        # Query params
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "apikey": FMP_API_KEY,
        }
        if limit is not None:
            params["limit"] = limit
        if period:
            params["period"] = period  # ví dụ: "FY" hoặc "annual"

        logger.debug(f"Gọi FMP API: {url} | params={params}")

        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                raw_list = response.json()

                if not isinstance(raw_list, list):
                    logger.warning(f"Dữ liệu FMP Financial Ratios cho {symbol} không phải list: {type(raw_list)}. URL: {url}")
                    return None
                if not raw_list:
                    logger.info(f"FMP trả về list rỗng cho Financial Ratios của {symbol}. URL: {url}")
                    return None

                logger.info(f"Lấy thành công {len(raw_list)} bản ghi Financial Ratios cho {symbol}.")

                # Parse sang Pydantic model; cho phép field thừa nên không sợ thiếu field
                parsed_items: List[FinancialRatiosItem] = []
                for i, item_raw in enumerate(raw_list):
                    if not isinstance(item_raw, dict):
                        logger.warning(f"Item thứ {i} trong Financial Ratios của {symbol} không phải dict: {type(item_raw)}")
                        continue
                    parsed_items.append(FinancialRatiosItem(**item_raw))

                if not parsed_items:
                    logger.info(f"Không parse được bản ghi hợp lệ nào cho Financial Ratios của {symbol}.")
                    return None

                logger.info(f"Parse thành công {len(parsed_items)} FinancialRatiosItem cho {symbol}.")
                return parsed_items

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"Lỗi HTTPStatusError khi lấy Financial Ratios cho {symbol}: "
                    f"{e.response.status_code} - {e.response.text[:200]}",
                    exc_info=False,
                )
            except httpx.RequestError as e:
                logger.error(f"Lỗi RequestError khi lấy Financial Ratios cho {symbol}: {e}", exc_info=True)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Lỗi JSONDecodeError khi lấy Financial Ratios cho {symbol}: {e}. "
                    f"Response: {response.text[:200] if 'response' in locals() else 'N/A'}",
                    exc_info=False,
                )
            except Exception:
                logger.exception(f"Lỗi không xác định khi lấy Financial Ratios cho {symbol}")

        return None
    
    async def get_key_metrics_ttm(
        self,
        symbol: str,
    ) -> Optional[KeyMetricsTTMItem]:
        """
        Lấy dữ liệu Key Metrics TTM từ FMP.
        FMP thường trả về một list chứa một object, chúng ta sẽ lấy object đó.
        """
        logger.info(f"Bắt đầu lấy Key Metrics TTM cho {symbol}")

        if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error(f"FMP API Key không được cấu hình cho Key Metrics TTM của {symbol}.")
            return None

        endpoint = f"/v3/key-metrics-ttm/{symbol.upper()}"
        params = {"apikey": FMP_API_KEY}
        url = f"{BASE_FMP_URL}{endpoint}"
        logger.debug(f"Gọi FMP API: {url}") 
        raw_data_list: Optional[List[Dict[str, Any]]] = None
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                raw_data_list = response.json()

                if not isinstance(raw_data_list, list):
                    logger.warning(f"Dữ liệu FMP Key Metrics TTM cho {symbol} không phải list: {type(raw_data_list)}. URL: {url}")
                    return None
                if not raw_data_list:
                    logger.info(f"FMP trả về list rỗng cho Key Metrics TTM của {symbol}. URL: {url}")
                    return None

                logger.info(f"Lấy thành công {len(raw_data_list)} bản ghi Key Metrics TTM (mong đợi 1) cho {symbol}.") 
                item_raw = raw_data_list[0]
                if not isinstance(item_raw, dict):
                    logger.warning(f"Item đầu tiên trong Key Metrics TTM cho {symbol} không phải dict: {type(item_raw)}")
                    return None

                parsed_item = KeyMetricsTTMItem(**item_raw)
                logger.info(f"Parse thành công KeyMetricsTTMItem cho {symbol}.")
                return parsed_item

            except httpx.HTTPStatusError as e:
                logger.error(f"Lỗi HTTPStatusError khi lấy Key Metrics TTM cho {symbol}: {e.response.status_code} - {e.response.text[:200]}", exc_info=False)
            except httpx.RequestError as e:
                logger.error(f"Lỗi RequestError khi lấy Key Metrics TTM cho {symbol}: {e}", exc_info=True)
            except json.JSONDecodeError as e:
                logger.error(f"Lỗi JSONDecodeError khi lấy Key Metrics TTM cho {symbol}: {e}. Response: {response.text[:200] if 'response' in locals() else 'N/A'}", exc_info=False)
            except IndexError:
                logger.warning(f"FMP trả về list rỗng cho Key Metrics TTM của {symbol} (sau khi kiểm tra list). URL: {url}")
            except Exception as e:
                logger.exception(f"Lỗi không xác định khi lấy Key Metrics TTM cho {symbol}")
        return None
    
    async def get_analyst_estimates(
        self,
        symbol: str,
        period: Optional[str] = None,
        limit: Optional[int] = None   
    ) -> Optional[List[AnalystEstimateItem]]:
        """
        Lấy dữ liệu ước tính của các nhà phân tích từ FMP.
        """
        logger.info(f"Bắt đầu lấy Analyst Estimates cho {symbol}, period: {period}, limit: {limit}")

        if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error(f"FMP API Key không được cấu hình cho Analyst Estimates của {symbol}.")
            return None

        endpoint = f"/v3/analyst-estimates/{symbol.upper()}"
        params = {"apikey": FMP_API_KEY}
        if period:
            params["period"] = period
        if limit:
            params["limit"] = limit
        
        log_params_display = {k:v for k,v in params.items() if k != "apikey"}
        url = f"{BASE_FMP_URL}{endpoint}"
        logger.debug(f"Gọi FMP API: {url} với params: {log_params_display}")

        raw_data_list: Optional[List[Dict[str, Any]]] = None
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                raw_data_list = response.json()

                if not isinstance(raw_data_list, list):
                    logger.warning(f"Dữ liệu FMP Analyst Estimates cho {symbol} không phải list: {type(raw_data_list)}. URL: {url}")
                    return None
                logger.info(f"Lấy thành công {len(raw_data_list)} bản ghi Analyst Estimates cho {symbol}.")

            except httpx.HTTPStatusError as e:
                logger.error(f"Lỗi HTTPStatusError khi lấy Analyst Estimates cho {symbol}: {e.response.status_code} - {e.response.text[:200]}", exc_info=False)
                return None
            except httpx.RequestError as e:
                logger.error(f"Lỗi RequestError khi lấy Analyst Estimates cho {symbol}: {e}", exc_info=True)
                return None
            except json.JSONDecodeError as e:
                logger.error(f"Lỗi JSONDecodeError khi lấy Analyst Estimates cho {symbol}: {e}. Response: {response.text[:200] if 'response' in locals() else 'N/A'}", exc_info=False)
                return None
            except Exception as e:
                logger.exception(f"Lỗi không xác định khi lấy Analyst Estimates cho {symbol}")
                return None

        if raw_data_list is None:
            return None
        if not raw_data_list:
             logger.info(f"Không có dữ liệu Analyst Estimates nào cho {symbol} với các tham số đã cho.")
             return []

        parsed_list: List[AnalystEstimateItem] = []
        for item_raw in raw_data_list:
            try:
                parsed_item = AnalystEstimateItem(**item_raw)
                parsed_list.append(parsed_item)
            except Exception as e_parse:
                logger.warning(f"Lỗi parse item cho AnalystEstimateItem (symbol: {symbol}): {item_raw}. Lỗi: {e_parse}")
        
        logger.info(f"Parse thành công {len(parsed_list)} items AnalystEstimateItem cho {symbol}.")
        return parsed_list
    
    
    # Fundamental tool
    async def get_key_metrics(
        self,
        symbol: str,
        limit: int = 1
    ) -> Optional[List[Dict[str, Any]]]:
        """Get key metrics including P/E, P/B, ROE, ROA, beta, etc."""
        
        if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error(f"FMP API Key not configured")
            return None
            
        endpoint = f"/v3/key-metrics/{symbol.upper()}"
        params = {"limit": limit, "apikey": FMP_API_KEY}
        url = f"{BASE_FMP_URL}{endpoint}"
        
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list):
                    logger.info(f"Successfully retrieved {len(data)} key metrics for {symbol}")
                    return data
                return None
            except Exception as e:
                logger.error(f"Error when getting key metrics {symbol}: {e}")
                return None

    async def get_income_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """Get Income Statement to calculate margins"""
        
        endpoint = f"/v3/income-statement/{symbol.upper()}"
        params = {"period": period, "limit": limit, "apikey": FMP_API_KEY}
        url = f"{BASE_FMP_URL}{endpoint}"
        
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"Income Statement Error {symbol}: {e}")
                return None

    async def get_balance_sheet(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """Get Balance Sheet to calculate D/E, current ratio"""
        
        endpoint = f"/v3/balance-sheet-statement/{symbol.upper()}"
        params = {"period": period, "limit": limit, "apikey": FMP_API_KEY}
        url = f"{BASE_FMP_URL}{endpoint}"
        
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"Balance Sheet Error {symbol}: {e}")
                return None

    async def get_cash_flow_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """Take Cash Flow Statement to calculate FCF"""
        
        endpoint = f"/v3/cash-flow-statement/{symbol.upper()}"
        params = {"period": period, "limit": limit, "apikey": FMP_API_KEY}
        url = f"{BASE_FMP_URL}{endpoint}"
        
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"Cash Flow Error {symbol}: {e}")
                return None

    async def get_quote(
        self,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Get quote data including current price and market cap"""
        
        endpoint = f"/v3/quote/{symbol.upper()}"
        params = {"apikey": FMP_API_KEY}
        url = f"{BASE_FMP_URL}{endpoint}"
        
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                return None
            except Exception as e:
                logger.error(f"Quote Error {symbol}: {e}")
                return None