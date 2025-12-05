import httpx
from typing import List, Optional, Dict, Any
from datetime import timedelta
import logging
import json 

from src.models.equity import FMPSearchResultItem
from src.utils.logger.set_up_log_dataFMP import setup_logger 

import os
from dotenv import load_dotenv

load_dotenv()

logger = setup_logger(__name__, log_level=logging.INFO)

FMP_API_KEY_FOR_SERVICE = os.getenv("FMP_API_KEY", "YOUR_FMP_API_KEY_PLACEHOLDER")
BASE_FMP_URL = "https://financialmodelingprep.com/api"

if not FMP_API_KEY_FOR_SERVICE or FMP_API_KEY_FOR_SERVICE == "YOUR_FMP_API_KEY_PLACEHOLDER":
    logger.warning("FMP_API_KEY is not set correctly in environment variables for SearchService.")

class SearchService:
    def __init__(self):
        self._discovery_cache: Dict[str, Dict[str, Any]] = {}

    async def search_symbols_fmp(self, query_term: str, limit_count: Optional[int] = 10) -> Optional[List[FMPSearchResultItem]]:
        logger.info(f"Searching FMP for symbols with query: '{query_term}', limit: {limit_count}")

        if not FMP_API_KEY_FOR_SERVICE or FMP_API_KEY_FOR_SERVICE == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error("FMP API Key not configured for symbol search.")
            return None

        endpoint_path = "/v3/search"
        fmp_params: Dict[str, Any] = {"query": query_term}
        if limit_count is not None and limit_count > 0:
            fmp_params["limit"] = limit_count

        all_params_for_fmp = fmp_params.copy()
        all_params_for_fmp["apikey"] = FMP_API_KEY_FOR_SERVICE

        url = f"{BASE_FMP_URL}{endpoint_path}"
        log_params = {k:v for k,v in all_params_for_fmp.items() if k != "apikey"}
        logger.debug(f"Calling FMP search API: {url} with params: {log_params}")

        raw_search_results: Optional[List[Dict[str, Any]]] = None

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(url, params=all_params_for_fmp)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list):
                    raw_search_results = data
                    # logger.debug(f"Successfully fetched {len(raw_search_results)} raw search results from FMP for '{query_term}'.")
                else:
                    logger.warning(f"FMP search for '{query_term}' did not return a list. Response type: {type(data)}, Response: {str(data)[:200]}. URL: {url}")
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTPStatusError during FMP search for '{query_term}': {e.response.status_code} - {e.response.text[:200]}. URL: {url}", exc_info=False)
            except httpx.RequestError as e:
                logger.error(f"RequestError during FMP search for '{query_term}': {e}. URL: {url}", exc_info=True)
            except json.JSONDecodeError as e:
                logger.error(f"JSONDecodeError during FMP search for '{query_term}': {e}. Response: {response.text[:200] if 'response' in locals() else 'N/A'}. URL: {url}", exc_info=False)
            except Exception as e:
                logger.exception(f"General exception during FMP search for '{query_term}'. URL: {url}")


        if raw_search_results is None:
            logger.error(f"No raw search results obtained for '{query_term}' due to a previous error.")
            return None 
        if not raw_search_results: 
            logger.info(f"No search results found from FMP for '{query_term}'.")
            return []

        search_results: List[FMPSearchResultItem] = []
        for item_idx, item_data in enumerate(raw_search_results):
            if isinstance(item_data, dict):
                try:
                    search_results.append(FMPSearchResultItem(**item_data))
                except Exception as e_pydantic: 
                    logger.warning(f"Could not parse FMP search item for symbol '{item_data.get('symbol', 'N/A')}' at index {item_idx} due to Pydantic error: {e_pydantic}. Data: {item_data}")
            else:
                logger.warning(f"Search result item at index {item_idx} is not a dict: {item_data}")

        logger.info(f"Successfully parsed {len(search_results)} search results for '{query_term}'.")
        return search_results