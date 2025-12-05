# import httpx
# from typing import List, Optional, Dict, Any
# import logging
# import json

# from src.models.equity import TickerTapeData
# from src.mappers.equity_mapper import EquityMapper
# from src.utils.logger.set_up_log_dataFMP import setup_logger 

# import os
# from dotenv import load_dotenv

# load_dotenv()

# logger = setup_logger(__name__, log_level=logging.INFO)

# FMP_API_KEY_FOR_SERVICE = os.getenv("FMP_API_KEY", "YOUR_FMP_API_KEY_PLACEHOLDER")
# BASE_FMP_URL = "https://financialmodelingprep.com/api"

# if not FMP_API_KEY_FOR_SERVICE or FMP_API_KEY_FOR_SERVICE == "YOUR_FMP_API_KEY_PLACEHOLDER":
#     logger.warning("FMP_API_KEY is not set correctly in environment variables for TickerTapeService.")

# class TickerTapeService:
#     @staticmethod 
#     async def get_ticker_tape_batch(symbols_list: List[str], provider: Optional[str] = None) -> List[Optional[TickerTapeData]]:
#         if not symbols_list:
#             logger.info("No symbols provided for ticker tape batch fetch.")
#             return []

#         if not FMP_API_KEY_FOR_SERVICE or FMP_API_KEY_FOR_SERVICE == "YOUR_FMP_API_KEY_PLACEHOLDER":
#             logger.error("FMP API Key not configured for ticker tape.")
#             return [None] * len(symbols_list)

#         symbols_str = ",".join(symbols_list).upper()
#         fmp_batch_quote_url = f"{BASE_FMP_URL}/v3/quote/{symbols_str}?apikey={FMP_API_KEY_FOR_SERVICE}"
#         logger.debug(f"Fetching FMP batch quote from URL: {fmp_batch_quote_url}")

#         raw_quote_data_list: Optional[List[Dict[str, Any]]] = None

#         async with httpx.AsyncClient(timeout=20.0) as client:
#             try:
#                 # logger.info(f"Fetching FMP batch quote for: {symbols_str[:200]}{'...' if len(symbols_str) > 200 else ''}")
#                 response = await client.get(fmp_batch_quote_url)
#                 response.raise_for_status()
#                 raw_quote_data_list = response.json()

#                 if not isinstance(raw_quote_data_list, list):
#                     logger.warning(f"FMP batch quote for {symbols_str[:100]} did not return a list. Response type: {type(raw_quote_data_list)}, Response: {str(raw_quote_data_list)[:200]}")
#                     raw_quote_data_list = None
#                 else:
#                     logger.debug(f"Successfully fetched {len(raw_quote_data_list)} raw quote items for batch: {symbols_str[:100]}")
#             except httpx.HTTPStatusError as e:
#                 logger.error(f"HTTPStatusError during FMP batch quote for {symbols_str[:100]}: {e.response.status_code} - {e.response.text[:200]}", exc_info=False)
#             except httpx.RequestError as e:
#                 logger.error(f"RequestError during FMP batch quote for {symbols_str[:100]}: {e}", exc_info=True)
#             except json.JSONDecodeError as e:
#                 logger.error(f"JSONDecodeError during FMP batch quote for {symbols_str[:100]}: {e}. Response: {response.text[:200] if 'response' in locals() else 'N/A'}", exc_info=False)
#             except Exception as e:
#                 logger.exception(f"General exception during FMP batch quote for {symbols_str[:100]}")

#         if raw_quote_data_list is None: 
#             logger.error(f"Failed to fetch any raw quote data for batch: {symbols_str[:100]}")
#             return [None] * len(symbols_list)
#         if not raw_quote_data_list: 
#              logger.info(f"FMP batch quote returned an empty list for: {symbols_str[:100]}")
#              return [None] * len(symbols_list)


#         results_dict: Dict[str, TickerTapeData] = {}
#         successful_maps = 0
#         for item_data in raw_quote_data_list:
#             if isinstance(item_data, dict) and item_data.get("symbol"):
#                 try:
#                     mapped_item = EquityMapper.map_fmp_quote_to_ticker_tape_item(item_data)
#                     if mapped_item:
#                         results_dict[mapped_item.symbol.upper()] = mapped_item 
#                         successful_maps +=1
#                     else:
#                         logger.warning(f"Failed to map FMP quote item to TickerTapeData for symbol '{item_data.get('symbol')}'. Data: {item_data}")
#                 except Exception as map_err:
#                     logger.error(f"Error mapping FMP quote item for symbol '{item_data.get('symbol')}': {map_err}. Data: {item_data}", exc_info=True)
#             else:
#                 logger.warning(f"Skipping invalid item in FMP batch quote response: {item_data}")

#         logger.debug(f"Successfully mapped {successful_maps} items out of {len(raw_quote_data_list)} raw quote items.")

#         final_results: List[Optional[TickerTapeData]] = []
#         for s_input in symbols_list:
#             found_item = results_dict.get(s_input.upper())
#             if found_item is None:
#                 logger.debug(f"No ticker tape data found for symbol '{s_input.upper()}' after mapping.")
#             final_results.append(found_item)

#         # logger.info(f"Ticker tape batch processing complete. Returning {len([r for r in final_results if r is not None])} valid items out of {len(symbols_list)} requested.")
#         return final_results


import httpx
from typing import List, Optional, Dict, Any
import logging
import json
from datetime import datetime, timedelta

from src.models.equity import TickerTapeData, ChartDataItem
from src.mappers.equity_mapper import EquityMapper
from src.utils.logger.set_up_log_dataFMP import setup_logger 

import os
from dotenv import load_dotenv

load_dotenv()

logger = setup_logger(__name__, log_level=logging.INFO)

FMP_API_KEY_FOR_SERVICE = os.getenv("FMP_API_KEY", "YOUR_FMP_API_KEY_PLACEHOLDER")
BASE_FMP_URL = "https://financialmodelingprep.com/api"

if not FMP_API_KEY_FOR_SERVICE or FMP_API_KEY_FOR_SERVICE == "YOUR_FMP_API_KEY_PLACEHOLDER":
    logger.warning("FMP_API_KEY is not set correctly in environment variables for TickerTapeService.")

class TickerTapeService:
    
    @staticmethod
    async def _fetch_intraday_chart(
        symbol: str,
        client: httpx.AsyncClient,
        timeframe: str = "1hour"
    ) -> List[ChartDataItem]:
        """
        Fetch intraday historical chart data from FMP API
        """
        # 1. Validation Checks
        if not FMP_API_KEY_FOR_SERVICE or FMP_API_KEY_FOR_SERVICE == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.warning(f"Cannot fetch chart data for {symbol}: FMP API Key not configured")
            return []
        
        # 2. Construct URL (No Date Filter to avoid timezone issues)
        fmp_chart_url = (
            f"{BASE_FMP_URL}/v3/historical-chart/{timeframe}/{symbol.upper()}"
            f"?apikey={FMP_API_KEY_FOR_SERVICE}"
        )
        
        try:
            response = await client.get(fmp_chart_url)
            
            # Handle non-200 responses gracefully
            if response.status_code == 404:
                return []
            response.raise_for_status()
            
            raw_chart_data = response.json()
            
            if not isinstance(raw_chart_data, list):
                logger.warning(f"FMP chart API for {symbol} returned non-list: {type(raw_chart_data)}")
                return []
            
            # 3. Optimize: Take only recent 24 points for Sparkline
            # FMP returns data sorted Newest -> Oldest
            recent_data = raw_chart_data[:24]
            
            chart_items = []
            for item in recent_data:
                # Ensure item is a dict and has required keys
                if isinstance(item, dict) and item.get("date") and item.get("close") is not None:
                    try:
                        # --- FIX IS HERE ---
                        # Map FMP 'close' to ChartDataItem 'value'
                        chart_items.append(
                            ChartDataItem(
                                time=item["date"],
                                value=float(item["close"]) # Changed from 'price' to 'value' to match Model
                            )
                        )
                    except Exception as e:
                        # Log error to see if mapping fails again
                        logger.error(f"Mapping error for {symbol} at {item.get('date')}: {e}")
                        continue
            
            # 4. Reverse to Chronological Order (Oldest -> Newest) for plotting
            chart_items.reverse()
            
            # logger.debug(f"Fetched {len(chart_items)} chart data points for {symbol}")
            return chart_items
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching chart for {symbol}: {e.response.status_code}")
            return []
        except Exception as e:
            logger.exception(f"Unexpected error fetching chart for {symbol}")
            return []
    
    @staticmethod 
    async def get_ticker_tape_batch(
        symbols_list: List[str], 
        provider: Optional[str] = None,
        include_chart: bool = True,
        chart_timeframe: str = "1hour"
    ) -> List[Optional[TickerTapeData]]:
        """
        Fetch ticker tape data with optional chart data
        
        Args:
            symbols_list: List of stock symbols
            provider: Data provider (currently only FMP supported)
            include_chart: Whether to fetch intraday chart data
            chart_timeframe: Chart timeframe for intraday data
            
        Returns:
            List of TickerTapeData objects (or None for failed symbols)
        """
        if not symbols_list:
            logger.info("No symbols provided for ticker tape batch fetch.")
            return []

        if not FMP_API_KEY_FOR_SERVICE or FMP_API_KEY_FOR_SERVICE == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error("FMP API Key not configured for ticker tape.")
            return [None] * len(symbols_list)

        symbols_str = ",".join(symbols_list).upper()
        fmp_batch_quote_url = f"{BASE_FMP_URL}/v3/quote/{symbols_str}?apikey={FMP_API_KEY_FOR_SERVICE}"
        logger.debug(f"Fetching FMP batch quote from URL: {fmp_batch_quote_url}")

        raw_quote_data_list: Optional[List[Dict[str, Any]]] = None

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                logger.info(f"Fetching FMP batch quote for: {symbols_str[:200]}{'...' if len(symbols_str) > 200 else ''}")
                response = await client.get(fmp_batch_quote_url)
                response.raise_for_status()
                raw_quote_data_list = response.json()

                if not isinstance(raw_quote_data_list, list):
                    logger.warning(f"FMP batch quote for {symbols_str[:100]} did not return a list. Response type: {type(raw_quote_data_list)}, Response: {str(raw_quote_data_list)[:200]}")
                    raw_quote_data_list = None
                else:
                    logger.debug(f"Successfully fetched {len(raw_quote_data_list)} raw quote items for batch: {symbols_str[:100]}")
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTPStatusError during FMP batch quote for {symbols_str[:100]}: {e.response.status_code} - {e.response.text[:200]}", exc_info=False)
            except httpx.RequestError as e:
                logger.error(f"RequestError during FMP batch quote for {symbols_str[:100]}: {e}", exc_info=True)
            except json.JSONDecodeError as e:
                logger.error(f"JSONDecodeError during FMP batch quote for {symbols_str[:100]}: {e}. Response: {response.text[:200] if 'response' in locals() else 'N/A'}", exc_info=False)
            except Exception as e:
                logger.exception(f"General exception during FMP batch quote for {symbols_str[:100]}")

            if raw_quote_data_list is None: 
                logger.error(f"Failed to fetch any raw quote data for batch: {symbols_str[:100]}")
                return [None] * len(symbols_list)
            if not raw_quote_data_list: 
                logger.info(f"FMP batch quote returned an empty list for: {symbols_str[:100]}")
                return [None] * len(symbols_list)

            # Map quote data to TickerTapeData
            results_dict: Dict[str, TickerTapeData] = {}
            successful_maps = 0
            for item_data in raw_quote_data_list:
                if isinstance(item_data, dict) and item_data.get("symbol"):
                    try:
                        mapped_item = EquityMapper.map_fmp_quote_to_ticker_tape_item(item_data)
                        if mapped_item:
                            results_dict[mapped_item.symbol.upper()] = mapped_item 
                            successful_maps += 1
                        else:
                            logger.warning(f"Failed to map FMP quote item to TickerTapeData for symbol '{item_data.get('symbol')}'. Data: {item_data}")
                    except Exception as map_err:
                        logger.error(f"Error mapping FMP quote item for symbol '{item_data.get('symbol')}': {map_err}. Data: {item_data}", exc_info=True)
                else:
                    logger.warning(f"Skipping invalid item in FMP batch quote response: {item_data}")

            logger.debug(f"Successfully mapped {successful_maps} items out of {len(raw_quote_data_list)} raw quote items.")

            # Fetch chart data for all symbols if requested
            if include_chart and results_dict:
                logger.info(f"Fetching chart data for {len(results_dict)} symbols with timeframe {chart_timeframe}")
                
                # Fetch charts concurrently for all symbols
                import asyncio
                chart_tasks = [
                    TickerTapeService._fetch_intraday_chart(symbol, client, chart_timeframe)
                    for symbol in results_dict.keys()
                ]
                chart_results = await asyncio.gather(*chart_tasks, return_exceptions=True)
                
                # Attach chart data to each TickerTapeData object
                for symbol, chart_data in zip(results_dict.keys(), chart_results):
                    if isinstance(chart_data, list):
                        results_dict[symbol].chartData = chart_data
                        logger.debug(f"Attached {len(chart_data)} chart points to {symbol}")
                    else:
                        logger.warning(f"Failed to fetch chart for {symbol}: {chart_data}")
                        results_dict[symbol].chartData = []

        # Build final results list matching input order
        final_results: List[Optional[TickerTapeData]] = []
        for s_input in symbols_list:
            found_item = results_dict.get(s_input.upper())
            if found_item is None:
                logger.debug(f"No ticker tape data found for symbol '{s_input.upper()}' after mapping.")
            final_results.append(found_item)

        logger.info(f"Ticker tape batch processing complete. Returning {len([r for r in final_results if r is not None])} valid items out of {len(symbols_list)} requested.")
        return final_results