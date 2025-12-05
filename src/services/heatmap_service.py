import httpx
import json
from typing import List, Optional, Dict, Any

from src.schemas.heatmap import SP500ConstituentItem, SP500QuoteDataItem, SP500QuoteWithSectorItem
from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings 


class SP500Service(LoggerMixin):
    def __init__(self):
        super().__init__()
        self.fmp_api_key = settings.FMP_API_KEY
        self.base_fmp_url = settings.BASE_FMP_URL


    async def _fetch_fmp_api(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Helper method to call FMP API and return a list of dicts."""

        if not self.fmp_api_key or self.fmp_api_key == "YOUR_FMP_API_KEY_PLACEHOLDER":
            self.logger.error(f"FMP API Key not configured for endpoint: {endpoint}")
            return None
        
        query_params = params.copy() if params else {}
        query_params["apikey"] = self.fmp_api_key
        url = f"{self.base_fmp_url}{endpoint}"

        try:
            response = await client.get(url, params=query_params, timeout=30.0) 
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and len(data) > 0:
                self.logger.warning(f"FMP API {endpoint} returned dict instead of list, possibly a single symbol: {data}")
                if "Error Message" in data:
                    self.logger.error(f"FMP API error at {endpoint}: {data['Error Message']}")
                    return None
                return [data] 
            
            self.logger.warning(f"FMP API {endpoint} returned unexpected type: {type(data)}")
            return None
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTPStatusError error when calling FMP API {endpoint}: {e.response.status_code} - {e.response.text[:200]}", exc_info=False)
        except httpx.RequestError as e:
            self.logger.error(f"RequestError error when calling FMP API {endpoint}: {e}", exc_info=True)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSONDecodeError error when calling FMP API {endpoint}: {e}. Response text: {response.text[:200] if 'response' in locals() else 'N/A'}", exc_info=False)
        except Exception:
            self.logger.exception(f"Unexpected error when calling FMP API {endpoint}")
        return None


    async def get_sp500_constituents_with_quotes(self) -> Optional[List[SP500QuoteWithSectorItem]]:
        """
        Fetch the list of S&P 500 constituents and their latest market quotes.
        Returns a list of SP500QuoteWithSectorItem, or None if the constituent fetch fails.
        """
        async with httpx.AsyncClient() as client:
            # Step 1: Call API to get data
            raw_list = await self._fetch_fmp_api(client, "/v3/sp500_constituent")
            if raw_list is None:
                self.logger.error("Failed to fetch S&P 500 constituents")
                return None
            if not raw_list:
                self.logger.info("No constituents returned")
                return []

            # Parse items and collect symbols
            constituents_data: List[SP500ConstituentItem] = []
            symbols_for_quote: List[str] = []
            symbol_to_sector_info_map: Dict[str, Dict[str, Optional[str]]] = {}

            for item_raw in raw_list:
                try:
                    # **item_raw: Automatically get all key–values in item and assign to corresponding parameters instead of item.get("field_name") for each field
                    constituent = SP500ConstituentItem(**item_raw)
                    constituents_data.append(constituent)

                    if constituent.symbol:
                        symbols_for_quote.append(constituent.symbol)
                        symbol_to_sector_info_map[constituent.symbol] = {
                            "sector": constituent.sector,
                            "subSector": constituent.subSector 
                        }

                except Exception as e_parse_constituent:
                    self.logger.warning(f"Error parsing S&P 500 constituent item: {item_raw}. Error: {e_parse_constituent}")

            if not symbols_for_quote:
                self.logger.warning("No valid symbols from S&P 500 constituents to get quote.")
                return []

            self.logger.info(f"Get {len(symbols_for_quote)} symbols from S&P 500 constituents. Start getting quotes.")

            # Step 2: Get quotes for symbols (try with 500 symbols at a time)
            batch_size = 510 
            all_quotes_data_raw: List[Dict[str, Any]] = []

            for start in range(0, len(symbols_for_quote), batch_size):
                batch_symbols = symbols_for_quote[start : start + batch_size]

                symbols_str = ",".join(batch_symbols)

                quotes_batch_raw = await self._fetch_fmp_api(client, f"/v3/quote/{symbols_str}")

                if quotes_batch_raw:
                    all_quotes_data_raw.extend(quotes_batch_raw)
                else:
                    self.logger.warning(f"Quote batch failed for symbol: {batch_symbols[0] if batch_symbols else 'N/A'}")
            
            if not all_quotes_data_raw:
                self.logger.error("Unable to get quote data for S&P 500 symbols.")
                results_no_quote: List[SP500QuoteWithSectorItem] = []

                for symbol, sector_info in symbol_to_sector_info_map.items():
                    # name_from_constituent = next((c.name for c in constituents_data if c.symbol == symbol), symbol)
                    # Build a symbol → name map once
                    name_map = {}
                    for item in constituents_data:
                        name_map[item.symbol] = item.name

                    name_from_constituent = name_map.get(symbol) or symbol

                    results_no_quote.append(SP500QuoteWithSectorItem(symbol=symbol, 
                                                 name=name_from_constituent, 
                                                 sector=sector_info.get("sector"), 
                                                 subSector=sector_info.get("subSector"))
                                            )
                if results_no_quote:
                    return results_no_quote
                else:
                    return []

            quotes_map: Dict[str, SP500QuoteDataItem] = {}
            for quote_raw in all_quotes_data_raw:
                try:
                    quote_item = SP500QuoteDataItem(**quote_raw)
                    if quote_item.symbol:
                        quotes_map[quote_item.symbol] = quote_item
                except Exception as e_parse_quote:
                    self.logger.warning(f"Error parsing quote item: {quote_raw}. Error: {e_parse_quote}")

            # Step 3: Combine data
            name_map = {}
            for item in constituents_data:
                name_map[item.symbol] = item.name

            final_results: List[SP500QuoteWithSectorItem] = []
            for symbol, sector_info in symbol_to_sector_info_map.items():
                quote_data = quotes_map.get(symbol)

                name_from_constituent = name_map.get(symbol)

                if quote_data:
                    final_results.append(
                        SP500QuoteWithSectorItem(
                            symbol=quote_data.symbol,
                            name=quote_data.name or name_from_constituent or symbol, 
                            sector=sector_info.get("sector"),
                            subSector=sector_info.get("subSector"),
                            price=quote_data.price,
                            changesPercentage=quote_data.changesPercentage,
                            change=quote_data.change,
                            dayLow=quote_data.dayLow,
                            dayHigh=quote_data.dayHigh,
                            yearHigh=quote_data.yearHigh,
                            yearLow=quote_data.yearLow,
                            marketCap=quote_data.marketCap,
                            priceAvg50=quote_data.priceAvg50,
                            priceAvg200=quote_data.priceAvg200,
                            exchange=quote_data.exchange,
                            volume=quote_data.volume,
                            avgVolume=quote_data.avgVolume,
                            open_price=quote_data.open_price,
                            previousClose=quote_data.previousClose,
                            eps=quote_data.eps,
                            pe=quote_data.pe
                        )
                    )
                else:
                    self.logger.warning(f"No quote data found for symbol: {symbol}, only basic information returned.")
                    final_results.append(SP500QuoteWithSectorItem(symbol=symbol, 
                                                                  name=name_from_constituent or symbol, 
                                                                  sector=sector_info["sector"]))

            self.logger.info(f"Completed S&P 500 data fetching and aggregation. Total {len(final_results)} items.")
            
            return final_results