import asyncio
from datetime import timedelta
from typing import Any, Dict, List, Optional
import httpx
import logging

from src.mappers.equity_mapper import EquityMapper
from src.models.equity import MarketOverviewData
from src.services.equity_service import EquityService
from src.utils.logger.set_up_log_dataFMP import setup_logger 

logger = setup_logger(__name__, log_level=logging.INFO)

equity_service_instance = EquityService()

class MarketOverviewService:
    def __init__(self):
        self._discovery_cache: Dict[str, Dict[str, Any]] = {}
        self._CACHE_DURATION_DISCOVERY = timedelta(minutes=15)

    async def _fetch_and_map_one_symbol_overview_fmp(self, symbol: str, client: httpx.AsyncClient) -> Optional[MarketOverviewData]:
        logger.debug(f"Fetching and mapping overview for symbol: {symbol}")
        fmp_quote_data = await equity_service_instance.fetch_fmp_data_helper(client, "/v3/quote/{symbol}", symbol)
        latest_sma_50_val, latest_sma_200_val = None, None
        if fmp_quote_data and isinstance(fmp_quote_data, dict):
            latest_sma_50_val = fmp_quote_data.get('priceAvg50')
            latest_sma_200_val = fmp_quote_data.get('priceAvg200')
        else:
            logger.debug(f"No quote data or not a dict for SMA extraction for {symbol}.")


        fmp_profile_data = await equity_service_instance.fetch_fmp_data_helper(client, "/v3/profile/{symbol}", symbol)
        fmp_pre_post_data = await equity_service_instance.fetch_fmp_data_helper(client, "/v4/batch-pre-post-market/{symbol}", symbol)

        if not fmp_quote_data and not fmp_profile_data:
            logger.warning(f"Insufficient FMP data (quote/profile) for {symbol}. Skipping overview.")
            return None

        try:
            mapped_data = EquityMapper.map_fmp_to_market_overview(
                symbol=symbol, fmp_quote_data=fmp_quote_data, fmp_profile_data=fmp_profile_data,
                fmp_pre_post_data=fmp_pre_post_data,
                latest_sma_50=latest_sma_50_val, latest_sma_200=latest_sma_200_val
            )
            return mapped_data
        except Exception as e:
            logger.error(f"Error mapping FMP data to MarketOverviewData for {symbol}: {e}", exc_info=True)
            return None


    async def get_market_overview(self, symbols: List[str], provider: Optional[str] = None) -> List[Optional[MarketOverviewData]]:
        logger.info(f"Fetching market overview for symbols: {symbols} using provider: {provider if provider else 'default_fmp'}")
        results: List[Optional[MarketOverviewData]] = []
        if not symbols:
            logger.warning("No symbols provided for market overview.")
            return []

        async with httpx.AsyncClient(timeout=35.0) as client:
            tasks = [self._fetch_and_map_one_symbol_overview_fmp(symbol, client) for symbol in symbols]
            results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)

        for i, res_or_exc in enumerate(results_or_exceptions):
            if isinstance(res_or_exc, Exception):
                logger.error(f"Error during asyncio.gather for market overview of symbol '{symbols[i]}': {res_or_exc}", exc_info=True)
                results.append(None)
            elif res_or_exc is None:
                logger.debug(f"No market overview data returned for symbol '{symbols[i]}'.")
                results.append(None)
            else:
                results.append(res_or_exc)
        logger.info(f"Market overview fetch complete. Got {len([r for r in results if r is not None])} valid items out of {len(symbols)} requested.")
        return results