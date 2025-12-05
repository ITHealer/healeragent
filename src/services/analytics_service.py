import asyncio
import json
import httpx
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Any
import logging

from src.helpers.redis_cache import get_cache, set_cache
from src.models.equity import KeyStatsOutput, StockNewsSentimentItem
from src.services.news_service import NewsService
from src.utils.config import settings
import aioredis

FMP_API_KEY = settings.FMP_API_KEY
BASE_FMP_URL = settings.BASE_FMP_URL
FMP_URL_STABLE = settings.FMP_URL_STABLE
logger = logging.getLogger(__name__)

class AnalyticsService:
    def __init__(self, news_service: NewsService, redis_client: Optional[aioredis.Redis]):
        self.news_service = news_service
        self.redis_client = redis_client
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def _get_historical_prices(self, symbol: str, start_date: str, end_date: str) -> Dict[str, float]:
        symbol_upper = symbol.upper()
        cache_key = f"historical_prices_{symbol_upper}_{start_date}_{end_date}"
        if self.redis_client:
            try:
                cached_json = await self.redis_client.get(cache_key)
                if cached_json:
                    return json.loads(cached_json)
            except Exception as e:
                logger.error(f"Redis GET error for {cache_key}: {e}", exc_info=True)

        url = f"{BASE_FMP_URL}/v3/historical-price-full/{symbol_upper}?from={start_date}&to={end_date}&apikey={FMP_API_KEY}"
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            data = response.json()
            if data and "historical" in data and isinstance(data["historical"], list):
                price_map = {item['date']: item['close'] for item in data['historical']}
                if self.redis_client and price_map:
                    cache_ttl = getattr(settings, 'CACHE_TTL_HISTORY', 86400 * 7)
                    await self.redis_client.set(name=cache_key, value=json.dumps(price_map), ex=int(cache_ttl))
                return price_map
        except Exception as e:
            logger.error(f"Failed to get historical prices for {symbol_upper} from FMP: {e}", exc_info=True)
        return {}

    async def _calculate_future_return(self, symbol: str, event_date_str: str, days_ahead: int) -> Optional[Tuple[float, bool]]:
        try:
            event_date = datetime.strptime(event_date_str, "%Y-%m-%d").date()
            start_date_req = event_date - timedelta(days=5)
            end_date_req = event_date + timedelta(days=40)
            
            prices = await self._get_historical_prices(symbol, start_date_req.isoformat(), end_date_req.isoformat())
            if not prices: return None

            trading_days = sorted(prices.keys())
            event_trading_day = next((day for day in trading_days if day >= event_date.isoformat()), None)
            if not event_trading_day: return None

            start_index = trading_days.index(event_trading_day)
            end_index = start_index + days_ahead
            if end_index >= len(trading_days): return None
            
            start_price = prices[trading_days[start_index]]
            end_price = prices[trading_days[end_index]]
            if start_price == 0: return None
            
            percent_return = ((end_price - start_price) / start_price) * 100
            is_win = end_price > start_price
            return percent_return, is_win
        except (ValueError, IndexError, TypeError):
            return None

    async def _calculate_all_positive_earnings_stats(self) -> Optional[float]:
        logger.info("Calculating 'Positive Earnings Win Rate' from general earnings calendar.")

        to_date = datetime.now().date() - timedelta(days=30)
        from_date = to_date - timedelta(days=90)
        
        url = f"{FMP_URL_STABLE}/earnings-calendar?from={from_date.isoformat()}&to={to_date.isoformat()}&apikey={FMP_API_KEY}"
        
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            all_earnings_events = response.json()
            if not isinstance(all_earnings_events, list):
                logger.warning(f"Earnings calendar returned non-list data: {type(all_earnings_events)}")
                return 0.0
        except Exception as e:
            logger.error(f"Failed to fetch general earnings calendar: {e}")
            return None

        positive_surprise_events = []
        for event in all_earnings_events:
            try:
                eps = event.get('epsActual')
                eps_estimated = event.get('epsEstimated')
                if eps is not None and eps_estimated is not None and eps > eps_estimated:
                    positive_surprise_events.append(event)
            except (ValueError, TypeError):
                continue
        
        if not positive_surprise_events:
            logger.warning("No positive earnings surprises found in the specified date range.")
            return 0.0
        
        logger.info(f"Found {len(positive_surprise_events)} positive earnings events. Taking a sample to analyze.")
        
        sample_to_analyze = positive_surprise_events[:40]

        return_calculation_tasks = [
            self._calculate_future_return(event['symbol'], event['date'], 20)
            for event in sample_to_analyze if 'symbol' in event and 'date' in event
        ]

        if not return_calculation_tasks:
            return 0.0

        return_results = await asyncio.gather(*return_calculation_tasks)
        valid_results = [res for res in return_results if res is not None]

        if not valid_results:
            logger.warning("Could not calculate future returns for any sampled positive earnings event.")
            return 0.0

        win_count = sum(1 for _, is_win in valid_results if is_win)
        win_rate = (win_count / len(valid_results)) * 100
        
        logger.info(f"Positive Earnings Win Rate (from Calendar): Wins={win_count}, Total Analyzed Events={len(valid_results)}, Rate={win_rate:.2f}%")
        return win_rate

    async def _calculate_chosen_picks_stats(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Tính Tỷ lệ thắng và Lợi nhuận trung bình.
        LOGIC MỚI: Tính toán dựa trên hiệu suất 20 ngày gần nhất của một danh sách các cổ phiếu lớn, tiêu biểu.
        """
        logger.info("Calculating 'Chosen Picks' stats based on recent 20-day performance of major stocks.")
        chosen_picks_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "V", "WMT", "UNH"
        ]
        
        logger.info(f"Analyzing recent performance for: {chosen_picks_symbols}")

        event_date = datetime.now().date() - timedelta(days=40)
        event_date_str = event_date.strftime('%Y-%m-%d')
        
        tasks = [
            self._calculate_future_return(symbol, event_date_str, 20)
            for symbol in chosen_picks_symbols
        ]

        results = await asyncio.gather(*tasks)
        
        valid_results = [res for res in results if res is not None]
        if not valid_results:
            logger.warning("Could not calculate recent 20-day returns for any of the chosen picks.")
            return 0.0, 0.0

        win_count = sum(1 for _, is_win in valid_results if is_win)
        total_returns = sum(percent_return for percent_return, _ in valid_results)

        win_rate = (win_count / len(valid_results)) * 100
        avg_return = total_returns / len(valid_results)
        
        logger.info(f"Chosen Picks Recent Performance: Analyzed={len(valid_results)}, Win Rate={win_rate:.2f}%, Avg Return={avg_return:.2f}%")
        return win_rate, avg_return

    async def get_key_stats(self) -> Optional[KeyStatsOutput]:
        cache_key = "key_stats_v3"
        if self.redis_client:
            cached_data = await get_cache(self.redis_client, cache_key, KeyStatsOutput)
            if cached_data:
                logger.info("Key Stats: Cache HIT.")
                return cached_data

        logger.info("Key Stats: Cache MISS. Starting calculations...")
        
        stat1_task = self._calculate_all_positive_earnings_stats()
        stats_2_3_task = self._calculate_chosen_picks_stats()
        
        results = await asyncio.gather(stat1_task, stats_2_3_task)
        win_rate_all_earnings, (win_rate_picks, avg_return_picks) = results

        if win_rate_all_earnings is None or win_rate_picks is None or avg_return_picks is None:
            logger.error("Failed to calculate one or more key stats.")
            return None

        key_stats = KeyStatsOutput(
            winRateAllPositiveEarnings=win_rate_all_earnings,
            winRateChosenPicks=win_rate_picks,
            avgReturnChosenPicks=avg_return_picks
        )

        if self.redis_client:
            await set_cache(self.redis_client, cache_key, key_stats, expiry=86400)
            logger.info("Key Stats: Cached new results for 24 hours.")

        return key_stats

    async def close(self):
        await self.http_client.aclose()