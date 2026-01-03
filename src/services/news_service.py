import asyncio
from datetime import datetime
import httpx
from typing import List, Optional, Dict, Any
import logging
import json 

from src.models.news import NewsItem
from src.models.equity import NewsItemOutput, PressReleaseItem, SocialSentimentItem, StockNewsSentimentItem
from src.mappers.news_mapper import NewsMapper
from src.utils.logger.set_up_log_dataFMP import setup_logger 
from src.utils.config import settings

FMP_API_KEY = settings.FMP_API_KEY
BASE_FMP_URL = settings.BASE_FMP_URL
FMP_URL_STABLE = settings.FMP_URL_STABLE

logger = setup_logger(__name__, log_level=logging.INFO)

if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
    logger.warning("FMP_API_KEY is not set or using placeholder for NewsService.")

class NewsService:
    
    @staticmethod
    def _create_optimized_client() -> httpx.AsyncClient:
        """
        Tạo httpx.AsyncClient với connection pool được tối ưu.
        """
        limits = httpx.Limits(
            max_keepalive_connections=50,
            max_connections=200,
            keepalive_expiry=30.0
        )
        
        timeout = httpx.Timeout(
            connect=5.0,
            read=10.0,
            write=5.0,
            pool=5.0
        )
        
        return httpx.AsyncClient(limits=limits, timeout=timeout)
    
    async def _fetch_fmp_data(
        self, 
        client: httpx.AsyncClient, 
        endpoint_path: str, 
        params: Dict[str, Any],
        max_retries: int = 2
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch data từ FMP với retry logic và optimized timeout.
        """
        if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error(f"FMP API Key not configured for endpoint {endpoint_path}.")
            return None

        params["apikey"] = FMP_API_KEY
        url = f"{BASE_FMP_URL}{endpoint_path}"
        log_params = {k: v for k, v in params.items() if k != "apikey"}
        
        # ========== RETRY LOGIC ==========
        for attempt in range(max_retries):
            try:
                # Tăng timeout theo exponential backoff
                current_timeout = 8.0 * (1.5 ** attempt)  # 25s, 37.5s, 56.25s
                
                logger.debug(
                    f"Fetching FMP data from {endpoint_path}, "
                    f"attempt {attempt + 1}/{max_retries}, timeout={current_timeout}s"
                )
                
                response = await client.get(url, params=params, timeout=current_timeout)
                response.raise_for_status()
                data = response.json()
                
                if isinstance(data, list):
                    # logger.debug(f"✅ Successfully fetched news {len(data)} items from FMP ({endpoint_path}) on attempt {attempt + 1}")
                    return data
                else:
                    logger.warning(
                        f"Unexpected data format from FMP ({endpoint_path}). "
                        f"Expected list, got {type(data)}. URL: {url}, Params: {log_params}"
                    )
                    return None
            
            except (httpx.ConnectTimeout, httpx.ReadTimeout) as timeout_err:
                if attempt < max_retries - 1:
                    wait_time = 0.5 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"⏱️ Timeout fetching FMP data ({endpoint_path}) on attempt {attempt + 1}/{max_retries}. "
                        f"Error: {type(timeout_err).__name__}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"❌ Final timeout for FMP endpoint {endpoint_path} after {max_retries} attempts. "
                        f"URL: {url}, Params: {log_params}"
                    )
                    return None
            
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"❌ HTTPStatusError fetching FMP data ({endpoint_path}): "
                    f"{e.response.status_code} - {e.response.text[:200]}. "
                    f"URL: {url}, Params: {log_params}", 
                    exc_info=False
                )
                return None  # HTTP error không retry
            
            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"⚠️ RequestError fetching FMP data ({endpoint_path}) on attempt {attempt + 1}/{max_retries}. "
                        f"Retrying in {wait_time}s... Error: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"❌ Final RequestError for FMP endpoint {endpoint_path} after {max_retries} attempts. "
                        f"URL: {url}, Params: {log_params}", 
                        exc_info=True
                    )
                    return None
            
            except json.JSONDecodeError as e:
                logger.error(
                    f"❌ JSONDecodeError processing FMP response ({endpoint_path}): {e}. "
                    f"Response: {response.text[:200] if 'response' in locals() else 'N/A'}. "
                    f"URL: {url}, Params: {log_params}", 
                    exc_info=False
                )
                return None  # JSON error không retry
            
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"⚠️ Error fetching FMP data ({endpoint_path}) on attempt {attempt + 1}/{max_retries}. "
                        f"Retrying in {wait_time}s... Error: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.exception(
                        f"❌ Final error fetching FMP data ({endpoint_path}) after {max_retries} attempts. "
                        f"URL: {url}, Params: {log_params}"
                    )
                    return None
        
        return None
    
    async def get_stock_news_sentiments_rss(self, page: int = 0) -> Optional[List[StockNewsSentimentItem]]:
        """Lấy stock news sentiments RSS feed từ FMP."""
        logger.info(f"Fetching stock news sentiments RSS feed from FMP, page {page}")
        endpoint_path = "/v4/stock-news-sentiments-rss-feed"
        fmp_params = {"page": page}

        async with self._create_optimized_client() as client:
            raw_sentiment_news_list = await self._fetch_fmp_data(client, endpoint_path, fmp_params)

        if raw_sentiment_news_list is None:
            logger.error(f"Failed to fetch stock news sentiments for page {page}.")
            return None 
        if not raw_sentiment_news_list:
            logger.info(f"No stock news sentiments found for page {page}.")
            return [] 

        mapped_sentiment_news: List[StockNewsSentimentItem] = []
        for item_idx, item_data in enumerate(raw_sentiment_news_list):
            if not isinstance(item_data, dict):
                logger.warning(f"Skipping non-dict item in sentiment news at index {item_idx}: {item_data}")
                continue
            try:
                mapped_item = StockNewsSentimentItem(**item_data)
                mapped_sentiment_news.append(mapped_item)
            except Exception as map_err: 
                item_title = item_data.get("title", "N/A")
                logger.error(
                    f"Error mapping stock news sentiment item at index {item_idx} (Title: {item_title}): {map_err}. "
                    f"Data: {item_data}", 
                    exc_info=False
                )

        logger.info(f"Successfully mapped {len(mapped_sentiment_news)} stock news sentiment items for page {page}.")
        return mapped_sentiment_news

    async def get_general_news(self, page: int = 0) -> Optional[List[NewsItemOutput]]:
        """Lấy general news từ FMP."""
        logger.info(f"Fetching general news from FMP, page {page}")
        endpoint_path = "/v4/general_news"
        fmp_params = {"page": page}

        async with self._create_optimized_client() as client:
            raw_news_list = await self._fetch_fmp_data(client, endpoint_path, fmp_params)

        if raw_news_list is None:
            logger.error(f"Failed to fetch general news for page {page}.")
            return None
        if not raw_news_list:
            logger.info(f"No general news found for page {page}.")
            return []

        mapped_news: List[NewsItemOutput] = []
        for item_idx, item_data in enumerate(raw_news_list):
            try:
                mapped_item = NewsMapper.fmp_item_to_news_output(item_data, news_type="latest", category="general")
                if mapped_item:
                    mapped_news.append(mapped_item)
                else:
                    logger.warning(f"Failed to map general news item at index {item_idx}. Data: {item_data}")
            except Exception as map_err:
                logger.error(f"Error mapping general news item at index {item_idx}: {map_err}. Data: {item_data}", exc_info=True)

        logger.info(f"Successfully mapped {len(mapped_news)} general news items for page {page}.")
        return mapped_news

    async def get_company_news(
        self, 
        symbol: str, 
        limit: int = 10,
        from_date: Optional[str] = None, 
        to_date: Optional[str] = None 
    ) -> Optional[List[NewsItemOutput]]:
        """Lấy company news từ FMP với date range filter."""
        logger.info(f"Fetching company news from FMP for symbol {symbol}, limit {limit}, from {from_date} to {to_date}")
        endpoint_path = "/v3/stock_news"
        
        fmp_params = {
            "tickers": symbol.upper(), 
            "limit": limit
        }
        
        if from_date:
            fmp_params["from"] = from_date
        if to_date:
            fmp_params["to"] = to_date

        async with self._create_optimized_client() as client:
            raw_news_list = await self._fetch_fmp_data(client, endpoint_path, fmp_params)

        if raw_news_list is None:
            logger.error(f"Failed to fetch company news for symbol {symbol}.")
            return None
        if not raw_news_list:
            logger.info(f"No company news found for symbol {symbol}.")
            return []

        mapped_news: List[NewsItemOutput] = []
        for item_idx, item_data in enumerate(raw_news_list):
            try:
                mapped_item = NewsMapper.fmp_item_to_news_output(item_data, news_type="latest", category="company_specific")
                if mapped_item:
                    mapped_news.append(mapped_item)
                else:
                    logger.warning(f"Failed to map company news item for {symbol} at index {item_idx}. Data: {item_data}")
            except Exception as map_err:
                logger.error(
                    f"Error mapping company news item for {symbol} at index {item_idx}: {map_err}. Data: {item_data}", 
                    exc_info=True
                )

        logger.info(f"Successfully mapped {len(mapped_news)} company news items for symbol {symbol}.")
        return mapped_news
    
    async def get_historical_social_sentiment(
        self,
        symbol: str,
        page: int = 0
    ) -> Optional[List[SocialSentimentItem]]:
        """Lấy dữ liệu cảm xúc xã hội lịch sử cho một mã cổ phiếu từ FMP."""
        logger.info(f"Fetching historical social sentiment for symbol: {symbol}, page: {page}")
        endpoint_path = "/v4/historical/social-sentiment"
        fmp_params = {"symbol": symbol.upper(), "page": page}
        
        async with self._create_optimized_client() as client:
            raw_sentiment_list = await self._fetch_fmp_data(client, endpoint_path, fmp_params)

        if raw_sentiment_list is None:
            logger.error(f"Failed to fetch historical social sentiment for {symbol}, page {page}.")
            return None 
        if not raw_sentiment_list:
            logger.info(f"No historical social sentiment data found for {symbol}, page {page}.")
            return [] 

        mapped_sentiments: List[SocialSentimentItem] = []
        for item_idx, item_data in enumerate(raw_sentiment_list):
            if not isinstance(item_data, dict):
                logger.warning(
                    f"Skipping non-dict item in historical social sentiment at index {item_idx} for {symbol}: {item_data}"
                )
                continue
            try:
                sentiment_item = SocialSentimentItem(**item_data)
                mapped_sentiments.append(sentiment_item)
            except Exception as map_err:
                logger.error(
                    f"Error mapping historical social sentiment item for {symbol} at index {item_idx}: {map_err}. "
                    f"Data: {item_data}",
                    exc_info=False 
                )

        logger.info(f"Successfully mapped {len(mapped_sentiments)} historical social sentiment items for {symbol}, page {page}.")
        return mapped_sentiments
    
    async def get_latest_press_releases(self, page: int, limit: int) -> Optional[List[PressReleaseItem]]:
        """Lấy danh sách các thông cáo báo chí mới nhất từ FMP."""
        endpoint_path = "/news/press-releases-latest"
        url = f"{FMP_URL_STABLE}{endpoint_path}?page={page}&limit={limit}&apikey={FMP_API_KEY}"
        
        logger.info(f"Fetching latest press releases from FMP for page {page}, limit {limit}.")
        
        async with self._create_optimized_client() as client:
            try:
                response = await client.get(url, timeout=40.0)
                response.raise_for_status()
                data = response.json()
                
                if not isinstance(data, list):
                    logger.warning(f"Press releases API returned non-list data: {type(data)}")
                    return None

                return [PressReleaseItem(**item) for item in data if isinstance(item, dict)]
            except httpx.HTTPStatusError as hse:
                logger.error(f"HTTP error fetching press releases: {hse.response.status_code} - {hse.response.text[:200]}")
                return None
            except Exception as e:
                logger.error(f"Failed to fetch or process press releases: {e}", exc_info=True)
                return None
            
    async def get_news_for_multiple_symbols(
        self,
        limit: int,
        symbols: List[str]
    ) -> Optional[List[NewsItem]]:
        """
        Lấy tin tức cho danh sách mã và ánh xạ sang model NewsItem tùy chỉnh.
        """
        if not symbols:
            return []
        
        async def _fetch_batch(symbols_batch: List[str], client: httpx.AsyncClient):
            symbols_str = ",".join(symbols_batch)
            url = f"{FMP_URL_STABLE}/news/stock?symbols={symbols_str}&limit={limit}&apikey={FMP_API_KEY}"
            try:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                return data if isinstance(data, list) else []
            except Exception as e:
                logger.error(f"Failed to fetch news batch for symbols {symbols_str}: {e}")
                return []

        BATCH_SIZE = 5
        symbol_chunks = [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]
        
        all_raw_news: List[Dict] = []
        async with self._create_optimized_client() as client:
            tasks = [_fetch_batch(chunk, client) for chunk in symbol_chunks]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, list):
                    all_raw_news.extend(result)
        
        # --- BƯỚC ÁNH XẠ DỮ LIỆU ---
        final_news_list: List[NewsItem] = []
        for item_dict in all_raw_news:
            if not isinstance(item_dict, dict):
                continue
            
            try:
                # Chuyển đổi 'publishedDate' từ string sang datetime
                published_date_obj = datetime.strptime(
                    item_dict.get("publishedDate", ""), 
                    "%Y-%m-%d %H:%M:%S"
                )

                # Chuyển đổi 'symbol' từ string sang list[string]
                symbol_str = item_dict.get("symbol")
                symbols_list = [symbol_str] if symbol_str else []

                # Tạo đối tượng NewsItem từ dữ liệu đã được ánh xạ
                custom_item = NewsItem(
                    title=item_dict.get("title"),
                    publisher_name=item_dict.get("publisher"),  # Ánh xạ 'publisher' -> 'publisher_name'
                    published_date=published_date_obj,
                    url=item_dict.get("url"),
                    symbols=symbols_list,
                    text=item_dict.get("text")
                )
                final_news_list.append(custom_item)
            except (ValueError, TypeError) as date_err:
                logger.warning(f"Could not parse date for news item, skipping. Error: {date_err}. Data: {item_dict}")
            except Exception as p_error:
                logger.warning(f"Could not map news item to custom model: {p_error}. Data: {item_dict}")
        
        final_news_list.sort(key=lambda x: x.published_date, reverse=True)

        return final_news_list