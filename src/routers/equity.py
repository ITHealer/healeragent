import asyncio
import json
import logging

import aioredis
from fastapi import APIRouter, Depends, Query, HTTPException, WebSocket, WebSocketDisconnect
from datetime import date 
from typing import Any, Dict, List, Optional
import httpx
import hashlib
from starlette.websockets import WebSocketState
# from src.models.hedge_fund import HedgeFundRequest, HedgeFundResponse
from src.models.news import NewsItem
from src.handlers.api_key_authenticator_handler import APIKeyAuth

from src.core.websocket_fetchers import _alchemy_transactions_fetcher, _bitcoin_transactions_fetcher, _custom_chart_fetcher, _discovery_fetcher, _equity_detail_fetcher, _list_fetcher, _market_overview_fetcher, _ticker_tape_fetcher
from src.core.websocket_manager import ensure_fetcher_task_running, stop_fetcher_task_if_unneeded, global_connection_manager
from src.scheduler.get_data_scheduler import ACTIVES_CACHE_KEY, CRYPTO_CACHE_KEY, DEFAULT_SCREENER_CACHE_KEY, ETFS_CACHE_KEY, GAINERS_CACHE_KEY, LOSERS_CACHE_KEY, STOCKS_CACHE_KEY
from src.utils.logger.set_up_log_dataFMP import setup_logger

from src.helpers.websocket_connection_helpers import ConnectionManager
from src.helpers.redis_cache import get_list_cache, get_paginated_cache, get_redis_client, set_list_cache, set_cache, get_cache, set_paginated_cache

# from src.services.hedge_fund_service import hedge_fund_service
from src.services.price_target_service import price_target_service
from src.services.government_trading_service import senate_trading_service
from src.services.analytics_service import AnalyticsService
from src.services.regional_screener_service import MarketIndicesService
from src.services.tool_call_service import ToolCallService
from src.services.financial_report_service import FinancialStatementsService
from src.services.heatmap_service import SP500Service
from src.services.crypto_spotlight_service import CryptoSpotlightService
from src.services.discovery_service import DiscoveryService
from src.services.equity_detail_service import EquityDetailService
from src.services.market_overview_service import MarketOverviewService
from src.services.profile_service import ProfileService
from src.services.search_service import SearchService
from src.services.ticker_tape_service import TickerTapeService
from src.services.list_service import ListService
from src.services.equity_service import EquityService
from src.services.news_service import NewsService

from src.schemas.heatmap import SP500QuoteWithSectorItem
from src.models.equity import AnalystEstimateItem, CompanyDescription, CryptoNewsItem, CryptoSpotlightItem, CustomListWsParams, DiscoveryWsParams, ListPageWsParams, OnchainSubscriptionParams, PaginatedData, TickerTapeWSParams, EquityDetailWsParams, FMPCompanyOutlookProfile, FMPSearchResultItem, FinancialStatementGrowthItem, FinancialStatementsData, KeyMetricsTTMItem, KeyStatsOutput, LogoData, MarketRegionEnum, MarketRegionParams, NewsItemOutput, PressReleaseItem, PriceTargetWithChartOutput, ProfileResponseWrapper, ScreenerOutput, ScreenerStep1Data, ScreenerWSParams, SenateTradingItem, SocialSentimentItem, StockDetailPayload, StockNewsSentimentItem, TickerTapeData, WebSocketRequest
from src.models.equity import APIResponseData, DiscoveryItemOutput, MarketOverviewData, APIResponse, CompanyProfile
from src.utils.config import settings

from fastapi import WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from starlette.websockets import WebSocketState

from dotenv import load_dotenv

load_dotenv()

# --- Logging Setup ---
logger = setup_logger(__name__, log_level=logging.INFO)

router = APIRouter()
news_service = NewsService()
list_service = ListService()
equity_service = EquityService()
profile_service = ProfileService()
market_overview_service = MarketOverviewService()
crypto_spotlight_service = CryptoSpotlightService()
equity_detail_service = EquityDetailService()
search_service = SearchService()
manager = ConnectionManager()
discovery_service = DiscoveryService()
fs_service_instance = FinancialStatementsService()
tool_call_service = ToolCallService()
indices_service = MarketIndicesService()
api_auth = APIKeyAuth()

FMP_API_KEY = settings.FMP_API_KEY
BASE_FMP_URL = settings.BASE_FMP_URL
FMP_URL_STABLE = settings.FMP_URL_STABLE

TRUSTED_NEWS_SOURCES = {
    "reuters.com",
    "apnews.com",
    "wsj.com",
    "bloomberg.com",
    "cnbc.com",
    "marketwatch.com",
    "investors.com", 
    "forbes.com",
    "ft.com",
    "benzinga.com",
    "zacks.com",
    "themotleyfool.com",
    "investopedia.com",
    "seekingalpha.com",
    "yahoo.com",
    "youtube.com",
    "businessinsider.com",
    "foxbusiness.com",
    "schaeffersresearch.com",
    "invezz.com",
    "barrons.com",
    
}

@router.get("/list/stocks",
            response_model=APIResponse[DiscoveryItemOutput],
            summary="Lấy danh sách cổ phiếu đã được cache sẵn và phân trang")
async def get_stock_list_endpoint(
    pageNumber: int = Query(1, ge=1, description="Số trang bắt đầu từ 1"),
    pageSize: int = Query(20, ge=1, le=100, description="Số lượng item trên mỗi trang"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):

    cached_paginated_object: Optional[PaginatedData[DiscoveryItemOutput]] = await get_paginated_cache(
        redis_client, STOCKS_CACHE_KEY, PaginatedData[DiscoveryItemOutput]
    )

    if cached_paginated_object is None or cached_paginated_object.data is None:
        logger.warning(f"Cache MISS hoặc dữ liệu cache rỗng cho key: {STOCKS_CACHE_KEY}. Scheduler có thể chưa chạy.")
        return APIResponse[DiscoveryItemOutput](
            data=APIResponseData[DiscoveryItemOutput](data=[]),
            totalRows=0,
            message="Dữ liệu đang được chuẩn bị, vui lòng thử lại sau giây lát.",
            provider_used="cache_miss",
            status="202"
        )
    
    logger.info(f"Cache HIT cho key chính: {STOCKS_CACHE_KEY}. Phân trang cho page={pageNumber}, pageSize={pageSize}")

    full_data_list = cached_paginated_object.data
    total_rows = cached_paginated_object.totalRows

    start_index = (pageNumber - 1) * pageSize
    end_index = start_index + pageSize
    paginated_data_slice = full_data_list[start_index:end_index]
    
    response_data = APIResponseData[DiscoveryItemOutput](data=paginated_data_slice)
    return APIResponse[DiscoveryItemOutput](
        provider_used="prefetched_cache",
        data=response_data,
        message="OK",
        totalRows=total_rows 
    )

@router.get("/list/etfs",
            response_model=APIResponse[DiscoveryItemOutput],
            summary="Lấy danh sách ETF đã được cache sẵn và phân trang")
async def get_etfs_list_endpoint(
    pageNumber: int = Query(1, ge=1, description="Số trang bắt đầu từ 1"),
    pageSize: int = Query(20, ge=1, le=100, description="Số lượng item trên mỗi trang"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    cached_paginated_object: Optional[PaginatedData[DiscoveryItemOutput]] = await get_paginated_cache(
        redis_client, ETFS_CACHE_KEY, PaginatedData[DiscoveryItemOutput]
    )

    if cached_paginated_object is None or cached_paginated_object.data is None:
        logger.warning(f"Cache MISS cho key chính: {ETFS_CACHE_KEY}. Scheduler có thể chưa chạy.")
        return APIResponse[DiscoveryItemOutput](
            data=APIResponseData[DiscoveryItemOutput](data=[]),
            totalRows=0,
            message="Dữ liệu đang được chuẩn bị, vui lòng thử lại sau giây lát.",
            provider_used="cache_miss",
            status="202"
        )
    
    logger.info(f"Cache HIT cho key chính: {ETFS_CACHE_KEY}. Phân trang cho page={pageNumber}, pageSize={pageSize}")
    
    full_data_list = cached_paginated_object.data
    total_rows = cached_paginated_object.totalRows
    
    start_index = (pageNumber - 1) * pageSize
    end_index = start_index + pageSize
    paginated_data_slice = full_data_list[start_index:end_index]
    
    response_data = APIResponseData[DiscoveryItemOutput](data=paginated_data_slice)
    return APIResponse[DiscoveryItemOutput](
        provider_used="prefetched_cache",
        data=response_data,
        message="OK",
        totalRows=total_rows
    )

@router.get("/list/crypto",
            response_model=APIResponse[DiscoveryItemOutput],
            summary="Lấy danh sách crypto đã được cache sẵn và phân trang")
async def get_crypto_list_endpoint(
    pageNumber: int = Query(1, ge=1, description="Số trang bắt đầu từ 1"),
    pageSize: int = Query(20, ge=1, le=100, description="Số lượng item trên mỗi trang"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    cached_paginated_object: Optional[PaginatedData[DiscoveryItemOutput]] = await get_paginated_cache(
        redis_client, CRYPTO_CACHE_KEY, PaginatedData[DiscoveryItemOutput]
    )

    if cached_paginated_object is None or cached_paginated_object.data is None:
        logger.warning(f"Cache MISS cho key chính: {CRYPTO_CACHE_KEY}. Scheduler có thể chưa chạy.")
        return APIResponse[DiscoveryItemOutput](
            data=APIResponseData[DiscoveryItemOutput](data=[]),
            totalRows=0,
            message="Dữ liệu đang được chuẩn bị, vui lòng thử lại sau giây lát.",
            provider_used="cache_miss",
            status="202"
        )
    
    logger.info(f"Cache HIT cho key chính: {CRYPTO_CACHE_KEY}. Phân trang cho page={pageNumber}, pageSize={pageSize}")

    full_data_list = cached_paginated_object.data
    total_rows = cached_paginated_object.totalRows

    start_index = (pageNumber - 1) * pageSize
    end_index = start_index + pageSize
    paginated_data_slice = full_data_list[start_index:end_index]
    
    response_data = APIResponseData[DiscoveryItemOutput](data=paginated_data_slice)
    return APIResponse[DiscoveryItemOutput](
        provider_used="prefetched_cache",
        data=response_data,
        message="OK",
        totalRows=total_rows
    )

#region --- Websocket ---

@router.websocket("/ws/gateway")
async def websocket_gateway(
    websocket: WebSocket,
    auth_data: Dict[str, Any] = Depends(api_auth.get_current_auth_for_ws),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "Unknown"

    user_id = auth_data.get("user_id")
    org_id = auth_data.get("effective_organization_id")
    
    subscribed_topics = set()

    # Flag để tracking connection state
    is_connected = True

    try:
        while is_connected:
            # payload_json = await websocket.receive_json()
            try:
                # Thêm timeout để không bị block vĩnh viễn
                payload_json = await asyncio.wait_for(
                    websocket.receive_json(), 
                    timeout=30.0  # 30 seconds timeout
                )
            except asyncio.TimeoutError:
                # Send ping để check connection
                try:
                    await websocket.send_json({"event": "ping"})
                    continue
                # Nếu ping fail → client đã disconnect
                except (ConnectionError, RuntimeError, Exception) as e:
                    self.logger.debug(f"WebSocket ping failed: {e}")
                    is_connected = False
                    break

            request = WebSocketRequest(**payload_json)
            
            event = request.event
            payload = request.payload
            
            topic = ""
            fetcher_coro = None

            if event == "subscribe_ticker_tape":
                params = TickerTapeWSParams(**payload)
                topic = f"ticker_{'_'.join(sorted(params.symbols.split(',')))}"
                fetcher_coro = _ticker_tape_fetcher(topic, params)

            elif event == "subscribe_equity_detail":
                params = EquityDetailWsParams(**payload)
                topic = f"equity_detail_{params.symbol.upper()}_{params.timeframe}_{params.from_date}_{params.to_date}"
                fetcher_coro = _equity_detail_fetcher(topic, params)

            elif event.startswith("subscribe_list_"):
                asset_type = event.replace("subscribe_list_", "")
                if asset_type in ["stocks", "etfs", "crypto"]:
                    params = ListPageWsParams(**payload)
                    topic = f"list_{asset_type}_p{params.page}_l{params.limit}"
                    fetcher_coro = _list_fetcher(topic, params, asset_type, redis_client)

            elif event.startswith("subscribe_discovery_"):
                mover_type = event.replace("subscribe_discovery_", "")
                if mover_type in ["gainers", "losers", "actives"]:
                    params = DiscoveryWsParams(**payload)
                    topic = f"discovery_{mover_type}_l{params.limit}"
                    fetcher_coro = _discovery_fetcher(topic, params, mover_type, redis_client)

            elif event == "subscribe_market_overview":
                params = MarketRegionParams(**payload)
                topic = f"market_overview_{params.region.value}"
                fetcher_coro = _market_overview_fetcher(topic, params)

            elif event == "subscribe_custom_chart":
                try:
                    params = CustomListWsParams(**payload)
                    symbols_key = "_".join(sorted(params.symbols))
                    topic = f"custom_chart_{hash(symbols_key)}"
                    fetcher_coro = _custom_chart_fetcher(topic, params, redis_client)
                except Exception as e:
                    logger.error(f"Invalid payload for 'subscribe_custom_chart': {e}", exc_info=True)
                    await websocket.send_json({"event": "error", "payload": {"message": f"Invalid payload for {event}: {e}"}})

            elif event == "subscribe_onchain_transactions":
                try:
                    params = OnchainSubscriptionParams(**payload)
                    
                    sorted_addresses = sorted([addr.lower() for addr in params.addresses])
                    addresses_key = "_".join(sorted_addresses)
                    topic = f"onchain_tx_{addresses_key}"
                    
                    fetcher_coro = _alchemy_transactions_fetcher(topic, params)
                except Exception as e:
                    logger.error(f"Invalid payload for '{event}': {e}")
                    await websocket.send_json({"event": "error", "payload": {"message": f"Invalid payload for {event}"}})
                    continue

            elif event == "subscribe_onchain_btc":
                topic = "onchain_btc_all_tx"
                fetcher_coro = _bitcoin_transactions_fetcher(topic)
                
                subscribed_topics.add(topic)
                await global_connection_manager.subscribe(websocket, topic)
                await ensure_fetcher_task_running(topic, fetcher_coro)

            elif event == "unsubscribe":
                topic_to_leave = payload.get("topic")
                if topic_to_leave and topic_to_leave in subscribed_topics:
                    subscribed_topics.discard(topic_to_leave)
                    if await global_connection_manager.unsubscribe(websocket, topic_to_leave):
                        await stop_fetcher_task_if_unneeded(topic_to_leave)
            
            if topic and fetcher_coro:
                subscribed_topics.add(topic)
                await global_connection_manager.subscribe(websocket, topic)
                await ensure_fetcher_task_running(topic, fetcher_coro)
            elif event != "unsubscribe":
                await websocket.send_json({"event": "error", "payload": {"message": f"Unknown event or invalid params: {event}"}})

    except WebSocketDisconnect:
        logger.info(f"[WS Gateway] Client {client_host} disconnected.")
        is_connected = False
    except ConnectionResetError:
        logger.info(f"[WS Gateway] Connection reset for client {client_host}.")
        is_connected = False
    except BrokenPipeError:
        logger.info(f"[WS Gateway] Broken pipe for client {client_host}.")
        is_connected = False
    except Exception as e:
        logger.exception(f"[WS Gateway] Unexpected error for client {client_host}: {e}")
        is_connected = False
    finally:
        logger.info(f"Cleaning up connections for {client_host}. Unsubscribing from all topics.")

        # 1. Remove từ tất cả subscriptions
        topics_that_might_be_empty = await global_connection_manager.disconnect(websocket)

        # 2. Stop ALL fetcher tasks NGAY LẬP TỨC
        stop_tasks = []
        for topic in topics_that_might_be_empty:
            stop_tasks.append(stop_fetcher_task_if_unneeded(topic))
        
        # Chờ tất cả tasks stop xong (với timeout)
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # for topic in topics_that_might_be_empty:
        #     await stop_fetcher_task_if_unneeded(topic)
        
        # Đảm bảo websocket được đóng properly
        try:
            if hasattr(websocket, 'client_state') and websocket.client_state != WebSocketState.DISCONNECTED:
                await websocket.close(code=1000)  # Normal closure
        except Exception as e:
            logger.debug(f"Error closing websocket for {client_host}: {e}")
            pass 
        
        logger.info(f"Connection closed for {client_host}.")

        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
        logger.info(f"Connection closed for {client_host}.")

#region --- Ticker Tape and Tops---
@router.get("/ticker_tape",
            response_model=APIResponse[TickerTapeData],
            summary="Lấy dữ liệu ticker tape cho danh sách mã chứng khoán từ FMP")
async def http_ticker_tape(
    symbols: str = Query("^GSPC,AAPL,MSFT", description="Danh sách các mã, cách nhau bởi dấu phẩy"),
    provider: Optional[str] = Query(settings.DEFAULT_PROVIDER, description="Nhà cung cấp dữ liệu"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    symbols_list = sorted([s.strip().upper() for s in symbols.split(',') if s.strip()])
    if not symbols_list:
        logger.warning("HTTP Ticker Tape: No valid symbols provided.")
        raise HTTPException(status_code=400, detail="Không có mã hợp lệ nào được cung cấp.")

    MAX_TICKER_SYMBOLS = 50 
    if len(symbols_list) > MAX_TICKER_SYMBOLS:
        logger.warning(f"HTTP Ticker Tape: Too many symbols requested: {len(symbols_list)}. Max: {MAX_TICKER_SYMBOLS}")
        raise HTTPException(status_code=400, detail=f"Chỉ cho phép tối đa {MAX_TICKER_SYMBOLS} mã cho ticker tape.")

    cache_key = f"ticker_tape_{'_'.join(symbols_list)}_{provider}"
    logger.debug(f"HTTP Ticker Tape: Request for symbols {symbols_list}, provider {provider}. Cache key: {cache_key}")


    cached_data: Optional[List[TickerTapeData]] = await get_list_cache(redis_client, cache_key, TickerTapeData)
    if cached_data:
        logger.info(f"HTTP Ticker Tape: Cache HIT for {cache_key}")
        response_data_payload = APIResponseData[TickerTapeData](data=cached_data)
        return APIResponse[TickerTapeData](
            message="OK (cached)",
            status="200",
            provider_used="fmp_cached",
            data=response_data_payload
        )
    logger.info(f"HTTP Ticker Tape: Cache MISS for {cache_key}. Fetching new data.")

    ticker_data_with_none = await TickerTapeService.get_ticker_tape_batch(symbols_list, provider=provider)
    valid_data: List[TickerTapeData] = [item for item in ticker_data_with_none if item is not None]

    actual_provider_used = provider
    response_data_payload = APIResponseData[TickerTapeData](data=valid_data)

    message_to_return = "OK"
    if not valid_data and symbols_list:
        message_to_return = f"Không tìm thấy dữ liệu ticker tape cho các mã được yêu cầu từ {actual_provider_used}."
        logger.info(f"HTTP Ticker Tape: No valid data found for {symbols_list} from {actual_provider_used}")

    if valid_data:
        await set_list_cache(redis_client, cache_key, valid_data, expiry=settings.CACHE_TTL_TICKER)
        logger.info(f"HTTP Ticker Tape: Cached new data for {cache_key}")


    return APIResponse[TickerTapeData](
        message=message_to_return,
        status="200",
        provider_used=actual_provider_used,
        data=response_data_payload
    )

#region --- Gainers, Losers, Actives ---
@router.get("/discovery/losers",
            response_model=APIResponse[DiscoveryItemOutput],
            summary="Lấy top thị trường giảm giá (từ cache)")
async def get_losers_http(
    pageNumber: int = Query(1, ge=1, description="Số trang bắt đầu từ 1"),
    pageSize: int = Query(10, ge=1, le=100, description="Số lượng item trên mỗi trang"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    cached_paginated_object: Optional[PaginatedData[DiscoveryItemOutput]] = await get_paginated_cache(
        redis_client, LOSERS_CACHE_KEY, PaginatedData[DiscoveryItemOutput]
    )

    if cached_paginated_object is None or cached_paginated_object.data is None:
        logger.warning(f"Cache MISS cho key chính: {LOSERS_CACHE_KEY}. Scheduler có thể chưa chạy hoặc đang lỗi.")
        return APIResponse[DiscoveryItemOutput](
            data=APIResponseData[DiscoveryItemOutput](data=[]),
            totalRows=0,
            message="Dữ liệu đang được chuẩn bị, vui lòng thử lại sau 1 phút.",
            provider_used="cache_miss",
            status="202"
        )
    
    logger.info(f"Cache HIT cho key chính: {LOSERS_CACHE_KEY}. Phân trang cho page={pageNumber}, pageSize={pageSize}")
    
    full_data_list = cached_paginated_object.data
    total_rows = cached_paginated_object.totalRows

    start_index = (pageNumber - 1) * pageSize
    end_index = start_index + pageSize
    paginated_data_slice = full_data_list[start_index:end_index]
    
    response_data = APIResponseData[DiscoveryItemOutput](data=paginated_data_slice)
    return APIResponse[DiscoveryItemOutput](
        provider_used="prefetched_cache",
        data=response_data,
        message="OK",
        totalRows=total_rows
    )

@router.get("/discovery/gainers",
            response_model=APIResponse[DiscoveryItemOutput],
            summary="Lấy top thị trường tăng giá (từ cache)")
async def get_gainers_http(
    pageNumber: int = Query(1, ge=1, description="Số trang bắt đầu từ 1"),
    pageSize: int = Query(10, ge=1, le=100, description="Số lượng item trên mỗi trang"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    cached_paginated_object: Optional[PaginatedData[DiscoveryItemOutput]] = await get_paginated_cache(
        redis_client, GAINERS_CACHE_KEY, PaginatedData[DiscoveryItemOutput]
    )

    if cached_paginated_object is None or cached_paginated_object.data is None:
        logger.warning(f"Cache MISS cho key chính: {GAINERS_CACHE_KEY}. Scheduler có thể chưa chạy hoặc đang lỗi.")
        return APIResponse[DiscoveryItemOutput](
            data=APIResponseData[DiscoveryItemOutput](data=[]),
            totalRows=0,
            message="Dữ liệu đang được chuẩn bị, vui lòng thử lại sau 1 phút.",
            provider_used="cache_miss",
            status="202"
        )
    
    logger.info(f"Cache HIT cho key chính: {GAINERS_CACHE_KEY}. Phân trang cho page={pageNumber}, pageSize={pageSize}")

    full_data_list = cached_paginated_object.data
    total_rows = cached_paginated_object.totalRows
    
    start_index = (pageNumber - 1) * pageSize
    end_index = start_index + pageSize
    paginated_data_slice = full_data_list[start_index:end_index]
    
    response_data = APIResponseData[DiscoveryItemOutput](data=paginated_data_slice)
    return APIResponse[DiscoveryItemOutput](
        provider_used="prefetched_cache",
        data=response_data,
        message="OK",
        totalRows=total_rows
    )

@router.get("/discovery/active",
            response_model=APIResponse[DiscoveryItemOutput],
            summary="Lấy các mã giao dịch sôi động nhất (từ cache)")
async def get_actives_http(
    pageNumber: int = Query(1, ge=1, description="Số trang bắt đầu từ 1"),
    pageSize: int = Query(10, ge=1, le=100, description="Số lượng item trên mỗi trang"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    cached_paginated_object: Optional[PaginatedData[DiscoveryItemOutput]] = await get_paginated_cache(
        redis_client, ACTIVES_CACHE_KEY, PaginatedData[DiscoveryItemOutput]
    )
    
    if cached_paginated_object is None or cached_paginated_object.data is None:
        logger.warning(f"Cache MISS cho key chính: {ACTIVES_CACHE_KEY}. Scheduler có thể chưa chạy hoặc đang lỗi.")
        return APIResponse[DiscoveryItemOutput](
            data=APIResponseData[DiscoveryItemOutput](data=[]),
            totalRows=0,
            message="Dữ liệu đang được chuẩn bị, vui lòng thử lại sau 1 phút.",
            provider_used="cache_miss",
            status="202"
        )
    
    logger.info(f"Cache HIT cho key chính: {ACTIVES_CACHE_KEY}. Phân trang cho page={pageNumber}, pageSize={pageSize}")

    full_data_list = cached_paginated_object.data
    total_rows = cached_paginated_object.totalRows
    
    start_index = (pageNumber - 1) * pageSize
    end_index = start_index + pageSize
    paginated_data_slice = full_data_list[start_index:end_index]
    
    response_data = APIResponseData[DiscoveryItemOutput](data=paginated_data_slice)
    return APIResponse[DiscoveryItemOutput](
        provider_used="prefetched_cache",
        data=response_data,
        message="OK",
        totalRows=total_rows
    )

#region --- Screener ---
@router.get("/screener",
            response_model=APIResponse[ScreenerOutput],
            summary="Lấy kết quả sàng lọc cổ phiếu, hỗ trợ phân trang và cache")
async def get_screener_http(
    pageNumber: int = Query(1, ge=1, description="Số trang, bắt đầu từ 1"),
    pageSize: int = Query(20, ge=1, le=200, description="Số lượng item trên mỗi trang"),
    market_cap_more_than: Optional[float] = Query(None, alias="marketCapMoreThan"),
    market_cap_lower_than: Optional[float] = Query(None, alias="marketCapLowerThan"),
    price_more_than: Optional[float] = Query(None, alias="priceMoreThan"),
    price_lower_than: Optional[float] = Query(None, alias="priceLowerThan"),
    beta_more_than: Optional[float] = Query(None, alias="betaMoreThan"),
    beta_lower_than: Optional[float] = Query(None, alias="betaLowerThan"),
    volume_more_than: Optional[float] = Query(None, alias="volumeMoreThan"),
    volume_lower_than: Optional[float] = Query(None, alias="volumeLowerThan"),
    dividend_more_than: Optional[float] = Query(None, alias="dividendMoreThan"),
    dividend_lower_than: Optional[float] = Query(None, alias="dividendLowerThan"),
    is_etf: Optional[bool] = Query(None, alias="isEtf"),
    is_fund: Optional[bool] = Query(None, alias="isFund"),
    is_actively_trading: Optional[bool] = Query(True, alias="isActivelyTrading"),
    sector: Optional[str] = Query(None),
    industry: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    exchange: Optional[str] = Query(None),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    filter_params = {
        "market_cap_more_than": market_cap_more_than, "market_cap_lower_than": market_cap_lower_than,
        "price_more_than": price_more_than, "price_lower_than": price_lower_than,
        "beta_more_than": beta_more_than, "beta_lower_than": beta_lower_than,
        "volume_more_than": volume_more_than, "volume_lower_than": volume_lower_than,
        "dividend_more_than": dividend_more_than, "dividend_lower_than": dividend_lower_than,
        "is_etf": is_etf, "is_fund": is_fund, "is_actively_trading": is_actively_trading,
        "sector": sector, "industry": industry, "country": country, "exchange": exchange,
    }
    active_filters = {k: v for k, v in filter_params.items() if v is not None}
    is_custom_filter_request = bool(active_filters)

    full_data_list: Optional[List[ScreenerOutput]] = None
    total_rows: int = 0
    provider_used: str = ""
    cache_key = ""

    if not is_custom_filter_request:
        cache_key = DEFAULT_SCREENER_CACHE_KEY
        logger.info(f"Screener: Default request. Reading from key: {cache_key}")
        cached_paginated_object = await get_paginated_cache(redis_client, cache_key, PaginatedData[ScreenerOutput])
        if cached_paginated_object:
            full_data_list = cached_paginated_object.data
            total_rows = cached_paginated_object.totalRows
            provider_used = "fmp_prefetched_cache"
    else:
        filters_for_key = active_filters.copy()
        filters_for_key['limit'] = 100 
        sorted_filters_str = json.dumps(filters_for_key, sort_keys=True)
        hashed_filters = hashlib.md5(sorted_filters_str.encode('utf-8')).hexdigest()
        cache_key = f"screener_custom_{hashed_filters}"
        
        cached_paginated_object = await get_paginated_cache(redis_client, cache_key, PaginatedData[ScreenerOutput])
        if cached_paginated_object:
            full_data_list = cached_paginated_object.data
            total_rows = cached_paginated_object.totalRows
            provider_used = "fmp_screener_cached"

    if full_data_list is None:
        logger.info(f"Screener: Cache MISS for key '{cache_key}'. Fetching live data.")
        if not is_custom_filter_request:
            return APIResponse[ScreenerOutput](
                data=APIResponseData[ScreenerOutput](data=[]), totalRows=0,
                message="Dữ liệu mặc định chưa sẵn sàng, vui lòng thử lại sau.",
                provider_used="cache_miss", status="202"
            )

        filters_for_service = active_filters.copy()
        filters_for_service['limit'] = 200

        data_from_service = await discovery_service.get_screener_http_compatible(**filters_for_service)
        
        if data_from_service is not None:
            full_data_list = data_from_service
            total_rows = len(full_data_list)
            provider_used = "fmp_screener_live"
            paginated_result = PaginatedData(totalRows=total_rows, data=full_data_list)
            await set_paginated_cache(redis_client, cache_key, paginated_result, expiry=settings.CACHE_DEFAULT_TTL)
        else:
            logger.error(f"Screener: Service error for filters {active_filters}")
            raise HTTPException(status_code=502, detail="Lỗi khi lấy dữ liệu sàng lọc từ nhà cung cấp.")

    if not full_data_list:
        return APIResponse[ScreenerOutput](
            data=APIResponseData[ScreenerOutput](data=[]),
            totalRows=0,
            message="Không tìm thấy kết quả nào khớp với tiêu chí.",
            provider_used=provider_used, status="200"
        )
    
    start_index = (pageNumber - 1) * pageSize
    end_index = start_index + pageSize
    paginated_data = full_data_list[start_index:end_index]

    response_data_payload = APIResponseData[ScreenerOutput](data=paginated_data)
    return APIResponse[ScreenerOutput](
        provider_used=provider_used,
        data=response_data_payload,
        message="OK",
        status="200",
        totalRows=total_rows
    )

#region --- Market Indices by Region ---
@router.get("/market-region",
            response_model=APIResponse[DiscoveryItemOutput],
            summary="Lấy dữ liệu tổng quan các chỉ số chính theo khu vực")
async def get_market_indices_by_region_http(
    region: Optional[MarketRegionEnum] = Query(None, description="Chọn một khu vực cụ thể hoặc bỏ trống để lấy tất cả."),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Endpoint này trả về dữ liệu giá cho các chỉ số và hợp đồng tương lai chính
    của một khu vực thị trường (US, Europe, Asia).
    """
    is_all_regions = region is None or region == MarketRegionEnum.ALL
    
    if is_all_regions:
        cache_key = "market_indices_all"
    else:
        cache_key = f"market_indices_{region.value.lower()}"

    cache_ttl = 60

    if redis_client:
        try:
            cached_data: Optional[List[DiscoveryItemOutput]] = await get_list_cache(
                redis_client, cache_key, DiscoveryItemOutput
            )
            if cached_data is not None:
                logger.info(f"Cache HIT cho market indices ({region.value}): {cache_key}")
                response_data_payload = APIResponseData[DiscoveryItemOutput](data=cached_data)
                return APIResponse[DiscoveryItemOutput](
                    message="OK (cached)",
                    provider_used="cached_fmp",
                    data=response_data_payload
                )
        except Exception as e_cache_get:
            logger.error(f"Lỗi Redis GET cho market indices ({region.value}, key: {cache_key}): {e_cache_get}", exc_info=True)

    logger.info(f"Cache MISS cho market indices ({region.value}): {cache_key}. Đang lấy từ FMP.")

    data_list: Optional[List[DiscoveryItemOutput]]
    if is_all_regions:
        data_list = await indices_service.get_and_fetch_all_indices(redis_client=redis_client) 
    else:
        data_list = await indices_service.get_and_fetch_indices_by_region(redis_client=redis_client,region=region.value)

    if data_list is None:
        logger.error(f"Service không thể lấy được dữ liệu cho region: {region.value}.")
        raise HTTPException(
            status_code=502,
            detail=f"Không thể lấy dữ liệu chỉ số từ nhà cung cấp cho region: {region.value}."
        )

    response_data_payload = APIResponseData[DiscoveryItemOutput](data=data_list)
    message_to_return = "OK"
    if not data_list:
        message_to_return = f"Không tìm thấy dữ liệu chỉ số nào cho region: {region.value}."
        logger.info(message_to_return)

    api_response_obj = APIResponse[DiscoveryItemOutput](
        message=message_to_return,
        status="200",
        provider_used="fmp_direct",
        data=response_data_payload
    )

    if redis_client and data_list is not None:
        try:
            await set_list_cache(
                redis_client,
                cache_key,
                data_list,
                expiry=cache_ttl
            )
            logger.info(f"Đã cache dữ liệu market indices cho key: {cache_key} với TTL {cache_ttl}s")
        except Exception as e_cache_set:
            logger.error(f"Lỗi Redis SET cho market indices ({region.value}, key: {cache_key}): {e_cache_set}", exc_info=True)

    return api_response_obj

#region --- Search --- 
@router.get("/equity/search",
            response_model=APIResponse[FMPSearchResultItem],
            summary="Tìm kiếm mã cổ phiếu và các tài sản khác bằng FMP")
async def search_symbols_http(
    query: str = Query(..., min_length=1, description="Từ khóa tìm kiếm (ví dụ: AAPL, Apple, Bitcoin)"),
    limit: Optional[int] = Query(10, ge=1, le=10000, description="Giới hạn số lượng kết quả tìm kiếm (ví dụ: 5, 10)"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    cache_key = f"equity_search_{query.lower()}_limit_{limit if limit else 'all'}"
    logger.debug(f"Equity Search: Query '{query}', Limit {limit}. Cache key: {cache_key}")

    cached_data: Optional[APIResponse[FMPSearchResultItem]] = await get_cache(
        redis_client, cache_key, APIResponse[FMPSearchResultItem]
    )
    if cached_data and cached_data.data:
        logger.info(f"Equity Search: Cache HIT for {cache_key}")
        cached_data.message = "OK (cached)"
        cached_data.provider_used = "fmp_cached"
        return cached_data
    logger.info(f"Equity Search: Cache MISS for {cache_key}. Fetching new data.")


    actual_provider_used = "fmp"

    if not query.strip():
        logger.warning("Equity Search: Empty search query provided.")
        raise HTTPException(status_code=400, detail="Từ khóa tìm kiếm không được để trống.")

    search_results_list: Optional[List[FMPSearchResultItem]] = await search_service.search_symbols_fmp(
        query_term=query,
        limit_count=limit
    )

    if search_results_list is None:
        logger.error(f"Equity Search: Failed to perform search for '{query}' via {actual_provider_used}.")
        raise HTTPException(status_code=502, detail=f"Không thể thực hiện tìm kiếm cho '{query}' qua nhà cung cấp {actual_provider_used}.")

    items_to_return = search_results_list
    response_data_payload = APIResponseData[FMPSearchResultItem](data=items_to_return)

    message_to_return = "OK"
    if not items_to_return:
        message_to_return = f"Không tìm thấy kết quả nào cho '{query}'."
        logger.info(f"Equity Search: No results found for '{query}'.")


    api_response = APIResponse[FMPSearchResultItem](
        message=message_to_return,
        status="200",
        provider_used=actual_provider_used,
        data=response_data_payload
    )
    await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_LISTS)
    logger.info(f"Equity Search: Cached results for {cache_key} (Count: {len(items_to_return)})")
    return api_response

#region --- Company Profile---
@router.get("/equity/profile",
            response_model=APIResponse[FMPCompanyOutlookProfile],
            summary="Lấy thông tin profile chi tiết công ty từ FMP (sử dụng tối đa 3 API calls)")
async def get_company_profile_http(
    symbol: str = Query(..., description="Mã cổ phiếu"),
    provider: Optional[str] = Query(settings.DEFAULT_PROVIDER, description="Nhà cung cấp (hiện tại sẽ dùng FMP)"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    upper_symbol = symbol.upper()
    cache_key = f"equity_profile_v2{upper_symbol}"
    logger.debug(f"Company Profile: Symbol '{upper_symbol}'. Cache key: {cache_key}")

    cached_data: Optional[APIResponse[FMPCompanyOutlookProfile]] = await get_cache(
        redis_client, cache_key, APIResponse[FMPCompanyOutlookProfile]
    )
    if cached_data and cached_data.data:
        logger.info(f"Company Profile: Cache HIT for {cache_key}")
        cached_data.message = "OK (cached)"
        cached_data.provider_used = "fmp_cached"
        return cached_data
    logger.info(f"Company Profile: Cache MISS for {cache_key}. Fetching new data.")


    actual_provider_used = provider
    profile_detail_obj: Optional[FMPCompanyOutlookProfile] = await profile_service.get_company_profile(
        symbol=upper_symbol, 
        provider=actual_provider_used
    )

    if not profile_detail_obj:
        logger.warning(f"Company Profile: No profile data found for {upper_symbol} from {actual_provider_used}.")
        raise HTTPException(status_code=404, detail=f"Không tìm thấy dữ liệu profile chi tiết cho {upper_symbol} từ {actual_provider_used}.")

    # wrapper_payload = ProfileResponseWrapper(
    #     symbol=upper_symbol,
    #     symbol_name=profile_detail_obj.name if profile_detail_obj.name else upper_symbol,
    #     data=[profile_detail_obj]
    # )

    # response_data_obj = APIResponseData[ProfileResponseWrapper](data=[wrapper_payload])
    # api_response = APIResponse[ProfileResponseWrapper](
    #     provider_used=actual_provider_used,
    #     data=response_data_obj,
    #     message="OK"
    # )
    
    profile_detail_obj.symbol = upper_symbol
    profile_detail_obj.symbol_name = profile_detail_obj.name if profile_detail_obj.name else upper_symbol
    
    response_data_obj = APIResponseData[FMPCompanyOutlookProfile](data=[profile_detail_obj])
    api_response = APIResponse[FMPCompanyOutlookProfile](
        provider_used=actual_provider_used,
        data=response_data_obj,
        message="OK",
    )

    await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_PROFILE)
    logger.info(f"Company Profile: Cached new data for {cache_key}")
    return api_response

#region --- Market Overview ---
@router.get("/equity/overview_market",
            response_model=APIResponse[MarketOverviewData],
            summary="Get detailed market overview for specified symbols")
async def get_market_overview_http(
    symbols: str = Query("^GSPC", description="Danh sách các mã, cách nhau bởi dấu phẩy. Ví dụ: AAPL,MSFT,^GSPC"),
    provider: Optional[str] = Query(settings.DEFAULT_PROVIDER, description=f"Nhà cung cấp dữ liệu. Mặc định: {settings.DEFAULT_PROVIDER}"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    symbols_list = sorted([s.strip().upper() for s in symbols.split(',') if s.strip()])
    if not symbols_list:
        logger.warning("Market Overview: No valid symbols provided.")
        raise HTTPException(status_code=400, detail="Không có mã hợp lệ nào được cung cấp.")

    MAX_SYMBOLS_OVERVIEW = 10
    if len(symbols_list) > MAX_SYMBOLS_OVERVIEW:
        logger.warning(f"Market Overview: Too many symbols requested: {len(symbols_list)}. Max: {MAX_SYMBOLS_OVERVIEW}")
        raise HTTPException(status_code=400, detail=f"Quá nhiều mã. Tối đa {MAX_SYMBOLS_OVERVIEW} mã cho mỗi yêu cầu market overview.")

    cache_key = f"market_overview_{'_'.join(symbols_list)}_{provider}"
    logger.debug(f"Market Overview: Symbols {symbols_list}, Provider {provider}. Cache key: {cache_key}")

    cached_data: Optional[APIResponse[MarketOverviewData]] = await get_cache(
        redis_client, cache_key, APIResponse[MarketOverviewData]
    )
    if cached_data and cached_data.data:
        logger.info(f"Market Overview: Cache HIT for {cache_key}")
        cached_data.message = "OK (cached)"
        cached_data.provider_used = f"{provider}_cached"
        return cached_data
    logger.info(f"Market Overview: Cache MISS for {cache_key}. Fetching new data.")


    overview_data_with_none = await market_overview_service.get_market_overview(symbols=symbols_list, provider=provider)
    valid_data: List[MarketOverviewData] = [item for item in overview_data_with_none if item is not None]

    message_to_return = "OK"
    if not valid_data and symbols_list:
        logger.warning(f"Market Overview: No valid data for {symbols_list} from provider {provider}")
        message_to_return = f"Không tìm thấy dữ liệu tổng quan thị trường hợp lệ cho các mã đã cho từ {provider}."


    response_data_payload = APIResponseData[MarketOverviewData](data=valid_data)
    api_response = APIResponse[MarketOverviewData](
        provider_used=provider,
        data=response_data_payload,
        message=message_to_return
    )

    await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_DETAILS)
    logger.info(f"Market Overview: Cached results for {cache_key} (Count: {len(valid_data)})")
    return api_response

#region --- Company News ---

@router.get("/news/by-symbols",
            response_model=APIResponse[NewsItem],
            summary="Lấy tin tức cho một danh sách các mã cổ phiếu")
async def get_news_by_symbols_http(
    # Client sẽ truyền các mã dưới dạng: ?symbols=AAPL,MSFT,TSLA,NVDA,GOOG,AMD
    symbols: str = Query(..., description="Danh sách các mã, cách nhau bởi dấu phẩy."),
    limit: int = Query(5, description="Số lượng tin tức tối đa cho mỗi mã (mặc định 5)"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Lấy các tin tức mới nhất cho một danh sách các mã cổ phiếu được cung cấp.
    Tối đa 5 mã mỗi lần gọi API, hệ thống sẽ tự động chia nhỏ và gọi nhiều lần nếu cần.
    """
    # Chuẩn hóa danh sách mã để tạo cache key nhất quán
    symbols_list = sorted([s.strip().upper() for s in symbols.split(',') if s.strip()])
    if not symbols_list:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp ít nhất một mã hợp lệ.")

    symbols_key = "_".join(symbols_list)
    cache_key = f"news_by_symbols_{symbols_key}"
    
    # Kiểm tra cache
    if redis_client:
        cached_data: Optional[List[NewsItem]] = await get_list_cache(
            redis_client, cache_key, NewsItem
        )
        if cached_data is not None:
            return APIResponse[NewsItem](
                message="OK (cached)",
                provider_used="cached_fmp",
                data=APIResponseData[NewsItem](data=cached_data)
            )


    # Gọi service
    data_list = await news_service.get_news_for_multiple_symbols(limit=limit, symbols=symbols_list)

    if data_list is None:
        raise HTTPException(status_code=502, detail="Không thể lấy dữ liệu tin tức từ nhà cung cấp.")

    # Lưu vào cache
    if redis_client and data_list:
        await set_list_cache(redis_client, cache_key, data_list, expiry=settings.CACHE_TTL_NEWS)

    # Trả về response
    return APIResponse[NewsItem](
        message="OK",
        status="200",
        provider_used="fmp_direct",
        data=APIResponseData[NewsItem](data=data_list)
    )

@router.get("/news/company",
            response_model=APIResponse[NewsItemOutput],
            summary="Lấy tin tức cho một công ty cụ thể từ FMP")
async def get_company_news_http(
    symbol: str = Query(..., description="Mã cổ phiếu (ví dụ: AAPL)"),
    limit: int = Query(10, ge=1, le=10000, description="Số lượng tin tức tối đa"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    upper_symbol = symbol.upper()
    cache_key = f"company_news_{upper_symbol}_limit_{limit}"
    logger.debug(f"Company News: Symbol '{upper_symbol}', Limit {limit}. Cache key: {cache_key}")


    cached_data: Optional[APIResponse[NewsItemOutput]] = await get_cache(
        redis_client, cache_key, APIResponse[NewsItemOutput]
    )
    if cached_data and cached_data.data:
        logger.info(f"Company News: Cache HIT for {cache_key}")
        cached_data.message = "OK (cached)"
        cached_data.provider_used = "fmp_cached"
        return cached_data
    logger.info(f"Company News: Cache MISS for {cache_key}. Fetching new data.")


    actual_provider_used = "fmp"
    news_list: Optional[List[NewsItemOutput]] = await news_service.get_company_news(upper_symbol, limit)

    if news_list is None:
        logger.error(f"Company News: Failed to get news for {upper_symbol} from {actual_provider_used}.")
        raise HTTPException(status_code=502, detail=f"Không thể lấy tin tức cho {upper_symbol} từ {actual_provider_used}.")

    items_to_return = news_list
    response_data_payload = APIResponseData[NewsItemOutput](data=items_to_return)
    message = "OK" if items_to_return else f"Không tìm thấy tin tức nào cho {upper_symbol}."
    if not items_to_return:
        logger.info(f"Company News: No news found for {upper_symbol}.")


    api_response = APIResponse[NewsItemOutput](
        message=message,
        status="200",
        provider_used=actual_provider_used,
        data=response_data_payload
    )
    await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_NEWS)
    logger.info(f"Company News: Cached results for {cache_key} (Count: {len(items_to_return)})")
    return api_response

@router.get("/list/news",
            response_model=APIResponse[NewsItemOutput],
            summary="Lấy tin tức thị trường chung từ FMP")
async def get_general_news_http(
    page: int = Query(0, ge=0, description="Số trang để lấy (bắt đầu từ 0)"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    cache_key = f"general_news_page_{page}"
    logger.debug(f"General News: Page {page}. Cache key: {cache_key}")

    cached_data: Optional[APIResponse[NewsItemOutput]] = await get_cache(
        redis_client, cache_key, APIResponse[NewsItemOutput]
    )
    if cached_data and cached_data.data:
        logger.info(f"General News: Cache HIT for {cache_key}")
        cached_data.message = "OK (cached)"
        cached_data.provider_used = "fmp_cached"
        return cached_data
    logger.info(f"General News: Cache MISS for {cache_key}. Fetching new data.")


    actual_provider_used = "fmp"
    news_list: Optional[List[NewsItemOutput]] = await news_service.get_general_news(page=page)

    if news_list is None:
        logger.error(f"General News: Failed to get general news (page {page}) from {actual_provider_used}.")
        raise HTTPException(status_code=502, detail=f"Không thể lấy tin tức chung từ {actual_provider_used}.")

    items_to_return = news_list
    response_data_payload = APIResponseData[NewsItemOutput](data=items_to_return)
    message = "OK" if items_to_return else f"Không tìm thấy tin tức chung nào cho trang {page}."
    if not items_to_return:
        logger.info(f"General News: No news found for page {page}.")

    api_response = APIResponse[NewsItemOutput](
        message=message,
        status="200",
        provider_used=actual_provider_used,
        data=response_data_payload
    )
    await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_NEWS)
    return api_response

@router.get("/news/crypto",
            response_model=APIResponse[CryptoNewsItem],
            summary="Get the latest stock news from FMP")
async def get_crypto_news(
    pageNumber: int = Query(0, ge=0, description="Page number for news results."),
    pageSize: int = Query(20, ge=1, le=100, description="Number of news articles per page (1-100)."),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
        logger.error("Crypto News: FMP API key is not configured.")
        return APIResponse[CryptoNewsItem](
            data=None, 
            message="FMP API key is not configured on the server.",
            provider_used="server_config_error",
            status="500"
        )

    cache_key = f"crypto_news_page_{pageNumber}_limit_{pageSize}"
    logger.debug(f"Crypto News: Page {pageNumber}, Limit {pageSize}. Cache key: {cache_key}")


    cached_response: Optional[APIResponse[CryptoNewsItem]] = await get_cache(
        redis_client, cache_key, APIResponse[CryptoNewsItem]
    )
    if cached_response and cached_response.data and cached_response.data.data:
        logger.info(f"Cache HIT for crypto news: {cache_key}")
        cached_response.message = "OK (cached)"
        cached_response.provider_used = "fmp_direct_cached"
        return cached_response
    logger.info(f"Cache MISS for crypto news: {cache_key}. Fetching from FMP.")


    fmp_url = f"{BASE_FMP_URL}/v4/crypto_news?page={pageNumber}&limit={pageSize}&apikey={FMP_API_KEY}"

    async with httpx.AsyncClient() as http_client:
        try:
            response = await http_client.get(fmp_url)
            response.raise_for_status()
            fmp_data = response.json()

            if not isinstance(fmp_data, list):
                logger.error(f"Crypto News: Invalid response format from FMP. Expected list, got {type(fmp_data)}. URL: {fmp_url}")
                raise HTTPException(status_code=502, detail="Invalid response format from upstream FMP API for crypto news.")

            processed_news: List[CryptoNewsItem] = []
            for item in fmp_data:
                try:
                    processed_news.append(CryptoNewsItem(**item))
                except Exception as e: 
                    logger.error(f"Error parsing crypto news item: {item}. Error: {e}")
                    continue

            message = f"Successfully retrieved {len(processed_news)} crypto news articles."
            if not processed_news and fmp_data:
                message = "FMP returned data, but no valid crypto news items could be processed."
                logger.warning(f"Crypto News: FMP returned data but no items could be processed. Page: {pageNumber}, Limit: {pageSize}")
            elif not processed_news:
                 logger.info(f"Crypto News: No news items found for Page: {pageNumber}, Limit: {pageSize}")


            response_data_payload = APIResponseData[CryptoNewsItem](data=processed_news)
            api_response = APIResponse[CryptoNewsItem](
                data=response_data_payload,
                message=message,
                provider_used="fmp_direct",
                status="200"
            )

            await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_NEWS)
            logger.info(f"Crypto News: Cached results for {cache_key} (Count: {len(processed_news)})")

            return api_response

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching crypto news from FMP: {e.response.status_code} - {e.response.text}. URL: {fmp_url}", exc_info=True)
            return APIResponse[CryptoNewsItem](
                data=None,
                message=f"Error fetching crypto news from FMP: {e.response.status_code}",
                provider_used="fmp_direct_error",
                status=str(e.response.status_code)
            )
        except Exception as e:
            logger.exception(f"An unexpected error occurred while fetching crypto news for page {pageNumber}, limit {pageSize}: {e}")
            return APIResponse[CryptoNewsItem](
                data=None,
                message=f"An unexpected error occurred: {str(e)}",
                provider_used="server_error",
                status="500"
            )

@router.get("/news/stock",
            response_model=APIResponse[CryptoNewsItem],
            summary="Get the latest stock news from FMP")
async def get_stock_news(
    pageNumber: int = Query(0, ge=0, description="Page number for news results."),
    pageSize: int = Query(20, ge=1, le=100, description="Number of news articles per page (1-100)."),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
        logger.error("Crypto News: FMP API key is not configured.")
        return APIResponse[CryptoNewsItem](
            data=None, 
            message="FMP API key is not configured on the server.",
            provider_used="server_config_error",
            status="500"
        )

    cache_key = f"stock_news_page_{pageNumber}_limit_{pageSize}"
    logger.debug(f"Stock News: Page {pageNumber}, Limit {pageSize}. Cache key: {cache_key}")

    cached_response: Optional[APIResponse[CryptoNewsItem]] = await get_cache(
        redis_client, cache_key, APIResponse[CryptoNewsItem]
    )
    if cached_response and cached_response.data and cached_response.data.data:
        logger.info(f"Cache HIT for stock news: {cache_key}")
        cached_response.message = "OK (cached)"
        cached_response.provider_used = "fmp_direct_cached"
        return cached_response
    logger.info(f"Cache MISS for stock news: {cache_key}. Fetching from FMP.")


    fmp_url = f"{FMP_URL_STABLE}/news/stock-latest?page={pageNumber}&limit={pageSize}&apikey={FMP_API_KEY}"

    async with httpx.AsyncClient() as http_client:
        try:
            response = await http_client.get(fmp_url)
            response.raise_for_status()
            fmp_data = response.json()

            if not isinstance(fmp_data, list):
                logger.error(f"Stock News: Invalid response format from FMP. Expected list, got {type(fmp_data)}. URL: {fmp_url}")
                raise HTTPException(status_code=502, detail="Invalid response format from upstream FMP API for stock news.")

            processed_news: List[CryptoNewsItem] = []
            untrusted_sources_found = set()

            for item in fmp_data:
                site = item.get("site")

                if site and site in TRUSTED_NEWS_SOURCES:
                    try:
                        processed_news.append(CryptoNewsItem(**item))
                    except Exception as e: 
                        logger.error(f"Error parsing trusted news item: {item}. Error: {e}")
                else:
                    if site:
                        untrusted_sources_found.add(site)
            if untrusted_sources_found:
                logger.debug(f"Skipped articles from untrusted sources: {untrusted_sources_found}")

            message = f"Successfully retrieved {len(processed_news)} stock news articles."
            if not processed_news and fmp_data:
                message = "FMP returned data, but no valid stock news items could be processed."
                logger.warning(f"Stock News: FMP returned data but no items could be processed. Page: {pageNumber}, Limit: {pageSize}")
            elif not processed_news:
                 logger.info(f"Stock News: No news items found for Page: {pageNumber}, Limit: {pageSize}")


            response_data_payload = APIResponseData[CryptoNewsItem](data=processed_news)
            api_response = APIResponse[CryptoNewsItem](
                data=response_data_payload,
                message=message,
                provider_used="fmp_direct",
                status="200"
            )

            await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_NEWS)
            logger.info(f"Stock News: Cached results for {cache_key} (Count: {len(processed_news)})")

            return api_response

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching stock news from FMP: {e.response.status_code} - {e.response.text}. URL: {fmp_url}", exc_info=True)
            return APIResponse[CryptoNewsItem](
                data=None,
                message=f"Error fetching stock news from FMP: {e.response.status_code}",
                provider_used="fmp_direct_error",
                status=str(e.response.status_code)
            )
        except Exception as e:
            logger.exception(f"An unexpected error occurred while fetching stock news for page {pageNumber}, limit {pageSize}: {e}")
            return APIResponse[CryptoNewsItem](
                data=None,
                message=f"An unexpected error occurred: {str(e)}",
                provider_used="server_error",
                status="500"
            )
        
@router.get("/news/sentiments-rss",
            response_model=APIResponse[StockNewsSentimentItem],
            summary="Lấy tin tức cảm xúc thị trường dạng RSS từ FMP")
async def get_stock_news_sentiments_rss_feed_http(
    page: int = Query(0, ge=0, description="Số trang để lấy (bắt đầu từ 0)"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
        logger.error("FMP API key is not configured for news sentiments RSS.")
        error_data_payload = APIResponseData[StockNewsSentimentItem](data=[])
        return APIResponse[StockNewsSentimentItem](
            message="FMP API key is not configured on the server.",
            status="500",
            provider_used="server_config_error",
            data=error_data_payload
        )

    cache_key = f"news_sentiments_rss_page_{page}"
    if redis_client:
        try:
            cached_api_response: Optional[APIResponse[StockNewsSentimentItem]] = await get_cache(
                redis_client, cache_key, APIResponse[StockNewsSentimentItem] 
            )
            if cached_api_response and cached_api_response.data and cached_api_response.data.data:
                logger.info(f"Cache HIT for news sentiments RSS: {cache_key}")
                cached_api_response.message = "OK (cached)"
                cached_api_response.provider_used = "cached_fmp"
                return cached_api_response
        except Exception as e_cache_get:
            logger.error(f"Redis GET error for news sentiments RSS {cache_key}: {e_cache_get}", exc_info=True)

    logger.info(f"Cache MISS for news sentiments RSS: {cache_key}. Fetching from FMP.")

    sentiment_news_list: Optional[List[StockNewsSentimentItem]] = await news_service.get_stock_news_sentiments_rss(page=page)

    actual_provider_used = "fmp_direct"

    if sentiment_news_list is None:
        logger.error(f"Service call failed to fetch stock news sentiments for page {page}.")
        error_data_payload = APIResponseData[StockNewsSentimentItem](data=[])
        return APIResponse[StockNewsSentimentItem](
            message=f"Không thể lấy dữ liệu tin tức cảm xúc từ nhà cung cấp cho trang {page}.",
            status="502",
            provider_used=actual_provider_used + "_error",
            data=error_data_payload
        )

    items_to_return = sentiment_news_list
    response_data_payload = APIResponseData[StockNewsSentimentItem](data=items_to_return)
    message_to_return = "OK"
    if not items_to_return:
        message_to_return = f"Không tìm thấy tin tức cảm xúc nào cho trang {page}."
        logger.info(message_to_return)

    api_response = APIResponse[StockNewsSentimentItem](
        message=message_to_return,
        status="200",
        provider_used=actual_provider_used,
        data=response_data_payload
    )

    if redis_client and items_to_return: 
        try:
            await set_cache(
                redis_client,
                cache_key,
                api_response, 
                expiry=settings.CACHE_TTL_NEWS
            )
            logger.info(f"Successfully prepared and attempted to cache news sentiments RSS for {cache_key}")
        except Exception as e_cache_set:
            logger.error(f"Redis SET error (after Pydantic serialization) for news sentiments RSS {cache_key}: {e_cache_set}", exc_info=True)

    return api_response

@router.get("/news/event",
            response_model=APIResponse[PressReleaseItem],
            summary="Lấy các thông cáo báo chí mới nhất")
async def get_latest_press_releases_http(
    pageNumber: int = Query(0, ge=0, description="Số trang để lấy (bắt đầu từ 0)."),
    pageSize: int = Query(20, ge=1, le=100, description="Số lượng kết quả mỗi trang."),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Lấy danh sách các thông cáo báo chí (Press Releases) mới nhất từ FMP.
    Kết quả được cache lại để tăng tốc độ.
    """
    cache_key = f"press_releases_latest_p{pageNumber}_l{pageSize}"
    
    if redis_client:
        cached_data: Optional[List[PressReleaseItem]] = await get_list_cache(
            redis_client, cache_key, PressReleaseItem
        )
        if cached_data is not None:
            logger.info(f"Cache HIT for latest press releases: {cache_key}")
            return APIResponse[PressReleaseItem](
                message="OK (cached)",
                provider_used="cached_fmp",
                data=APIResponseData[PressReleaseItem](data=cached_data)
            )

    logger.info(f"Cache MISS for press releases: {cache_key}. Fetching new data.")

    data_list = await news_service.get_latest_press_releases(page=pageNumber, limit=pageSize)

    if data_list is None:
        raise HTTPException(
            status_code=502,
            detail="Không thể lấy dữ liệu thông cáo báo chí từ nhà cung cấp."
        )

    if redis_client and data_list:
        await set_list_cache(redis_client, cache_key, data_list, expiry=settings.CACHE_TTL_NEWS)
        logger.info(f"Cached new press releases data for {cache_key}")

    response_data_payload = APIResponseData[PressReleaseItem](data=data_list)
    return APIResponse[PressReleaseItem](
        message="OK",
        status="200",
        provider_used="fmp_direct",
        data=response_data_payload
    )

#region --- Crypto Spotlight and Detail ---
@router.get("/crypto-spotlight",
            response_model=APIResponse[CryptoSpotlightItem],
            summary="Lấy danh sách crypto nổi bật với chi tiết và logo từ FMP")
async def get_crypto_spotlight_http(
    limit: int = Query(10, ge=1, le=30, description="Số lượng crypto nổi bật trả về (tối đa 30)"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    cache_key = f"crypto_spotlight_limit_{limit}"
    logger.debug(f"Crypto Spotlight: Limit {limit}. Cache key: {cache_key}")

    cached_data: Optional[APIResponse[CryptoSpotlightItem]] = await get_cache(
        redis_client, cache_key, APIResponse[CryptoSpotlightItem]
    )
    if cached_data and cached_data.data: 
        logger.info(f"Crypto Spotlight: Cache HIT for {cache_key}")
        cached_data.message = "OK (cached)"
        cached_data.provider_used = "fmp_cached"
        return cached_data
    logger.info(f"Crypto Spotlight: Cache MISS for {cache_key}. Fetching new data.")


    actual_provider_used = "fmp" 
    spotlight_data: Optional[List[CryptoSpotlightItem]] = await crypto_spotlight_service.get_crypto_spotlight(limit=limit)

    if spotlight_data is None:
        logger.error(f"Crypto Spotlight: Failed to get spotlight data (limit {limit}) from {actual_provider_used}.")
        raise HTTPException(status_code=502, detail=f"Không thể lấy dữ liệu crypto spotlight từ {actual_provider_used}.")

    items_to_return = spotlight_data
    response_data_payload = APIResponseData[CryptoSpotlightItem](data=items_to_return)
    message = "OK" if items_to_return else "Không tìm thấy dữ liệu crypto spotlight."
    if not items_to_return:
        logger.info(f"Crypto Spotlight: No data found for limit {limit}.")


    api_response = APIResponse[CryptoSpotlightItem](
        message=message,
        status="200",
        provider_used=actual_provider_used,
        data=response_data_payload
    )
    await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_NEWS)
    logger.info(f"Crypto Spotlight: Cached results for {cache_key} (Count: {len(items_to_return)})")
    return api_response

# @router.get("/detail/equity",
#             response_model=APIResponse[StockDetailPayload],
#             summary="Lấy thông tin chi tiết cổ phiếu bao gồm quote và dữ liệu chart từ FMP")
# async def get_stock_detail_http(
#     symbol: str = Query(..., description="Mã cổ phiếu (ví dụ: AAPL)"),
#     timeframe: str = Query(..., description="Khung thời gian cho chart (ví dụ: 5min, 1hour). Phải khớp với FMP hỗ trợ."),
#     from_date: Optional[date] = Query(None, alias="from", description="Ngày bắt đầu cho dữ liệu lịch sử (YYYY-MM-DD)"),
#     to_date: Optional[date] = Query(None, alias="to", description="Ngày kết thúc cho dữ liệu lịch sử (YYYY-MM-DD)"),
#     redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
# ):
#     upper_symbol = symbol.upper()
#     cache_key = f"equity_detail_{upper_symbol}_{timeframe}_{from_date}_{to_date}"
#     logger.debug(f"Equity Detail: Symbol '{upper_symbol}', Timeframe '{timeframe}', From '{from_date}', To '{to_date}'. Cache key: {cache_key}")


#     cached_data: Optional[APIResponse[StockDetailPayload]] = await get_cache(
#         redis_client, cache_key, APIResponse[StockDetailPayload]
#     )
#     if cached_data and cached_data.data:
#         # logger.info(f"Equity Detail: Cache HIT for {cache_key}")
#         cached_data.message = "OK (cached)"
#         cached_data.provider_used = "fmp_cached"
#         return cached_data
#     logger.info(f"Equity Detail: Cache MISS for {cache_key}. Fetching new data.")


#     actual_provider_used = "fmp" 
#     from_date_str = from_date.isoformat()
#     to_date_str = to_date.isoformat()

#     valid_fmp_timeframes = ["1min", "5min", "15min", "30min", "1hour", "4hour", "daily"]
#     if timeframe not in valid_fmp_timeframes:
#         logger.warning(f"Equity Detail: Timeframe '{timeframe}' might not be directly supported by FMP /v3/historical-chart/. Proceeding with call.")


#     async with httpx.AsyncClient() as client:
#         stock_detail_data: Optional[StockDetailPayload] = await equity_detail_service.get_equity_detail(
#             symbol=upper_symbol,
#             timeframe=timeframe,
#             from_date_str=from_date_str,
#             to_date_str=to_date_str,
#             client=client
#         )

#     if not stock_detail_data:
#         logger.warning(f"Equity Detail: No data found for {upper_symbol}, timeframe {timeframe}, from {from_date_str} to {to_date_str} using {actual_provider_used}.")
#         raise HTTPException(status_code=404, detail=f"Không thể lấy dữ liệu chi tiết cho mã {upper_symbol} với các tham số đã cho từ {actual_provider_used}.")

#     response_data_payload = APIResponseData[StockDetailPayload](data=[stock_detail_data])
#     api_response = APIResponse[StockDetailPayload](
#         message="OK",
#         status="200",
#         provider_used=actual_provider_used,
#         data=response_data_payload
#     )
#     await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_CHART)
#     # logger.info(f"Equity Detail: Cached new data for {cache_key}")
#     return api_response

@router.get("/detail/equity",
            response_model=APIResponse[StockDetailPayload],
            summary="Lấy thông tin chi tiết cổ phiếu bao gồm quote và dữ liệu chart từ FMP")
async def get_stock_detail_http(
    symbol: str = Query(..., description="Mã cổ phiếu (ví dụ: AAPL)"),
    timeframe: str = Query(..., description="Khung thời gian cho chart (ví dụ: 5min, 1hour). Phải khớp với FMP hỗ trợ."),
    from_date: Optional[date] = Query(None, alias="from", description="Ngày bắt đầu cho dữ liệu lịch sử (YYYY-MM-DD)"),
    to_date: Optional[date] = Query(None, alias="to", description="Ngày kết thúc cho dữ liệu lịch sử (YYYY-MM-DD)"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    upper_symbol = symbol.upper()
    
    # FIX: Chỉ convert sang string nếu có giá trị
    # Nếu None, để FMP tự động lấy dữ liệu ngày gần nhất
    from_date_str = from_date.isoformat() if from_date is not None else None
    to_date_str = to_date.isoformat() if to_date is not None else None
    
    # ✅ Cache key xử lý None: dùng "auto" khi FMP tự động chọn ngày
    from_key = from_date_str if from_date_str else "auto"
    to_key = to_date_str if to_date_str else "auto"
    cache_key = f"equity_detail_{upper_symbol}_{timeframe}_{from_key}_{to_key}"
    logger.debug(f"Equity Detail: Symbol '{upper_symbol}', Timeframe '{timeframe}', From '{from_date_str or 'auto'}', To '{to_date_str or 'auto'}'. Cache key: {cache_key}")

    cached_data: Optional[APIResponse[StockDetailPayload]] = await get_cache(
        redis_client, cache_key, APIResponse[StockDetailPayload]
    )
    if cached_data and cached_data.data:
        cached_data.message = "OK (cached)"
        cached_data.provider_used = "fmp_cached"
        return cached_data
    logger.info(f"Equity Detail: Cache MISS for {cache_key}. Fetching new data.")

    actual_provider_used = "fmp"

    valid_fmp_timeframes = ["1min", "5min", "15min", "30min", "1hour", "4hour", "daily"]
    if timeframe not in valid_fmp_timeframes:
        logger.warning(f"Equity Detail: Timeframe '{timeframe}' might not be directly supported by FMP /v3/historical-chart/. Proceeding with call.")

    async with httpx.AsyncClient() as client:
        stock_detail_data: Optional[StockDetailPayload] = await equity_detail_service.get_equity_detail(
            symbol=upper_symbol,
            timeframe=timeframe,
            from_date_str=from_date_str,
            to_date_str=to_date_str,
            client=client
        )

    if not stock_detail_data:
        logger.warning(f"Equity Detail: No data found for {upper_symbol}, timeframe {timeframe}, from {from_date_str} to {to_date_str} using {actual_provider_used}.")
        raise HTTPException(status_code=404, detail=f"Không thể lấy dữ liệu chi tiết cho mã {upper_symbol} với các tham số đã cho từ {actual_provider_used}.")

    response_data_payload = APIResponseData[StockDetailPayload](data=[stock_detail_data])
    api_response = APIResponse[StockDetailPayload](
        message="OK",
        status="200",
        provider_used=actual_provider_used,
        data=response_data_payload
    )
    await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_CHART)
    return api_response


@router.get("/detail/profile",
    response_model=APIResponse[CompanyProfile],
    summary="Lấy Company Profile từ FMP"
)
async def get_stock_profile_http(
    symbol: str = Query(..., description="Mã cổ phiếu (vd: AAPL)"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    upper_symbol = symbol.upper()
    cache_key = f"equity_profile_stable_{upper_symbol}"

    # 1) Cache
    cached = await get_cache(redis_client, cache_key, APIResponse[CompanyProfile])
    if cached and cached.data and cached.data.data:
        cached.message = "OK (cached)"
        cached.provider_used = "fmp_cached"
        return cached

    # 2) Call FMP Stable + parse Pydantic
    async with httpx.AsyncClient() as client:
        profiles = await equity_detail_service.get_equity_profile(
            symbol=upper_symbol, client=client
        )

    if not profiles:
        logger.warning(f"Stable Profile: No data for {upper_symbol}")
        raise HTTPException(
            status_code=404,
            detail=f"Không thể lấy Stable Profile cho {upper_symbol} từ FMP."
        )

    # 3) Build APIResponse (data: List[CompanyProfile])
    payload = APIResponseData[CompanyProfile](data=profiles)
    resp = APIResponse[CompanyProfile](
        message="OK",
        status="200",
        provider_used="fmp_stable",
        totalRows=len(profiles),
        data=payload
    )

    # 4) Cache
    await set_cache(redis_client, cache_key, resp, expiry=settings.CACHE_TTL_CHART)
    return resp

#region --- GET LOGO ---
@router.get("/logo",
            response_model=APIResponse[LogoData],
            summary="Lấy logo cho một mã cổ phiếu cụ thể từ FMP")
async def get_logo_http(
    symbol: str = Query(..., description="Mã cổ phiếu (ví dụ: AAPL)"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    upper_symbol = symbol.upper()
    cache_key = f"logo_{upper_symbol}"
    logger.debug(f"Logo: Symbol '{upper_symbol}'. Cache key: {cache_key}")

    cached_data: Optional[APIResponse[LogoData]] = await get_cache(
        redis_client, cache_key, APIResponse[LogoData]
    )
    if cached_data and cached_data.data and cached_data.data.data: 
        logger.info(f"Logo: Cache HIT for {cache_key}")
        cached_data.message = "OK (cached)"
        cached_data.provider_used = "fmp_cached"
        return cached_data
    logger.info(f"Logo: Cache MISS for {cache_key}. Fetching new data.")


    actual_provider_used = "fmp"
    logo_url: Optional[str] = await equity_service.get_logo(symbol=upper_symbol)

    if not logo_url:
        logger.warning(f"Logo: No logo found for {upper_symbol} from {actual_provider_used}.")
        raise HTTPException(status_code=404, detail=f"Không tìm thấy logo cho mã {upper_symbol} từ {actual_provider_used}.")

    logo_data_item = LogoData(logo_url=logo_url)
    response_data_payload = APIResponseData[LogoData](data=[logo_data_item])
    api_response = APIResponse[LogoData](
        message="OK",
        status="200",
        provider_used=actual_provider_used,
        data=response_data_payload
    )
    await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_LISTS)
    logger.info(f"Logo: Cached new logo URL for {cache_key}")
    return api_response

#region --- Heatmap ---
@router.get("/heatmap",
            response_model=APIResponse[SP500QuoteWithSectorItem],
            summary="Lấy danh sách S&P 500 constituents kèm theo dữ liệu quote chi tiết")
async def get_sp500_quotes_with_sector_http(
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
        logger.error("FMP API key không được cấu hình cho S&P 500 quotes.")
        error_data_payload = APIResponseData[SP500QuoteWithSectorItem](data=[])
        return APIResponse[SP500QuoteWithSectorItem](
            message="FMP API key is not configured on the server.",
            status="500",
            provider_used="server_config_error",
            data=error_data_payload
        )

    cache_key = "sp500_quotes_with_sector_v1"

    if redis_client:
        try:
            cached_data: Optional[List[SP500QuoteWithSectorItem]] = await get_list_cache(
                redis_client, cache_key, SP500QuoteWithSectorItem
            )
            if cached_data is not None: 
                logger.info(f"Cache HIT cho S&P 500 quotes: {cache_key}")
                response_data_payload = APIResponseData[SP500QuoteWithSectorItem](data=cached_data)
                return APIResponse[SP500QuoteWithSectorItem](
                    message="OK (dữ liệu S&P 500 quotes lấy từ cache)",
                    status="200",
                    provider_used="cached_fmp",
                    data=response_data_payload
                )
        except Exception as e_cache_get:
            logger.error(f"Lỗi Redis GET cho S&P 500 quotes (key: {cache_key}): {e_cache_get}", exc_info=True)

    logger.info(f"Cache MISS cho S&P 500 quotes: {cache_key}. Đang lấy từ FMP.")

    sp500_service_instance = SP500Service()
    data_list: Optional[List[SP500QuoteWithSectorItem]] = await sp500_service_instance.get_sp500_constituents_with_quotes()

    actual_provider_used = "fmp_direct"

    if data_list is None: 
        logger.error("Service không thể lấy được dữ liệu S&P 500 quotes.")
        error_data_payload = APIResponseData[SP500QuoteWithSectorItem](data=[])
        return APIResponse[SP500QuoteWithSectorItem](
            message="Không thể lấy dữ liệu S&P 500 quotes từ nhà cung cấp.",
            status="502",
            provider_used=actual_provider_used + "_error",
            data=error_data_payload
        )
    response_data_payload = APIResponseData[SP500QuoteWithSectorItem](data=data_list)
    message_to_return = "OK"
    if not data_list:
        message_to_return = "Không tìm thấy dữ liệu S&P 500 quotes hoặc không có constituents nào."
        logger.info(message_to_return)

    api_response_obj = APIResponse[SP500QuoteWithSectorItem](
        message=message_to_return,
        status="200",
        provider_used=actual_provider_used,
        data=response_data_payload
    )

    if redis_client and data_list: 
        try:
            await set_list_cache(
                redis_client,
                cache_key,
                data_list,
                expiry=settings.CACHE_TTL_LONG 
            )
            logger.info(f"Đã cache dữ liệu S&P 500 quotes cho key: {cache_key}")
        except Exception as e_cache_set:
            logger.error(f"Lỗi Redis SET cho S&P 500 quotes (key: {cache_key}): {e_cache_set}", exc_info=True)

    return api_response_obj

#region --- Financial Report ---
@router.get("/financials/{symbol}",
            response_model=APIResponse[FinancialStatementsData],
            summary="Lấy báo cáo tài chính (Income, Balance, CashFlow - Annual & Quarterly) cho một mã cổ phiếu")
async def get_all_financial_statements_http(
    symbol: str,
    annual_limit: int = Query(5, ge=1, le=100, description="Số năm báo cáo thường niên muốn lấy"),
    quarterly_limit: int = Query(4, ge=1, le=40, description="Số quý báo cáo hàng quý muốn lấy"),
    period_type_for_cache: str = Query("annual", enum=["annual", "quarterly"], description="Loại kỳ báo cáo chính để tạo cache key"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
        logger.error(f"FMP API key không được cấu hình cho financials của {symbol}.")
        error_data_payload = APIResponseData[FinancialStatementsData](data=[])
        return APIResponse[FinancialStatementsData](
            message="FMP API key is not configured on the server.",
            status="500",
            provider_used="server_config_error",
            data=error_data_payload
        )
    cache_key_base = f"financial_statements_{symbol.upper()}"
    cache_key = f"{cache_key_base}_{period_type_for_cache}_limits_a{annual_limit}q{quarterly_limit}"


    if redis_client:
        try:
            cached_data_obj: Optional[FinancialStatementsData] = await get_cache(
                redis_client, cache_key, FinancialStatementsData
            )
            if cached_data_obj:
                logger.info(f"Cache HIT cho financial statements ({symbol}): {cache_key}")
                response_data_payload = APIResponseData[FinancialStatementsData](data=[cached_data_obj])
                return APIResponse[FinancialStatementsData](
                    message="OK (dữ liệu báo cáo tài chính lấy từ cache)",
                    status="200",
                    provider_used="cached_fmp",
                    data=response_data_payload
                )
        except Exception as e_cache_get:
            logger.error(f"Lỗi Redis GET cho financial statements ({symbol}, key: {cache_key}): {e_cache_get}", exc_info=True)

    logger.info(f"Cache MISS cho financial statements ({symbol}): {cache_key}. Đang lấy từ FMP.")
 
    all_statements_data: Optional[FinancialStatementsData] = await fs_service_instance.get_all_financial_statements(
        symbol=symbol,
        annual_limit=annual_limit,
        quarterly_limit=quarterly_limit
    )

    actual_provider_used = "fmp_direct"

    if all_statements_data is None:
        logger.error(f"Service không thể lấy được dữ liệu báo cáo tài chính cho {symbol}.")
        error_data_payload = APIResponseData[FinancialStatementsData](data=[])
        raise HTTPException(
            status_code=502,
            detail=f"Không thể lấy dữ liệu báo cáo tài chính từ nhà cung cấp cho {symbol}."
        )
    response_data_payload = APIResponseData[FinancialStatementsData](data=[all_statements_data])
    message_to_return = "OK"
    if not (all_statements_data.income_statements_annual or
            all_statements_data.income_statements_quarterly or
            all_statements_data.balance_sheets_annual or
            all_statements_data.balance_sheets_quarterly or
            all_statements_data.cash_flow_statements_annual or
            all_statements_data.cash_flow_statements_quarterly):
        message_to_return = f"Không tìm thấy dữ liệu báo cáo tài chính nào cho {symbol} với các giới hạn đã cho."
        logger.info(message_to_return)


    api_response_obj = APIResponse[FinancialStatementsData](
        message=message_to_return,
        status="200",
        provider_used=actual_provider_used,
        data=response_data_payload
    )

    if redis_client and all_statements_data:
        try:
            cache_ttl = settings.CACHE_TTL_FINANCIALS_ANNUAL if period_type_for_cache == "annual" else settings.CACHE_TTL_FINANCIALS_QUARTERLY
            if cache_ttl is None: 
                cache_ttl = settings.CACHE_TTL_LONG 
            await set_cache(
                redis_client,
                cache_key,
                all_statements_data, 
                expiry=int(cache_ttl)
            )
            logger.info(f"Đã cache dữ liệu báo cáo tài chính cho {symbol} (key: {cache_key}) với TTL {cache_ttl}s")
        except Exception as e_cache_set:
            logger.error(f"Lỗi Redis SET cho financial statements ({symbol}, key: {cache_key}): {e_cache_set}", exc_info=True)

    return api_response_obj

#region --- Tool call ---
@router.get("/tool-call/sentiment/{symbol}",
            response_model=APIResponse[SocialSentimentItem],
            summary="Lấy dữ liệu cảm xúc xã hội lịch sử từ FMP cho một mã cổ phiếu")
async def get_historical_social_sentiment_http(
    symbol: str,
    page: int = Query(0, ge=0, description="Số trang để lấy (bắt đầu từ 0)"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
        logger.error(f"FMP API key không được cấu hình cho historical social sentiment của {symbol}.")
        error_data_payload = APIResponseData[SocialSentimentItem](data=[])
        return APIResponse[SocialSentimentItem](
            message="FMP API key is not configured on the server.",
            status="500",
            provider_used="server_config_error",
            data=error_data_payload
        )
    cache_key = f"social_sentiment_{symbol.upper()}_page_{page}"

    if redis_client:
        try:
            cached_api_response: Optional[APIResponse[SocialSentimentItem]] = await get_cache(
                redis_client, cache_key, APIResponse[SocialSentimentItem]
            )
            if cached_api_response and cached_api_response.data and cached_api_response.data.data:
                logger.info(f"Cache HIT cho historical social sentiment ({symbol}, page {page}): {cache_key}")
                cached_api_response.message = "OK (cached)"
                cached_api_response.provider_used = "cached_fmp"
                return cached_api_response
        except Exception as e_cache_get:
            logger.error(f"Lỗi Redis GET cho historical social sentiment ({symbol}, key: {cache_key}): {e_cache_get}", exc_info=True)

    logger.info(f"Cache MISS cho historical social sentiment ({symbol}, page {page}): {cache_key}. Đang lấy từ FMP.")
    sentiment_data_list: Optional[List[SocialSentimentItem]] = await news_service.get_historical_social_sentiment(
        symbol=symbol, page=page
    )

    actual_provider_used = "fmp_direct"

    if sentiment_data_list is None:
        logger.error(f"Service không thể lấy được dữ liệu historical social sentiment cho {symbol}, page {page}.")
        error_data_payload = APIResponseData[SocialSentimentItem](data=[])
        return APIResponse[SocialSentimentItem](
            message=f"Không thể lấy dữ liệu cảm xúc xã hội lịch sử từ nhà cung cấp cho {symbol}, page {page}.",
            status="502",
            provider_used=actual_provider_used + "_error",
            data=error_data_payload
        )
    response_data_payload = APIResponseData[SocialSentimentItem](data=sentiment_data_list)
    message_to_return = "OK"
    if not sentiment_data_list:
        message_to_return = f"Không tìm thấy dữ liệu cảm xúc xã hội lịch sử nào cho {symbol}, page {page}."
        logger.info(message_to_return)

    api_response_obj = APIResponse[SocialSentimentItem](
        message=message_to_return,
        status="200",
        provider_used=actual_provider_used,
        data=response_data_payload
    )

    if redis_client and sentiment_data_list:
        try:
            await set_cache(
                redis_client,
                cache_key,
                api_response_obj,
                expiry=int(settings.CACHE_TTL_NEWS) 
            )
            logger.info(f"Đã gọi set_cache cho historical social sentiment ({symbol}, page {page}, key: {cache_key})")
        except Exception as e_cache_set:
            logger.error(f"Lỗi Redis SET cho historical social sentiment ({symbol}, key: {cache_key}): {e_cache_set}", exc_info=True)

    return api_response_obj

@router.get("/tool-call/fundamental/{symbol}",
            response_model=APIResponse[FinancialStatementGrowthItem],
            summary="Lấy dữ liệu tăng trưởng báo cáo tài chính từ FMP")
async def get_financial_statement_growth_http(
    symbol: str,
    period: str = Query("annual", enum=["annual", "quarter"], description="Kỳ báo cáo (annual hoặc quarter)"),
    limit: int = Query(10, ge=1, le=100, description="Số kỳ báo cáo muốn lấy (FMP thường giới hạn số kỳ có sẵn)"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
        logger.error(f"FMP API key không được cấu hình cho financial statement growth của {symbol}.")
        error_data_payload = APIResponseData[FinancialStatementGrowthItem](data=[])
        return APIResponse[FinancialStatementGrowthItem](
            message="FMP API key is not configured on the server.",
            status="500",
            provider_used="server_config_error",
            data=error_data_payload
        )

    cache_key = f"financial_growth_{symbol.upper()}_{period}_limit_{limit}"

    if redis_client:
        try:
            cached_api_response: Optional[APIResponse[FinancialStatementGrowthItem]] = await get_cache(
                redis_client, cache_key, APIResponse[FinancialStatementGrowthItem]
            )
            if cached_api_response and cached_api_response.data and cached_api_response.data.data is not None: 
                logger.info(f"Cache HIT cho financial statement growth ({symbol}, {period}): {cache_key}")
                cached_api_response.message = "OK (cached)"
                cached_api_response.provider_used = "cached_fmp"
                return cached_api_response
        except Exception as e_cache_get:
            logger.error(f"Lỗi Redis GET cho financial statement growth ({symbol}, key: {cache_key}): {e_cache_get}", exc_info=True)

    logger.info(f"Cache MISS cho financial statement growth ({symbol}, {period}): {cache_key}. Đang lấy từ FMP.")
    growth_data_list: Optional[List[FinancialStatementGrowthItem]] = await tool_call_service.get_financial_statement_growth(
        symbol=symbol, period=period, limit=limit
    )

    actual_provider_used = "fmp_direct"

    if growth_data_list is None:
        logger.error(f"Service không thể lấy được dữ liệu financial statement growth cho {symbol}, {period}.")
        error_data_payload = APIResponseData[FinancialStatementGrowthItem](data=[])
        return APIResponse[FinancialStatementGrowthItem](
            message=f"Không thể lấy dữ liệu tăng trưởng BCTC từ nhà cung cấp cho {symbol}, {period}.",
            status="502",
            provider_used=actual_provider_used + "_error",
            data=error_data_payload
        )

    response_data_payload = APIResponseData[FinancialStatementGrowthItem](data=growth_data_list)
    message_to_return = "OK"
    if not growth_data_list:
        message_to_return = f"Không tìm thấy dữ liệu tăng trưởng BCTC nào cho {symbol}, {period} với các tham số đã cho."
        logger.info(message_to_return)

    api_response_obj = APIResponse[FinancialStatementGrowthItem](
        message=message_to_return,
        status="200",
        provider_used=actual_provider_used,
        data=response_data_payload
    )

    if redis_client and growth_data_list is not None: 
        try:
            cache_ttl = settings.CACHE_TTL_FINANCIALS_ANNUAL if period == "annual" else settings.CACHE_TTL_FINANCIALS_QUARTERLY
            if cache_ttl is None: cache_ttl = settings.CACHE_DEFAULT_TTL 

            await set_cache(
                redis_client,
                cache_key,
                api_response_obj, 
                expiry=int(cache_ttl)
            )
            logger.info(f"Đã gọi set_cache cho financial statement growth ({symbol}, {period}, key: {cache_key}) với TTL {cache_ttl}s")
        except Exception as e_cache_set:
            logger.error(f"Lỗi Redis SET cho financial statement growth ({symbol}, key: {cache_key}): {e_cache_set}", exc_info=True)

    return api_response_obj

@router.get("/tool_call/stats/{symbol}",
            response_model=APIResponse[KeyMetricsTTMItem], 
            summary="Lấy dữ liệu Key Metrics TTM (Trailing Twelve Months) từ FMP")
async def get_key_metrics_ttm_http(
    symbol: str,
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
        logger.error(f"FMP API key không được cấu hình cho key-metrics-ttm của {symbol}.")
        error_data_payload = APIResponseData[KeyMetricsTTMItem](data=[])
        return APIResponse[KeyMetricsTTMItem](
            message="FMP API key is not configured on the server.",
            status="500",
            provider_used="server_config_error",
            data=error_data_payload
        )

    cache_key = f"key_metrics_ttm_{symbol.upper()}"

    if redis_client:
        try:
            cached_api_response: Optional[APIResponse[KeyMetricsTTMItem]] = await get_cache(
                redis_client, cache_key, APIResponse[KeyMetricsTTMItem]
            )
            if cached_api_response and cached_api_response.data and cached_api_response.data.data is not None:
                logger.info(f"Cache HIT cho key-metrics-ttm ({symbol}): {cache_key}")
                cached_api_response.message = "OK (cached)"
                cached_api_response.provider_used = "cached_fmp"
                return cached_api_response
        except Exception as e_cache_get:
            logger.error(f"Lỗi Redis GET cho key-metrics-ttm ({symbol}, key: {cache_key}): {e_cache_get}", exc_info=True)

    logger.info(f"Cache MISS cho key-metrics-ttm ({symbol}): {cache_key}. Đang lấy từ FMP.")
    key_metrics_ttm_data: Optional[KeyMetricsTTMItem] = await tool_call_service.get_key_metrics_ttm(symbol=symbol)

    actual_provider_used = "fmp_direct"

    if key_metrics_ttm_data is None:
        logger.warning(f"Service không lấy được dữ liệu key-metrics-ttm cho {symbol}.")
        response_data_payload = APIResponseData[KeyMetricsTTMItem](data=[])
        message_to_return = f"Không tìm thấy dữ liệu Key Metrics TTM cho {symbol}."
        status_code = "404"
    else:
        response_data_payload = APIResponseData[KeyMetricsTTMItem](data=[key_metrics_ttm_data])
        message_to_return = "OK"
        status_code = "200"


    api_response_obj = APIResponse[KeyMetricsTTMItem](
        message=message_to_return,
        status=status_code,
        provider_used=actual_provider_used if status_code == "200" else actual_provider_used + "_error",
        data=response_data_payload
    )

    if redis_client and key_metrics_ttm_data: 
        try:
            cache_ttl = settings.CACHE_TTL_FINANCIALS_QUARTERLY 
            if cache_ttl is None: cache_ttl = settings.CACHE_DEFAULT_TTL

            await set_cache(
                redis_client,
                cache_key,
                api_response_obj, 
                expiry=int(cache_ttl)
            )
            logger.info(f"Đã gọi set_cache cho key-metrics-ttm ({symbol}, key: {cache_key}) với TTL {cache_ttl}s")
        except Exception as e_cache_set:
            logger.error(f"Lỗi Redis SET cho key-metrics-ttm ({symbol}, key: {cache_key}): {e_cache_set}", exc_info=True)

    return api_response_obj

@router.get("/tool_call/estimates/{symbol}",
            response_model=APIResponse[AnalystEstimateItem], 
            summary="Lấy dữ liệu ước tính của các nhà phân tích từ FMP")
async def get_analyst_estimates_http(
    symbol: str,
    period: Optional[str] = Query("quarter", enum=["quarter", "annual"], description="Lọc theo kỳ (quarter hoặc annual) - tùy thuộc hỗ trợ của FMP"),
    limit: Optional[int] = Query(1, ge=1, le=100, description="Giới hạn số lượng kết quả - tùy thuộc hỗ trợ của FMP"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
        logger.error(f"FMP API key không được cấu hình cho analyst estimates của {symbol}.")
        error_data_payload = APIResponseData[AnalystEstimateItem](data=[])
        return APIResponse[AnalystEstimateItem](
            message="FMP API key is not configured on the server.",
            status="500",
            provider_used="server_config_error",
            data=error_data_payload
        )

    cache_key_parts = ["analyst_estimates", symbol.upper()]
    if period:
        cache_key_parts.append(f"period_{period}")
    if limit:
        cache_key_parts.append(f"limit_{limit}")
    cache_key = "_".join(cache_key_parts)


    if redis_client:
        try:
            cached_api_response: Optional[APIResponse[AnalystEstimateItem]] = await get_cache(
                redis_client, cache_key, APIResponse[AnalystEstimateItem]
            )
            if cached_api_response and cached_api_response.data and cached_api_response.data.data is not None:
                logger.info(f"Cache HIT cho analyst estimates ({symbol}): {cache_key}")
                cached_api_response.message = "OK (cached)"
                cached_api_response.provider_used = "cached_fmp"
                return cached_api_response
        except Exception as e_cache_get:
            logger.error(f"Lỗi Redis GET cho analyst estimates ({symbol}, key: {cache_key}): {e_cache_get}", exc_info=True)

    logger.info(f"Cache MISS cho analyst estimates ({symbol}): {cache_key}. Đang lấy từ FMP.")
    estimate_data_list: Optional[List[AnalystEstimateItem]] = await tool_call_service.get_analyst_estimates(
        symbol=symbol, period=period, limit=limit
    )

    actual_provider_used = "fmp_direct"

    if estimate_data_list is None:
        logger.error(f"Service không thể lấy được dữ liệu analyst estimates cho {symbol}.")
        error_data_payload = APIResponseData[AnalystEstimateItem](data=[])
        return APIResponse[AnalystEstimateItem](
            message=f"Không thể lấy dữ liệu ước tính từ nhà cung cấp cho {symbol}.",
            status="502",
            provider_used=actual_provider_used + "_error",
            data=error_data_payload
        )

    response_data_payload = APIResponseData[AnalystEstimateItem](data=estimate_data_list)
    message_to_return = "OK"
    if not estimate_data_list:
        message_to_return = f"Không tìm thấy dữ liệu ước tính nào cho {symbol} với các tham số đã cho."
        logger.info(message_to_return)

    api_response_obj = APIResponse[AnalystEstimateItem](
        message=message_to_return,
        status="200",
        provider_used=actual_provider_used,
        data=response_data_payload
    )

    if redis_client and estimate_data_list is not None: 
        try:
            cache_ttl = settings.CACHE_TTL_NEWS 
            if cache_ttl is None: cache_ttl = settings.CACHE_DEFAULT_TTL

            await set_cache(
                redis_client,
                cache_key,
                api_response_obj,
                expiry=int(cache_ttl)
            )
            logger.info(f"Đã gọi set_cache cho analyst estimates ({symbol}, key: {cache_key}) với TTL {cache_ttl}s")
        except Exception as e_cache_set:
            logger.error(f"Lỗi Redis SET cho analyst estimates ({symbol}, key: {cache_key}): {e_cache_set}", exc_info=True)

    return api_response_obj

#region Profile 
@router.get(
    "/description/{symbol}",
    response_model=APIResponse[CompanyDescription],
    summary="Lấy mô tả chi tiết của một công ty"
)
async def get_company_description_http(
    symbol: str,
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Endpoint để lấy mô tả (description) của một công ty dựa trên mã cổ phiếu.
    Kết quả sẽ được cache lại để tăng tốc độ cho các lần gọi sau.
    """
    cache_key = f"company_description_{symbol.upper()}"

    if redis_client:
        try:
            cached_response: Optional[APIResponse[CompanyDescription]] = await get_cache(
                redis_client, cache_key, APIResponse[CompanyDescription]
            )
            if cached_response and cached_response.data and cached_response.data.data:
                logger.info(f"Cache HIT cho description của {symbol}: {cache_key}")
                cached_response.message = "OK (cached)"
                cached_response.provider_used = "cached_fmp"
                return cached_response
        except Exception as e_cache_get:
            logger.error(f"Lỗi Redis GET cho description của {symbol} (key: {cache_key}): {e_cache_get}", exc_info=True)

    logger.info(f"Cache MISS cho description của {symbol}: {cache_key}. Đang lấy từ FMP.")

    # Khởi tạo hoặc sử dụng instance service đã có
    profile_service_instance = ProfileService()
    description_text = await profile_service_instance.get_company_description(symbol)

    if description_text is None:
        logger.warning(f"Không tìm thấy mô tả cho {symbol}.")
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy mô tả cho mã cổ phiếu: {symbol}"
        )

    # Tạo đối tượng Pydantic để trả về
    description_data = CompanyDescription(symbol=symbol.upper(), description=description_text)
    response_data_payload = APIResponseData[CompanyDescription](data=[description_data])
    api_response_obj = APIResponse[CompanyDescription](
        message="OK",
        status="200",
        provider_used="fmp_direct",
        data=response_data_payload
    )

    if redis_client:
        try:
            cache_ttl = 86400 * 7 # 7 ngày
            await set_cache(
                redis_client,
                cache_key,
                api_response_obj,
                expiry=int(cache_ttl)
            )
            logger.info(f"Đã cache description cho {symbol} (key: {cache_key}) với TTL {cache_ttl}s")
        except Exception as e_cache_set:
            logger.error(f"Lỗi Redis SET cho description của {symbol} (key: {cache_key}): {e_cache_set}", exc_info=True)

    return api_response_obj
    
#region Key Stat
@router.get("/key-stats",
            response_model=APIResponse[KeyStatsOutput],
            summary="Tính toán các chỉ số thống kê quan trọng về hiệu suất")
async def get_key_stats_http(
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    # Khởi tạo các service cần thiết
    news_service_instance = NewsService()
    analytics_service = AnalyticsService(news_service=news_service_instance, redis_client=redis_client)
    
    try:
        key_stats_data = await analytics_service.get_key_stats()
    finally:
        await analytics_service.close()

    if not key_stats_data:
        raise HTTPException(status_code=503, detail="Không thể tính toán các chỉ số thống kê vào lúc này.")

    response_data_payload = APIResponseData[KeyStatsOutput](data=[key_stats_data])
    return APIResponse[KeyStatsOutput](
        message="OK",
        status="200",
        provider_used="fmp_analytics_service",
        data=response_data_payload
    )

#region Target Price
@router.get("/target-price",
            response_model=APIResponse[PriceTargetWithChartOutput],
            summary="Lấy các mục tiêu giá mới nhất kèm theo dữ liệu biểu đồ")
async def get_price_targets_with_chart_http(
    page: int = Query(1, ge=1, description="Số trang để lấy (bắt đầu từ 1)."),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    cache_key = f"price_targets_page_{page}"
    if redis_client:
        cached_data: Optional[List[PriceTargetWithChartOutput]] = await get_list_cache(
            redis_client, cache_key, PriceTargetWithChartOutput
        )
        if cached_data is not None:
            logger.info(f"Cache HIT for price targets with chart: {cache_key}")
            response_data_payload = APIResponseData[PriceTargetWithChartOutput](data=cached_data)
            return APIResponse[PriceTargetWithChartOutput](
                message="OK (cached)",
                provider_used="cached_fmp",
                data=response_data_payload
            )

    logger.info(f"Cache MISS for price targets with chart: {cache_key}. Fetching new data.")

    data_list = await price_target_service.get_price_targets_with_charts(page=page, redis_client=redis_client)

    if data_list is None:
        raise HTTPException(
            status_code=502,
            detail="Không thể lấy dữ liệu mục tiêu giá từ nhà cung cấp."
        )

    if redis_client and data_list:
        await set_list_cache(redis_client, cache_key, data_list, expiry=settings.CACHE_TTL_NEWS)
        logger.info(f"Cached new price targets data for {cache_key}")

    response_data_payload = APIResponseData[PriceTargetWithChartOutput](data=data_list)
    return APIResponse[PriceTargetWithChartOutput](
        message="OK",
        status="200",
        provider_used="fmp_direct",
        data=response_data_payload
    )

#region Senate Trading
@router.get("/senate-trading",
            response_model=APIResponse[SenateTradingItem],
            summary="Lấy các giao dịch mới nhất của Thượng viện Mỹ")
async def get_latest_senate_trades_http(
    pageNumber: int = Query(0, ge=0, description="Số trang để lấy (bắt đầu từ 0)."),
    pageSize: int = Query(100, ge=1, le=1000, description="Số lượng kết quả mỗi trang."),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    cache_key = f"senate_trading_latest_p{pageNumber}_l{pageSize}"
    if redis_client:
        cached_data: Optional[List[SenateTradingItem]] = await get_list_cache(
            redis_client, cache_key, SenateTradingItem
        )
        if cached_data is not None:
            logger.info(f"Cache HIT for latest senate trades: {cache_key}")
            response_data_payload = APIResponseData[SenateTradingItem](data=cached_data)
            return APIResponse[SenateTradingItem](
                message="OK (cached)",
                provider_used="cached_fmp",
                data=response_data_payload
            )

    logger.info(f"Cache MISS for latest senate trades: {cache_key}. Fetching new data.")

    data_list = await senate_trading_service.get_latest_senate_trades(page=pageNumber, limit=pageSize)

    if data_list is None:
        raise HTTPException(
            status_code=502,
            detail="Không thể lấy dữ liệu giao dịch Thượng viện từ nhà cung cấp."
        )

    if redis_client and data_list:
        await set_list_cache(redis_client, cache_key, data_list, expiry=settings.CACHE_TTL_NEWS)
        logger.info(f"Cached new senate trades data for {cache_key}")

    response_data_payload = APIResponseData[SenateTradingItem](data=data_list)
    return APIResponse[SenateTradingItem](
        message="OK",
        status="200",
        provider_used="fmp_direct",
        data=response_data_payload
    )

@router.get("/top-politicians",
            response_model=APIResponse[SenateTradingItem],
            summary="Lấy danh sách giao dịch theo tên của Thượng nghị sĩ")
async def get_senate_trades_by_name_http(
    name: str = Query(..., description="Tên của Thượng nghị sĩ cần tra cứu, ví dụ: 'Jerry Moran' hoặc 'Sheldon Whitehouse'"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Lấy danh sách tất cả các giao dịch được báo cáo bởi một Thượng nghị sĩ cụ thể.
    """
    normalized_name = name.lower().replace(" ", "_")
    cache_key = f"senate_trades_by_name_{normalized_name}"
    
    if redis_client:
        cached_data: Optional[List[SenateTradingItem]] = await get_list_cache(
            redis_client, cache_key, SenateTradingItem
        )
        if cached_data is not None:
            logger.info(f"Cache HIT for senate trades by name: {cache_key}")
            return APIResponse[SenateTradingItem](
                message="OK (cached)",
                provider_used="cached_fmp",
                data=APIResponseData[SenateTradingItem](data=cached_data)
            )

    logger.info(f"Cache MISS for senate trades by name: {cache_key}. Fetching new data.")

    data_list = await senate_trading_service.get_trades_by_senator_name(name)

    if data_list is None:
        raise HTTPException(
            status_code=502, 
            detail=f"Không thể lấy dữ liệu giao dịch Thượng viện cho tên '{name}' từ nhà cung cấp."
        )

    if redis_client:
        await set_list_cache(redis_client, cache_key, data_list, expiry=86400)
        logger.info(f"Cached new senate trades data for {cache_key}")

    response_data_payload = APIResponseData[SenateTradingItem](data=data_list)
    return APIResponse[SenateTradingItem](
        message="OK",
        status="200",
        provider_used="fmp_direct",
        data=response_data_payload
    )
