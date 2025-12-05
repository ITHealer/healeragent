import json
from typing import Optional, Any, List, Type, get_args, TypeVar, AsyncGenerator
import logging 

from pydantic import BaseModel
import aioredis
from contextlib import asynccontextmanager
from src.models.equity import PaginatedData
from src.utils.config import settings
from src.utils.logger.set_up_log_dataFMP import setup_logger

logger = setup_logger(__name__, log_level=logging.INFO)
T = TypeVar("T", bound=BaseModel)

async def _make_redis_url(db_index: Optional[int] = None) -> str:
    host = settings.REDIS_HOST
    port = settings.REDIS_PORT
    pwd  = getattr(settings, "REDIS_PASSWORD", None)
    dbi  = settings.REDIS_DB if db_index is None else db_index
    if pwd:
        return f"redis://:{pwd}@{host}:{port}/{dbi}"
    return f"redis://{host}:{port}/{dbi}"

async def get_cache(redis_client: Optional[aioredis.Redis], cache_key: str, response_model: Type[T]) -> Optional[T]:
    """
    Hàm tiện ích chung để lấy dữ liệu (đơn lẻ) từ cache Redis.

    Args:
        redis_client: Instance của Redis client.
        cache_key: Key để lấy dữ liệu.
        response_model: Pydantic model của dữ liệu mong đợi.

    Returns:
        Một instance của response_model hoặc None nếu không có trong cache hoặc lỗi.
    """
    if redis_client:
        try:
            cached_data_bytes = await redis_client.get(cache_key)
            if cached_data_bytes:
                logger.info(f"Cache HIT cho key: {cache_key}")
                cached_json_string = cached_data_bytes.decode('utf-8')
                return response_model.model_validate_json(cached_json_string)
        except aioredis.RedisError as re:
            logger.error(f"Redis GET error cho key {cache_key}: {re}", exc_info=True)
        except json.JSONDecodeError as jde:
            logger.error(f"Lỗi JSONDecodeError khi parse cache cho key {cache_key}: {jde}. Data: {cached_data_bytes[:200] if cached_data_bytes else 'N/A'}", exc_info=False)
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi get cache cho key {cache_key}: {e}", exc_info=True)
    else:
        logger.debug(f"Redis client không có sẵn, bỏ qua get_cache cho key: {cache_key}")
    return None

async def get_list_cache(redis_client: Optional[aioredis.Redis], cache_key: str, item_response_model: Type[T]) -> Optional[List[T]]:
    """
    Hàm tiện ích chung để lấy danh sách dữ liệu từ cache Redis.

    Args:
        redis_client: Instance của Redis client.
        cache_key: Key để lấy dữ liệu.
        item_response_model: Pydantic model của các item trong danh sách.

    Returns:
        Một danh sách các instance của item_response_model hoặc None nếu không có trong cache hoặc lỗi.
    """
    if redis_client:
        try:
            cached_data_bytes = await redis_client.get(cache_key)
            if cached_data_bytes:
                logger.info(f"Cache HIT cho list key: {cache_key}")
                cached_json_string = cached_data_bytes.decode('utf-8')
                list_of_dicts = json.loads(cached_json_string)
                if isinstance(list_of_dicts, list):
                    return [item_response_model.model_validate(item_dict) for item_dict in list_of_dicts]
                else:
                    logger.error(f"Dữ liệu cache cho list key {cache_key} không phải là list sau khi parse JSON: {type(list_of_dicts)}")
                    return None
        except aioredis.RedisError as re:
            logger.error(f"Redis GET error cho list key {cache_key}: {re}", exc_info=True)
        except json.JSONDecodeError as jde:
            logger.error(f"Lỗi JSONDecodeError khi parse cache cho list key {cache_key}: {jde}. Data: {cached_data_bytes[:200] if cached_data_bytes else 'N/A'}", exc_info=False)
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi get_list_cache cho key {cache_key}: {e}", exc_info=True)
    else:
        logger.debug(f"Redis client không có sẵn, bỏ qua get_list_cache cho key: {cache_key}")
    return None

async def set_cache(redis_client: Optional[aioredis.Redis], cache_key: str, data_to_cache: BaseModel, expiry: int):
    """
    Hàm tiện ích chung để lưu dữ liệu Pydantic model vào cache Redis.

    Args:
        redis_client: Instance của Redis client.
        cache_key: Key để lưu dữ liệu.
        data_to_cache: Dữ liệu Pydantic model cần lưu.
        expiry: Thời gian hết hạn (TTL) tính bằng giây.
    """
    if redis_client and data_to_cache:
        try:
            json_string_to_store = data_to_cache.model_dump_json()
            await redis_client.set(cache_key, json_string_to_store, ex=expiry)
            logger.info(f"Đã cache dữ liệu cho key {cache_key} với TTL {expiry}s (sử dụng model_dump_json)")
        except aioredis.RedisError as re:
            logger.error(f"Redis SET error cho key {cache_key}: {re}", exc_info=True)
            raise 
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi set_cache cho key {cache_key}: {e}", exc_info=True)
            raise 
    elif not redis_client:
        logger.warning(f"Redis client không có sẵn, không thể set_cache cho key: {cache_key}")
    elif not data_to_cache:
        logger.warning(f"Dữ liệu rỗng, không set_cache cho key: {cache_key}")


async def set_list_cache(redis_client: Optional[aioredis.Redis], cache_key: str, data_list_to_cache: List[BaseModel], expiry: int):
    """
    Hàm tiện ích chung để lưu danh sách các Pydantic model vào cache Redis.

    Args:
        redis_client: Instance của Redis client.
        cache_key: Key để lưu dữ liệu.
        data_list_to_cache: Danh sách các Pydantic model cần lưu.
        expiry: Thời gian hết hạn (TTL) tính bằng giây.
    """
    if redis_client and data_list_to_cache:
        try:
            list_of_dicts = [item.model_dump(mode='json') for item in data_list_to_cache]
            json_string_to_store = json.dumps(list_of_dicts)
            await redis_client.set(cache_key, json_string_to_store, ex=expiry)
            logger.info(f"Đã cache danh sách dữ liệu cho key {cache_key} với TTL {expiry}s")
        except aioredis.RedisError as re:
            logger.error(f"Redis SET error cho list key {cache_key}: {re}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi set_list_cache cho key {cache_key}: {e}", exc_info=True)
            raise
    elif not redis_client:
        logger.warning(f"Redis client không có sẵn, không thể set_list_cache cho key: {cache_key}")
    elif not data_list_to_cache:
        logger.info(f"Danh sách dữ liệu rỗng, không set_list_cache cho key: {cache_key}")

async def get_redis_client_llm() -> Optional[aioredis.Redis]:
    """Trả về một aioredis.Redis client hoặc None, sử dụng cho các mục đích không phải FastAPI dependency."""
    if not settings.REDIS_HOST:
        logger.warning("Redis host (LLM) không được cấu hình. Cache sẽ bị vô hiệu hóa.")
        return None
    try:
        redis_conn = await aioredis.from_url(await _make_redis_url())
        logger.info(f"Đã kết nối tới Redis (LLM) {settings.REDIS_HOST}:{settings.REDIS_PORT} db={settings.REDIS_DB}")
        return redis_conn
    except (aioredis.exceptions.ConnectionError, ConnectionRefusedError, TimeoutError) as e:
        logger.error(f"Không thể kết nối tới Redis (LLM): {e}. Cache sẽ bị vô hiệu hóa.")
        return None
    except Exception as e:
        logger.error(f"Không thể kết nối tới Redis (LLM): {e}. Cache sẽ bị vô hiệu hóa.")
        return None

# async def get_redis_client() -> Optional[aioredis.Redis]:
#     """
#     Dependency cho FastAPI. Chỉ dùng với `Depends()`.
#     Nó quản lý kết nối cho một request HTTP.
#     """
#     if not settings.REDIS_HOST:
#         logger.warning("Redis host not configured. Cache will be disabled for FastAPI request.")
#         yield None
#         return
        
#     redis_conn = None
#     try:
#         redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"
#         if getattr(settings, "REDIS_PASSWORD", None):
#             redis_url = f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}"
        
#         redis_conn = await aioredis.from_url(redis_url, db=15)
#         yield redis_conn
    
#     except Exception as e:
#         logger.error(f"Could not connect to Redis for FastAPI request: {e}")
#         yield None
#     finally:
#         if redis_conn:
#             await redis_conn.close()

# async def get_redis_client() -> AsyncGenerator[Optional[aioredis.Redis], None]:
#     """
#     Dependency cho FastAPI với Depends()
#     """
#     if not settings.REDIS_HOST:
#         logger.warning("Redis host not configured. Cache disabled.")
#         yield None
#         return
#     redis_conn = None
#     try:
#         redis_conn = await aioredis.from_url(
#             await _make_redis_url(),
#             encoding="utf-8",
#             decode_responses=False,
#             socket_connect_timeout=5,
#             max_connections=50
#         )
#         logger.debug("Redis connection established")
#         yield redis_conn
#     except Exception as e:
#         logger.error(f"Redis connection error: {e}", exc_info=True)
#         yield None
#     finally:
#         if redis_conn:
#             try:
#                 await redis_conn.close()
#                 logger.debug("Redis connection closed")
#             except Exception as e:
#                 logger.debug(f"Error closing Redis: {e}")

async def get_redis_client() -> AsyncGenerator[Optional[aioredis.Redis], None]:
    """
    Dependency cho FastAPI với Depends()
    """
    if not settings.REDIS_HOST:
        logger.warning("Redis host not configured. Cache disabled.")
        yield None
        return
    
    redis_conn = None
    
    # Try to establish connection
    try:
        redis_conn = await aioredis.from_url(
            await _make_redis_url(),
            encoding="utf-8",
            decode_responses=False,
            socket_connect_timeout=5,
            max_connections=50
        )
        logger.debug("Redis connection established")
    except Exception as e:
        logger.error(f"Redis connection error: {e}", exc_info=True)
        yield None
        return
    
    # Yield connection và cleanup
    try:
        yield redis_conn
    finally:
        # ENSURE CONNECTION IS CLOSED
        if redis_conn:
            try:
                await redis_conn.close()
                logger.debug("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}", exc_info=True)

@asynccontextmanager
async def get_redis_client_for_scheduler() -> Optional[aioredis.Redis]:
    """
    Context Manager cho các tác vụ nền (scheduler). 
    Dùng với `async with` để đảm bảo kết nối Redis được quản lý và đóng đúng cách.
    Cho phép chọn DB index riêng biệt (mặc định lấy từ settings.REDIS_DB).
    """
    if not settings.REDIS_HOST:
        logger.warning("Redis host not configured. Cache will be disabled for scheduler task.")
        yield None
        return

    redis_conn: Optional[aioredis.Redis] = None
    try:
        redis_url = await _make_redis_url()
        redis_conn = await aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=False,
            socket_connect_timeout=5,
            max_connections=50
        )
        logger.debug(f"Scheduler Redis connected (db={settings.REDIS_DB})")
        yield redis_conn
    except Exception as e:
        logger.error(f"Could not connect to Redis for scheduler task: {e}", exc_info=True)
        yield None
    finally:
        if redis_conn:
            try:
                await redis_conn.close()
                logger.debug("Scheduler Redis connection closed.")
            except Exception as e:
                logger.warning(f"Error closing Redis for scheduler task: {e}", exc_info=True)

async def set_paginated_cache(
    redis_client: Optional[aioredis.Redis],
    cache_key: str,
    paginated_data: PaginatedData, 
    expiry: int
):
    """Lưu dữ liệu có phân trang (bao gồm totalRows và data) vào Redis."""
    if redis_client and paginated_data:
        try:
            json_string_to_store = paginated_data.model_dump_json()
            await redis_client.set(cache_key, json_string_to_store, ex=expiry)
            # logger.info(f"Đã cache dữ liệu phân trang cho key {cache_key} với TTL {expiry}s")
        except Exception as e:
            logger.error(f"Lỗi Redis SET cho paginated key {cache_key}: {e}", exc_info=True)
            raise

async def get_paginated_cache(
    redis_client: Optional[aioredis.Redis],
    cache_key: str,
    response_model: Type[PaginatedData] 
) -> Optional[PaginatedData]:
    """Lấy dữ liệu có phân trang từ Redis."""
    if redis_client:
        try:
            cached_data_bytes = await redis_client.get(cache_key)
            if cached_data_bytes:
                logger.info(f"Cache HIT cho paginated key: {cache_key}")
                cached_json_string = cached_data_bytes.decode('utf-8')
                return response_model.model_validate_json(cached_json_string)
        except Exception as e:
            logger.error(f"Lỗi Redis GET cho paginated key {cache_key}: {e}", exc_info=True)
    return None
