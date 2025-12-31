import json
import asyncio
from typing import Optional, Any, List, Type, TypeVar, AsyncGenerator
import logging

from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.models.equity import PaginatedData
from src.utils.config import settings
from src.utils.logger.set_up_log_dataFMP import setup_logger

# Import aioredis/redis with BlockingConnectionPool support
# Works with both old aioredis and new redis-py (4.2+)
try:
    import aioredis
    # Try modern redis-py path first (aioredis was merged into redis-py)
    try:
        from aioredis.connection import BlockingConnectionPool
    except ImportError:
        try:
            from aioredis import BlockingConnectionPool
        except ImportError:
            # Fallback: redis-py asyncio module
            from redis.asyncio.connection import BlockingConnectionPool
    HAS_BLOCKING_POOL = True
except ImportError:
    # Last resort: use regular aioredis without BlockingConnectionPool
    import aioredis
    HAS_BLOCKING_POOL = False

logger = setup_logger(__name__, log_level=logging.INFO)
T = TypeVar("T", bound=BaseModel)

# Timeout settings
REDIS_CONNECT_TIMEOUT = 10  # seconds - was 5
REDIS_SOCKET_TIMEOUT = 10   # seconds
REDIS_CLOSE_TIMEOUT = 8     # seconds - for graceful close

# Connection pool settings - PRODUCTION READY
REDIS_MAX_CONNECTIONS = 100  # Increased for high traffic (was 50)
REDIS_HEALTH_CHECK_INTERVAL = 30  # seconds
REDIS_POOL_TIMEOUT = 20  # seconds - time to wait for available connection

# Retry settings
REDIS_RETRY_ATTEMPTS = 3
REDIS_RETRY_BASE_DELAY = 0.5  # seconds - exponential backoff base


# =============================================================================
# SINGLETON REDIS CLIENT FOR LLM OPERATIONS
# =============================================================================

_redis_llm_client: Optional[aioredis.Redis] = None
_redis_llm_lock: asyncio.Lock = asyncio.Lock()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def _make_redis_url(db_index: Optional[int] = None) -> str:
    """Build Redis URL from settings"""
    host = settings.REDIS_HOST
    port = settings.REDIS_PORT
    pwd = getattr(settings, "REDIS_PASSWORD", None)
    dbi = settings.REDIS_DB if db_index is None else db_index
    if pwd:
        return f"redis://:{pwd}@{host}:{port}/{dbi}"
    return f"redis://{host}:{port}/{dbi}"


async def _create_redis_connection(
    redis_url: str,
    decode_responses: bool = False
) -> Optional[aioredis.Redis]:
    """
    Create a Redis client with BlockingConnectionPool for production.

    BlockingConnectionPool WAITS for available connection instead of throwing
    ConnectionError when pool is exhausted. Critical for high-traffic production.

    Returns:
        Redis client or None if connection failed
    """
    try:
        if HAS_BLOCKING_POOL:
            # PRODUCTION MODE: Use BlockingConnectionPool
            # WAITS for connection instead of throwing error when pool exhausted
            # Critical for handling thousands of concurrent users
            pool = BlockingConnectionPool.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=decode_responses,
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
                socket_timeout=REDIS_SOCKET_TIMEOUT,
                max_connections=REDIS_MAX_CONNECTIONS,
                timeout=REDIS_POOL_TIMEOUT,  # Wait up to 20s for available connection
                retry_on_timeout=True,
                health_check_interval=REDIS_HEALTH_CHECK_INTERVAL
            )
            client = aioredis.Redis(connection_pool=pool)
            logger.info(f"Redis BlockingConnectionPool created (max={REDIS_MAX_CONNECTIONS}, timeout={REDIS_POOL_TIMEOUT}s)")
        else:
            # FALLBACK MODE: Regular ConnectionPool (will throw error if exhausted)
            logger.warning("BlockingConnectionPool not available, using regular pool (may throw ConnectionError under load)")
            client = await aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=decode_responses,
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
                socket_timeout=REDIS_SOCKET_TIMEOUT,
                max_connections=REDIS_MAX_CONNECTIONS,
                retry_on_timeout=True,
                health_check_interval=REDIS_HEALTH_CHECK_INTERVAL
            )
        return client
    except Exception as e:
        logger.error(f"Failed to create Redis connection: {e}")
        return None


async def _close_redis_safely(redis_client: Optional[aioredis.Redis]) -> None:
    """
    Safely close Redis connection with timeout to prevent hanging.
    
    This fixes the "Timed out closing connection after 5" error.
    """
    if redis_client is None:
        return
    
    try:
        # Use asyncio.wait_for to prevent indefinite hanging
        await asyncio.wait_for(
            redis_client.close(),
            timeout=REDIS_CLOSE_TIMEOUT
        )
        # Wait for connection pool to close
        await asyncio.wait_for(
            redis_client.connection_pool.disconnect(),
            timeout=REDIS_CLOSE_TIMEOUT
        )
        logger.debug("Redis connection closed successfully")
    except asyncio.TimeoutError:
        logger.warning("Redis close timed out, connection may leak")
    except asyncio.CancelledError:
        # Gracefully handle task cancellation
        logger.debug("Redis close task was cancelled")
    except Exception as e:
        logger.warning(f"Error closing Redis connection: {e}")


# =============================================================================
# CACHE GET/SET FUNCTIONS WITH RETRY LOGIC
# =============================================================================

async def get_cache(
    redis_client: Optional[aioredis.Redis], 
    cache_key: str, 
    response_model: Type[T]
) -> Optional[T]:
    """
    Get single data item from Redis cache with error handling.
    
    Args:
        redis_client: Redis client instance
        cache_key: Cache key to retrieve
        response_model: Pydantic model for response
        
    Returns:
        Parsed model instance or None
    """
    if redis_client is None:
        logger.debug(f"Redis client not available, cache MISS for: {cache_key}")
        return None
    
    try:
        # Use timeout wrapper for GET operation
        cached_data_bytes = await asyncio.wait_for(
            redis_client.get(cache_key),
            timeout=REDIS_SOCKET_TIMEOUT
        )
        
        if cached_data_bytes:
            logger.info(f"Cache HIT for key: {cache_key}")
            # Handle both bytes and str (depending on decode_responses setting)
            if isinstance(cached_data_bytes, bytes):
                cached_json_string = cached_data_bytes.decode('utf-8')
            else:
                cached_json_string = cached_data_bytes
            return response_model.model_validate_json(cached_json_string)
            
    except asyncio.TimeoutError:
        logger.error(f"Redis GET timeout for key: {cache_key}")
    except aioredis.RedisError as re:
        logger.error(f"Redis GET error for key {cache_key}: {re}")
    except json.JSONDecodeError as jde:
        logger.error(f"JSON decode error for cache key {cache_key}: {jde}")
    except Exception as e:
        logger.error(f"Unexpected error getting cache for key {cache_key}: {e}")
    
    return None


async def set_cache(
    redis_client: Optional[aioredis.Redis],
    cache_key: str,
    data_to_cache: BaseModel,
    expiry: int
) -> bool:
    """
    Set single data item in Redis cache.
    
    Args:
        redis_client: Redis client instance
        cache_key: Cache key
        data_to_cache: Pydantic model to cache
        expiry: TTL in seconds
        
    Returns:
        True if successful, False otherwise
    """
    if redis_client is None:
        logger.warning(f"Redis client not available, cannot cache key: {cache_key}")
        return False
    
    if data_to_cache is None:
        logger.warning(f"Empty data, not caching key: {cache_key}")
        return False
    
    try:
        json_string_to_store = data_to_cache.model_dump_json()
        await asyncio.wait_for(
            redis_client.set(cache_key, json_string_to_store, ex=expiry),
            timeout=REDIS_SOCKET_TIMEOUT
        )
        logger.info(f"Cached data for key {cache_key} with TTL {expiry}s")
        return True
        
    except asyncio.TimeoutError:
        logger.error(f"Redis SET timeout for key: {cache_key}")
    except aioredis.RedisError as re:
        logger.error(f"Redis SET error for key {cache_key}: {re}")
    except Exception as e:
        logger.error(f"Unexpected error setting cache for key {cache_key}: {e}")
    
    return False


async def get_list_cache(
    redis_client: Optional[aioredis.Redis], 
    cache_key: str, 
    item_response_model: Type[T]
) -> Optional[List[T]]:
    """
    Get list of items from Redis cache.
    
    Args:
        redis_client: Redis client instance
        cache_key: Cache key
        item_response_model: Pydantic model for each list item
        
    Returns:
        List of parsed models or None
    """
    if redis_client is None:
        logger.debug(f"Redis client not available, cache MISS for list key: {cache_key}")
        return None
    
    try:
        cached_data_bytes = await asyncio.wait_for(
            redis_client.get(cache_key),
            timeout=REDIS_SOCKET_TIMEOUT
        )
        
        if cached_data_bytes:
            logger.info(f"Cache HIT for list key: {cache_key}")
            # Handle both bytes and str (depending on decode_responses setting)
            if isinstance(cached_data_bytes, bytes):
                cached_json_string = cached_data_bytes.decode('utf-8')
            else:
                cached_json_string = cached_data_bytes
            list_of_dicts = json.loads(cached_json_string)
            return [item_response_model.model_validate(item) for item in list_of_dicts]
            
    except asyncio.TimeoutError:
        logger.error(f"Redis GET timeout for list key: {cache_key}")
    except aioredis.RedisError as re:
        logger.error(f"Redis GET error for list key {cache_key}: {re}")
    except json.JSONDecodeError as jde:
        logger.error(f"JSON decode error for list cache key {cache_key}: {jde}")
    except Exception as e:
        logger.error(f"Unexpected error getting list cache for key {cache_key}: {e}")
    
    return None


async def set_list_cache(
    redis_client: Optional[aioredis.Redis], 
    cache_key: str, 
    data_list_to_cache: List[BaseModel], 
    expiry: int
) -> bool:
    """
    Set list of items in Redis cache.
    
    Args:
        redis_client: Redis client instance  
        cache_key: Cache key
        data_list_to_cache: List of Pydantic models to cache
        expiry: TTL in seconds
        
    Returns:
        True if successful, False otherwise
    """
    if redis_client is None:
        logger.warning(f"Redis client not available, cannot cache list key: {cache_key}")
        return False
    
    if not data_list_to_cache:
        logger.info(f"Empty list, not caching key: {cache_key}")
        return False
    
    try:
        list_of_dicts = [item.model_dump(mode='json') for item in data_list_to_cache]
        json_string_to_store = json.dumps(list_of_dicts)
        
        await asyncio.wait_for(
            redis_client.set(cache_key, json_string_to_store, ex=expiry),
            timeout=REDIS_SOCKET_TIMEOUT
        )
        logger.info(f"Cached list data for key {cache_key} with TTL {expiry}s")
        return True
        
    except asyncio.TimeoutError:
        logger.error(f"Redis SET timeout for list key: {cache_key}")
    except aioredis.RedisError as re:
        logger.error(f"Redis SET error for list key {cache_key}: {re}")
    except Exception as e:
        logger.error(f"Unexpected error setting list cache for key {cache_key}: {e}")
    
    return False


# =============================================================================
# PAGINATED CACHE FUNCTIONS
# =============================================================================

async def get_paginated_cache(
    redis_client: Optional[aioredis.Redis],
    cache_key: str,
    response_model: Type[PaginatedData]
) -> Optional[PaginatedData]:
    """
    Get paginated data from Redis cache with improved error handling.
    
    Args:
        redis_client: Redis client instance
        cache_key: Cache key
        response_model: PaginatedData model type
        
    Returns:
        PaginatedData instance or None
    """
    if redis_client is None:
        logger.warning(f"Redis client not available, cache MISS for paginated key: {cache_key}")
        return None
    
    try:
        cached_data_bytes = await asyncio.wait_for(
            redis_client.get(cache_key),
            timeout=REDIS_SOCKET_TIMEOUT
        )
        
        if cached_data_bytes:
            logger.info(f"Cache HIT for paginated key: {cache_key}")
            # Handle both bytes and str (depending on decode_responses setting)
            if isinstance(cached_data_bytes, bytes):
                cached_json_string = cached_data_bytes.decode('utf-8')
            else:
                cached_json_string = cached_data_bytes
            return response_model.model_validate_json(cached_json_string)
            
    except asyncio.TimeoutError:
        logger.error(f"Redis GET timeout for paginated key: {cache_key}")
    except aioredis.ConnectionError as ce:
        logger.error(f"Redis connection error for paginated key {cache_key}: {ce}")
    except aioredis.RedisError as re:
        logger.error(f"Redis error for paginated key {cache_key}: {re}")
    except Exception as e:
        logger.error(f"Unexpected error for paginated key {cache_key}: {e}")
    
    return None


async def set_paginated_cache(
    redis_client: Optional[aioredis.Redis],
    cache_key: str,
    paginated_data: PaginatedData,
    expiry: int
) -> bool:
    """
    Set paginated data in Redis cache.
    
    Args:
        redis_client: Redis client instance
        cache_key: Cache key
        paginated_data: PaginatedData instance to cache
        expiry: TTL in seconds
        
    Returns:
        True if successful, False otherwise
    """
    if redis_client is None:
        logger.warning(f"Redis client not available, cannot cache paginated key: {cache_key}")
        return False
    
    if paginated_data is None:
        logger.warning(f"Empty paginated data, not caching key: {cache_key}")
        return False
    
    try:
        json_string_to_store = paginated_data.model_dump_json()
        await asyncio.wait_for(
            redis_client.set(cache_key, json_string_to_store, ex=expiry),
            timeout=REDIS_SOCKET_TIMEOUT
        )
        logger.info(f"Cached paginated data for key {cache_key} with TTL {expiry}s")
        return True
        
    except asyncio.TimeoutError:
        logger.error(f"Redis SET timeout for paginated key: {cache_key}")
    except aioredis.RedisError as re:
        logger.error(f"Redis SET error for paginated key {cache_key}: {re}")
    except Exception as e:
        logger.error(f"Unexpected error setting paginated cache for key {cache_key}: {e}")
    
    return False


# =============================================================================
# CONNECTION CONTEXT MANAGERS WITH RETRY LOGIC
# =============================================================================

@asynccontextmanager
async def get_redis_client_with_retry(
    max_retries: int = REDIS_RETRY_ATTEMPTS
) -> AsyncGenerator[Optional[aioredis.Redis], None]:
    """
    Context manager that gets Redis connection with retry logic.
    
    Usage:
        async with get_redis_client_with_retry() as redis:
            if redis:
                await redis.get("key")
    
    Args:
        max_retries: Maximum number of connection attempts
        
    Yields:
        Redis client or None if all attempts failed
    """
    if not settings.REDIS_HOST:
        logger.warning("Redis host not configured. Cache disabled.")
        yield None
        return
    
    redis_conn = None
    last_error = None
    
    for attempt in range(max_retries):
        try:
            redis_url = await _make_redis_url()
            redis_conn = await _create_redis_connection(redis_url, decode_responses=False)
            
            if redis_conn:
                # Test connection with PING
                await asyncio.wait_for(
                    redis_conn.ping(),
                    timeout=REDIS_CONNECT_TIMEOUT
                )
                logger.debug(f"Redis connected on attempt {attempt + 1}")
                break
                
        except (aioredis.ConnectionError, asyncio.TimeoutError, OSError) as e:
            last_error = e
            logger.warning(
                f"Redis connection attempt {attempt + 1}/{max_retries} failed: {e}"
            )
            
            if attempt < max_retries - 1:
                # Exponential backoff
                delay = REDIS_RETRY_BASE_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)
            else:
                logger.error(f"Failed to connect to Redis after {max_retries} attempts: {last_error}")
                redis_conn = None
    
    try:
        yield redis_conn
    finally:
        await _close_redis_safely(redis_conn)


async def get_redis_client() -> AsyncGenerator[Optional[aioredis.Redis], None]:
    """
    FastAPI Dependency for Redis connection with improved error handling.
    
    Usage with FastAPI:
        @app.get("/endpoint")
        async def endpoint(redis: Optional[aioredis.Redis] = Depends(get_redis_client)):
            ...
    
    Yields:
        Redis client or None
    """
    if not settings.REDIS_HOST:
        logger.warning("Redis host not configured. Cache disabled.")
        yield None
        return
    
    redis_conn = None
    
    try:
        redis_url = await _make_redis_url()
        redis_conn = await _create_redis_connection(redis_url, decode_responses=False)
        
        if redis_conn:
            # Verify connection is alive
            await asyncio.wait_for(
                redis_conn.ping(),
                timeout=REDIS_CONNECT_TIMEOUT
            )
            logger.debug("Redis connection established for request")
            
    except (aioredis.ConnectionError, asyncio.TimeoutError, OSError) as e:
        logger.error(f"Redis connection error: {e}")
        redis_conn = None
    except Exception as e:
        logger.error(f"Unexpected Redis error: {e}")
        redis_conn = None
    
    try:
        yield redis_conn
    finally:
        await _close_redis_safely(redis_conn)


@asynccontextmanager
async def get_redis_client_for_scheduler() -> AsyncGenerator[Optional[aioredis.Redis], None]:
    """
    Context Manager for background scheduler tasks with retry logic.
    
    Usage:
        async with get_redis_client_for_scheduler() as redis:
            if redis:
                await redis.set("key", "value")
    
    Yields:
        Redis client or None
    """
    if not settings.REDIS_HOST:
        logger.warning("Redis host not configured. Cache disabled for scheduler.")
        yield None
        return
    
    redis_conn = None
    
    try:
        redis_url = await _make_redis_url()
        redis_conn = await _create_redis_connection(redis_url, decode_responses=False)
        
        if redis_conn:
            await asyncio.wait_for(
                redis_conn.ping(),
                timeout=REDIS_CONNECT_TIMEOUT
            )
            logger.debug(f"Scheduler Redis connected (db={settings.REDIS_DB})")
            
    except Exception as e:
        logger.error(f"Could not connect to Redis for scheduler: {e}")
        redis_conn = None
    
    try:
        yield redis_conn
    finally:
        await _close_redis_safely(redis_conn)


async def get_redis_client_llm() -> Optional[aioredis.Redis]:
    """
    Get SINGLETON Redis client for LLM operations.

    Uses a shared connection to prevent connection leaks.
    DO NOT close this connection after use - it's shared across the application.

    Returns:
        Redis client singleton or None
    """
    global _redis_llm_client

    if not settings.REDIS_HOST:
        logger.warning("Redis host (LLM) not configured. Cache disabled.")
        return None

    # Fast path - already connected
    if _redis_llm_client is not None:
        try:
            # Quick health check
            await asyncio.wait_for(_redis_llm_client.ping(), timeout=2.0)
            return _redis_llm_client
        except Exception:
            # Connection lost, need to reconnect
            _redis_llm_client = None

    # Slow path - create new connection with lock
    async with _redis_llm_lock:
        # Double-check after acquiring lock
        if _redis_llm_client is not None:
            return _redis_llm_client

        try:
            redis_url = await _make_redis_url()
            redis_conn = await _create_redis_connection(redis_url, decode_responses=True)

            if redis_conn:
                await asyncio.wait_for(
                    redis_conn.ping(),
                    timeout=REDIS_CONNECT_TIMEOUT
                )
                logger.info(f"Connected to Redis (LLM) {settings.REDIS_HOST}:{settings.REDIS_PORT}")
                _redis_llm_client = redis_conn
                return _redis_llm_client

        except (aioredis.ConnectionError, ConnectionRefusedError, asyncio.TimeoutError) as e:
            logger.error(f"Cannot connect to Redis (LLM): {e}. Cache disabled.")
        except Exception as e:
            logger.error(f"Unexpected error connecting to Redis (LLM): {e}")

    return None


async def close_redis_llm_client() -> None:
    """
    Close the singleton Redis LLM client.
    Call this on application shutdown.
    """
    global _redis_llm_client

    async with _redis_llm_lock:
        if _redis_llm_client is not None:
            try:
                await asyncio.wait_for(
                    _redis_llm_client.close(),
                    timeout=REDIS_CLOSE_TIMEOUT
                )
                logger.info("Redis LLM client closed")
            except Exception as e:
                logger.warning(f"Error closing Redis LLM client: {e}")
            finally:
                _redis_llm_client = None


# =============================================================================
# REDIS CONNECTION MANAGER SINGLETON (Optional - for advanced use cases)
# =============================================================================

class RedisConnectionManager:
    """
    Singleton Redis connection manager for advanced lifecycle management.
    
    Usage:
        manager = RedisConnectionManager()
        client = await manager.get_client()
        # ... use client ...
        await manager.close()  # Call on application shutdown
    """
    _instance = None
    _client: Optional[aioredis.Redis] = None
    _lock: asyncio.Lock = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._lock = asyncio.Lock()
        return cls._instance
    
    async def get_client(self) -> Optional[aioredis.Redis]:
        """Get or create Redis client"""
        async with self._lock:
            if self._client is None:
                if not settings.REDIS_HOST:
                    return None
                    
                redis_url = await _make_redis_url()
                self._client = await _create_redis_connection(redis_url)
                
                if self._client:
                    try:
                        await self._client.ping()
                    except Exception:
                        self._client = None
            
            return self._client
    
    async def close(self) -> None:
        """Close Redis connection gracefully"""
        async with self._lock:
            if self._client:
                await _close_redis_safely(self._client)
                self._client = None
    
    async def health_check(self) -> bool:
        """Check if Redis connection is healthy"""
        try:
            client = await self.get_client()
            if client:
                await asyncio.wait_for(client.ping(), timeout=5.0)
                return True
        except Exception:
            pass
        return False


# Global instance (optional)
redis_manager = RedisConnectionManager()