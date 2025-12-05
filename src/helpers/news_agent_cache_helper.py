"""
News Agent Cache Helper
=======================
Module này cung cấp các utility functions để cache responses của News Agent APIs.

Chiến lược cache:
- Daily News: TTL 15 phút (tin tức chung thay đổi nhanh)
- Personalized News: TTL 30 phút (có user context, cache được lâu hơn)
- Symbol News: TTL 15 phút (tin cổ phiếu cập nhật liên tục)
- Sector Analysis: TTL 60 phút (phân tích ngành ổn định hơn)
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional, List
from datetime import datetime

import aioredis

logger = logging.getLogger(__name__)

# ========================
# Cache TTL Constants (seconds)
# ========================
ONE_DAY_IN_SECONDS = 24 * 60 * 60

CACHE_TTL_DAILY_NEWS = ONE_DAY_IN_SECONDS #15 * 60  # 15 minutes
CACHE_TTL_PERSONALIZED_NEWS = 30 * 60  # 30 minutes
CACHE_TTL_SYMBOL_NEWS = ONE_DAY_IN_SECONDS  # 15 minutes
CACHE_TTL_SECTOR_ANALYSIS = ONE_DAY_IN_SECONDS  # 60 minutes


# ========================
# Cache Key Generation
# ========================

def _generate_hash(*args) -> str:
    """
    Generate MD5 hash from arguments.
    Sử dụng để tạo cache key ngắn gọn và deterministic.
    """
    # Convert all args to string and join
    content = "|".join(str(arg) for arg in args if arg is not None)
    return hashlib.md5(content.encode()).hexdigest()[:12]


def generate_daily_news_cache_key(
    topics: Optional[str],
    max_searches: int,
    target_language: str,
    max_results_per_search: int,
    processing_method: str
) -> str:
    """
    Generate cache key for Daily News endpoint.
    
    Pattern: news_daily:{hash}
    """
    topics_normalized = (topics or "default").strip().lower()
    
    key_hash = _generate_hash(
        topics_normalized,
        max_searches,
        target_language,
        max_results_per_search,
        processing_method
    )
    
    return f"news_daily:{key_hash}"


def generate_personalized_news_cache_key(
    user_id: int,
    topics: Optional[str],
    additional_topics: Optional[str],
    max_searches: int,
    target_language: str,
    max_results_per_search: int,
    processing_method: str
) -> str:
    """
    Generate cache key for Personalized News endpoint.
    
    Pattern: news_personalized:{user_id}:{hash}
    """
    topics_normalized = (topics or "").strip().lower()
    additional_normalized = (additional_topics or "").strip().lower()
    
    key_hash = _generate_hash(
        topics_normalized,
        additional_normalized,
        max_searches,
        target_language,
        max_results_per_search,
        processing_method
    )
    
    return f"news_personalized:{user_id}:{key_hash}"


def generate_symbol_news_cache_key(
    user_id: int,
    symbols: Optional[List[str]],
    additional_symbols: Optional[List[str]],
    topics: Optional[str],
    max_searches: int,
    target_language: str,
    max_results_per_search: int,
    processing_method: str
) -> str:
    """
    Generate cache key for Symbol News endpoint.
    
    Pattern: news_symbol:{user_id}:{hash}
    """
    # Normalize symbols
    symbols_str = ",".join(sorted(symbols)) if symbols else ""
    additional_str = ",".join(sorted(additional_symbols)) if additional_symbols else ""
    topics_normalized = (topics or "").strip().lower()
    
    key_hash = _generate_hash(
        symbols_str,
        additional_str,
        topics_normalized,
        max_searches,
        target_language,
        max_results_per_search,
        processing_method
    )
    
    return f"news_symbol:{user_id}:{key_hash}"


def generate_sector_analysis_cache_key(
    sectors: List[str],
    analysis_depth: str,
    topics: Optional[str],
    target_language: str,
    user_id: Optional[int],
    max_results_per_search: int,
    processing_method: str
) -> str:
    """
    Generate cache key for Sector Analysis endpoint.
    
    Pattern: news_sector:{hash} hoặc news_sector:user:{user_id}:{hash}
    """
    # Normalize sectors (sort để đảm bảo consistent key)
    sectors_str = ",".join(sorted(s.strip().lower() for s in sectors))
    topics_normalized = (topics or "").strip().lower()
    
    key_hash = _generate_hash(
        sectors_str,
        analysis_depth,
        topics_normalized,
        target_language,
        max_results_per_search,
        processing_method
    )
    
    if user_id:
        return f"news_sector:user:{user_id}:{key_hash}"
    else:
        return f"news_sector:{key_hash}"


# ========================
# Cache Operations
# ========================

async def get_news_cache(
    redis_client: Optional[aioredis.Redis],
    cache_key: str
) -> Optional[Dict[str, Any]]:
    """
    Get cached news response from Redis.
    
    Returns:
        Dict with structure matching execute_* function returns, or None if not found.
    """
    if not redis_client:
        logger.debug("Redis client not available, skipping cache get")
        return None
    
    try:
        cached_data = await redis_client.get(cache_key)
        if cached_data:
            logger.info(f"✅ Cache HIT: {cache_key}")
            # Decode and parse JSON
            data_dict = json.loads(cached_data.decode('utf-8'))
            return data_dict
        else:
            logger.debug(f"❌ Cache MISS: {cache_key}")
            return None
            
    except aioredis.RedisError as e:
        logger.error(f"Redis error when getting cache key {cache_key}: {e}", exc_info=True)
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for cache key {cache_key}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error when getting cache key {cache_key}: {e}", exc_info=True)
        return None


async def set_news_cache(
    redis_client: Optional[aioredis.Redis],
    cache_key: str,
    data: Dict[str, Any],
    ttl: int
) -> bool:
    """
    Set news response cache in Redis.
    
    Args:
        redis_client: Redis client instance
        cache_key: Cache key
        data: Response data dict from execute_* functions
        ttl: Time to live in seconds
        
    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        logger.debug("Redis client not available, skipping cache set")
        return False
    
    try:
        # Serialize to JSON
        json_data = json.dumps(data)
        
        # Set with TTL
        await redis_client.set(cache_key, json_data, ex=ttl)
        logger.info(f"💾 Cache SET: {cache_key} (TTL: {ttl}s)")
        return True
        
    except aioredis.RedisError as e:
        logger.error(f"Redis error when setting cache key {cache_key}: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error when setting cache key {cache_key}: {e}", exc_info=True)
        return False


async def set_news_cache(
    redis_client: Optional[aioredis.Redis],
    cache_key: str,
    data: Dict[str, Any],
    ttl: int
) -> bool:
    """
    Set news response cache in Redis.
    ONLY cache if results are not empty.
    
    Args:
        redis_client: Redis client instance
        cache_key: Cache key
        data: Response data dict from execute_* functions
        ttl: Time to live in seconds
        
    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        logger.debug("Redis client not available, skipping cache set")
        return False
    
    # ============ VALIDATION: Don't cache empty results ============
    summaries = data.get("summaries", [])
    total_articles = data.get("processing_stats", {}).get("total_articles", 0)
    
    if not summaries or total_articles == 0:
        logger.info(f"⚠️  Skipping cache for {cache_key}: No articles found (empty result)")
        return False
    # ================================================================
    
    try:
        # Serialize to JSON
        json_data = json.dumps(data)
        
        # Set with TTL
        await redis_client.set(cache_key, json_data, ex=ttl)
        logger.info(f"💾 Cache SET: {cache_key} (TTL: {ttl}s, Articles: {total_articles})")
        return True
        
    except aioredis.RedisError as e:
        logger.error(f"Redis error when setting cache key {cache_key}: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error when setting cache key {cache_key}: {e}", exc_info=True)
        return False

async def invalidate_user_news_cache(
    redis_client: Optional[aioredis.Redis],
    user_id: int,
    cache_pattern: str = "news_*"
) -> int:
    """
    Invalidate all news cache for a specific user.
    Useful when user updates preferences.
    
    Args:
        redis_client: Redis client
        user_id: User ID
        cache_pattern: Pattern to match (default: all news cache)
        
    Returns:
        Number of keys deleted
    """
    if not redis_client:
        return 0
    
    try:
        # Scan for keys matching pattern with user_id
        user_pattern = f"*:{user_id}:*"
        cursor = 0
        deleted_count = 0
        
        while True:
            cursor, keys = await redis_client.scan(
                cursor,
                match=user_pattern,
                count=100
            )
            
            if keys:
                deleted = await redis_client.delete(*keys)
                deleted_count += deleted
                logger.info(f"🗑️ Invalidated {deleted} cache keys for user {user_id}")
            
            if cursor == 0:
                break
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error invalidating cache for user {user_id}: {e}", exc_info=True)
        return 0


# ========================
# Cache Statistics
# ========================

async def get_cache_stats(
    redis_client: Optional[aioredis.Redis]
) -> Dict[str, Any]:
    """
    Get cache statistics for news agent.
    
    Returns:
        Dict with cache stats (total keys, memory usage, etc.)
    """
    if not redis_client:
        return {"available": False}
    
    try:
        # Count keys by pattern
        patterns = {
            "daily": "news_daily:*",
            "personalized": "news_personalized:*",
            "symbol": "news_symbol:*",
            "sector": "news_sector:*"
        }
        
        stats = {"available": True, "timestamp": datetime.utcnow().isoformat()}
        
        for name, pattern in patterns.items():
            cursor = 0
            count = 0
            
            while True:
                cursor, keys = await redis_client.scan(cursor, match=pattern, count=1000)
                count += len(keys)
                if cursor == 0:
                    break
            
            stats[f"{name}_keys"] = count
        
        # Get Redis info
        info = await redis_client.info("memory")
        stats["memory_used_mb"] = round(info.get("used_memory", 0) / 1024 / 1024, 2)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}", exc_info=True)
        return {"available": False, "error": str(e)}