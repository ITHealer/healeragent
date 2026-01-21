"""
Scanner Result Cache Helper

Manages caching of individual scanner step results for synthesis.
Each step's result is cached with appropriate TTL based on data freshness requirements.

Cache Key Format: scanner:{symbol}:{step_name}:{time_bucket}
Time Bucket: 3-minute intervals for consistency across requests

Usage:
    from src.helpers.scanner_cache_helper import (
        save_scanner_result,
        get_scanner_result,
        get_all_scanner_results
    )

    # Save step result
    await save_scanner_result(symbol, "technical", result_dict)

    # Get single step
    cached = await get_scanner_result(symbol, "technical")

    # Get all available for synthesis
    all_results = await get_all_scanner_results(symbol)
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from src.helpers.redis_cache import get_redis_client_llm
from src.utils.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# CACHE TTL CONFIGURATION (seconds)
# =============================================================================

SCANNER_STEP_TTL = {
    "technical":   180,    # 3 minutes - price data changes frequently
    "position":    300,    # 5 minutes - RS vs benchmark
    "risk":        300,    # 5 minutes - stop loss levels tied to price
    "sentiment":   600,    # 10 minutes - news/sentiment less volatile
    "fundamental": 900,    # 15 minutes - fundamentals rarely change intraday
}

# All supported scanner steps
SCANNER_STEPS = ["technical", "position", "risk", "sentiment", "fundamental"]

# Time bucket interval (seconds) - used for cache key consistency
CACHE_TIME_BUCKET_INTERVAL = 180  # 3 minutes


# =============================================================================
# CACHE KEY GENERATION
# =============================================================================

def _get_time_bucket() -> int:
    """
    Get current time bucket for cache key.
    Groups requests within 3-minute windows to same cache key.
    """
    return int(datetime.now().timestamp()) // CACHE_TIME_BUCKET_INTERVAL


def _make_scanner_cache_key(symbol: str, step_name: str) -> str:
    """
    Generate cache key for scanner result.

    Format: scanner:{SYMBOL}:{step_name}:{time_bucket}

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        step_name: Analysis step (technical, position, risk, sentiment, fundamental)

    Returns:
        Cache key string
    """
    symbol_upper = symbol.upper().strip()
    bucket = _get_time_bucket()
    return f"scanner:{symbol_upper}:{step_name}:{bucket}"


def _make_scanner_cache_key_pattern(symbol: str, step_name: str) -> str:
    """
    Generate pattern to find any cached result for symbol+step.
    Used when we want to find cache regardless of time bucket.

    Format: scanner:{SYMBOL}:{step_name}:*
    """
    symbol_upper = symbol.upper().strip()
    return f"scanner:{symbol_upper}:{step_name}:*"


# =============================================================================
# CACHE OPERATIONS
# =============================================================================

async def save_scanner_result(
    symbol: str,
    step_name: str,
    result: Dict[str, Any]
) -> bool:
    """
    Save scanner step result to Redis cache.

    Args:
        symbol: Stock symbol
        step_name: Step name (technical, position, risk, sentiment, fundamental)
        result: Result dictionary containing:
            - content: The LLM response text
            - raw_data: Raw data from tools (optional)
            - timestamp: When the analysis was generated

    Returns:
        True if saved successfully, False otherwise
    """
    if step_name not in SCANNER_STEPS:
        logger.warning(f"[ScannerCache] Unknown step: {step_name}")
        return False

    redis = await get_redis_client_llm()
    if redis is None:
        logger.debug("[ScannerCache] Redis not available, cache disabled")
        return False

    cache_key = _make_scanner_cache_key(symbol, step_name)
    ttl = SCANNER_STEP_TTL.get(step_name, 300)

    try:
        # Add metadata to result
        cache_data = {
            "symbol": symbol.upper(),
            "step": step_name,
            "cached_at": datetime.now().isoformat(),
            "content": result.get("content", ""),
            "raw_data": result.get("raw_data"),
            "metadata": result.get("metadata", {})
        }

        json_data = json.dumps(cache_data, default=str, ensure_ascii=False)
        await redis.set(cache_key, json_data, ex=ttl)

        logger.info(f"[ScannerCache] Saved {step_name} for {symbol} (TTL={ttl}s)")
        return True

    except Exception as e:
        logger.error(f"[ScannerCache] Save error for {symbol}/{step_name}: {e}")
        return False


async def get_scanner_result(
    symbol: str,
    step_name: str
) -> Optional[Dict[str, Any]]:
    """
    Get scanner step result from cache.

    Args:
        symbol: Stock symbol
        step_name: Step name

    Returns:
        Cached result dict or None if not found/expired
    """
    if step_name not in SCANNER_STEPS:
        return None

    redis = await get_redis_client_llm()
    if redis is None:
        return None

    cache_key = _make_scanner_cache_key(symbol, step_name)

    try:
        cached = await redis.get(cache_key)
        if cached:
            logger.info(f"[ScannerCache] HIT {step_name} for {symbol}")
            return json.loads(cached)

        logger.debug(f"[ScannerCache] MISS {step_name} for {symbol}")
        return None

    except Exception as e:
        logger.error(f"[ScannerCache] Get error for {symbol}/{step_name}: {e}")
        return None


async def get_all_scanner_results(symbol: str) -> Dict[str, Any]:
    """
    Get all available scanner results for a symbol.
    Used by synthesis step to check what's already cached.

    Args:
        symbol: Stock symbol

    Returns:
        Dictionary with:
            - available: List of step names that have cached results
            - missing: List of step names that need to be run
            - data: Dict mapping step_name -> cached result
            - cache_ages: Dict mapping step_name -> age in seconds
    """
    available = []
    missing = []
    data = {}
    cache_ages = {}

    now = datetime.now()

    for step_name in SCANNER_STEPS:
        result = await get_scanner_result(symbol, step_name)

        if result:
            available.append(step_name)
            data[step_name] = result

            # Calculate cache age
            cached_at = result.get("cached_at")
            if cached_at:
                try:
                    cached_time = datetime.fromisoformat(cached_at)
                    age_seconds = (now - cached_time).total_seconds()
                    cache_ages[step_name] = int(age_seconds)
                except (ValueError, TypeError):
                    cache_ages[step_name] = 0
        else:
            missing.append(step_name)

    return {
        "symbol": symbol.upper(),
        "available": available,
        "missing": missing,
        "data": data,
        "cache_ages": cache_ages,
        "checked_at": now.isoformat()
    }


async def clear_scanner_cache(symbol: str, step_name: Optional[str] = None) -> int:
    """
    Clear scanner cache for a symbol.

    Args:
        symbol: Stock symbol
        step_name: Specific step to clear, or None to clear all steps

    Returns:
        Number of keys deleted
    """
    redis = await get_redis_client_llm()
    if redis is None:
        return 0

    deleted_count = 0

    try:
        steps_to_clear = [step_name] if step_name else SCANNER_STEPS

        for step in steps_to_clear:
            if step not in SCANNER_STEPS:
                continue

            # Find and delete all matching keys (any time bucket)
            pattern = _make_scanner_cache_key_pattern(symbol, step)
            cursor = 0

            while True:
                cursor, keys = await redis.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted = await redis.delete(*keys)
                    deleted_count += deleted
                if cursor == 0:
                    break

        if deleted_count > 0:
            logger.info(f"[ScannerCache] Cleared {deleted_count} keys for {symbol}")

        return deleted_count

    except Exception as e:
        logger.error(f"[ScannerCache] Clear error for {symbol}: {e}")
        return 0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_step_ttl(step_name: str) -> int:
    """Get TTL in seconds for a step."""
    return SCANNER_STEP_TTL.get(step_name, 300)


def get_step_freshness_label(age_seconds: int, step_name: str) -> str:
    """
    Get human-readable freshness label for cached data.

    Args:
        age_seconds: Age of cache in seconds
        step_name: Step name to compare against TTL

    Returns:
        Freshness label: "fresh", "stale", "expired"
    """
    ttl = SCANNER_STEP_TTL.get(step_name, 300)

    if age_seconds < ttl * 0.5:
        return "fresh"
    elif age_seconds < ttl:
        return "stale"
    else:
        return "expired"


def format_cache_status_message(cache_status: Dict[str, Any]) -> str:
    """
    Format cache status into human-readable message.

    Args:
        cache_status: Result from get_all_scanner_results()

    Returns:
        Formatted status message
    """
    available = cache_status.get("available", [])
    missing = cache_status.get("missing", [])
    cache_ages = cache_status.get("cache_ages", {})

    lines = [f"Scanner Cache Status for {cache_status.get('symbol', 'Unknown')}:"]

    if available:
        lines.append(f"  Available ({len(available)}): {', '.join(available)}")
        for step in available:
            age = cache_ages.get(step, 0)
            freshness = get_step_freshness_label(age, step)
            lines.append(f"    - {step}: {age}s old ({freshness})")

    if missing:
        lines.append(f"  Missing ({len(missing)}): {', '.join(missing)}")

    return "\n".join(lines)
