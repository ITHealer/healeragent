"""
URL Reader Job Queue Service
============================

Redis-based priority queue for background URL content reading jobs.

Features:
- Priority queue using Redis ZSET (BZPOPMAX)
- Job status tracking with Redis hashes
- Result storage with TTL
- Atomic operations for job lifecycle

Redis Key Structure:
- url_reader:priority          - Sorted set for priority queue
- url_reader:status:{job_id}   - Hash for job status
- url_reader:result:{job_id}   - String for job results
- url_reader:data:{job_id}     - String for original request

Usage:
    queue = await get_url_reader_queue()
    job_id = await queue.submit_job(request)
    result = await queue.pop_job(timeout=30)
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Redis client - compatible with both aioredis and redis.asyncio
try:
    import aioredis as redis
except ImportError:
    import redis.asyncio as redis

from src.news_aggregator.schemas.url_reader_job import (
    URLReaderJobRequest,
    URLReaderJobResult,
    URLReaderJobStatus,
    URLReaderQueueStats,
)

logger = logging.getLogger(__name__)


# Redis key prefixes
KEY_PREFIX = "url_reader"
QUEUE_KEY = f"{KEY_PREFIX}:priority"
STATUS_KEY_PREFIX = f"{KEY_PREFIX}:status"
RESULT_KEY_PREFIX = f"{KEY_PREFIX}:result"
DATA_KEY_PREFIX = f"{KEY_PREFIX}:data"

# TTLs
DATA_TTL = 60 * 60 * 24  # 24 hours for request data
RESULT_TTL = 60 * 60 * 24 * 7  # 7 days for results
STATUS_TTL = 60 * 60 * 24 * 7  # 7 days for status


class URLReaderQueueService:
    """
    Redis-based priority queue for URL reader jobs.

    Uses ZSET for priority ordering and atomic pop operations.
    """

    def __init__(
        self,
        redis_host: str = None,
        redis_port: int = None,
        redis_password: str = None,
        redis_db: int = None,
    ):
        """
        Initialize queue service.

        Args:
            redis_host: Redis host (default: REDIS_HOST env var)
            redis_port: Redis port (default: REDIS_PORT env var)
            redis_password: Redis password (default: REDIS_PASSWORD env var)
            redis_db: Redis database (default: REDIS_DB env var or 4)
        """
        self.redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        self.redis_port = redis_port or int(os.getenv("REDIS_PORT", 6379))
        self.redis_password = redis_password or os.getenv("REDIS_PASSWORD", "")
        self.redis_db = redis_db or int(os.getenv("REDIS_DB_URL_READER", 4))

        self._redis: Optional[redis.Redis] = None
        self.logger = logger

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        if self._redis is None:
            self._redis = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password or None,
                db=self.redis_db,
                decode_responses=True,
            )
        return self._redis

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    def _generate_job_id(self, request_id: int) -> str:
        """Generate unique job ID."""
        suffix = uuid.uuid4().hex[:8]
        return f"url_reader_{request_id}_{suffix}"

    async def submit_job(self, request: URLReaderJobRequest) -> str:
        """
        Submit a URL reader job to the queue.

        Args:
            request: URL reader job request

        Returns:
            Job ID
        """
        r = await self._get_redis()

        job_id = self._generate_job_id(request.request_id)
        now = datetime.utcnow()

        # Store request data
        data_key = f"{DATA_KEY_PREFIX}:{job_id}"
        await r.set(data_key, request.model_dump_json(), ex=DATA_TTL)

        # Initialize status
        status_key = f"{STATUS_KEY_PREFIX}:{job_id}"
        await r.hset(
            status_key,
            mapping={
                "status": URLReaderJobStatus.PENDING.value,
                "request_id": str(request.request_id),
                "total_urls": str(len(request.urls)),
                "processed_urls": "0",
                "created_at": now.isoformat(),
                "callback_url": request.callback_url,
            },
        )
        await r.expire(status_key, STATUS_TTL)

        # Add to priority queue (higher priority = higher score)
        score = request.priority + (time.time() / 1e10)
        await r.zadd(QUEUE_KEY, {job_id: score})

        self.logger.info(
            f"[URLReaderQueue] Job submitted: {job_id} | "
            f"urls={len(request.urls)} | priority={request.priority}"
        )

        return job_id

    async def pop_job(self, timeout: float = 30.0) -> Optional[Tuple[str, URLReaderJobRequest]]:
        """
        Pop highest priority job from queue.

        Uses BZPOPMAX for blocking pop with priority ordering.

        Args:
            timeout: Timeout in seconds

        Returns:
            Tuple of (job_id, request) or None if timeout
        """
        r = await self._get_redis()

        # BZPOPMAX returns (key, member, score) or None
        result = await r.bzpopmax(QUEUE_KEY, timeout=timeout)

        if result is None:
            return None

        _, job_id, priority = result

        # Update status to processing
        await self.update_status(job_id, URLReaderJobStatus.PROCESSING)

        # Get request data
        data_key = f"{DATA_KEY_PREFIX}:{job_id}"
        request_json = await r.get(data_key)

        if not request_json:
            self.logger.error(f"[URLReaderQueue] No data for job: {job_id}")
            return None

        try:
            request = URLReaderJobRequest.model_validate_json(request_json)
        except Exception as e:
            self.logger.error(f"[URLReaderQueue] Invalid request data: {e}")
            return None

        self.logger.info(
            f"[URLReaderQueue] Job popped: {job_id} | "
            f"priority={priority:.1f} | urls={len(request.urls)}"
        )

        return job_id, request

    async def update_status(
        self,
        job_id: str,
        status: URLReaderJobStatus,
        processed_urls: int = None,
        error: str = None,
        result: URLReaderJobResult = None,
    ) -> None:
        """
        Update job status.

        Args:
            job_id: Job ID
            status: New status
            processed_urls: Number of URLs processed so far
            error: Error message (if failed)
            result: Job result (if completed)
        """
        r = await self._get_redis()
        status_key = f"{STATUS_KEY_PREFIX}:{job_id}"
        now = datetime.utcnow()

        updates = {"status": status.value}

        if status == URLReaderJobStatus.PROCESSING:
            updates["started_at"] = now.isoformat()
        elif status in (URLReaderJobStatus.COMPLETED, URLReaderJobStatus.FAILED):
            updates["completed_at"] = now.isoformat()

        if processed_urls is not None:
            updates["processed_urls"] = str(processed_urls)

        if error:
            updates["error"] = error

        await r.hset(status_key, mapping=updates)

        # Store result if provided
        if result:
            result_key = f"{RESULT_KEY_PREFIX}:{job_id}"
            await r.set(result_key, result.model_dump_json(), ex=RESULT_TTL)

        self.logger.debug(f"[URLReaderQueue] Status updated: {job_id} → {status.value}")

    async def update_progress(self, job_id: str, processed_urls: int) -> None:
        """Update job progress (number of URLs processed)."""
        r = await self._get_redis()
        status_key = f"{STATUS_KEY_PREFIX}:{job_id}"
        await r.hset(status_key, "processed_urls", str(processed_urls))

    async def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status.

        Args:
            job_id: Job ID

        Returns:
            Status dict or None if not found
        """
        r = await self._get_redis()
        status_key = f"{STATUS_KEY_PREFIX}:{job_id}"
        status_data = await r.hgetall(status_key)

        if not status_data:
            return None

        return status_data

    async def get_result(self, job_id: str) -> Optional[URLReaderJobResult]:
        """
        Get job result.

        Args:
            job_id: Job ID

        Returns:
            URLReaderJobResult or None if not found
        """
        r = await self._get_redis()
        result_key = f"{RESULT_KEY_PREFIX}:{job_id}"
        result_json = await r.get(result_key)

        if not result_json:
            return None

        try:
            return URLReaderJobResult.model_validate_json(result_json)
        except Exception as e:
            self.logger.error(f"[URLReaderQueue] Invalid result data: {e}")
            return None

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending job.

        Args:
            job_id: Job ID

        Returns:
            True if cancelled, False if not found or already processing
        """
        r = await self._get_redis()

        # Check current status
        status_data = await self.get_status(job_id)
        if not status_data:
            return False

        if status_data.get("status") != URLReaderJobStatus.PENDING.value:
            return False

        # Remove from queue
        removed = await r.zrem(QUEUE_KEY, job_id)

        if removed:
            await self.update_status(job_id, URLReaderJobStatus.CANCELLED)
            self.logger.info(f"[URLReaderQueue] Job cancelled: {job_id}")
            return True

        return False

    async def get_queue_length(self) -> int:
        """Get number of jobs in queue."""
        r = await self._get_redis()
        return await r.zcard(QUEUE_KEY)

    async def get_queue_stats(self) -> URLReaderQueueStats:
        """Get queue statistics."""
        r = await self._get_redis()

        queue_length = await r.zcard(QUEUE_KEY)

        # Count by status
        pending_count = 0
        processing_count = 0
        total_urls_pending = 0

        # Get all status keys (limited scan)
        cursor = 0
        status_keys = []
        while True:
            cursor, keys = await r.scan(cursor, match=f"{STATUS_KEY_PREFIX}:*", count=100)
            status_keys.extend(keys)
            if cursor == 0 or len(status_keys) >= 1000:
                break

        for key in status_keys[:1000]:
            status_data = await r.hgetall(key)
            status = status_data.get("status")
            if status == URLReaderJobStatus.PENDING.value:
                pending_count += 1
                total_urls_pending += int(status_data.get("total_urls", 0))
            elif status == URLReaderJobStatus.PROCESSING.value:
                processing_count += 1

        return URLReaderQueueStats(
            queue_length=queue_length,
            pending_count=pending_count,
            processing_count=processing_count,
            total_urls_pending=total_urls_pending,
        )

    async def get_pending_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get list of pending jobs.

        Args:
            limit: Max number of jobs to return

        Returns:
            List of job info dicts
        """
        r = await self._get_redis()

        # Get top jobs from queue
        jobs = await r.zrevrange(QUEUE_KEY, 0, limit - 1, withscores=True)

        result = []
        for job_id, score in jobs:
            status_data = await self.get_status(job_id)
            if status_data:
                result.append({
                    "job_id": job_id,
                    "priority": int(score),
                    **status_data,
                })

        return result


# Singleton instance
_url_reader_queue: Optional[URLReaderQueueService] = None


async def get_url_reader_queue() -> URLReaderQueueService:
    """Get singleton URL reader queue instance."""
    global _url_reader_queue
    if _url_reader_queue is None:
        _url_reader_queue = URLReaderQueueService()
    return _url_reader_queue


async def close_url_reader_queue() -> None:
    """Close URL reader queue connection."""
    global _url_reader_queue
    if _url_reader_queue:
        await _url_reader_queue.close()
        _url_reader_queue = None
