"""
Task Queue Service
==================

Redis-based priority queue for async task processing.

Features:
- Priority-based task ordering (BZPOPMAX)
- Task status tracking (pending, processing, completed, failed)
- Scheduled tasks with delay
- Job result storage
- Retry logic for failed jobs

Redis Keys:
- news_tasks:priority     - Sorted set for priority queue
- news_tasks:status:{id}  - Hash for task status
- news_tasks:result:{id}  - Hash for task result
- news_tasks:data:{id}    - String for task data

Usage:
    queue = TaskQueueService()
    await queue.initialize()

    # Submit task
    job_id = await queue.submit_task(task_request)

    # Worker pulls task
    task_data = await queue.pop_task()

    # Update status
    await queue.update_status(job_id, TaskStatus.COMPLETED, result)
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.news_aggregator.schemas.task import (
    TaskRequest,
    TaskStatus,
    TaskResult,
    TaskStatusResponse,
    TaskSubmitResponse,
)
from src.utils.config import settings

logger = logging.getLogger(__name__)

# Redis key prefixes
QUEUE_KEY = "news_tasks:priority"
STATUS_KEY_PREFIX = "news_tasks:status"
RESULT_KEY_PREFIX = "news_tasks:result"
DATA_KEY_PREFIX = "news_tasks:data"

# TTL for stored data
STATUS_TTL = 86400 * 7  # 7 days
RESULT_TTL = 86400 * 7  # 7 days
DATA_TTL = 86400        # 1 day


class TaskQueueService:
    """
    Redis-based task queue with priority support.

    Uses Redis sorted sets for priority ordering:
    - Higher score = higher priority = processed first
    - BZPOPMAX blocks and returns highest priority task
    """

    def __init__(self):
        """Initialize task queue service."""
        self.logger = logger
        self._redis_client = None
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize Redis connection.

        Returns:
            True if initialized successfully
        """
        if self._initialized:
            return True

        try:
            # Import here to avoid circular imports
            try:
                import aioredis
            except ImportError:
                import redis.asyncio as aioredis

            # Build Redis URL
            host = settings.REDIS_HOST
            port = settings.REDIS_PORT
            password = getattr(settings, "REDIS_PASSWORD", None)
            db = getattr(settings, "REDIS_DB_LLM", 0)

            if password:
                redis_url = f"redis://:{password}@{host}:{port}/{db}"
            else:
                redis_url = f"redis://{host}:{port}/{db}"

            self._redis_client = await aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=10.0,
                socket_connect_timeout=10.0,
            )

            # Test connection
            await self._redis_client.ping()

            self._initialized = True
            self.logger.info(f"[TaskQueue] Connected to Redis at {host}:{port}")
            return True

        except Exception as e:
            self.logger.error(f"[TaskQueue] Failed to connect to Redis: {e}")
            return False

    async def close(self):
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None
            self._initialized = False

    def _ensure_initialized(self):
        """Ensure service is initialized."""
        if not self._initialized or not self._redis_client:
            raise RuntimeError("TaskQueueService not initialized. Call initialize() first.")

    def _status_key(self, job_id: str) -> str:
        """Get Redis key for task status."""
        return f"{STATUS_KEY_PREFIX}:{job_id}"

    def _result_key(self, job_id: str) -> str:
        """Get Redis key for task result."""
        return f"{RESULT_KEY_PREFIX}:{job_id}"

    def _data_key(self, job_id: str) -> str:
        """Get Redis key for task data."""
        return f"{DATA_KEY_PREFIX}:{job_id}"

    async def submit_task(
        self,
        request: TaskRequest,
    ) -> TaskSubmitResponse:
        """
        Submit a new task to the queue.

        Args:
            request: Task request with symbols, options, callback info

        Returns:
            TaskSubmitResponse with job_id and queue position
        """
        self._ensure_initialized()

        # Generate job ID
        job_id = request.generate_job_id()
        now = datetime.utcnow()

        try:
            # Calculate effective priority
            # If scheduled for future, reduce priority temporarily
            priority = request.priority
            if request.scheduled_at and request.scheduled_at > now:
                # Scheduled tasks get lower priority until their time
                delay_seconds = (request.scheduled_at - now).total_seconds()
                if delay_seconds > 3600:  # More than 1 hour away
                    priority = max(1, priority - 10)

            # Store task data
            task_data = {
                "job_id": job_id,
                "request": request.model_dump_json(),
                "created_at": now.isoformat(),
                "status": TaskStatus.PENDING.value,
            }

            # Store in Redis
            pipe = self._redis_client.pipeline()

            # Add to priority queue
            pipe.zadd(QUEUE_KEY, {job_id: priority})

            # Store task data
            pipe.set(
                self._data_key(job_id),
                json.dumps(task_data),
                ex=DATA_TTL,
            )

            # Store status
            pipe.hset(
                self._status_key(job_id),
                mapping={
                    "status": TaskStatus.PENDING.value,
                    "request_id": str(request.request_id),
                    "created_at": now.isoformat(),
                    "priority": str(priority),
                },
            )
            pipe.expire(self._status_key(job_id), STATUS_TTL)

            await pipe.execute()

            # Get queue position
            queue_position = await self._redis_client.zrevrank(QUEUE_KEY, job_id)

            self.logger.info(
                f"[TaskQueue] Task submitted: {job_id} | "
                f"priority={priority} | position={queue_position}"
            )

            return TaskSubmitResponse(
                job_id=job_id,
                request_id=request.request_id,
                status=TaskStatus.PENDING,
                message="Task queued successfully",
                queue_position=queue_position,
            )

        except Exception as e:
            self.logger.error(f"[TaskQueue] Submit error: {e}")
            raise

    async def pop_task(
        self,
        timeout: float = 30.0,
    ) -> Optional[Tuple[str, TaskRequest]]:
        """
        Pop the highest priority task from queue.

        Blocks until a task is available or timeout.

        Args:
            timeout: Timeout in seconds

        Returns:
            Tuple of (job_id, TaskRequest) or None if timeout
        """
        self._ensure_initialized()

        try:
            # BZPOPMAX returns (key, member, score) for highest score
            result = await self._redis_client.bzpopmax(QUEUE_KEY, timeout=timeout)

            if not result:
                return None

            # result = (queue_key, job_id, score)
            _, job_id, score = result

            # Get task data
            data_json = await self._redis_client.get(self._data_key(job_id))
            if not data_json:
                self.logger.warning(f"[TaskQueue] No data for job {job_id}")
                return None

            task_data = json.loads(data_json)
            request = TaskRequest.model_validate_json(task_data["request"])

            # Update status to processing
            await self.update_status(job_id, TaskStatus.PROCESSING)

            self.logger.info(
                f"[TaskQueue] Task popped: {job_id} | "
                f"priority={score} | symbols={request.symbols}"
            )

            return (job_id, request)

        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.logger.error(f"[TaskQueue] Pop error: {e}")
            return None

    async def update_status(
        self,
        job_id: str,
        status: TaskStatus,
        result: Optional[TaskResult] = None,
        error: Optional[str] = None,
    ) -> bool:
        """
        Update task status.

        Args:
            job_id: Job identifier
            status: New status
            result: Task result (for completed tasks)
            error: Error message (for failed tasks)

        Returns:
            True if updated successfully
        """
        self._ensure_initialized()

        try:
            now = datetime.utcnow()

            # Build status update
            status_update = {
                "status": status.value,
                "updated_at": now.isoformat(),
            }

            if status == TaskStatus.PROCESSING:
                status_update["started_at"] = now.isoformat()
            elif status == TaskStatus.COMPLETED:
                status_update["completed_at"] = now.isoformat()
            elif status == TaskStatus.FAILED:
                status_update["completed_at"] = now.isoformat()
                if error:
                    status_update["error"] = error[:500]

            # Update status hash
            await self._redis_client.hset(
                self._status_key(job_id),
                mapping=status_update,
            )

            # Store result if provided
            if result:
                await self._redis_client.set(
                    self._result_key(job_id),
                    result.model_dump_json(),
                    ex=RESULT_TTL,
                )

            self.logger.debug(f"[TaskQueue] Status updated: {job_id} → {status.value}")
            return True

        except Exception as e:
            self.logger.error(f"[TaskQueue] Status update error: {e}")
            return False

    async def get_status(self, job_id: str) -> Optional[TaskStatusResponse]:
        """
        Get task status.

        Args:
            job_id: Job identifier

        Returns:
            TaskStatusResponse or None if not found
        """
        self._ensure_initialized()

        try:
            status_data = await self._redis_client.hgetall(self._status_key(job_id))

            if not status_data:
                return None

            return TaskStatusResponse(
                job_id=job_id,
                request_id=int(status_data.get("request_id", 0)),
                status=TaskStatus(status_data.get("status", "pending")),
                message=status_data.get("error"),
                created_at=datetime.fromisoformat(status_data["created_at"]),
                started_at=(
                    datetime.fromisoformat(status_data["started_at"])
                    if status_data.get("started_at")
                    else None
                ),
            )

        except Exception as e:
            self.logger.error(f"[TaskQueue] Get status error: {e}")
            return None

    async def get_result(self, job_id: str) -> Optional[TaskResult]:
        """
        Get task result.

        Args:
            job_id: Job identifier

        Returns:
            TaskResult or None if not found
        """
        self._ensure_initialized()

        try:
            result_json = await self._redis_client.get(self._result_key(job_id))
            if not result_json:
                return None

            return TaskResult.model_validate_json(result_json)

        except Exception as e:
            self.logger.error(f"[TaskQueue] Get result error: {e}")
            return None

    async def get_queue_length(self) -> int:
        """Get number of tasks in queue."""
        self._ensure_initialized()
        return await self._redis_client.zcard(QUEUE_KEY)

    async def get_pending_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get list of pending tasks.

        Args:
            limit: Max tasks to return

        Returns:
            List of task info dicts
        """
        self._ensure_initialized()

        try:
            # Get top tasks by priority
            tasks = await self._redis_client.zrevrange(
                QUEUE_KEY,
                0,
                limit - 1,
                withscores=True,
            )

            result = []
            for job_id, score in tasks:
                status_data = await self._redis_client.hgetall(self._status_key(job_id))
                result.append({
                    "job_id": job_id,
                    "priority": score,
                    "status": status_data.get("status", "unknown"),
                    "request_id": status_data.get("request_id"),
                    "created_at": status_data.get("created_at"),
                })

            return result

        except Exception as e:
            self.logger.error(f"[TaskQueue] Get pending error: {e}")
            return []

    async def cancel_task(self, job_id: str) -> bool:
        """
        Cancel a pending task.

        Args:
            job_id: Job identifier

        Returns:
            True if cancelled, False if already processing/completed
        """
        self._ensure_initialized()

        try:
            # Check current status
            status_data = await self._redis_client.hgetall(self._status_key(job_id))

            if not status_data:
                return False

            current_status = status_data.get("status")
            if current_status != TaskStatus.PENDING.value:
                self.logger.warning(
                    f"[TaskQueue] Cannot cancel {job_id}: status={current_status}"
                )
                return False

            # Remove from queue and update status
            pipe = self._redis_client.pipeline()
            pipe.zrem(QUEUE_KEY, job_id)
            pipe.hset(
                self._status_key(job_id),
                mapping={
                    "status": TaskStatus.CANCELLED.value,
                    "updated_at": datetime.utcnow().isoformat(),
                },
            )
            await pipe.execute()

            self.logger.info(f"[TaskQueue] Task cancelled: {job_id}")
            return True

        except Exception as e:
            self.logger.error(f"[TaskQueue] Cancel error: {e}")
            return False


# Singleton instance
_task_queue: Optional[TaskQueueService] = None


async def get_task_queue() -> TaskQueueService:
    """Get singleton task queue instance."""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueueService()
        await _task_queue.initialize()
    return _task_queue


async def close_task_queue():
    """Close task queue connection."""
    global _task_queue
    if _task_queue:
        await _task_queue.close()
        _task_queue = None
