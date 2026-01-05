"""
Persistent Working Memory - Redis-backed storage for session state.

Implements the "3 Golden Rules" from Weaviate + Anthropic:
1. KEEP CONTEXT INFORMATIVE YET TIGHT - Essential info within reach
2. DESIGN AROUND KV-CACHE - Stable prefix, dynamic at end
3. TREAT FILE SYSTEM AS INFINITE CONTEXT - External storage for recoverable info

Features:
- Redis persistence for cross-restart recovery
- Task State Offloading for long-running tasks
- Automatic fallback to in-memory when Redis unavailable
- Session recovery after server restart
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.redis_cache import get_redis_client_llm


# ============================================================================
# CONSTANTS
# ============================================================================

# Redis key prefixes
REDIS_WM_PREFIX = "wm"  # Working Memory
REDIS_TASK_PREFIX = "task"  # Task State

# TTL settings
DEFAULT_WM_TTL = 3600 * 4  # 4 hours for working memory
DEFAULT_TASK_TTL = 3600 * 2  # 2 hours for task state
DEEP_RESEARCH_TASK_TTL = 3600 * 8  # 8 hours for deep research


# ============================================================================
# ENUMS
# ============================================================================

class TaskStatus(str, Enum):
    """Task execution status for offloading."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"  # For resumable tasks


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TaskState:
    """
    Task state for offloading - saves intermediate progress.

    Used for:
    - Deep Research Mode (5-30 minutes)
    - Multi-step analysis with dependencies
    - Screener -> Analysis -> Report flow
    """
    task_id: str
    session_id: str
    user_id: str

    # Task metadata
    task_type: str  # "deep_research", "analysis", "screener", etc.
    status: TaskStatus
    created_at: str  # ISO format
    updated_at: str

    # Progress tracking
    current_step: int = 0
    total_steps: int = 0
    progress_percent: float = 0.0

    # Intermediate results
    intermediate_results: Dict[str, Any] = None

    # Error recovery
    last_error: Optional[str] = None
    retry_count: int = 0

    # Context for resume
    query: str = ""
    symbols: List[str] = None
    tool_categories: List[str] = None

    def __post_init__(self):
        if self.intermediate_results is None:
            self.intermediate_results = {}
        if self.symbols is None:
            self.symbols = []
        if self.tool_categories is None:
            self.tool_categories = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskState":
        """Create from dict."""
        data["status"] = TaskStatus(data["status"])
        return cls(**data)


@dataclass
class WorkingMemorySnapshot:
    """
    Snapshot of working memory for Redis persistence.

    Only stores essential state, not full entries.
    """
    session_id: str
    user_id: str

    # Core state
    current_turn: int
    symbols: List[str]

    # Query context
    last_intent: Optional[str] = None
    last_language: str = "auto"
    last_query_type: Optional[str] = None

    # Timestamps
    created_at: str = None
    updated_at: str = None

    # Summary of recent activity (compact)
    recent_tools: List[str] = None  # Last 5 tool names used

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()
        if self.recent_tools is None:
            self.recent_tools = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkingMemorySnapshot":
        return cls(**data)


# ============================================================================
# PERSISTENT WORKING MEMORY SERVICE
# ============================================================================

class PersistentWorkingMemoryService(LoggerMixin):
    """
    Redis-backed persistence layer for Working Memory.

    Golden Rule #3: Treat File System as Infinite Context
    - External storage for recoverable information
    - Working memory as scratchpad

    Usage:
        service = PersistentWorkingMemoryService()

        # Save snapshot
        await service.save_snapshot(session_id, snapshot)

        # Recover after restart
        snapshot = await service.load_snapshot(session_id)

        # Task state offloading
        await service.save_task_state(task_state)
        task = await service.load_task_state(task_id)
    """

    def __init__(self):
        super().__init__()
        self._redis_available = None  # Lazy check

    async def _get_redis(self):
        """Get Redis client with availability check."""
        try:
            redis = await get_redis_client_llm()
            if redis:
                self._redis_available = True
                return redis
            self._redis_available = False
            return None
        except Exception as e:
            self.logger.warning(f"[PERSISTENT_WM] Redis unavailable: {e}")
            self._redis_available = False
            return None

    def _make_wm_key(self, session_id: str) -> str:
        """Create Redis key for working memory snapshot."""
        return f"{REDIS_WM_PREFIX}:{session_id}"

    def _make_task_key(self, task_id: str) -> str:
        """Create Redis key for task state."""
        return f"{REDIS_TASK_PREFIX}:{task_id}"

    def _make_session_tasks_key(self, session_id: str) -> str:
        """Create Redis key for session's task list."""
        return f"{REDIS_TASK_PREFIX}:session:{session_id}"

    # ========================================================================
    # WORKING MEMORY SNAPSHOT METHODS
    # ========================================================================

    async def save_snapshot(
        self,
        snapshot: WorkingMemorySnapshot,
        ttl: int = DEFAULT_WM_TTL,
    ) -> bool:
        """
        Save working memory snapshot to Redis.

        Args:
            snapshot: Working memory state to save
            ttl: Time-to-live in seconds

        Returns:
            True if saved successfully
        """
        redis = await self._get_redis()
        if not redis:
            self.logger.debug("[PERSISTENT_WM] Redis unavailable, skip save")
            return False

        try:
            key = self._make_wm_key(snapshot.session_id)
            snapshot.updated_at = datetime.now().isoformat()

            data = json.dumps(snapshot.to_dict())
            await redis.set(key, data, ex=ttl)

            self.logger.debug(
                f"[PERSISTENT_WM] Saved snapshot for session {snapshot.session_id[:8]}... "
                f"(turn={snapshot.current_turn}, symbols={snapshot.symbols})"
            )
            return True

        except Exception as e:
            self.logger.error(f"[PERSISTENT_WM] Failed to save snapshot: {e}")
            return False

    async def load_snapshot(
        self,
        session_id: str,
    ) -> Optional[WorkingMemorySnapshot]:
        """
        Load working memory snapshot from Redis.

        Args:
            session_id: Session to load

        Returns:
            Snapshot or None if not found
        """
        redis = await self._get_redis()
        if not redis:
            return None

        try:
            key = self._make_wm_key(session_id)
            data = await redis.get(key)

            if data:
                snapshot = WorkingMemorySnapshot.from_dict(json.loads(data))
                self.logger.info(
                    f"[PERSISTENT_WM] Recovered snapshot for session {session_id[:8]}... "
                    f"(turn={snapshot.current_turn}, symbols={snapshot.symbols})"
                )
                return snapshot

            return None

        except Exception as e:
            self.logger.error(f"[PERSISTENT_WM] Failed to load snapshot: {e}")
            return None

    async def delete_snapshot(self, session_id: str) -> bool:
        """Delete working memory snapshot."""
        redis = await self._get_redis()
        if not redis:
            return False

        try:
            key = self._make_wm_key(session_id)
            await redis.delete(key)
            return True
        except Exception as e:
            self.logger.error(f"[PERSISTENT_WM] Failed to delete snapshot: {e}")
            return False

    async def extend_snapshot_ttl(
        self,
        session_id: str,
        ttl: int = DEFAULT_WM_TTL,
    ) -> bool:
        """Extend TTL for active session."""
        redis = await self._get_redis()
        if not redis:
            return False

        try:
            key = self._make_wm_key(session_id)
            await redis.expire(key, ttl)
            return True
        except Exception:
            return False

    # ========================================================================
    # TASK STATE OFFLOADING METHODS
    # ========================================================================

    async def save_task_state(
        self,
        task: TaskState,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Save task state for offloading/recovery.

        Task State Offloading allows:
        - Resume multi-step tasks after interruption
        - Prevent context window pollution during long tasks
        - Recovery from server restarts

        Args:
            task: Task state to save
            ttl: Custom TTL (default based on task type)

        Returns:
            True if saved successfully
        """
        redis = await self._get_redis()
        if not redis:
            self.logger.debug("[PERSISTENT_WM] Redis unavailable, task state not persisted")
            return False

        try:
            # Determine TTL
            if ttl is None:
                if task.task_type == "deep_research":
                    ttl = DEEP_RESEARCH_TASK_TTL
                else:
                    ttl = DEFAULT_TASK_TTL

            # Update timestamp
            task.updated_at = datetime.now().isoformat()

            # Save task state
            task_key = self._make_task_key(task.task_id)
            data = json.dumps(task.to_dict())
            await redis.set(task_key, data, ex=ttl)

            # Add to session's task list (for listing)
            session_key = self._make_session_tasks_key(task.session_id)
            await redis.sadd(session_key, task.task_id)
            await redis.expire(session_key, ttl)

            self.logger.info(
                f"[PERSISTENT_WM] Saved task state: {task.task_id[:8]}... "
                f"(type={task.task_type}, status={task.status.value}, "
                f"step={task.current_step}/{task.total_steps})"
            )
            return True

        except Exception as e:
            self.logger.error(f"[PERSISTENT_WM] Failed to save task state: {e}")
            return False

    async def load_task_state(
        self,
        task_id: str,
    ) -> Optional[TaskState]:
        """
        Load task state for resumption.

        Args:
            task_id: Task ID to load

        Returns:
            TaskState or None if not found
        """
        redis = await self._get_redis()
        if not redis:
            return None

        try:
            key = self._make_task_key(task_id)
            data = await redis.get(key)

            if data:
                task = TaskState.from_dict(json.loads(data))
                self.logger.info(
                    f"[PERSISTENT_WM] Loaded task state: {task_id[:8]}... "
                    f"(status={task.status.value}, step={task.current_step})"
                )
                return task

            return None

        except Exception as e:
            self.logger.error(f"[PERSISTENT_WM] Failed to load task state: {e}")
            return None

    async def update_task_progress(
        self,
        task_id: str,
        current_step: int,
        progress_percent: float,
        intermediate_result: Optional[Dict[str, Any]] = None,
        step_key: Optional[str] = None,
    ) -> bool:
        """
        Update task progress (for long-running tasks).

        Args:
            task_id: Task to update
            current_step: Current step number
            progress_percent: Progress percentage (0-100)
            intermediate_result: Result from current step
            step_key: Key for storing result (default: step_{n})

        Returns:
            True if updated successfully
        """
        task = await self.load_task_state(task_id)
        if not task:
            return False

        task.current_step = current_step
        task.progress_percent = progress_percent
        task.status = TaskStatus.IN_PROGRESS

        if intermediate_result:
            key = step_key or f"step_{current_step}"
            task.intermediate_results[key] = intermediate_result

        return await self.save_task_state(task)

    async def complete_task(
        self,
        task_id: str,
        final_result: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Mark task as completed."""
        task = await self.load_task_state(task_id)
        if not task:
            return False

        task.status = TaskStatus.COMPLETED
        task.progress_percent = 100.0
        task.current_step = task.total_steps

        if final_result:
            task.intermediate_results["final"] = final_result

        return await self.save_task_state(task)

    async def fail_task(
        self,
        task_id: str,
        error: str,
    ) -> bool:
        """Mark task as failed."""
        task = await self.load_task_state(task_id)
        if not task:
            return False

        task.status = TaskStatus.FAILED
        task.last_error = error
        task.retry_count += 1

        return await self.save_task_state(task)

    async def pause_task(
        self,
        task_id: str,
    ) -> bool:
        """Pause task for later resumption."""
        task = await self.load_task_state(task_id)
        if not task:
            return False

        task.status = TaskStatus.PAUSED
        return await self.save_task_state(task)

    async def get_session_tasks(
        self,
        session_id: str,
        status_filter: Optional[TaskStatus] = None,
    ) -> List[TaskState]:
        """
        Get all tasks for a session.

        Args:
            session_id: Session to get tasks for
            status_filter: Optional status filter

        Returns:
            List of task states
        """
        redis = await self._get_redis()
        if not redis:
            return []

        try:
            # Get task IDs for session
            session_key = self._make_session_tasks_key(session_id)
            task_ids = await redis.smembers(session_key)

            if not task_ids:
                return []

            # Load each task
            tasks = []
            for task_id in task_ids:
                task = await self.load_task_state(task_id)
                if task:
                    if status_filter is None or task.status == status_filter:
                        tasks.append(task)

            return tasks

        except Exception as e:
            self.logger.error(f"[PERSISTENT_WM] Failed to get session tasks: {e}")
            return []

    async def get_resumable_tasks(
        self,
        session_id: str,
    ) -> List[TaskState]:
        """Get tasks that can be resumed (paused or in_progress)."""
        tasks = await self.get_session_tasks(session_id)
        return [
            t for t in tasks
            if t.status in [TaskStatus.PAUSED, TaskStatus.IN_PROGRESS]
        ]

    async def delete_task_state(
        self,
        task_id: str,
        session_id: str,
    ) -> bool:
        """Delete task state."""
        redis = await self._get_redis()
        if not redis:
            return False

        try:
            # Delete task
            task_key = self._make_task_key(task_id)
            await redis.delete(task_key)

            # Remove from session list
            session_key = self._make_session_tasks_key(session_id)
            await redis.srem(session_key, task_id)

            return True
        except Exception:
            return False

    # ========================================================================
    # HEALTH & STATS
    # ========================================================================

    async def is_available(self) -> bool:
        """Check if Redis persistence is available."""
        redis = await self._get_redis()
        return redis is not None

    async def get_stats(self) -> Dict[str, Any]:
        """Get persistence service stats."""
        redis = await self._get_redis()

        stats = {
            "redis_available": redis is not None,
            "working_memory_count": 0,
            "task_count": 0,
        }

        if redis:
            try:
                # Count working memory keys
                wm_keys = []
                async for key in redis.scan_iter(f"{REDIS_WM_PREFIX}:*"):
                    wm_keys.append(key)
                stats["working_memory_count"] = len(wm_keys)

                # Count task keys
                task_keys = []
                async for key in redis.scan_iter(f"{REDIS_TASK_PREFIX}:*"):
                    if ":session:" not in key:  # Exclude session lists
                        task_keys.append(key)
                stats["task_count"] = len(task_keys)

            except Exception as e:
                self.logger.error(f"[PERSISTENT_WM] Failed to get stats: {e}")

        return stats


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_persistent_wm_service: Optional[PersistentWorkingMemoryService] = None


def get_persistent_wm_service() -> PersistentWorkingMemoryService:
    """Get singleton persistent working memory service."""
    global _persistent_wm_service
    if _persistent_wm_service is None:
        _persistent_wm_service = PersistentWorkingMemoryService()
    return _persistent_wm_service


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def create_task_state(
    task_id: str,
    session_id: str,
    user_id: str,
    task_type: str,
    query: str,
    symbols: List[str] = None,
    tool_categories: List[str] = None,
    total_steps: int = 1,
) -> TaskState:
    """
    Create and save a new task state.

    Usage:
        task = await create_task_state(
            task_id="task_123",
            session_id="sess_abc",
            user_id="user_xyz",
            task_type="deep_research",
            query="Phân tích kỹ thuật AAPL",
            symbols=["AAPL"],
            tool_categories=["technical", "price"],
            total_steps=5,
        )
    """
    task = TaskState(
        task_id=task_id,
        session_id=session_id,
        user_id=user_id,
        task_type=task_type,
        status=TaskStatus.PENDING,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        total_steps=total_steps,
        query=query,
        symbols=symbols or [],
        tool_categories=tool_categories or [],
    )

    service = get_persistent_wm_service()
    await service.save_task_state(task)

    return task


async def save_working_memory_snapshot(
    session_id: str,
    user_id: str,
    current_turn: int,
    symbols: List[str],
    last_intent: Optional[str] = None,
    last_language: str = "auto",
    last_query_type: Optional[str] = None,
    recent_tools: List[str] = None,
) -> bool:
    """
    Save working memory snapshot for recovery.

    Call this periodically during session to enable cross-restart recovery.
    """
    snapshot = WorkingMemorySnapshot(
        session_id=session_id,
        user_id=user_id,
        current_turn=current_turn,
        symbols=symbols,
        last_intent=last_intent,
        last_language=last_language,
        last_query_type=last_query_type,
        recent_tools=recent_tools or [],
    )

    service = get_persistent_wm_service()
    return await service.save_snapshot(snapshot)


async def recover_working_memory(
    session_id: str,
) -> Optional[WorkingMemorySnapshot]:
    """
    Recover working memory from Redis after server restart.

    Returns:
        Snapshot if found, None otherwise
    """
    service = get_persistent_wm_service()
    return await service.load_snapshot(session_id)
