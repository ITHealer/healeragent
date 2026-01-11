"""
Task API Router
===============

REST API endpoints for the News Analysis Task System.

Endpoints:
- POST /submit       - Submit new analysis task
- GET /{job_id}/status  - Get task status
- GET /{job_id}/result  - Get task result
- DELETE /{job_id}      - Cancel pending task
- GET /queue/stats      - Get queue statistics

Usage (BE .NET):
    1. Submit task:
       POST /api/v1/tasks/submit
       { "request_id": 1792, "symbols": ["TSLA", "BTC"], ... }

    2. Poll status (optional):
       GET /api/v1/tasks/{job_id}/status

    3. Receive callback when complete:
       POST /api/v1/user-task/submit-generation-result (on BE side)
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.news_aggregator.schemas.task import (
    TaskRequest,
    TaskType,
    TaskPriority,
    TaskStatus,
    TaskSubmitResponse,
    TaskStatusResponse,
    TaskResult,
)
from src.news_aggregator.services.task_queue import get_task_queue

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["News Analysis Tasks"])


# =============================================================================
# API Request/Response Models
# =============================================================================

class TaskSubmitRequest(BaseModel):
    """Request body for task submission."""

    request_id: int = Field(
        ...,
        description="Reference ID from BE .NET system",
        examples=[1792],
    )
    symbols: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of symbols to analyze (stocks, crypto)",
        examples=[["TSLA", "BTC", "NVDA"]],
    )
    task_type: str = Field(
        default="news_analysis",
        description="Type of analysis: news_analysis, market_summary, custom",
    )
    target_language: str = Field(
        default="vi",
        description="Target language for output (vi, en, etc.)",
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="Webhook URL to receive results when complete",
        examples=["https://api.example.com/api/v1/user-task/submit-generation-result"],
    )
    priority: int = Field(
        default=TaskPriority.NORMAL,
        ge=1,
        le=100,
        description="Task priority (1-100, higher = more urgent)",
    )
    scheduled_at: Optional[str] = Field(
        default=None,
        description="ISO datetime to schedule task for later execution",
    )
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional options for the analysis",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "request_id": 1792,
                    "symbols": ["TSLA", "BTC", "NVDA"],
                    "task_type": "news_analysis",
                    "target_language": "vi",
                    "callback_url": "https://api.example.com/api/v1/user-task/submit-generation-result",
                    "priority": 50,
                }
            ]
        }
    }


class TaskSubmitResponseAPI(BaseModel):
    """Response after task submission."""

    success: bool
    job_id: str
    request_id: int
    status: str
    message: str
    queue_position: Optional[int] = None


class TaskStatusResponseAPI(BaseModel):
    """Response for task status query."""

    job_id: str
    request_id: int
    status: str
    message: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class QueueStatsResponse(BaseModel):
    """Response for queue statistics."""

    queue_length: int
    pending_tasks: List[Dict[str, Any]]


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/submit", response_model=TaskSubmitResponseAPI)
async def submit_task(request: TaskSubmitRequest) -> TaskSubmitResponseAPI:
    """
    Submit a new news analysis task.

    The task will be queued and processed by background workers.
    Results will be sent to the callback_url when complete.

    **Flow:**
    1. Task is added to Redis priority queue
    2. Worker pulls task and processes through pipeline
    3. Result is sent to callback_url via POST

    **Callback Payload:**
    ```json
    {
        "requestId": 1792,
        "content": "{ ...TaskResult JSON... }"
    }
    ```
    """
    try:
        # Get task queue
        queue = await get_task_queue()

        # Build internal request
        task_type = TaskType.NEWS_ANALYSIS
        if request.task_type == "market_summary":
            task_type = TaskType.MARKET_SUMMARY
        elif request.task_type == "custom":
            task_type = TaskType.CUSTOM

        internal_request = TaskRequest(
            request_id=request.request_id,
            symbols=request.symbols,
            task_type=task_type,
            target_language=request.target_language,
            callback_url=request.callback_url,
            priority=request.priority,
            options=request.options or {},
        )

        # Submit to queue
        result = await queue.submit_task(internal_request)

        logger.info(
            f"[TaskAPI] Task submitted: {result.job_id} | "
            f"request_id={request.request_id} | symbols={request.symbols}"
        )

        return TaskSubmitResponseAPI(
            success=True,
            job_id=result.job_id,
            request_id=result.request_id,
            status=result.status.value,
            message=result.message or "Task queued successfully",
            queue_position=result.queue_position,
        )

    except Exception as e:
        logger.error(f"[TaskAPI] Submit error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit task: {str(e)}",
        )


@router.get("/{job_id}/status", response_model=TaskStatusResponseAPI)
async def get_task_status(job_id: str) -> TaskStatusResponseAPI:
    """
    Get the status of a submitted task.

    **Status Values:**
    - pending: Task is in queue waiting to be processed
    - processing: Task is currently being processed
    - completed: Task finished successfully (check /result endpoint)
    - failed: Task failed (error in message field)
    - cancelled: Task was cancelled before processing
    """
    try:
        queue = await get_task_queue()
        status = await queue.get_status(job_id)

        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Task {job_id} not found",
            )

        return TaskStatusResponseAPI(
            job_id=status.job_id,
            request_id=status.request_id,
            status=status.status.value,
            message=status.message,
            created_at=status.created_at.isoformat() if status.created_at else None,
            started_at=status.started_at.isoformat() if status.started_at else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TaskAPI] Status error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {str(e)}",
        )


@router.get("/{job_id}/result")
async def get_task_result(job_id: str) -> Dict[str, Any]:
    """
    Get the result of a completed task.

    Returns the full analysis result including:
    - Per-symbol analyses with sentiment and insights
    - Overall sentiment across all symbols
    - Key themes identified
    - Summary in target language

    **Note:** Result is only available for completed tasks.
    For failed tasks, check the /status endpoint for error message.
    """
    try:
        queue = await get_task_queue()

        # First check status
        status = await queue.get_status(job_id)
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Task {job_id} not found",
            )

        if status.status == TaskStatus.PENDING:
            return {
                "job_id": job_id,
                "status": "pending",
                "message": "Task is still in queue",
            }

        if status.status == TaskStatus.PROCESSING:
            return {
                "job_id": job_id,
                "status": "processing",
                "message": "Task is being processed",
            }

        if status.status == TaskStatus.FAILED:
            return {
                "job_id": job_id,
                "status": "failed",
                "message": status.message or "Task failed",
            }

        if status.status == TaskStatus.CANCELLED:
            return {
                "job_id": job_id,
                "status": "cancelled",
                "message": "Task was cancelled",
            }

        # Get result for completed task
        result = await queue.get_result(job_id)

        if not result:
            return {
                "job_id": job_id,
                "status": "completed",
                "message": "Result not available (may have expired)",
            }

        return result.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TaskAPI] Result error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task result: {str(e)}",
        )


@router.delete("/{job_id}")
async def cancel_task(job_id: str) -> Dict[str, Any]:
    """
    Cancel a pending task.

    Only tasks in 'pending' status can be cancelled.
    Tasks that are already processing cannot be cancelled.

    **Returns:**
    - success: true if cancelled, false otherwise
    - message: Explanation of result
    """
    try:
        queue = await get_task_queue()
        cancelled = await queue.cancel_task(job_id)

        if cancelled:
            logger.info(f"[TaskAPI] Task cancelled: {job_id}")
            return {
                "success": True,
                "job_id": job_id,
                "message": "Task cancelled successfully",
            }
        else:
            return {
                "success": False,
                "job_id": job_id,
                "message": "Cannot cancel task (not found or already processing)",
            }

    except Exception as e:
        logger.error(f"[TaskAPI] Cancel error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel task: {str(e)}",
        )


@router.get("/queue/stats", response_model=QueueStatsResponse)
async def get_queue_stats(
    limit: int = Query(default=10, ge=1, le=50, description="Max pending tasks to return"),
) -> QueueStatsResponse:
    """
    Get queue statistics.

    Returns:
    - Current queue length
    - List of pending tasks with their priority

    Useful for monitoring and debugging.
    """
    try:
        queue = await get_task_queue()

        queue_length = await queue.get_queue_length()
        pending_tasks = await queue.get_pending_tasks(limit=limit)

        return QueueStatsResponse(
            queue_length=queue_length,
            pending_tasks=pending_tasks,
        )

    except Exception as e:
        logger.error(f"[TaskAPI] Stats error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get queue stats: {str(e)}",
        )
