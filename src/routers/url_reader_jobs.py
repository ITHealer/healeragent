"""
URL Reader Jobs API Router
==========================

API endpoints for submitting and managing background URL reading jobs.

This processes news article URLs and returns summarized content.
Useful for processing FMP news URLs from user watchlist.

Endpoints:
- POST /api/v2/url-reader/jobs/submit - Submit URL reader job
- GET /api/v2/url-reader/jobs/{job_id}/status - Get job status
- GET /api/v2/url-reader/jobs/{job_id}/result - Get job result
- DELETE /api/v2/url-reader/jobs/{job_id} - Cancel a pending job
- GET /api/v2/url-reader/jobs/queue/stats - Get queue statistics

Usage:
    # Submit a URL reader job
    POST /api/v2/url-reader/jobs/submit
    {
        "request_id": 123,
        "urls": [
            "https://www.marketwatch.com/story/tesla-news",
            "https://finance.yahoo.com/news/nvda"
        ],
        "callback_url": "https://api.be.net/callback",
        "target_language": "vi"
    }
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from src.news_aggregator.schemas.url_reader_job import (
    URLReaderJobStatus,
    URLReaderJobRequest,
    URLReaderJobSubmitResponse,
    URLReaderJobStatusResponse,
    URLReaderJobResult,
    URLReaderQueueStats,
)
from src.news_aggregator.services.url_reader_queue import (
    get_url_reader_queue,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/url-reader/jobs")


# =============================================================================
# Request Models
# =============================================================================

class URLReaderJobSubmitRequest(BaseModel):
    """API request model for submitting a URL reader job."""

    # Required fields
    request_id: int = Field(..., description="Reference ID from BE .NET")
    urls: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of URLs to process (1-20 URLs)"
    )
    callback_url: str = Field(..., description="URL to receive job results")

    # Processing options
    target_language: Optional[str] = Field(
        default=None,
        description="Target language code (en, vi, zh, etc.). If not provided, keeps source language"
    )
    include_original: bool = Field(
        default=False,
        description="Include original content (transcript/article text) in response"
    )

    # LLM configuration
    model_name: str = Field(default="gpt-4.1-nano", description="LLM model name")
    provider_type: str = Field(default="openai", description="LLM provider")

    # Priority
    priority: int = Field(default=50, ge=1, le=100, description="Job priority (1-100)")

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": 1234,
                "urls": [
                    "https://www.marketwatch.com/story/tesla-stock-rises",
                    "https://finance.yahoo.com/news/nvidia-earnings-report"
                ],
                "callback_url": "https://api.example.com/callback/url-reader",
                "target_language": "vi",
                "include_original": False,
                "priority": 50
            }
        }


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/submit", response_model=URLReaderJobSubmitResponse)
async def submit_url_reader_job(request: URLReaderJobSubmitRequest) -> URLReaderJobSubmitResponse:
    """
    Submit a URL reader job for background processing.

    Processes one or more URLs to extract and summarize content.
    Results will be sent to the callback_url when complete.

    **Features:**
    - Processes video URLs (YouTube, TikTok, etc.)
    - Processes article URLs (news articles, blog posts)
    - Automatic content type detection
    - Multi-language summarization
    - Translation support

    **Response:**
    Returns immediately with job_id. Poll /status or wait for callback.
    """
    try:
        # Validate URLs
        if not request.urls:
            raise HTTPException(
                status_code=400,
                detail="At least one URL is required"
            )

        # Build internal request
        job_request = URLReaderJobRequest(
            request_id=request.request_id,
            urls=request.urls,
            callback_url=request.callback_url,
            target_language=request.target_language,
            include_original=request.include_original,
            model_name=request.model_name,
            provider_type=request.provider_type,
            priority=request.priority,
        )

        # Submit to queue
        queue = await get_url_reader_queue()
        job_id = await queue.submit_job(job_request)

        # Get queue position
        queue_length = await queue.get_queue_length()

        logger.info(
            f"[URLReaderJobs] Job submitted: {job_id} | "
            f"urls={len(request.urls)} | request_id={request.request_id}"
        )

        return URLReaderJobSubmitResponse(
            job_id=job_id,
            request_id=request.request_id,
            status=URLReaderJobStatus.PENDING,
            total_urls=len(request.urls),
            queue_position=queue_length,
            message=f"Job submitted successfully. Processing {len(request.urls)} URL(s).",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[URLReaderJobs] Submit error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")


@router.get("/{job_id}/status", response_model=URLReaderJobStatusResponse)
async def get_job_status(job_id: str) -> URLReaderJobStatusResponse:
    """
    Get the status of a URL reader job.

    **Statuses:**
    - `pending`: Job is waiting in queue
    - `processing`: Job is being processed
    - `completed`: Job finished successfully
    - `failed`: Job failed with error
    - `cancelled`: Job was cancelled

    **Progress:**
    - `processed_urls`: Number of URLs processed so far
    - `progress_percent`: Progress percentage (0-100)
    """
    try:
        queue = await get_url_reader_queue()
        status_data = await queue.get_status(job_id)

        if not status_data:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        total_urls = int(status_data.get("total_urls", 0))
        processed_urls = int(status_data.get("processed_urls", 0))
        progress_percent = int((processed_urls / total_urls * 100) if total_urls > 0 else 0)

        return URLReaderJobStatusResponse(
            job_id=job_id,
            request_id=int(status_data.get("request_id", 0)),
            status=URLReaderJobStatus(status_data.get("status", "pending")),
            total_urls=total_urls,
            processed_urls=processed_urls,
            created_at=status_data.get("created_at"),
            started_at=status_data.get("started_at"),
            completed_at=status_data.get("completed_at"),
            progress_percent=progress_percent,
            error=status_data.get("error"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[URLReaderJobs] Status error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/{job_id}/result", response_model=URLReaderJobResult)
async def get_job_result(job_id: str) -> URLReaderJobResult:
    """
    Get the result of a completed URL reader job.

    Returns full result including per-URL summaries.
    Only available for completed or failed jobs.

    **Response includes:**
    - Per-URL results with summaries
    - Success/failure counts
    - Processing times
    - Content types detected
    """
    try:
        queue = await get_url_reader_queue()

        # Check status first
        status_data = await queue.get_status(job_id)
        if not status_data:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        status = status_data.get("status")
        if status not in (URLReaderJobStatus.COMPLETED.value, URLReaderJobStatus.FAILED.value):
            raise HTTPException(
                status_code=400,
                detail=f"Job not complete. Current status: {status}"
            )

        # Get result
        result = await queue.get_result(job_id)
        if not result:
            raise HTTPException(
                status_code=404,
                detail="Result not found. It may have expired."
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[URLReaderJobs] Result error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get result: {str(e)}")


@router.delete("/{job_id}")
async def cancel_job(job_id: str) -> Dict[str, Any]:
    """
    Cancel a pending URL reader job.

    Only pending jobs can be cancelled.
    Jobs that are already processing cannot be cancelled.
    """
    try:
        queue = await get_url_reader_queue()
        cancelled = await queue.cancel_job(job_id)

        if cancelled:
            logger.info(f"[URLReaderJobs] Job cancelled: {job_id}")
            return {
                "status": "success",
                "message": f"Job {job_id} cancelled",
                "job_id": job_id,
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="Cannot cancel job. It may not exist or is already processing."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[URLReaderJobs] Cancel error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@router.get("/queue/stats", response_model=URLReaderQueueStats)
async def get_queue_stats() -> URLReaderQueueStats:
    """
    Get statistics about the URL reader job queue.

    Useful for monitoring queue health and load.
    """
    try:
        queue = await get_url_reader_queue()
        stats = await queue.get_queue_stats()
        return stats

    except Exception as e:
        logger.error(f"[URLReaderJobs] Stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/pending")
async def get_pending_jobs(limit: int = 10) -> Dict[str, Any]:
    """
    Get list of pending jobs in the queue.

    Args:
        limit: Maximum number of jobs to return (default: 10)
    """
    try:
        queue = await get_url_reader_queue()
        jobs = await queue.get_pending_jobs(limit=limit)

        return {
            "count": len(jobs),
            "jobs": jobs,
        }

    except Exception as e:
        logger.error(f"[URLReaderJobs] Pending jobs error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get pending jobs: {str(e)}")
