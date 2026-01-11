"""
URL Reader Job Schemas
======================

Models for background URL content reading/processing jobs.

This system processes URLs (news articles, videos) from user watchlist
and returns summarized content via callback.

Features:
- Single URL or batch URL processing
- Content extraction and summarization
- Translation support
- Callback to BE .NET

Usage:
    request = URLReaderJobRequest(
        request_id=123,
        urls=["https://example.com/news/article1"],
        callback_url="https://api.example.com/callback",
        target_language="vi",
    )
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl


class URLReaderJobStatus(str, Enum):
    """Status of a URL reader job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class URLContentType(str, Enum):
    """Type of content detected from URL."""
    VIDEO = "video"
    ARTICLE = "article"
    UNKNOWN = "unknown"


class URLProcessResult(BaseModel):
    """Result of processing a single URL."""
    url: str = Field(..., description="Original URL")
    status: str = Field(..., description="success or error")
    content_type: Optional[str] = Field(None, description="video, article, or unknown")

    # Content results
    title: Optional[str] = Field(None, description="Content title if available")
    summary: Optional[str] = Field(None, description="Generated summary")
    original_content: Optional[str] = Field(None, description="Original text/transcript")
    translation: Optional[str] = Field(None, description="Translated content if applicable")

    # Language info
    source_language: Optional[str] = Field(None, description="Detected source language")
    target_language: Optional[str] = Field(None, description="Target language for output")
    translation_needed: bool = Field(default=False, description="Whether translation was performed")

    # Metadata
    processing_time_ms: Optional[int] = Field(None, description="Processing time in ms")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Error info
    error: Optional[str] = Field(None, description="Error message if failed")


class URLReaderJobRequest(BaseModel):
    """
    Request to submit a URL reader job for background processing.

    Processes one or more URLs to extract and summarize content.
    Results are sent to callback_url when complete.
    """
    # Job identification
    request_id: int = Field(..., description="Reference ID from BE .NET")

    # URLs to process
    urls: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of URLs to process (1-20 URLs)"
    )

    # Callback configuration
    callback_url: str = Field(..., description="URL to receive job results")

    # Processing options
    target_language: Optional[str] = Field(
        default=None,
        description="Target language code (en, vi, zh, ja, ko, etc.). If not provided, keeps source language"
    )
    include_original: bool = Field(
        default=False,
        description="Include original content in response"
    )

    # LLM configuration
    model_name: str = Field(default="gpt-4.1-nano", description="LLM model name")
    provider_type: str = Field(default="openai", description="LLM provider")

    # Priority
    priority: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Job priority (1-100, higher = more urgent)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": 1234,
                "urls": [
                    "https://www.marketwatch.com/story/tesla-stock-news",
                    "https://finance.yahoo.com/news/nvda-earnings"
                ],
                "callback_url": "https://api.be.net/callback/url-reader",
                "target_language": "vi",
                "include_original": False,
                "priority": 50
            }
        }


class URLReaderJobSubmitResponse(BaseModel):
    """Response when a URL reader job is submitted."""
    job_id: str = Field(..., description="Unique job identifier")
    request_id: int = Field(..., description="Original request ID")
    status: URLReaderJobStatus = Field(default=URLReaderJobStatus.PENDING)
    total_urls: int = Field(..., description="Number of URLs to process")
    queue_position: Optional[int] = Field(None, description="Position in queue")
    message: str = Field(default="Job submitted successfully")


class URLReaderJobStatusResponse(BaseModel):
    """Response for job status check."""
    job_id: str
    request_id: int
    status: URLReaderJobStatus
    total_urls: int
    processed_urls: int = Field(default=0)
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percent: int = Field(default=0, ge=0, le=100)
    error: Optional[str] = None


class URLReaderJobResult(BaseModel):
    """
    Complete result of a URL reader job.

    This is what gets stored and sent to callback URL.
    """
    # Job identification
    job_id: str
    request_id: int
    status: URLReaderJobStatus

    # Timing
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[int] = None

    # Results
    total_urls: int = Field(..., description="Total URLs submitted")
    successful_count: int = Field(default=0, description="Successfully processed URLs")
    failed_count: int = Field(default=0, description="Failed URLs")
    results: List[URLProcessResult] = Field(default_factory=list, description="Per-URL results")

    # Error (if entire job failed)
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "url_reader_1234_abc123",
                "request_id": 1234,
                "status": "completed",
                "created_at": "2024-01-15T10:00:00Z",
                "completed_at": "2024-01-15T10:02:00Z",
                "processing_time_ms": 120000,
                "total_urls": 2,
                "successful_count": 2,
                "failed_count": 0,
                "results": [
                    {
                        "url": "https://example.com/news/1",
                        "status": "success",
                        "content_type": "article",
                        "summary": "Tesla stock rises...",
                        "source_language": "en",
                        "target_language": "vi",
                        "processing_time_ms": 15000
                    }
                ]
            }
        }


class URLReaderQueueStats(BaseModel):
    """Statistics about the URL reader job queue."""
    queue_length: int = Field(..., description="Total jobs in queue")
    pending_count: int = Field(..., description="Jobs waiting to be processed")
    processing_count: int = Field(..., description="Jobs currently being processed")
    total_urls_pending: int = Field(default=0, description="Total URLs pending across all jobs")
