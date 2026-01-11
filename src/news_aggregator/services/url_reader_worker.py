"""
URL Reader Job Worker Service
=============================

Background worker that processes URL reader jobs from the Redis queue.

Pipeline:
1. Pop job from Redis priority queue (BZPOPMAX)
2. Process each URL using ContentProcessor
3. Store result and update status
4. Send callback to BE .NET

Features:
- Graceful shutdown handling
- Per-URL error handling (job continues even if some URLs fail)
- Progress tracking
- Configurable concurrent workers
- Non-blocking URL processing

Usage:
    worker = URLReaderWorker(num_workers=2)
    await worker.start()
    # ... on shutdown ...
    await worker.stop()

Environment Variables:
    URL_READER_WORKER_COUNT: Number of concurrent workers (default: 2)
    URL_READER_TIMEOUT_PER_URL: Timeout per URL in seconds (default: 60)
    URL_READER_CALLBACK_TIMEOUT: Timeout for callback (default: 60)
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import List, Optional

from src.news_aggregator.schemas.url_reader_job import (
    URLReaderJobRequest,
    URLReaderJobResult,
    URLReaderJobStatus,
    URLProcessResult,
)
from src.news_aggregator.services.url_reader_queue import (
    URLReaderQueueService,
    get_url_reader_queue,
    close_url_reader_queue,
)
from src.news_aggregator.services.callback_service import (
    CallbackService,
    get_callback_service,
)

logger = logging.getLogger(__name__)


class URLReaderWorker:
    """
    Background worker for processing URL reader jobs.

    Pulls jobs from Redis queue and processes each URL
    to extract and summarize content.
    """

    # Default configuration
    DEFAULT_NUM_WORKERS = 2
    POP_TIMEOUT = 30.0  # Seconds to wait for jobs

    # Timeouts (can be overridden via env vars)
    TIMEOUT_PER_URL = int(os.getenv("URL_READER_TIMEOUT_PER_URL", "60"))
    CALLBACK_TIMEOUT = int(os.getenv("URL_READER_CALLBACK_TIMEOUT", "60"))

    def __init__(
        self,
        num_workers: int = None,
        pop_timeout: float = None,
    ):
        """
        Initialize URL reader worker.

        Args:
            num_workers: Number of concurrent worker tasks
            pop_timeout: Timeout for queue pop operation
        """
        self.num_workers = num_workers or int(
            os.getenv("URL_READER_WORKER_COUNT", str(self.DEFAULT_NUM_WORKERS))
        )
        self.pop_timeout = pop_timeout or self.POP_TIMEOUT
        self.logger = logger

        # Services (initialized in start())
        self._queue: Optional[URLReaderQueueService] = None
        self._callback_service: Optional[CallbackService] = None
        self._content_processor = None

        # Worker state
        self._running = False
        self._worker_tasks = []
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the worker and begin processing jobs."""
        if self._running:
            self.logger.warning("[URLReaderWorker] Already running")
            return

        self.logger.info(f"[URLReaderWorker] Starting with {self.num_workers} workers...")
        self.logger.info(
            f"[URLReaderWorker] Timeouts - per_url:{self.TIMEOUT_PER_URL}s, "
            f"callback:{self.CALLBACK_TIMEOUT}s"
        )

        # Initialize services
        await self._initialize_services()

        # Set running flag
        self._running = True
        self._shutdown_event.clear()

        # Start worker tasks
        for i in range(self.num_workers):
            task = asyncio.create_task(
                self._worker_loop(worker_id=i),
                name=f"url-reader-worker-{i}",
            )
            self._worker_tasks.append(task)

        self.logger.info(f"[URLReaderWorker] Started {len(self._worker_tasks)} workers")

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        if not self._running:
            return

        self.logger.info("[URLReaderWorker] Stopping...")
        self._running = False
        self._shutdown_event.set()

        # Wait for workers to complete current jobs
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            self._worker_tasks.clear()

        # Close services
        await self._close_services()

        self.logger.info("[URLReaderWorker] Stopped")

    async def _initialize_services(self) -> None:
        """Initialize all required services."""
        try:
            # Job queue
            self._queue = await get_url_reader_queue()
            self.logger.info("[URLReaderWorker] Job queue initialized")

            # Callback service
            self._callback_service = get_callback_service()
            self.logger.info("[URLReaderWorker] Callback service initialized")

        except Exception as e:
            self.logger.error(f"[URLReaderWorker] Service initialization failed: {e}")
            raise

    async def _close_services(self) -> None:
        """Close all services."""
        if self._callback_service:
            await self._callback_service.close()

        await close_url_reader_queue()

    def _get_content_processor(self, model_name: str, provider_type: str):
        """Get or create content processor with caching."""
        try:
            from src.media.handlers.content_processor_manager import processor_manager
            from src.providers.provider_factory import ProviderType

            # Get API key based on provider
            api_key = None
            if provider_type.lower() == ProviderType.OLLAMA:
                api_key = None
            elif provider_type.lower() == ProviderType.OPENAI:
                api_key = os.getenv("OPENAI_API_KEY")
            elif provider_type.lower() == ProviderType.GEMINI:
                api_key = os.getenv("GEMINI_API_KEY")

            processor = processor_manager.get_processor(
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key
            )
            return processor

        except Exception as e:
            self.logger.error(f"[URLReaderWorker] Failed to get processor: {e}")
            return None

    async def _worker_loop(self, worker_id: int) -> None:
        """
        Main worker loop - pulls and processes jobs.

        Args:
            worker_id: Worker identifier for logging
        """
        self.logger.info(f"[URLWorker-{worker_id}] Starting loop")

        while self._running and not self._shutdown_event.is_set():
            try:
                # Try to pop a job
                result = await self._queue.pop_job(timeout=self.pop_timeout)

                if result is None:
                    # No job available, continue waiting
                    continue

                job_id, request = result
                self.logger.info(
                    f"[URLWorker-{worker_id}] Processing job {job_id} | "
                    f"urls={len(request.urls)} | request_id={request.request_id}"
                )

                # Process the job
                await self._process_job(worker_id, job_id, request)

            except asyncio.CancelledError:
                self.logger.info(f"[URLWorker-{worker_id}] Cancelled")
                break
            except Exception as e:
                self.logger.error(f"[URLWorker-{worker_id}] Loop error: {e}")
                # Brief pause before retrying
                await asyncio.sleep(1)

        self.logger.info(f"[URLWorker-{worker_id}] Stopped")

    async def _process_job(
        self,
        worker_id: int,
        job_id: str,
        request: URLReaderJobRequest,
    ) -> None:
        """
        Process a single URL reader job.

        Args:
            worker_id: Worker identifier
            job_id: Job ID
            request: Job request
        """
        start_time = time.time()
        created_at = datetime.utcnow()

        results: List[URLProcessResult] = []
        successful_count = 0
        failed_count = 0

        try:
            # Get content processor
            processor = self._get_content_processor(
                model_name=request.model_name,
                provider_type=request.provider_type,
            )

            if not processor:
                raise RuntimeError("Failed to initialize content processor")

            # Process each URL
            total_urls = len(request.urls)
            for i, url in enumerate(request.urls):
                url_start_time = time.time()

                self.logger.info(
                    f"[URLWorker-{worker_id}] [{job_id}] Processing URL {i+1}/{total_urls}: {url[:60]}..."
                )

                try:
                    # Process single URL with timeout
                    url_result = await asyncio.wait_for(
                        self._process_single_url(
                            processor=processor,
                            url=url,
                            target_language=request.target_language,
                            include_original=request.include_original,
                        ),
                        timeout=self.TIMEOUT_PER_URL,
                    )

                    url_processing_time = int((time.time() - url_start_time) * 1000)
                    url_result.processing_time_ms = url_processing_time

                    if url_result.status == "success":
                        successful_count += 1
                    else:
                        failed_count += 1

                    results.append(url_result)

                except asyncio.TimeoutError:
                    failed_count += 1
                    results.append(URLProcessResult(
                        url=url,
                        status="error",
                        error=f"URL processing timeout after {self.TIMEOUT_PER_URL}s",
                        processing_time_ms=int((time.time() - url_start_time) * 1000),
                    ))
                    self.logger.warning(
                        f"[URLWorker-{worker_id}] [{job_id}] URL timeout: {url[:60]}"
                    )

                except Exception as e:
                    failed_count += 1
                    results.append(URLProcessResult(
                        url=url,
                        status="error",
                        error=str(e),
                        processing_time_ms=int((time.time() - url_start_time) * 1000),
                    ))
                    self.logger.error(
                        f"[URLWorker-{worker_id}] [{job_id}] URL error: {url[:60]} - {e}"
                    )

                # Update progress
                await self._queue.update_progress(job_id, i + 1)

            # Build job result
            processing_time_ms = int((time.time() - start_time) * 1000)

            job_result = URLReaderJobResult(
                job_id=job_id,
                request_id=request.request_id,
                status=URLReaderJobStatus.COMPLETED,
                created_at=created_at,
                started_at=created_at,
                completed_at=datetime.utcnow(),
                processing_time_ms=processing_time_ms,
                total_urls=total_urls,
                successful_count=successful_count,
                failed_count=failed_count,
                results=results,
            )

            # Update status to completed
            await self._queue.update_status(
                job_id=job_id,
                status=URLReaderJobStatus.COMPLETED,
                processed_urls=total_urls,
                result=job_result,
            )

            self.logger.info(
                f"[URLWorker-{worker_id}] [{job_id}] Job COMPLETED | "
                f"success={successful_count}, failed={failed_count} | "
                f"time={processing_time_ms}ms"
            )

            # Send callback
            if request.callback_url:
                await self._send_callback(
                    worker_id=worker_id,
                    job_id=job_id,
                    request=request,
                    result=job_result,
                )

        except Exception as e:
            # Handle job failure
            processing_time_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)

            self.logger.error(
                f"[URLWorker-{worker_id}] [{job_id}] FAILED: {error_msg} | "
                f"time={processing_time_ms}ms",
                exc_info=True,
            )

            # Create failed result
            job_result = URLReaderJobResult(
                job_id=job_id,
                request_id=request.request_id,
                status=URLReaderJobStatus.FAILED,
                created_at=created_at,
                started_at=created_at,
                completed_at=datetime.utcnow(),
                processing_time_ms=processing_time_ms,
                total_urls=len(request.urls),
                successful_count=successful_count,
                failed_count=failed_count + (len(request.urls) - len(results)),
                results=results,
                error=error_msg,
            )

            await self._queue.update_status(
                job_id=job_id,
                status=URLReaderJobStatus.FAILED,
                error=error_msg,
                result=job_result,
            )

            # Send error callback
            if request.callback_url:
                await self._send_callback(
                    worker_id=worker_id,
                    job_id=job_id,
                    request=request,
                    result=job_result,
                )

    async def _process_single_url(
        self,
        processor,
        url: str,
        target_language: Optional[str],
        include_original: bool,
    ) -> URLProcessResult:
        """
        Process a single URL using ContentProcessor.

        Args:
            processor: ContentProcessor instance
            url: URL to process
            target_language: Target language for summary
            include_original: Whether to include original content

        Returns:
            URLProcessResult with processing results
        """
        try:
            # Call processor
            result = await processor.process_url(
                url=url,
                target_language=target_language,
                print_progress=False,
            )

            # Handle error from processor
            if result.get("status") == "error":
                return URLProcessResult(
                    url=url,
                    status="error",
                    error=result.get("error", "Processing failed"),
                )

            # Build success result
            content_type = result.get("type", "unknown")

            url_result = URLProcessResult(
                url=url,
                status="success",
                content_type=content_type,
                summary=result.get("summary"),
                source_language=result.get("source_language"),
                target_language=result.get("target_language"),
                translation_needed=result.get("translation_needed", False),
                translation=result.get("translation"),
                metadata={},
            )

            # Add title if available
            if content_type == "video" and result.get("video_info"):
                url_result.title = result["video_info"].get("title")
                url_result.metadata["video_info"] = result.get("video_info", {})
                if include_original:
                    url_result.original_content = result.get("transcript")
            elif content_type == "article":
                url_result.metadata["original_length"] = result.get("original_length", 0)
                url_result.metadata["chunks_created"] = result.get("chunks_created", 1)
                if include_original:
                    url_result.original_content = result.get("original_text")

            return url_result

        except Exception as e:
            return URLProcessResult(
                url=url,
                status="error",
                error=str(e),
            )

    async def _send_callback(
        self,
        worker_id: int,
        job_id: str,
        request: URLReaderJobRequest,
        result: URLReaderJobResult,
    ) -> None:
        """Send callback to BE .NET."""
        try:
            self.logger.info(
                f"[URLWorker-{worker_id}] [{job_id}] Sending callback to {request.callback_url}"
            )

            from src.news_aggregator.schemas.task import CallbackStatus

            # Build payload
            import json
            payload = {
                "requestId": request.request_id,
                "jobType": "url_reader",
                "content": result.model_dump_json(exclude_none=True),
            }

            # Send callback with retry
            import httpx
            async with httpx.AsyncClient(timeout=self.CALLBACK_TIMEOUT) as client:
                for attempt in range(3):
                    try:
                        response = await client.post(
                            request.callback_url,
                            json=payload,
                            headers={
                                "Content-Type": "application/json",
                                "X-Request-ID": str(request.request_id),
                                "X-Job-Type": "url_reader",
                            },
                        )

                        if 200 <= response.status_code < 300:
                            self.logger.info(
                                f"[URLWorker-{worker_id}] [{job_id}] Callback SUCCESS | "
                                f"status={response.status_code}"
                            )
                            return
                        else:
                            self.logger.warning(
                                f"[URLWorker-{worker_id}] [{job_id}] Callback failed: "
                                f"HTTP {response.status_code}"
                            )

                    except Exception as e:
                        self.logger.warning(
                            f"[URLWorker-{worker_id}] [{job_id}] Callback attempt {attempt+1} failed: {e}"
                        )

                    # Backoff before retry
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)

                self.logger.error(
                    f"[URLWorker-{worker_id}] [{job_id}] Callback FAILED after 3 attempts"
                )

        except Exception as e:
            self.logger.error(
                f"[URLWorker-{worker_id}] [{job_id}] Callback error: {e}"
            )


# Singleton worker instance
_url_reader_worker: Optional[URLReaderWorker] = None


async def get_url_reader_worker(num_workers: int = 2) -> URLReaderWorker:
    """Get or create singleton URL reader worker."""
    global _url_reader_worker
    if _url_reader_worker is None:
        _url_reader_worker = URLReaderWorker(num_workers=num_workers)
    return _url_reader_worker


async def start_url_reader_worker(num_workers: int = 2) -> URLReaderWorker:
    """Start the URL reader worker."""
    worker = await get_url_reader_worker(num_workers)
    await worker.start()
    return worker


async def stop_url_reader_worker() -> None:
    """Stop the URL reader worker."""
    global _url_reader_worker
    if _url_reader_worker:
        await _url_reader_worker.stop()
        _url_reader_worker = None
