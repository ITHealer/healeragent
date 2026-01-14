"""
Callback Service
================

Handles webhook callbacks to BE .NET when tasks complete.

Features:
- Retry with exponential backoff (3 attempts, 2s/4s/8s delays)
- Timeout handling (30s default)
- Error logging with request details

Callback API:
    POST /api/v1/user-task/submit-generation-result
    Body: { "requestId": 1792, "content": "JSON string of result" }

Usage:
    service = CallbackService()
    success = await service.send_callback(
        callback_url="https://api.example.com/api/v1/user-task/submit-generation-result",
        request_id=1792,
        result=task_result,
    )
"""

import json
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

from src.news_aggregator.schemas.task import (
    TaskResult,
    CallbackPayload,
    CallbackStatus,
)
from src.news_aggregator.services.markdown_formatter import (
    format_task_result_markdown,
    format_error_markdown,
)

logger = logging.getLogger(__name__)


class CallbackService:
    """
    Service for sending webhook callbacks to BE .NET.

    Implements retry logic with exponential backoff for reliability.
    """

    DEFAULT_TIMEOUT = 30.0
    MAX_RETRIES = 3
    BASE_DELAY = 2.0  # Seconds

    def __init__(
        self,
        timeout: float = None,
        max_retries: int = None,
        base_delay: float = None,
    ):
        """
        Initialize callback service.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
        """
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.max_retries = max_retries or self.MAX_RETRIES
        self.base_delay = base_delay or self.BASE_DELAY
        self._client: Optional[httpx.AsyncClient] = None
        self.logger = logger

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _build_payload(
        self,
        request_id: int,
        result: TaskResult,
        use_markdown: bool = True,
    ) -> Dict[str, Any]:
        """
        Build callback payload for BE .NET.

        Args:
            request_id: Original request ID from BE
            result: Task result with analysis
            use_markdown: If True, format as markdown; else JSON

        Returns:
            Dict with requestId and content (markdown or JSON string)
        """
        if use_markdown:
            content = format_task_result_markdown(result)
        else:
            content = result.model_dump_json(exclude_none=True)

        return {
            "requestId": request_id,
            "content": content,
        }

    async def send_callback(
        self,
        callback_url: str,
        request_id: int,
        result: TaskResult,
    ) -> CallbackStatus:
        """
        Send callback to BE .NET with retry logic.

        Args:
            callback_url: URL to POST callback to
            request_id: Original request ID
            result: Task result to send

        Returns:
            CallbackStatus with success/failure info
        """
        if not callback_url:
            self.logger.warning(f"[Callback] No callback URL for request {request_id}")
            return CallbackStatus(
                success=False,
                request_id=request_id,
                error="No callback URL provided",
                attempts=0,
            )

        payload = self._build_payload(request_id, result)
        client = await self._get_client()

        last_error = None
        attempts = 0

        for attempt in range(self.max_retries):
            attempts = attempt + 1
            try:
                self.logger.info(
                    f"[Callback] Sending to {callback_url} | "
                    f"request_id={request_id} | attempt={attempts}"
                )

                response = await client.post(
                    callback_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Request-ID": str(request_id),
                    },
                )

                # Check for success (2xx status)
                if 200 <= response.status_code < 300:
                    self.logger.info(
                        f"[Callback] Success | request_id={request_id} | "
                        f"status={response.status_code} | attempts={attempts}"
                    )
                    return CallbackStatus(
                        success=True,
                        request_id=request_id,
                        status_code=response.status_code,
                        attempts=attempts,
                        sent_at=datetime.utcnow(),
                    )

                # Non-2xx response
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                self.logger.warning(
                    f"[Callback] Non-success response | request_id={request_id} | "
                    f"status={response.status_code} | attempt={attempts}"
                )

            except httpx.TimeoutException as e:
                last_error = f"Timeout: {str(e)}"
                self.logger.warning(
                    f"[Callback] Timeout | request_id={request_id} | attempt={attempts}"
                )

            except httpx.RequestError as e:
                last_error = f"Request error: {str(e)}"
                self.logger.warning(
                    f"[Callback] Request error | request_id={request_id} | "
                    f"attempt={attempts} | error={e}"
                )

            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                self.logger.error(
                    f"[Callback] Unexpected error | request_id={request_id} | "
                    f"attempt={attempts} | error={e}"
                )

            # Exponential backoff before retry
            if attempt < self.max_retries - 1:
                delay = self.base_delay * (2 ** attempt)
                self.logger.debug(f"[Callback] Waiting {delay}s before retry")
                await asyncio.sleep(delay)

        # All retries failed
        self.logger.error(
            f"[Callback] All retries failed | request_id={request_id} | "
            f"attempts={attempts} | last_error={last_error}"
        )

        return CallbackStatus(
            success=False,
            request_id=request_id,
            error=last_error,
            attempts=attempts,
        )

    async def send_error_callback(
        self,
        callback_url: str,
        request_id: int,
        error_message: str,
        job_id: Optional[str] = None,
    ) -> CallbackStatus:
        """
        Send error callback when task processing fails.

        Args:
            callback_url: URL to POST callback to
            request_id: Original request ID
            error_message: Error description
            job_id: Optional job ID for reference

        Returns:
            CallbackStatus with success/failure info
        """
        if not callback_url:
            self.logger.warning(f"[Callback] No callback URL for error {request_id}")
            return CallbackStatus(
                success=False,
                request_id=request_id,
                error="No callback URL provided",
                attempts=0,
            )

        # Format error as markdown
        content = format_error_markdown(
            request_id=request_id,
            error_message=error_message,
            job_id=job_id,
        )

        payload = {
            "requestId": request_id,
            "content": content,
        }

        client = await self._get_client()
        last_error = None
        attempts = 0

        for attempt in range(self.max_retries):
            attempts = attempt + 1
            try:
                response = await client.post(
                    callback_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Request-ID": str(request_id),
                    },
                )

                if 200 <= response.status_code < 300:
                    self.logger.info(
                        f"[Callback] Error callback sent | request_id={request_id}"
                    )
                    return CallbackStatus(
                        success=True,
                        request_id=request_id,
                        status_code=response.status_code,
                        attempts=attempts,
                        sent_at=datetime.utcnow(),
                    )

                last_error = f"HTTP {response.status_code}"

            except Exception as e:
                last_error = str(e)
                self.logger.warning(
                    f"[Callback] Error callback failed | attempt={attempts} | {e}"
                )

            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.base_delay * (2 ** attempt))

        return CallbackStatus(
            success=False,
            request_id=request_id,
            error=last_error,
            attempts=attempts,
        )


# Singleton instance
_callback_service: Optional[CallbackService] = None


def get_callback_service() -> CallbackService:
    """Get singleton callback service instance."""
    global _callback_service
    if _callback_service is None:
        _callback_service = CallbackService()
    return _callback_service


async def close_callback_service():
    """Close callback service connection."""
    global _callback_service
    if _callback_service:
        await _callback_service.close()
        _callback_service = None
