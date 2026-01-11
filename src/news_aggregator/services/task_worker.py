"""
Task Worker Service
===================

Background worker that processes news analysis tasks from the Redis queue.

Pipeline:
1. Pop task from Redis priority queue (BZPOPMAX)
2. Fetch news for symbols from FMP
3. Extract full article content via Tavily Extract
4. Fetch market data (quotes + historical)
5. Run AI analysis with citations
6. Send callback to BE .NET

Features:
- Graceful shutdown handling
- Per-task error handling with callback
- Configurable concurrent workers
- Task status tracking

Usage:
    worker = TaskWorker(num_workers=2)
    await worker.start()
    # ... on shutdown ...
    await worker.stop()
"""

import time
import asyncio
import logging
import signal
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.news_aggregator.schemas.task import (
    TaskRequest,
    TaskResult,
    TaskStatus,
    TaskType,
    SymbolAnalysis,
    ArticleContent,
)
from src.news_aggregator.schemas.fmp_news import NewsCategory
from src.news_aggregator.schemas.unified_news import UnifiedNewsItem

from src.news_aggregator.providers.fmp_provider import FMPNewsProvider
from src.news_aggregator.services.task_queue import (
    TaskQueueService,
    get_task_queue,
    close_task_queue,
)
from src.news_aggregator.services.content_extractor import (
    ContentExtractorService,
    get_content_extractor,
)
from src.news_aggregator.services.market_data import (
    MarketDataService,
    get_market_data_service,
)
from src.news_aggregator.services.ai_analyzer import (
    AIAnalyzerService,
    get_ai_analyzer,
)
from src.news_aggregator.services.callback_service import (
    CallbackService,
    get_callback_service,
)

logger = logging.getLogger(__name__)


class TaskWorker:
    """
    Background worker for processing news analysis tasks.

    Pulls tasks from Redis queue and executes the full analysis pipeline.
    """

    DEFAULT_NUM_WORKERS = 2
    POP_TIMEOUT = 30.0  # Seconds to wait for tasks
    MAX_NEWS_PER_SYMBOL = 10  # Max news articles per symbol

    def __init__(
        self,
        num_workers: int = None,
        pop_timeout: float = None,
    ):
        """
        Initialize task worker.

        Args:
            num_workers: Number of concurrent worker tasks
            pop_timeout: Timeout for queue pop operation
        """
        self.num_workers = num_workers or self.DEFAULT_NUM_WORKERS
        self.pop_timeout = pop_timeout or self.POP_TIMEOUT
        self.logger = logger

        # Services (initialized in start())
        self._queue: Optional[TaskQueueService] = None
        self._fmp_provider: Optional[FMPNewsProvider] = None
        self._content_extractor: Optional[ContentExtractorService] = None
        self._market_data: Optional[MarketDataService] = None
        self._ai_analyzer: Optional[AIAnalyzerService] = None
        self._callback_service: Optional[CallbackService] = None

        # Worker state
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the worker and begin processing tasks."""
        if self._running:
            self.logger.warning("[TaskWorker] Already running")
            return

        self.logger.info(f"[TaskWorker] Starting with {self.num_workers} workers...")

        # Initialize services
        await self._initialize_services()

        # Set running flag
        self._running = True
        self._shutdown_event.clear()

        # Start worker tasks
        for i in range(self.num_workers):
            task = asyncio.create_task(
                self._worker_loop(worker_id=i),
                name=f"task-worker-{i}",
            )
            self._worker_tasks.append(task)

        self.logger.info(f"[TaskWorker] Started {len(self._worker_tasks)} workers")

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        if not self._running:
            return

        self.logger.info("[TaskWorker] Stopping...")
        self._running = False
        self._shutdown_event.set()

        # Wait for workers to complete current tasks
        if self._worker_tasks:
            # Give workers time to finish current task
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            self._worker_tasks.clear()

        # Close services
        await self._close_services()

        self.logger.info("[TaskWorker] Stopped")

    async def _initialize_services(self) -> None:
        """Initialize all required services."""
        try:
            # Task queue
            self._queue = await get_task_queue()
            self.logger.info("[TaskWorker] Task queue initialized")

            # FMP provider for news
            try:
                self._fmp_provider = FMPNewsProvider()
                self.logger.info("[TaskWorker] FMP provider initialized")
            except ValueError as e:
                self.logger.warning(f"[TaskWorker] FMP provider not available: {e}")
                self._fmp_provider = None

            # Content extractor
            self._content_extractor = get_content_extractor()
            self.logger.info("[TaskWorker] Content extractor initialized")

            # Market data
            self._market_data = get_market_data_service()
            self.logger.info("[TaskWorker] Market data service initialized")

            # AI analyzer
            self._ai_analyzer = get_ai_analyzer()
            self.logger.info("[TaskWorker] AI analyzer initialized")

            # Callback service
            self._callback_service = get_callback_service()
            self.logger.info("[TaskWorker] Callback service initialized")

        except Exception as e:
            self.logger.error(f"[TaskWorker] Service initialization failed: {e}")
            raise

    async def _close_services(self) -> None:
        """Close all services."""
        if self._fmp_provider:
            await self._fmp_provider.close()

        if self._content_extractor:
            await self._content_extractor.close()

        if self._market_data:
            await self._market_data.close()

        if self._callback_service:
            await self._callback_service.close()

        await close_task_queue()

    async def _worker_loop(self, worker_id: int) -> None:
        """
        Main worker loop - pulls and processes tasks.

        Args:
            worker_id: Worker identifier for logging
        """
        self.logger.info(f"[Worker-{worker_id}] Starting loop")

        while self._running and not self._shutdown_event.is_set():
            try:
                # Try to pop a task
                result = await self._queue.pop_task(timeout=self.pop_timeout)

                if result is None:
                    # No task available, continue waiting
                    continue

                job_id, request = result
                self.logger.info(
                    f"[Worker-{worker_id}] Processing task {job_id} | "
                    f"symbols={request.symbols}"
                )

                # Process the task
                await self._process_task(worker_id, job_id, request)

            except asyncio.CancelledError:
                self.logger.info(f"[Worker-{worker_id}] Cancelled")
                break
            except Exception as e:
                self.logger.error(f"[Worker-{worker_id}] Loop error: {e}")
                # Brief pause before retrying
                await asyncio.sleep(1)

        self.logger.info(f"[Worker-{worker_id}] Stopped")

    async def _process_task(
        self,
        worker_id: int,
        job_id: str,
        request: TaskRequest,
    ) -> None:
        """
        Process a single task through the full pipeline.

        Args:
            worker_id: Worker identifier
            job_id: Task job ID
            request: Task request with symbols and options
        """
        start_time = time.time()

        try:
            # Phase 1: Fetch news for symbols
            news_by_symbol = await self._fetch_news_for_symbols(request.symbols)

            if not any(news_by_symbol.values()):
                self.logger.warning(f"[Worker-{worker_id}] No news found for task {job_id}")
                # Still proceed to create result with empty analyses

            # Phase 2: Extract full article content
            articles_by_symbol = await self._extract_articles(news_by_symbol)

            # Phase 3: Fetch market data
            market_data = {}
            if self._market_data:
                market_data = await self._market_data.get_market_data(
                    symbols=request.symbols,
                    include_historical=True,
                )

            # Phase 4: Run AI analysis
            analysis_result = await self._run_analysis(
                articles_by_symbol=articles_by_symbol,
                market_data=market_data,
                target_language=request.target_language,
            )

            # Build task result
            processing_time_ms = int((time.time() - start_time) * 1000)

            task_result = TaskResult(
                job_id=job_id,
                request_id=request.request_id,
                success=True,
                analyses=analysis_result.get("analyses", []),
                overall_sentiment=analysis_result.get("overall_sentiment"),
                key_themes=analysis_result.get("key_themes", []),
                summary=analysis_result.get("summary"),
                processing_time_ms=processing_time_ms,
            )

            # Update status to completed
            await self._queue.update_status(
                job_id=job_id,
                status=TaskStatus.COMPLETED,
                result=task_result,
            )

            self.logger.info(
                f"[Worker-{worker_id}] Task {job_id} completed | "
                f"{len(task_result.analyses)} analyses | {processing_time_ms}ms"
            )

            # Phase 5: Send callback
            if request.callback_url:
                callback_status = await self._callback_service.send_callback(
                    callback_url=request.callback_url,
                    request_id=request.request_id,
                    result=task_result,
                )

                if not callback_status.success:
                    self.logger.error(
                        f"[Worker-{worker_id}] Callback failed for {job_id}: "
                        f"{callback_status.error}"
                    )

        except Exception as e:
            # Handle task failure
            processing_time_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)

            self.logger.error(
                f"[Worker-{worker_id}] Task {job_id} failed: {error_msg}",
                exc_info=True,
            )

            # Update status to failed
            await self._queue.update_status(
                job_id=job_id,
                status=TaskStatus.FAILED,
                error=error_msg,
            )

            # Send error callback
            if request.callback_url and self._callback_service:
                await self._callback_service.send_error_callback(
                    callback_url=request.callback_url,
                    request_id=request.request_id,
                    error_message=error_msg,
                    job_id=job_id,
                )

    async def _fetch_news_for_symbols(
        self,
        symbols: List[str],
    ) -> Dict[str, List[UnifiedNewsItem]]:
        """
        Fetch news articles for each symbol.

        Args:
            symbols: List of symbols to fetch news for

        Returns:
            Dict mapping symbol to list of news items
        """
        if not self._fmp_provider:
            self.logger.warning("[TaskWorker] FMP provider not available")
            return {symbol: [] for symbol in symbols}

        result: Dict[str, List[UnifiedNewsItem]] = {symbol: [] for symbol in symbols}

        try:
            # Determine categories based on symbols
            # Crypto symbols vs stock symbols
            crypto_symbols = set()
            stock_symbols = set()

            for symbol in symbols:
                symbol_upper = symbol.upper()
                if any(crypto in symbol_upper for crypto in ["BTC", "ETH", "DOGE", "SOL", "XRP"]):
                    crypto_symbols.add(symbol)
                else:
                    stock_symbols.add(symbol)

            # Fetch from appropriate categories
            categories = [NewsCategory.STOCK, NewsCategory.GENERAL]
            if crypto_symbols:
                categories.append(NewsCategory.CRYPTO)

            all_news = await self._fmp_provider.fetch_news(
                categories=categories,
                page=0,
                limit=50,  # Fetch more to filter by symbol
            )

            # Group by symbol
            for item in all_news:
                for symbol in symbols:
                    symbol_upper = symbol.upper()
                    # Check if symbol is in the news item
                    if (
                        symbol_upper in [s.upper() for s in item.symbols]
                        or symbol_upper in item.title.upper()
                        or (item.content and symbol_upper in item.content.upper())
                    ):
                        if len(result[symbol]) < self.MAX_NEWS_PER_SYMBOL:
                            result[symbol].append(item)

            # Log results
            for symbol, news_list in result.items():
                self.logger.info(f"[TaskWorker] Found {len(news_list)} news for {symbol}")

        except Exception as e:
            self.logger.error(f"[TaskWorker] News fetch error: {e}")

        return result

    async def _extract_articles(
        self,
        news_by_symbol: Dict[str, List[UnifiedNewsItem]],
    ) -> Dict[str, List[ArticleContent]]:
        """
        Extract full article content for each symbol's news.

        Args:
            news_by_symbol: Dict mapping symbol to news items

        Returns:
            Dict mapping symbol to extracted article contents
        """
        if not self._content_extractor:
            self.logger.warning("[TaskWorker] Content extractor not available")
            return {symbol: [] for symbol in news_by_symbol.keys()}

        result: Dict[str, List[ArticleContent]] = {}

        for symbol, news_items in news_by_symbol.items():
            if not news_items:
                result[symbol] = []
                continue

            try:
                # Convert UnifiedNewsItem to dict format expected by extractor
                news_dicts = [
                    {
                        "title": item.title,
                        "url": item.url,
                        "text": item.content or "",
                        "publishedDate": item.published_at.isoformat() if item.published_at else None,
                        "site": item.source_site,
                        "image": item.image_url,
                    }
                    for item in news_items
                ]

                # Extract full content
                articles = await self._content_extractor.extract_articles(
                    news_items=news_dicts,
                    query=symbol,
                )
                result[symbol] = articles

                self.logger.info(f"[TaskWorker] Extracted {len(articles)} articles for {symbol}")

            except Exception as e:
                self.logger.error(f"[TaskWorker] Extract error for {symbol}: {e}")
                result[symbol] = []

        return result

    async def _run_analysis(
        self,
        articles_by_symbol: Dict[str, List[ArticleContent]],
        market_data: Dict[str, Any],
        target_language: str,
    ) -> Dict[str, Any]:
        """
        Run AI analysis on extracted articles.

        Args:
            articles_by_symbol: Dict mapping symbol to articles
            market_data: Market data for symbols
            target_language: Target language for analysis

        Returns:
            Analysis result with per-symbol analyses
        """
        if not self._ai_analyzer:
            self.logger.warning("[TaskWorker] AI analyzer not available")
            return {"analyses": [], "key_themes": [], "summary": None}

        try:
            result = await self._ai_analyzer.analyze(
                articles_by_symbol=articles_by_symbol,
                market_data=market_data,
                target_language=target_language,
            )
            return result

        except Exception as e:
            self.logger.error(f"[TaskWorker] Analysis error: {e}")
            return {"analyses": [], "key_themes": [], "summary": None, "error": str(e)}


# Singleton worker instance
_task_worker: Optional[TaskWorker] = None


async def get_task_worker(num_workers: int = 2) -> TaskWorker:
    """Get or create singleton task worker."""
    global _task_worker
    if _task_worker is None:
        _task_worker = TaskWorker(num_workers=num_workers)
    return _task_worker


async def start_task_worker(num_workers: int = 2) -> TaskWorker:
    """Start the task worker."""
    worker = await get_task_worker(num_workers)
    await worker.start()
    return worker


async def stop_task_worker() -> None:
    """Stop the task worker."""
    global _task_worker
    if _task_worker:
        await _task_worker.stop()
        _task_worker = None
