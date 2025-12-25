import os
import sys
import signal
import asyncio
import logging

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("scheduler_worker")


# =============================================================================
# SCHEDULER WORKER
# =============================================================================

class SchedulerWorker:
    """
    Standalone scheduler worker
    
    Chạy các background jobs:
    - prefetch_stock_data
    - prefetch_crypto_data
    - prefetch_etf_data
    - prefetch_gainers_data
    - prefetch_losers_data
    - prefetch_actives_data
    - prefetch_default_screener_data
    """
    
    def __init__(self):
        self.scheduler = self._create_scheduler()
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Metrics
        self.jobs_executed = 0
        self.jobs_failed = 0
    
    def _create_scheduler(self):
        """Create APScheduler with same config as main app"""
        
        # Reuse existing scheduler config
        from src.scheduler.scheduler_config import create_scheduler
        scheduler = create_scheduler()
        
        # Add event listeners
        scheduler.add_listener(self._on_job_executed, EVENT_JOB_EXECUTED)
        scheduler.add_listener(self._on_job_error, EVENT_JOB_ERROR)
        scheduler.add_listener(self._on_job_missed, EVENT_JOB_MISSED)
        
        return scheduler
    
    def _on_job_executed(self, event):
        self.jobs_executed += 1
        logger.debug(f"Job completed: {event.job_id}")
    
    def _on_job_error(self, event):
        self.jobs_failed += 1
        logger.error(f"Job failed: {event.job_id} - {event.exception}")
    
    def _on_job_missed(self, event):
        logger.warning(f"Job missed: {event.job_id}")
    
    def register_jobs(self):
        """Register all background jobs (same as main.py)"""
        
        from src.scheduler.scheduler_config import JobConfig
        from src.scheduler.get_data_scheduler import (
            prefetch_stock_data,
            prefetch_etf_data,
            prefetch_crypto_data,
            prefetch_gainers_data,
            prefetch_losers_data,
            prefetch_actives_data,
            prefetch_default_screener_data
        )
        
        job_settings = JobConfig.JOB_SETTINGS
        
        # Same jobs as main.py
        jobs = [
            ("prefetch_stocks", prefetch_stock_data, job_settings['prefetch_stocks']),
            ("prefetch_etfs", prefetch_etf_data, job_settings['prefetch_etfs']),
            ("prefetch_crypto", prefetch_crypto_data, job_settings['prefetch_crypto']),
            ("prefetch_gainers", prefetch_gainers_data, job_settings['prefetch_gainers']),
            ("prefetch_losers", prefetch_losers_data, job_settings['prefetch_losers']),
            ("prefetch_actives", prefetch_actives_data, job_settings['prefetch_actives']),
            ("prefetch_screener", prefetch_default_screener_data, job_settings['prefetch_screener']),
        ]
        
        for job_id, func, settings_dict in jobs:
            interval = settings_dict.get('interval_seconds', 60)
            job_kwargs = {k: v for k, v in settings_dict.items() if k != 'interval_seconds'}
            
            self.scheduler.add_job(
                func,
                IntervalTrigger(seconds=interval),
                id=job_id,
                **job_kwargs
            )
            logger.info(f"Registered job: {job_id} (every {interval}s)")
        
        logger.info(f"Total {len(jobs)} jobs registered")
    
    async def start(self):
        """Start the scheduler worker"""
        
        logger.info("=" * 60)
        logger.info("SCHEDULER WORKER STARTING")
        logger.info(f"PID: {os.getpid()}")
        logger.info("=" * 60)
        
        self._running = True
        
        # Register jobs
        self.register_jobs()
        
        # Start scheduler
        self.scheduler.start()
        logger.info(f"Scheduler started with {len(self.scheduler.get_jobs())} jobs")
        
        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: self._shutdown_event.set())
        
        # Health check loop
        health_task = asyncio.create_task(self._health_loop())
        
        try:
            await self._shutdown_event.wait()
        finally:
            health_task.cancel()
            await self.stop()
    
    async def _health_loop(self):
        """Log health status periodically"""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(60)  # Every 60 seconds
            logger.info(
                f"[scheduler-worker] Health: jobs_executed={self.jobs_executed}, "
                f"jobs_failed={self.jobs_failed}, "
                f"scheduled={len(self.scheduler.get_jobs())}"
            )
    
    async def stop(self):
        """Stop the scheduler"""
        logger.info("=" * 60)
        logger.info("SCHEDULER WORKER STOPPING")
        logger.info("=" * 60)
        
        try:
            self.scheduler.shutdown(wait=True)
        except Exception as e:
            logger.warning(f"Shutdown error: {e}")
        
        self._running = False
        logger.info(f"Final stats: executed={self.jobs_executed}, failed={self.jobs_failed}")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Main entry point"""
    
    # Log environment
    scheduler_mode = os.getenv("SCHEDULER_MODE", "embedded")
    scheduler_process = os.getenv("SCHEDULER_PROCESS", "false")
    
    logger.info(f"SCHEDULER_MODE={scheduler_mode}")
    logger.info(f"SCHEDULER_PROCESS={scheduler_process}")
    
    if scheduler_mode != "standalone":
        logger.warning(f"Expected SCHEDULER_MODE=standalone, got '{scheduler_mode}'")
    
    # Start worker
    worker = SchedulerWorker()
    await worker.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker interrupted")
    except Exception as e:
        logger.error(f"Worker crashed: {e}", exc_info=True)
        sys.exit(1)