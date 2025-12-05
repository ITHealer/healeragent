import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from typing import Optional

from src.utils.config import settings
from src.utils.logger.custom_logging import LoggerMixin
from src.database import get_postgres_db
from src.services.data_providers.fmp.symbol_directory_service import get_symbol_directory_service

logger = LoggerMixin().logger


# ============================================================================
# JOB CONFIGURATION
# ============================================================================

# Monthly sync: Ngày 1 hàng tháng lúc 2 AM UTC
DEFAULT_SYNC_DAY = 1  # Day of month
DEFAULT_SYNC_HOUR = 2
DEFAULT_SYNC_MINUTE = 0


class SymbolDirectoryMonthlySync:
    """
    Monthly background job để sync symbol directory từ FMP vào PostgreSQL
    
    Features:
    - Monthly schedule (configurable)
    - Full sync: stocks, crypto, symbol_changes, delisted
    - Error handling và logging
    - Job status tracking
    """
    
    def __init__(
        self,
        sync_day: int = DEFAULT_SYNC_DAY,
        sync_hour: int = DEFAULT_SYNC_HOUR,
        sync_minute: int = DEFAULT_SYNC_MINUTE
    ):
        """
        Initialize monthly sync job
        
        Args:
            sync_day: Day of month to run (1-31)
            sync_hour: Hour to run (0-23, UTC)
            sync_minute: Minute to run (0-59)
        """
        self.sync_day = sync_day
        self.sync_hour = sync_hour
        self.sync_minute = sync_minute
        self.scheduler = AsyncIOScheduler()
        self.job_id = "symbol_directory_monthly_sync"
    
    async def run_sync(self):
        """
        Execute full monthly sync
        
        Syncs:
        1. Stock list (~15k stocks)
        2. Crypto list (~5k cryptos)
        3. Symbol changes (~1k)
        4. Delisted companies (~10k)
        
        Total: ~4 FMP API calls, ~30-60 seconds
        """
        logger.info("=" * 80)
        logger.info("SYMBOL DIRECTORY MONTHLY SYNC STARTED")
        logger.info(f"Timestamp: {datetime.utcnow().isoformat()}")
        logger.info("=" * 80)
        
        start_time = datetime.utcnow()
        
        try:
            # Get DB session
            session = get_postgres_db().get_session()
            
            try:
                # Get service với DB session
                service = get_symbol_directory_service(session)
                
                # Run full sync
                logger.info("Starting full sync from FMP to PostgreSQL...")
                results = await service.sync_all()
                
                # Log results
                logger.info("-" * 80)
                logger.info("SYNC RESULTS:")
                
                for result in results:
                    status = "✓ SUCCESS" if result.success else "✗ FAILED"
                    logger.info(
                        f"  {status} | {result.job_type:20} | "
                        f"Fetched: {result.records_fetched:6} | "
                        f"Saved: {result.records_cached:6} | "
                        f"Duration: {result.duration_seconds:.2f}s"
                    )
                    
                    if result.error_message:
                        logger.error(f"    Error: {result.error_message}")
                
                # Summary
                total_records = sum(r.records_cached for r in results)
                successful_jobs = sum(1 for r in results if r.success)
                total_jobs = len(results)
                total_fmp_calls = total_jobs
                
                logger.info("-" * 80)
                logger.info("SYNC SUMMARY:")
                logger.info(f"  Jobs: {successful_jobs}/{total_jobs} successful")
                logger.info(f"  Total records synced: {total_records}")
                logger.info(f"  FMP API calls: {total_fmp_calls}")
                
                duration = (datetime.utcnow() - start_time).total_seconds()
                logger.info(f"  Total duration: {duration:.2f}s")
                
                if successful_jobs == total_jobs:
                    logger.info("✓ MONTHLY SYNC COMPLETED SUCCESSFULLY")
                else:
                    logger.warning(f"⚠ SYNC COMPLETED WITH {total_jobs - successful_jobs} FAILURES")
                
            finally:
                session.close()
        
        except Exception as e:
            logger.error(f"MONTHLY SYNC FAILED WITH ERROR: {str(e)}", exc_info=True)
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Failed after {duration:.2f}s")
        
        finally:
            logger.info("=" * 80)
            logger.info(f"Next sync scheduled on day {self.sync_day} at {self.sync_hour:02d}:{self.sync_minute:02d} UTC")
            logger.info("=" * 80)
    
    def start(self):
        """
        Start the monthly scheduled job
        
        Cron: Day X của mỗi tháng lúc H:M UTC
        """
        # Create cron trigger for monthly sync
        trigger = CronTrigger(
            day=self.sync_day,
            hour=self.sync_hour,
            minute=self.sync_minute,
            timezone="UTC"
        )
        
        # Add job to scheduler
        self.scheduler.add_job(
            self.run_sync,
            trigger=trigger,
            id=self.job_id,
            name="Symbol Directory Monthly Sync",
            replace_existing=True
        )
        
        # Start scheduler
        self.scheduler.start()
        
        logger.info(
            f"Symbol Directory monthly sync job scheduled: "
            f"Day {self.sync_day} at {self.sync_hour:02d}:{self.sync_minute:02d} UTC"
        )
    
    def stop(self):
        """Stop the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("Symbol Directory monthly sync job stopped")
    
    async def run_now(self):
        """
        Run sync immediately (bypass schedule)
        
        Useful for:
        - Bootstrap lần đầu
        - Manual trigger
        - Testing
        """
        logger.info("Running immediate sync (not scheduled)")
        await self.run_sync()


# ============================================================================
# GLOBAL JOB INSTANCE
# ============================================================================
_monthly_sync_job_instance: Optional[SymbolDirectoryMonthlySync] = None


def get_monthly_sync_job(
    sync_day: int = DEFAULT_SYNC_DAY,
    sync_hour: int = DEFAULT_SYNC_HOUR,
    sync_minute: int = DEFAULT_SYNC_MINUTE
) -> SymbolDirectoryMonthlySync:
    """
    Get singleton instance của monthly sync job
    
    Args:
        sync_day: Day of month (1-31)
        sync_hour: Hour (0-23, UTC)
        sync_minute: Minute (0-59)
        
    Returns:
        SymbolDirectoryMonthlySync instance
    """
    global _monthly_sync_job_instance
    
    if _monthly_sync_job_instance is None:
        _monthly_sync_job_instance = SymbolDirectoryMonthlySync(
            sync_day=sync_day,
            sync_hour=sync_hour,
            sync_minute=sync_minute
        )
    
    return _monthly_sync_job_instance


def start_symbol_directory_monthly_sync(
    sync_day: int = DEFAULT_SYNC_DAY,
    sync_hour: int = DEFAULT_SYNC_HOUR,
    sync_minute: int = DEFAULT_SYNC_MINUTE
):
    """
    Start symbol directory monthly sync job
    
    Call trong application startup
    
    Args:
        sync_day: Day of month to run (1-31)
        sync_hour: Hour to run (0-23, UTC)
        sync_minute: Minute to run (0-59)
        
    Example:
        # In main.py startup event
        @app.on_event("startup")
        async def startup_event():
            # Start monthly sync (runs on day 1 at 2 AM UTC)
            start_symbol_directory_monthly_sync(
                sync_day=1,
                sync_hour=2,
                sync_minute=0
            )
    """
    job = get_monthly_sync_job(sync_day, sync_hour, sync_minute)
    job.start()
    logger.info("Symbol Directory monthly sync job started successfully")


def stop_symbol_directory_monthly_sync():
    """
    Stop symbol directory monthly sync job
    
    Call trong application shutdown
    
    Example:
        @app.on_event("shutdown")
        async def shutdown_event():
            stop_symbol_directory_monthly_sync()
    """
    global _monthly_sync_job_instance
    
    if _monthly_sync_job_instance:
        _monthly_sync_job_instance.stop()
        _monthly_sync_job_instance = None


async def run_symbol_directory_sync_now():
    """
    Run symbol directory sync immediately
    
    Useful for:
    - Bootstrap / first-time setup
    - Manual sync trigger
    - Testing
    
    Example:
        # Bootstrap sync trên first startup
        @app.on_event("startup")
        async def startup_event():
            # Run immediate sync first time
            await run_symbol_directory_sync_now()
            
            # Then start monthly scheduled job
            start_symbol_directory_monthly_sync()
    """
    job = get_monthly_sync_job()
    await job.run_now()


# ============================================================================
# CONFIGURATION FROM ENV
# ============================================================================

def get_sync_config_from_settings():
    """
    Get sync configuration từ settings/env vars
    
    Environment variables:
    - SYMBOL_DIRECTORY_SYNC_DAY (default: 1)
    - SYMBOL_DIRECTORY_SYNC_HOUR (default: 2)
    - SYMBOL_DIRECTORY_SYNC_MINUTE (default: 0)
    
    Returns:
        Tuple of (sync_day, sync_hour, sync_minute)
    """
    sync_day = getattr(settings, 'SYMBOL_DIRECTORY_SYNC_DAY', DEFAULT_SYNC_DAY)
    sync_hour = getattr(settings, 'SYMBOL_DIRECTORY_SYNC_HOUR', DEFAULT_SYNC_HOUR)
    sync_minute = getattr(settings, 'SYMBOL_DIRECTORY_SYNC_MINUTE', DEFAULT_SYNC_MINUTE)
    
    return sync_day, sync_hour, sync_minute


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
"""
INTEGRATION WITH FASTAPI APP:

# In src/main.py

from src.jobs.symbol_directory_monthly_sync import (
    start_symbol_directory_monthly_sync,
    stop_symbol_directory_monthly_sync,
    run_symbol_directory_sync_now,
    get_sync_config_from_settings
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting application...")
    
    # Get config from settings
    sync_day, sync_hour, sync_minute = get_sync_config_from_settings()
    
    # Option 1: Run bootstrap sync trên first startup (recommended)
    logger.info("Running bootstrap sync...")
    await run_symbol_directory_sync_now()
    
    # Option 2: Hoặc check xem DB đã có data chưa
    # from src.database import get_postgres_db
    # from src.repositories.symbol_directory_repository import SymbolDirectoryRepository
    # session = get_postgres_db().get_session()
    # repo = SymbolDirectoryRepository(session)
    # if repo.count_stocks() == 0:
    #     logger.info("DB empty, running bootstrap sync...")
    #     await run_symbol_directory_sync_now()
    # session.close()
    
    # Start monthly scheduled sync
    start_symbol_directory_monthly_sync(
        sync_day=sync_day,
        sync_hour=sync_hour,
        sync_minute=sync_minute
    )
    
    logger.info("Application startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application...")
    
    # Stop sync job
    stop_symbol_directory_monthly_sync()
    
    logger.info("Application shutdown complete")


MANUAL TRIGGER (for testing):

if __name__ == "__main__":
    # Run sync immediately
    asyncio.run(run_symbol_directory_sync_now())


CONFIGURATION IN .env:

# Symbol directory monthly sync configuration
SYMBOL_DIRECTORY_SYNC_DAY=1           # Day 1 of each month
SYMBOL_DIRECTORY_SYNC_HOUR=2          # 2 AM UTC
SYMBOL_DIRECTORY_SYNC_MINUTE=0        # :00 minutes
"""