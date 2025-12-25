import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from src.utils.logger.custom_logging import LoggerMixin


class SchedulerMode(str, Enum):
    """Scheduler execution modes"""
    
    # Run scheduler in same process as API (development)
    EMBEDDED = "embedded"
    
    # Run scheduler as separate process (production)
    STANDALONE = "standalone"
    
    # Scheduler disabled (for API-only instances)
    DISABLED = "disabled"


@dataclass
class SchedulerIsolationConfig:
    """Configuration for scheduler isolation"""
    
    # Mode
    mode: SchedulerMode = SchedulerMode.EMBEDDED
    
    # Standalone process settings
    worker_count: int = 1
    max_concurrent_jobs: int = 5
    health_check_interval: int = 30
    
    # IPC settings (for communication between API and scheduler)
    use_redis_queue: bool = False
    redis_queue_name: str = "scheduler_jobs"
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    @classmethod
    def from_env(cls) -> "SchedulerIsolationConfig":
        """Create config from environment variables"""
        
        mode_str = os.getenv("SCHEDULER_MODE", "embedded").lower()
        
        try:
            mode = SchedulerMode(mode_str)
        except ValueError:
            mode = SchedulerMode.EMBEDDED
        
        return cls(
            mode=mode,
            worker_count=int(os.getenv("SCHEDULER_WORKERS", "1")),
            max_concurrent_jobs=int(os.getenv("SCHEDULER_MAX_JOBS", "5")),
            health_check_interval=int(os.getenv("SCHEDULER_HEALTH_INTERVAL", "30")),
            use_redis_queue=os.getenv("SCHEDULER_USE_REDIS", "false").lower() == "true",
            redis_queue_name=os.getenv("SCHEDULER_REDIS_QUEUE", "scheduler_jobs"),
            enable_metrics=os.getenv("SCHEDULER_METRICS", "true").lower() == "true",
            metrics_port=int(os.getenv("SCHEDULER_METRICS_PORT", "9090"))
        )


# Global config instance
_config: Optional[SchedulerIsolationConfig] = None


def get_scheduler_config() -> SchedulerIsolationConfig:
    """Get scheduler isolation configuration"""
    global _config
    
    if _config is None:
        _config = SchedulerIsolationConfig.from_env()
    
    return _config


def get_scheduler_mode() -> SchedulerMode:
    """Get current scheduler mode"""
    return get_scheduler_config().mode


def should_run_scheduler() -> bool:
    """Check if scheduler should run in this process"""
    config = get_scheduler_config()
    
    # In embedded mode, always run scheduler
    if config.mode == SchedulerMode.EMBEDDED:
        return True
    
    # In standalone mode, check if this is the scheduler process
    if config.mode == SchedulerMode.STANDALONE:
        return os.getenv("SCHEDULER_PROCESS", "false").lower() == "true"
    
    # Disabled mode
    return False


def should_start_api_scheduler() -> bool:
    """Check if API process should start scheduler jobs"""
    config = get_scheduler_config()
    
    # Only in embedded mode does API run scheduler
    return config.mode == SchedulerMode.EMBEDDED


# ============================================================================
# Helper functions for main.py integration
# ============================================================================

logger = LoggerMixin().logger


def setup_scheduler_for_mode(scheduler):
    """
    Setup scheduler based on mode
    
    Call this in main.py lifespan:
        from src.scheduler.isolation import setup_scheduler_for_mode
        
        setup_scheduler_for_mode(scheduler)
    """
    config = get_scheduler_config()
    
    if config.mode == SchedulerMode.DISABLED:
        logger.info("Scheduler is DISABLED - no background jobs will run")
        return
    
    if config.mode == SchedulerMode.STANDALONE:
        logger.info(
            "Scheduler mode: STANDALONE - "
            "background jobs run in separate process"
        )
        # Don't start scheduler in API process
        return
    
    # Embedded mode - start scheduler
    logger.info("Scheduler mode: EMBEDDED - starting background jobs in API process")
    
    # Configure scheduler
    from src.scheduler.scheduler_config import JobConfig
    
    # Add jobs with isolation-aware settings
    _add_jobs_with_config(scheduler, config)
    
    scheduler.start()
    logger.info(f"Scheduler started with {len(scheduler.get_jobs())} jobs")


def _add_jobs_with_config(scheduler, config: SchedulerIsolationConfig):
    """Add jobs with isolation configuration"""
    
    from apscheduler.triggers.interval import IntervalTrigger
    from src.scheduler.get_data_scheduler import (
        prefetch_stock_data,
        prefetch_etf_data,
        prefetch_crypto_data,
        prefetch_gainers_data,
        prefetch_losers_data,
        prefetch_actives_data,
        prefetch_default_screener_data
    )
    
    jobs = [
        ("prefetch_stocks", prefetch_stock_data, 60),
        ("prefetch_etfs", prefetch_etf_data, 60),
        ("prefetch_crypto", prefetch_crypto_data, 60),
        ("prefetch_gainers", prefetch_gainers_data, 60),
        ("prefetch_losers", prefetch_losers_data, 60),
        ("prefetch_actives", prefetch_actives_data, 60),
        ("prefetch_screener", prefetch_default_screener_data, 90),
    ]
    
    for job_id, func, interval in jobs:
        scheduler.add_job(
            func,
            IntervalTrigger(seconds=interval),
            id=job_id,
            replace_existing=True,
            max_instances=2,
            misfire_grace_time=60,
            coalesce=True
        )