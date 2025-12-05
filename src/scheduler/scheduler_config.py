from typing import Dict, List
from collections import defaultdict
from dataclasses import dataclass, field

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.memory import MemoryJobStore
import logging

logger = logging.getLogger(__name__)


def create_scheduler() -> AsyncIOScheduler:
    """
    Create scheduler with optimized configuration for handling long-running jobs
    
    Key improvements:
    - Increased max_instances per job
    - Added misfire_grace_time for delayed jobs
    - Configured coalesce to prevent job pileup
    - Optimized executor settings
    """
    
    # Job stores configuration
    jobstores = {
        'default': MemoryJobStore()
    }
    
    # Executors configuration - allow more concurrent jobs
    executors = {
        'default': AsyncIOExecutor() 
    }
    
    # Job defaults - critical for preventing "max instances" issue
    job_defaults = {
        'coalesce': True,           # Combine multiple pending executions into one
        'max_instances': 3,         # Allow up to 3 instances per job (was 1 by default)
        'misfire_grace_time': 30,   # Allow jobs to run even if 30s late
        'replace_existing': True    # Replace job if already exists
    }
    
    scheduler = AsyncIOScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
        timezone='UTC'
    )
    
    return scheduler


# Job configuration constants - increase intervals to reduce load
class JobConfig:
    """Configuration for different job types"""
    
    # Data prefetch jobs - increased from 20s to reduce overlapping
    PREFETCH_INTERVAL_SECONDS = 60  # Was 20s - now 60s
    
    # Discovery jobs (gainers, losers, actives)
    DISCOVERY_INTERVAL_SECONDS = 60  # Was 20s - now 60s
    
    # Screener jobs
    SCREENER_INTERVAL_SECONDS = 60  # Was 20s - now 90s (heavier operation)
    
    # Twitter scraping jobs
    TWITTER_INTERVAL_HOURS = 8
    
    # Specific job settings
    JOB_SETTINGS = {
        'prefetch_stocks': {
            'interval_seconds': 60,
            'max_instances': 2,
            'misfire_grace_time': 60,
            'coalesce': True
        },
        'prefetch_etfs': {
            'interval_seconds': 60,
            'max_instances': 2,
            'misfire_grace_time': 60,
            'coalesce': True,
            'jitter': 5                 # Delay 5s after stocks
        },
        'prefetch_crypto': {
            'interval_seconds': 60,
            'max_instances': 2,
            'misfire_grace_time': 30,
            'coalesce': True,
            'jitter': 10                # Delay 10s after stocks
        },
        'prefetch_gainers': {
            'interval_seconds': 60,
            'max_instances': 3,         # Allow more instances for lighter jobs
            'misfire_grace_time': 60,
            'coalesce': True,
            'jitter': 15                # Delay 15s
        },
        'prefetch_losers': {
            'interval_seconds': 60,
            'max_instances': 3,
            'misfire_grace_time': 90,
            'coalesce': True,
            'jitter': 20                # Delay 20s
        },
        'prefetch_actives': {
            'interval_seconds': 60,
            'max_instances': 3,
            'misfire_grace_time': 90,
            'coalesce': True,
            'jitter': 25                # Delay 25s
        },
        'prefetch_screener': {
            'interval_seconds': 90,     # Longer interval for heavy operation
            'max_instances': 2,
            'misfire_grace_time': 120,
            'coalesce': True,
            'jitter': 30                # Delay 30s
        }
    }

    @classmethod
    def get_job_config(cls, job_name: str) -> dict:
        """Helper to get config for a specific job"""
        return cls.JOB_SETTINGS.get(job_name, {
            'interval_seconds': 60,
            'max_instances': 1,
            'misfire_grace_time': 30,
            'coalesce': True,
            'jitter': 0
        })