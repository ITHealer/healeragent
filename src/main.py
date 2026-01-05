import os
import logging
import uvicorn
import secrets
import subprocess
import sys
from typing import Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from apscheduler.triggers.interval import IntervalTrigger

from src.middleware.rate_limiter import RateLimitMiddleware, RateLimitConfig

from src.utils.config import settings
from src.utils.constants import CODEMIND_LLM
from src.app import IncludeAPIRouter, logger_instance
from src.utils.config_loader import ConfigReaderInstance
from src.jobs.symbol_directory_monthly_sync import (
    start_symbol_directory_monthly_sync,
    run_symbol_directory_sync_now
)

from dotenv import load_dotenv
load_dotenv()

logger = logger_instance.get_logger(__name__)

# Read configuration from YAML file
api_config = ConfigReaderInstance.yaml.read_config_from_file(settings.API_CONFIG_FILENAME)
logging_config = ConfigReaderInstance.yaml.read_config_from_file(settings.LOG_CONFIG_FILENAME)

# Generate a security key (used to encrypt the session).
secret_key = secrets.token_urlsafe(32)

def env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")

def get_scheduler_mode() -> str:
    """
    Get scheduler mode from environment
    
    Modes:
    - embedded: Scheduler runs in same process as API (development)
    - standalone: Scheduler runs in separate process (production)
    - disabled: No scheduler (API-only instance)
    """
    return os.getenv("SCHEDULER_MODE", "embedded").lower()


def should_run_scheduler_in_api() -> bool:
    """Determine if API process should run scheduler"""
    mode = get_scheduler_mode()
    
    if mode == "disabled":
        return False
    
    if mode == "standalone":
        # In standalone mode, API doesn't run scheduler
        return False
    
    # Embedded mode - API runs scheduler
    return True

# lifespan (app lifecycle management, default is None).
def get_application(lifespan: Any = None):
    IS_PROD = settings.ENV_STATE == "prod"

    _app = FastAPI(lifespan=lifespan,
                   title=api_config.get('API_NAME'),
                   description=api_config.get('API_DESCRIPTION'),
                   version=api_config.get('API_VERSION'),
                   debug=api_config.get('API_DEBUG_MODE'),
                   # Disable docs & openapi when in production
                    docs_url=None if IS_PROD else "/docs",
                    redoc_url=None if IS_PROD else "/redoc",
                    openapi_url=None if IS_PROD else "/openapi.json",
                   )
    
    _app.include_router(IncludeAPIRouter())

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _app.add_middleware(SessionMiddleware, secret_key=secret_key)

    # Add rate limiting middleware (production-ready)
    # Configurable via environment variables
    rate_limit_enabled = env_bool("RATE_LIMIT_ENABLED", default=IS_PROD)
    if rate_limit_enabled:
        rate_config = RateLimitConfig(
            default_limit=int(os.getenv("RATE_LIMIT_DEFAULT", "100")),
            chat_limit=int(os.getenv("RATE_LIMIT_CHAT", "30")),
            stream_limit=int(os.getenv("RATE_LIMIT_STREAM", "20")),
            burst_limit=int(os.getenv("RATE_LIMIT_BURST", "10")),
            enabled=True,
            use_redis=True,
        )
        _app.add_middleware(RateLimitMiddleware, config=rate_config)
        logger.info(f"[RATE_LIMIT] Enabled: default={rate_config.default_limit}/min, chat={rate_config.chat_limit}/min")

    return _app

# Manage the lifecycle of asynchronous applications.
# Perform actions when the application starts and shuts down
@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Application lifecycle manager"""

    # Track resources for cleanup
    # consumer_process: Optional[subprocess.Popen] = None
    scheduler = None

    # ----------------------------------------------------
    # PHASE 1: STARTUP LOGIC (ALL code BEFORE the single 'yield')
    # ----------------------------------------------------
    logger.info("Starting application...")
    logger.info(f"Environment: {settings.ENV_STATE}")
    logger.info(f"Scheduler Mode: {get_scheduler_mode()}")
    
    # -------------------------------------------------------------------------
    # 1. Initialize Tool Registry
    # -------------------------------------------------------------------------
    from src.agents.tools import initialize_registry
    try:    
        registry = initialize_registry()
        summary = registry.get_summary()
        
        logger.info("[TOOL REGISTRY] Initialized successfully")
        logger.info(f"Total tools: {summary['total_tools']}")
        logger.info(f"Categories: {list(summary['categories'].keys())}")
        for cat, count in summary['categories'].items():
            logger.info(f"  - {cat}: {count} tools")
        
    except Exception as e:
        logger.error(f"Failed to initialize tool registry: {e}")
        logger.warning("Application will continue with legacy tools only")
    
    # -------------------------------------------------------------------------
    # 1.5 Initialize Symbol Cache (for asset disambiguation)
    # -------------------------------------------------------------------------
    try:
        from src.services.asset import initialize_symbol_cache
        symbol_cache = await initialize_symbol_cache()
        stats = symbol_cache.get_statistics()
        logger.info(
            f"[SYMBOL CACHE] Initialized: "
            f"crypto={stats['crypto_count']}, "
            f"stock={stats['stock_count']}, "
            f"ambiguous={stats['ambiguous_count']}"
        )
    except Exception as e:
        logger.warning(f"[SYMBOL CACHE] Failed to initialize: {e}")
        logger.warning("Symbol disambiguation will use defaults")

    # -------------------------------------------------------------------------
    # 1.6 Pre-warm UnifiedClassifier (avoid cold start on first request)
    # -------------------------------------------------------------------------
    try:
        from src.agents.classification import get_unified_classifier
        classifier = get_unified_classifier()
        logger.info(
            f"[CLASSIFIER] Pre-warmed: "
            f"model={classifier.model_name}, "
            f"vision={classifier.vision_model_name or 'None'}"
        )
    except Exception as e:
        logger.warning(f"[CLASSIFIER] Pre-warm failed: {e}")
        
    # -------------------------------------------------------------------------
    # 2. Start Consumer Manager Process
    # -------------------------------------------------------------------------
    # logger.info("Starting consumer manager process...")
    # try:
    #     consumer_process = subprocess.Popen([sys.executable, "-m", "src.run_consumers"])
    #     logger.info(f"Consumer manager process started with PID: {consumer_process.pid}")
    # except Exception as e:
    #     logger.error(f"Failed to start consumer manager process: {e}", exc_info=True)

    logger.info(CODEMIND_LLM)
    logger.info('event=app-startup')

    # -------------------------------------------------------------------------
    # 3. Configure CUDA/PyTorch
    # -------------------------------------------------------------------------
    try:
        import torch
        logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            torch.set_num_threads(4)
        else:
            logger.warning("CUDA not available. Using CPU for model inference.")
    except Exception as e:
        logger.error(f"Error checking CUDA: {str(e)}")

    # -------------------------------------------------------------------------
    # 4. Load Local Models
    # -------------------------------------------------------------------------
    try:
        from src.helpers.model_manager import model_manager
        
        logger.info("Start loading default models...")
        results = await model_manager.load_default_models()
        
        for model, success in results.items():
            if success:
                logger.info(f"Model {model} loaded successfully")
            else:
                logger.warning(f"Could not load model {model}")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

    # -------------------------------------------------------------------------
    # 5. Configure Scheduler (only in embedded mode)
    # -------------------------------------------------------------------------
    if should_run_scheduler_in_api():
        logger.info("Configuring background jobs (embedded mode)...")
        try:
            from src.scheduler.scheduler_config import create_scheduler, JobConfig
            from src.scheduler.get_data_scheduler import (
                prefetch_stock_data,
                prefetch_etf_data,
                prefetch_crypto_data,
                prefetch_gainers_data,
                prefetch_losers_data,
                prefetch_actives_data,
                prefetch_default_screener_data
            )
            
            scheduler = create_scheduler()
            job_settings = JobConfig.JOB_SETTINGS
            
            # Define jobs
            jobs = [
                ("prefetch_stocks", prefetch_stock_data, job_settings['prefetch_stocks']),
                ("prefetch_etfs", prefetch_etf_data, job_settings['prefetch_etfs']),
                ("prefetch_crypto", prefetch_crypto_data, job_settings['prefetch_crypto']),
                ("prefetch_gainers", prefetch_gainers_data, job_settings['prefetch_gainers']),
                ("prefetch_losers", prefetch_losers_data, job_settings['prefetch_losers']),
                ("prefetch_actives", prefetch_actives_data, job_settings['prefetch_actives']),
                ("prefetch_screener", prefetch_default_screener_data, job_settings['prefetch_screener']),
            ]
            
            # Register jobs
            for job_id, func, settings_dict in jobs:
                # Extract interval_seconds, use rest as kwargs
                interval = settings_dict.get('interval_seconds', 60)
                job_kwargs = {k: v for k, v in settings_dict.items() if k != 'interval_seconds'}
                
                scheduler.add_job(
                    func,
                    IntervalTrigger(seconds=interval),
                    id=job_id,
                    **job_kwargs
                )
            
            scheduler.start()
            logger.info(f"Scheduler started with {len(scheduler.get_jobs())} jobs (embedded mode)")
            
        except Exception as e:
            logger.error(f"Scheduler initialization failed: {e}")
            scheduler = None
    else:
        logger.info(f"Scheduler NOT started in API process (mode: {get_scheduler_mode()})")
        logger.info("Background jobs will be handled by scheduler_worker service")

    # Run symbol directory sync
    # await run_symbol_directory_sync_now()
    
    # # Start monthly job
    # start_symbol_directory_monthly_sync(sync_day=1, sync_hour=2, sync_minute=0)

    yield # Application START accepting requests HERE
    

    # ----------------------------------------------------
    # PHASE 2: SHUTDOWN LOGIC (ALL code AFTER the single 'yield')
    # ----------------------------------------------------

    # 1. Shutdown Scheduler
    logger.info("Shutting down scheduler...")
    if scheduler is not None:
        try:
            scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped")
        except Exception as e:
            logger.warning(f"Scheduler shutdown error: {e}")

    # 2. Terminate Consumer Process
    # if consumer_process:
    #     logger.info(f"Terminating consumer manager process with PID: {consumer_process.pid}...")
    #     consumer_process.terminate()
    #     try:
    #         # Wait for the process to terminate
    #         consumer_process.wait(timeout=5)
    #         logger.info("Consumer manager process terminated successfully.")
    #     except subprocess.TimeoutExpired:
    #         logger.warning("Consumer manager process did not terminate in time. Killing it.")
    #         consumer_process.kill()

    # 3. Close Redis LLM client
    logger.info("Closing Redis LLM client...")
    try:
        from src.helpers.redis_cache import close_redis_llm_client
        await close_redis_llm_client()
    except Exception as e:
        logger.warning(f"Redis LLM client close error: {e}")

    logger.info('event=app-shutdown message="All connections are closed."')


# Create FastAPI application object
app = get_application(lifespan=app_lifespan)


@app.get('/')
async def docs_redirect():
    if settings.ENV_STATE == "prod":
        return {"message": "Service running"}
    return RedirectResponse(url='/docs')

@app.get('/health')
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "scheduler-worker",
        "scheduler_mode": get_scheduler_mode()
    }

class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find('/health') == -1


if __name__ == '__main__':
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config['formatters']['access']['fmt'] = logging_config.get('UVICORN_FORMATTER')
    log_config['formatters']['default']['fmt'] = logging_config.get('UVICORN_FORMATTER')
    log_config['formatters']['access']['datefmt'] = logging_config.get('DATE_FORMATTER')
    log_config['formatters']['default']['datefmt'] = logging_config.get('DATE_FORMATTER')

    RELOAD = env_bool("UVICORN_RELOAD", False)
    
    uvicorn.run('src.main:app',
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        log_config=log_config,
        workers=settings.UVICORN_WORKERS,
        reload=RELOAD,
        reload_excludes=["models/*,cache/*,.venv/*"]
    )