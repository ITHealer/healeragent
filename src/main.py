import os
import logging
import uvicorn
import secrets
import subprocess
import sys
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from apscheduler.triggers.interval import IntervalTrigger
from src.scheduler.scheduler_config import create_scheduler, JobConfig
from src.scheduler.get_data_scheduler import (
    prefetch_actives_data, 
    prefetch_crypto_data, 
    prefetch_default_screener_data, 
    prefetch_etf_data, 
    prefetch_gainers_data, 
    prefetch_losers_data, 
    prefetch_stock_data
)
# from src.scheduler.twitter_jobs import scheduled_twitter_scrape_job

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

scheduler = create_scheduler()

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

    return _app

# Manage the lifecycle of asynchronous applications.
# Perform actions when the application starts and shuts down
@asynccontextmanager
async def app_lifespan(app: FastAPI):

    # ----------------------------------------------------
    # PHASE 1: STARTUP LOGIC (ALL code BEFORE the single 'yield')
    # ----------------------------------------------------
    logger.info("Starting application...")
    
    # ====================================================================
    # Initialize Tool Registry
    # ====================================================================
    from src.agents.tools import initialize_registry
    try:    
        registry = initialize_registry()
        summary = registry.get_summary()
        
        logger.info("=" * 60)
        logger.info("TOOL REGISTRY INITIALIZED")
        logger.info(f"Total tools: {summary['total_tools']}")
        logger.info(f"Categories: {list(summary['categories'].keys())}")
        for cat, count in summary['categories'].items():
            logger.info(f"  - {cat}: {count} tools")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to initialize tool registry: {e}")
        logger.warning("Application will continue with legacy tools only")
    
    # 0. ============= START CONSUMER MANAGER PROCESS =============
    logger.info("Starting consumer manager process...")
    consumer_process = None
    try:
        # Start the consumer manager in a separate process
        # We use sys.executable to ensure we're using the same python interpreter
        consumer_process = subprocess.Popen([sys.executable, "-m", "src.run_consumers"])
        logger.info(f"Consumer manager process started with PID: {consumer_process.pid}")
    except Exception as e:
        logger.error(f"Failed to start consumer manager process: {e}", exc_info=True)

    logger.info(CODEMIND_LLM)
    logger.info('event=app-startup')

    # 1. ============= CONFIGURE PYTORCH/CUDA ENV =============
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

    # 2. ============= LOAD LOCAL MODEL =============
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

    # 3. ============= JOB SCHEDULING =============
    logger.info("Configuring background jobs with custom settings...")
    job_settings = JobConfig.JOB_SETTINGS

    scheduler.add_job(
        prefetch_stock_data,
        IntervalTrigger(seconds=job_settings['prefetch_stocks']['interval_seconds']),
        id="prefetch_stocks",
        **{k: v for k, v in job_settings['prefetch_stocks'].items() if k != 'interval_seconds'}
    )

    scheduler.add_job(
        prefetch_etf_data,
        IntervalTrigger(seconds=job_settings['prefetch_etfs']['interval_seconds']),
        id="prefetch_etfs",
        **{k: v for k, v in job_settings['prefetch_etfs'].items() if k != 'interval_seconds'}
    )
    
    scheduler.add_job(
        prefetch_crypto_data,
        IntervalTrigger(seconds=job_settings['prefetch_crypto']['interval_seconds']),
        id="prefetch_crypto",
        **{k: v for k, v in job_settings['prefetch_crypto'].items() if k != 'interval_seconds'}
    )
    
    # Discovery jobs
    scheduler.add_job(
        prefetch_gainers_data,
        IntervalTrigger(seconds=job_settings['prefetch_gainers']['interval_seconds']),
        id="prefetch_gainers",
        **{k: v for k, v in job_settings['prefetch_gainers'].items() if k != 'interval_seconds'}
    )
    
    scheduler.add_job(
        prefetch_losers_data,
        IntervalTrigger(seconds=job_settings['prefetch_losers']['interval_seconds']),
        id="prefetch_losers",
        **{k: v for k, v in job_settings['prefetch_losers'].items() if k != 'interval_seconds'}
    )
    
    scheduler.add_job(
        prefetch_actives_data,
        IntervalTrigger(seconds=job_settings['prefetch_actives']['interval_seconds']),
        id="prefetch_actives",
        **{k: v for k, v in job_settings['prefetch_actives'].items() if k != 'interval_seconds'}
    )
    
    # Screener job
    scheduler.add_job(
        prefetch_default_screener_data,
        IntervalTrigger(seconds=job_settings['prefetch_screener']['interval_seconds']),
        id="prefetch_screener",
        **{k: v for k, v in job_settings['prefetch_screener'].items() if k != 'interval_seconds'}
    )

    # Twitter scraping job
    # scheduler.add_job(
    #     scheduled_twitter_scrape_job,
    #     IntervalTrigger(hours=JobConfig.TWITTER_INTERVAL_HOURS),
    #     id="scheduled_twitter_scrape",
    #     replace_existing=True,
    #     max_instances=1,
    #     misfire_grace_time=3600
    # )

    # ============= START SCHEDULER =============
    scheduler.start()

    logger.info(f"Configured {len(scheduler.get_jobs())} background jobs and started scheduler")

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
    scheduler.shutdown()

    # 2. Terminate Consumer Process
    if consumer_process:
        logger.info(f"Terminating consumer manager process with PID: {consumer_process.pid}...")
        consumer_process.terminate()
        try:
            # Wait for the process to terminate
            consumer_process.wait(timeout=5)
            logger.info("Consumer manager process terminated successfully.")
        except subprocess.TimeoutExpired:
            logger.warning("Consumer manager process did not terminate in time. Killing it.")
            consumer_process.kill()

    logger.info('event=app-shutdown message="All connections are closed."')


# Create FastAPI application object
app = get_application(lifespan=app_lifespan)


@app.get('/')
async def docs_redirect():
    if settings.ENV_STATE == "prod":
        return {"message": "Service running"}
    return RedirectResponse(url='/docs')


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
