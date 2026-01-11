"""
Production Logging System
========================

A comprehensive, production-ready logging system with:
- Category-based log files (app, error, api, agent, performance)
- Daily rotation with 15-day retention
- Request ID tracing via contextvars
- JSON format for production, colored text for development
- Automatic ERROR/CRITICAL mirroring to error/ directory

Usage:
------
```python
from src.core.logging import setup_logging, get_logger, RequestContext

# Initialize at app startup (once)
setup_logging()

# Get category-specific logger
logger = get_logger("agent")  # Logs to logs/agent/
logger = get_logger("api")    # Logs to logs/api/
logger = get_logger()         # Default: logs/app/

# In FastAPI middleware, set request context
with RequestContext(request_id="abc-123"):
    logger.info("Processing request")  # Includes [abc-123] in log
```

Directory Structure:
-------------------
logs/
├── app/            # General application logs
├── error/          # ERROR + CRITICAL only (quick debugging)
├── api/            # API request/response logs
├── agent/          # Agent + Tools + Memory flow
└── performance/    # Performance metrics

Each file: {category}_YYYY-MM-DD.log
Retention: 15 days (auto-cleanup)
"""

from src.core.logging.config import setup_logging, get_logger, shutdown_logging
from src.core.logging.context import (
    RequestContext,
    get_request_id,
    set_request_id,
    clear_request_id,
)
from src.core.logging.middleware import LoggingMiddleware

__all__ = [
    "setup_logging",
    "get_logger",
    "shutdown_logging",
    "RequestContext",
    "get_request_id",
    "set_request_id",
    "clear_request_id",
    "LoggingMiddleware",
]
