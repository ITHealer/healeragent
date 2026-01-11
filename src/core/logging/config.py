"""
Logging Configuration
====================

Main configuration module for the production logging system.

Environment Variables:
---------------------
- LOG_LEVEL: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- LOG_FORMAT: "json" for production, "text" for development (default)
- LOG_DIR: Base directory for log files (default: ./logs)
- LOG_RETENTION_DAYS: Days to keep log files (default: 15)
- LOG_CONSOLE: "true" to enable console output (default: true in dev)

Usage:
------
```python
from src.core.logging import setup_logging, get_logger

# Initialize at app startup (once)
setup_logging()

# Get loggers for different categories
app_logger = get_logger()           # → logs/app/
api_logger = get_logger("api")      # → logs/api/
agent_logger = get_logger("agent")  # → logs/agent/
perf_logger = get_logger("performance")  # → logs/performance/

# Use normally
agent_logger.info("Processing request")
agent_logger.error("Something failed", exc_info=True)
```
"""

import os
import sys
import logging
from typing import Dict, Optional

from src.core.logging.formatters import DevFormatter, JsonFormatter, FileFormatter
from src.core.logging.handlers import (
    CATEGORIES,
    ensure_log_directories,
    create_category_handler,
    create_error_handler,
    create_smart_routing_handler,
    cleanup_old_logs,
)


# Cache for configured loggers
_configured_loggers: Dict[str, logging.Logger] = {}
_logging_initialized = False

# Logger name to category mapping
LOGGER_CATEGORY_MAP = {
    # API category
    "api": "api",
    "uvicorn": "api",
    "uvicorn.access": "api",
    "uvicorn.error": "api",
    "fastapi": "api",
    # Agent category
    "agent": "agent",
    "unified_agent": "agent",
    "tool": "agent",
    "memory": "agent",
    "skill": "agent",
    "validation": "agent",
    "src.agents": "agent",
    # Performance category
    "performance": "performance",
    "perf": "performance",
    "metrics": "performance",
    # Default to app
}


def get_config() -> dict:
    """Get logging configuration from environment."""
    return {
        "level": os.environ.get("LOG_LEVEL", "INFO").upper(),
        "format": os.environ.get("LOG_FORMAT", "text").lower(),
        "log_dir": os.environ.get("LOG_DIR", "logs"),
        "retention_days": int(os.environ.get("LOG_RETENTION_DAYS", "15")),
        "console_enabled": os.environ.get("LOG_CONSOLE", "true").lower() == "true",
        "is_production": os.environ.get("ENV", "development").lower() == "production",
    }


def _get_category_for_logger(logger_name: str) -> str:
    """Determine the category for a logger name."""
    # Direct match
    if logger_name in LOGGER_CATEGORY_MAP:
        return LOGGER_CATEGORY_MAP[logger_name]

    # Prefix match
    for prefix, category in LOGGER_CATEGORY_MAP.items():
        if logger_name.startswith(prefix + ".") or logger_name.startswith(prefix):
            return category

    # Check for common patterns in logger name
    name_lower = logger_name.lower()
    if any(kw in name_lower for kw in ["agent", "tool", "memory", "skill", "unified"]):
        return "agent"
    if any(kw in name_lower for kw in ["api", "router", "endpoint", "http"]):
        return "api"
    if any(kw in name_lower for kw in ["perf", "metric", "timing", "duration"]):
        return "performance"

    # Default to app
    return "app"


def setup_logging(
    level: Optional[str] = None,
    use_json: Optional[bool] = None,
    console: Optional[bool] = None,
) -> None:
    """
    Initialize the logging system.

    Call this once at application startup.

    Args:
        level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_json: Override format (True for JSON, False for text)
        console: Override console output (True to enable)
    """
    global _logging_initialized

    if _logging_initialized:
        return

    config = get_config()

    # Apply overrides
    if level:
        config["level"] = level.upper()
    if use_json is not None:
        config["format"] = "json" if use_json else "text"
    if console is not None:
        config["console_enabled"] = console

    # Create log directories
    ensure_log_directories()

    # Set root logger level
    log_level = getattr(logging, config["level"], logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Allow all levels, handlers will filter

    # Clear existing handlers
    root_logger.handlers.clear()

    # Determine format
    use_json_format = config["format"] == "json" or config["is_production"]

    # Create console handler
    if config["console_enabled"]:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        if use_json_format:
            console_handler.setFormatter(JsonFormatter())
        else:
            console_handler.setFormatter(DevFormatter(use_colors=True))

        root_logger.addHandler(console_handler)

    # Create smart routing handler that auto-routes ALL logs to correct category
    # This means existing code using logging.getLogger(__name__) will work automatically
    smart_handler = create_smart_routing_handler(
        use_json=use_json_format,
        retention_days=config["retention_days"],
        level=logging.DEBUG,  # Capture all levels in files
    )
    root_logger.addHandler(smart_handler)

    # Create error handler (captures ERROR+ from all loggers)
    error_handler = create_error_handler(
        use_json=use_json_format,
        retention_days=config["retention_days"],
    )
    root_logger.addHandler(error_handler)

    # Configure third-party loggers to reduce noise
    _configure_third_party_loggers(log_level)

    # Run initial cleanup
    cleanup_old_logs(config["retention_days"])

    _logging_initialized = True

    # Log initialization
    root_logger.info(
        f"Logging initialized: level={config['level']}, "
        f"format={'json' if use_json_format else 'text'}, "
        f"dir={config['log_dir']}"
    )


def _configure_third_party_loggers(level: int) -> None:
    """Configure third-party library loggers to reduce noise."""
    noisy_loggers = [
        "httpx",
        "httpcore",
        "urllib3",
        "aiohttp",
        "asyncio",
        "websockets",
        "openai",
        "anthropic",
        "redis",
        "qdrant_client",
    ]

    for name in noisy_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(max(level, logging.WARNING))


def get_logger(
    name: Optional[str] = None,
    category: Optional[str] = None,
) -> logging.Logger:
    """
    Get a configured logger.

    The SmartRoutingHandler on root logger automatically routes logs to the
    correct category file based on logger name. The category parameter is
    optional and only used for explicit overrides.

    Args:
        name: Logger name (e.g., module name). If None, uses "app"
        category: Force a specific category prefix. If None, auto-detected from name

    Returns:
        Configured logger that writes to the appropriate category file

    Examples:
        get_logger("api.users")  → logs to api/
        get_logger("agent.tools")  → logs to agent/
        get_logger()  → logs to app/
        get_logger("mymodule", category="agent")  → logs to agent/ (via name prefix)
    """
    if not _logging_initialized:
        setup_logging()

    if name is None:
        name = "app"

    # If category is specified, prefix the name for correct routing
    # SmartRoutingHandler routes based on logger name keywords
    if category and category not in name.lower():
        name = f"{category}.{name}"

    # Check cache
    if name in _configured_loggers:
        return _configured_loggers[name]

    # Get or create logger - SmartRoutingHandler on root will auto-route
    logger = logging.getLogger(name)

    # Cache and return
    _configured_loggers[name] = logger
    return logger


def shutdown_logging() -> None:
    """
    Shutdown the logging system.

    Call this at application shutdown to ensure all logs are flushed.
    """
    global _logging_initialized

    logging.shutdown()
    _configured_loggers.clear()
    _logging_initialized = False


# Convenience function for getting common loggers
def get_api_logger(name: str = "api") -> logging.Logger:
    """Get logger for API operations."""
    return get_logger(name, category="api")


def get_agent_logger(name: str = "agent") -> logging.Logger:
    """Get logger for agent operations."""
    return get_logger(name, category="agent")


def get_perf_logger(name: str = "performance") -> logging.Logger:
    """Get logger for performance metrics."""
    return get_logger(name, category="performance")
