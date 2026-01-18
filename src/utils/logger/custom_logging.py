"""
Custom Logging - Bridge to Production Logging System
====================================================

This module provides backward compatibility with existing code while
using the new production logging system under the hood.

The new logging system provides:
- Category-based log files (app, error, api, agent, performance)
- Request ID tracing
- Daily rotation with 15-day retention
- JSON format for production

Usage (legacy - still works):
    from src.utils.logger.custom_logging import LogHandler, LoggerMixin

    # Option 1: LogHandler
    log_handler = LogHandler()
    logger = log_handler.get_logger(__name__)

    # Option 2: LoggerMixin (for classes)
    class MyClass(LoggerMixin):
        def my_method(self):
            self.logger.info("Hello")

Recommended (new way):
    from src.core.logging import get_logger

    logger = get_logger("agent")  # Logs to logs/agent/
    logger = get_logger("api")    # Logs to logs/api/
"""

import logging
from typing import List, Optional

from src.utils.config import settings

# Try to use new logging system, fallback to basic if not available
try:
    from src.core.logging import get_logger as _get_production_logger
    _USE_PRODUCTION_LOGGING = True
except ImportError:
    _USE_PRODUCTION_LOGGING = False
    _get_production_logger = None


def _detect_category(logger_name: str) -> str:
    """Detect log category from logger name."""
    name_lower = logger_name.lower()

    if any(kw in name_lower for kw in ["agent", "tool", "memory", "skill", "unified", "validation"]):
        return "agent"
    if any(kw in name_lower for kw in ["api", "router", "endpoint", "http"]):
        return "api"
    if any(kw in name_lower for kw in ["perf", "metric", "timing"]):
        return "performance"
    return "app"


class LogHandler(object):
    """
    Log handler that bridges to the production logging system.

    Maintains backward compatibility while using the new logging infrastructure.
    """

    def __init__(self):
        # Legacy handlers no longer needed - production system handles this
        self.available_handlers: List = []

    def get_logger(self, logger_name: str, category: Optional[str] = None):
        """
        Get a logger by name.

        Args:
            logger_name: Name of the logger (usually __name__)
            category: Force a specific category (agent, api, app, performance)

        Returns:
            Configured logger
        """
        if _USE_PRODUCTION_LOGGING and _get_production_logger:
            # Use new production logging system
            if category is None:
                category = _detect_category(logger_name)
            return _get_production_logger(logger_name, category=category)
        else:
            # Fallback to basic logging
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))
            return logger


class LoggerMixin:
    """
    Mixin class that provides a logger attribute.

    Classes inheriting from LoggerMixin get a self.logger attribute
    automatically configured with the production logging system.

    Example:
        class MyAgent(LoggerMixin):
            def __init__(self):
                super().__init__()  # Sets up self.logger
                self.logger.info("Agent initialized")
    """

    def __init__(self) -> None:
        # Get the actual class name for better log identification
        class_name = self.__class__.__name__
        module_name = self.__class__.__module__

        # Build a descriptive logger name
        logger_name = f"{module_name}.{class_name}"

        # Get logger from production system
        log_handler = LogHandler()
        self.logger = log_handler.get_logger(logger_name)


# Convenience function for quick logger access
def get_logger(name: str, category: Optional[str] = None) -> logging.Logger:
    """
    Quick function to get a configured logger.

    This is a convenience wrapper - prefer using src.core.logging.get_logger directly.

    Args:
        name: Logger name
        category: Optional category override

    Returns:
        Configured logger
    """
    return LogHandler().get_logger(name, category)
