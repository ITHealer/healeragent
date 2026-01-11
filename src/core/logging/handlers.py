"""
Custom Log Handlers
==================

Provides handlers for different log destinations:
- CategoryFileHandler: Rotates daily, separates by category
- ErrorMirrorHandler: Mirrors ERROR/CRITICAL to error/ directory
- PerformanceHandler: Special handler for performance metrics

Directory Structure:
-------------------
logs/
├── app/            # Default application logs
├── error/          # ERROR + CRITICAL only
├── api/            # API request/response
├── agent/          # Agent flow logs
└── performance/    # Performance metrics

File naming: {category}_YYYY-MM-DD.log
Retention: 15 days (configurable)
"""

import os
import glob
import logging
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from src.core.logging.formatters import FileFormatter, JsonFormatter


# Log categories
CATEGORIES = ["app", "error", "api", "agent", "performance"]


def get_log_dir() -> Path:
    """Get the base log directory."""
    # Use LOG_DIR env var or default to ./logs
    log_dir = os.environ.get("LOG_DIR", "logs")
    return Path(log_dir)


def ensure_log_directories() -> None:
    """Create all log category directories."""
    base_dir = get_log_dir()
    for category in CATEGORIES:
        category_dir = base_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)


def cleanup_old_logs(retention_days: int = 15) -> int:
    """
    Remove log files older than retention_days.

    Args:
        retention_days: Number of days to keep logs

    Returns:
        Number of files deleted
    """
    base_dir = get_log_dir()
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    deleted_count = 0

    for category in CATEGORIES:
        category_dir = base_dir / category
        if not category_dir.exists():
            continue

        # Find all .log files
        for log_file in category_dir.glob("*.log"):
            try:
                # Extract date from filename: {category}_YYYY-MM-DD.log
                date_str = log_file.stem.split("_")[-1]
                if len(date_str) == 10:  # YYYY-MM-DD format
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if file_date < cutoff_date:
                        log_file.unlink()
                        deleted_count += 1
            except (ValueError, IndexError):
                # Skip files that don't match the expected pattern
                continue

    return deleted_count


class DailyRotatingFileHandler(TimedRotatingFileHandler):
    """
    Custom rotating file handler with daily rotation.

    Features:
    - Rotates at midnight
    - Filename format: {category}_YYYY-MM-DD.log
    - Proper cleanup of old files
    """

    def __init__(
        self,
        category: str,
        retention_days: int = 15,
        use_json: bool = False,
    ):
        """
        Initialize daily rotating handler.

        Args:
            category: Log category (app, error, api, agent, performance)
            retention_days: Number of days to keep log files
            use_json: Use JSON format instead of text
        """
        self.category = category
        self.retention_days = retention_days

        # Build log file path
        base_dir = get_log_dir()
        category_dir = base_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Current date for initial filename
        today = datetime.now().strftime("%Y-%m-%d")
        filename = category_dir / f"{category}_{today}.log"

        super().__init__(
            filename=str(filename),
            when="midnight",
            interval=1,
            backupCount=retention_days,
            encoding="utf-8",
        )

        # Set formatter
        if use_json:
            self.setFormatter(JsonFormatter())
        else:
            self.setFormatter(FileFormatter())

    def doRollover(self):
        """Override to use our date-based naming convention."""
        if self.stream:
            self.stream.close()
            self.stream = None

        # Calculate new filename with today's date
        base_dir = get_log_dir()
        category_dir = base_dir / self.category
        today = datetime.now().strftime("%Y-%m-%d")
        new_filename = category_dir / f"{self.category}_{today}.log"

        self.baseFilename = str(new_filename)
        self.stream = self._open()

        # Cleanup old logs
        cleanup_old_logs(self.retention_days)


class ErrorMirrorFilter(logging.Filter):
    """Filter that only allows ERROR and CRITICAL level logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR


class CategoryFilter(logging.Filter):
    """Filter logs by logger name prefix."""

    def __init__(self, allowed_prefixes: list[str]):
        super().__init__()
        self.allowed_prefixes = allowed_prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        # Allow if logger name starts with any of the allowed prefixes
        return any(
            record.name.startswith(prefix) or record.name == prefix
            for prefix in self.allowed_prefixes
        )


def create_category_handler(
    category: str,
    level: int = logging.DEBUG,
    use_json: bool = False,
    retention_days: int = 15,
) -> DailyRotatingFileHandler:
    """
    Create a handler for a specific log category.

    Args:
        category: Log category name
        level: Minimum log level for this handler
        use_json: Use JSON format
        retention_days: Days to retain logs

    Returns:
        Configured DailyRotatingFileHandler
    """
    handler = DailyRotatingFileHandler(
        category=category,
        retention_days=retention_days,
        use_json=use_json,
    )
    handler.setLevel(level)
    return handler


def create_error_handler(
    use_json: bool = False,
    retention_days: int = 15,
) -> DailyRotatingFileHandler:
    """
    Create handler that captures all ERROR/CRITICAL logs.

    This handler receives logs from ALL loggers but only
    writes ERROR and CRITICAL level messages.
    """
    handler = DailyRotatingFileHandler(
        category="error",
        retention_days=retention_days,
        use_json=use_json,
    )
    handler.setLevel(logging.ERROR)
    handler.addFilter(ErrorMirrorFilter())
    return handler
