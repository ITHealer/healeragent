"""
Log Formatters
=============

Provides formatters for different environments:
- DevFormatter: Colored text format for development
- JsonFormatter: Structured JSON format for production

Format:
-------
Development (text):
    2026-01-11 12:00:00 | INFO  | agent.unified | [req-abc123] Processing request

Production (JSON):
    {"timestamp": "2026-01-11T12:00:00.000Z", "level": "INFO", "logger": "agent.unified", "request_id": "req-abc123", "message": "Processing request"}
"""

import json
import logging
from datetime import datetime
from typing import Optional

from src.core.logging.context import get_request_id


class DevFormatter(logging.Formatter):
    """
    Development formatter with colors and readable format.

    Format: {timestamp} | {level} | {logger} | [{request_id}] {message}
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\x1b[38;5;244m",    # Gray
        "INFO": "\x1b[38;5;39m",       # Blue
        "WARNING": "\x1b[38;5;208m",   # Orange
        "ERROR": "\x1b[38;5;196m",     # Red
        "CRITICAL": "\x1b[38;5;196;1m",  # Bold Red
    }
    RESET = "\x1b[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        # Get request ID from context
        request_id = get_request_id()
        request_id_str = f"[{request_id}] " if request_id else ""

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Format level (padded for alignment)
        level = record.levelname.ljust(5)

        # Format logger name (truncate if too long)
        logger_name = record.name
        if len(logger_name) > 25:
            logger_name = "..." + logger_name[-22:]

        # Build message
        message = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            message = f"{message}\n{exc_text}"

        # Format line
        line = f"{timestamp} | {level} | {logger_name.ljust(25)} | {request_id_str}{message}"

        # Apply colors for console
        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            line = f"{color}{line}{self.RESET}"

        return line


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for production/log aggregation.

    Outputs structured JSON for easy parsing by log management tools
    (ELK stack, CloudWatch, Datadog, etc.)
    """

    def format(self, record: logging.LogRecord) -> str:
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request ID if present
        request_id = get_request_id()
        if request_id:
            log_entry["request_id"] = request_id

        # Add source location for debugging
        log_entry["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields if present
        extra_fields = {
            k: v
            for k, v in record.__dict__.items()
            if k not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "taskName", "message",
            }
        }
        if extra_fields:
            log_entry["extra"] = extra_fields

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class FileFormatter(logging.Formatter):
    """
    Plain text formatter for file output (no colors).

    Format: {timestamp} | {level} | {logger} | [{request_id}] {message}
    """

    def format(self, record: logging.LogRecord) -> str:
        # Get request ID from context
        request_id = get_request_id()
        request_id_str = f"[{request_id}] " if request_id else ""

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S,%f"
        )[:23]  # Include milliseconds

        # Format level (padded for alignment)
        level = record.levelname.ljust(8)

        # Format logger name
        logger_name = record.name

        # Build message
        message = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            message = f"{message}\n{exc_text}"

        return f"{timestamp} | {level} | {logger_name} | {request_id_str}{message}"
