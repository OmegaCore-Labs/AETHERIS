"""
Logging Utilities

Structured logging with levels, colors, and file output.
"""

import logging
import sys
from typing import Optional
from datetime import datetime


class AetherisLogger:
    """
    Structured logger for AETHERIS operations.

    Features:
    - Multiple log levels
    - Colored console output
    - File logging
    - Structured JSON format
    """

    def __init__(
        self,
        name: str = "aetheris",
        level: str = "INFO",
        log_file: Optional[str] = None
    ):
        """
        Initialize logger.

        Args:
            name: Logger name
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for logs
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._get_console_formatter())
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self._get_file_formatter())
            self.logger.addHandler(file_handler)

    def _get_console_formatter(self) -> logging.Formatter:
        """Get colored console formatter."""
        class ColoredFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\033[36m',    # Cyan
                'INFO': '\033[32m',     # Green
                'WARNING': '\033[33m',  # Yellow
                'ERROR': '\033[31m',    # Red
                'CRITICAL': '\033[35m', # Magenta
            }
            RESET = '\033[0m'

            def format(self, record):
                color = self.COLORS.get(record.levelname, self.RESET)
                record.levelname = f"{color}{record.levelname}{self.RESET}"
                return super().format(record)

        formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return formatter

    def _get_file_formatter(self) -> logging.Formatter:
        """Get file formatter."""
        return logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log info message."""
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log error message."""
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)

    def progress(self, current: int, total: int, message: str = ""):
        """Log progress update."""
        percent = (current / total) * 100 if total > 0 else 0
        self.info(f"[{current}/{total}] {percent:.1f}% - {message}")

    def operation_start(self, operation: str, **kwargs):
        """Log operation start."""
        self.info(f"▶ {operation} started", extra={"operation": operation, **kwargs})

    def operation_end(self, operation: str, success: bool = True, **kwargs):
        """Log operation end."""
        status = "✓" if success else "✗"
        self.info(f"{status} {operation} {'completed' if success else 'failed'}", extra=kwargs)

    def json_log(self, level: str, data: dict):
        """Log structured JSON data."""
        import json
        message = json.dumps(data, default=str)
        getattr(self, level.lower())(message)


# Global logger instance
_default_logger = None


def get_logger(name: str = "aetheris") -> AetherisLogger:
    """Get global logger instance."""
    global _default_logger
    if _default_logger is None:
        _default_logger = AetherisLogger(name)
    return _default_logger
