"""
Logging Configuration for Footstep Audio Pipeline

Provides centralized logging with console and file handlers.
Replaces print() statements with professional logging.

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.debug("Detailed debug info")
    logger.warning("Something unexpected")
    logger.error("An error occurred")
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# Global log directory
LOG_DIR = Path("logs")


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: Optional[Path] = None
) -> None:
    """
    Configure logging for the entire pipeline.

    Call this once at the start of your application.

    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_to_file: Whether to save logs to file
        log_dir: Directory for log files (default: ./logs/)

    Example:
        >>> setup_logging(level="DEBUG", log_to_file=True)
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create log directory if needed
    if log_to_file:
        if log_dir is None:
            log_dir = LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatters
    # Console: Simple, user-friendly
    console_formatter = logging.Formatter(
        fmt='%(levelname)s: %(message)s'
    )

    # File: Detailed with timestamps
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (stdout for INFO+, stderr for WARNING+)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)

    # Filter: Only show INFO and DEBUG on stdout
    class InfoFilter(logging.Filter):
        def filter(self, record):
            return record.levelno <= logging.INFO

    console_handler.addFilter(InfoFilter())
    root_logger.addHandler(console_handler)

    # Error console handler (for WARNING, ERROR, CRITICAL)
    error_console_handler = logging.StreamHandler(sys.stderr)
    error_console_handler.setLevel(logging.WARNING)
    error_console_handler.setFormatter(console_formatter)
    root_logger.addHandler(error_console_handler)

    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Log the log file location
        root_logger.info(f"Logging to file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Usually __name__ of the calling module

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing video...")
    """
    return logging.getLogger(name)


# Convenience function for backward compatibility with print()
def log_info(message: str) -> None:
    """
    Quick info log (for migrating from print statements).

    Args:
        message: Message to log

    Example:
        >>> log_info("Processing complete")
    """
    logging.getLogger().info(message)


def log_debug(message: str) -> None:
    """Quick debug log."""
    logging.getLogger().debug(message)


def log_warning(message: str) -> None:
    """Quick warning log."""
    logging.getLogger().warning(message)


def log_error(message: str) -> None:
    """Quick error log."""
    logging.getLogger().error(message)


# Default setup (can be overridden by calling setup_logging)
if not logging.getLogger().handlers:
    setup_logging(level="INFO", log_to_file=False)
