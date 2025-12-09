"""
Unit Tests for Logging Module

Tests logging configuration and setup.
"""

import pytest
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging, get_logger, log_info, log_debug


@pytest.mark.unit
class TestLoggingSetup:
    """Test logging setup and configuration."""

    def test_setup_logging_creates_handlers(self):
        """Test that setup_logging creates log handlers."""
        setup_logging(level="INFO", log_to_file=False)

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_logging_levels(self):
        """Test that different logging levels work."""
        setup_logging(level="DEBUG", log_to_file=False)

        logger = get_logger("test")

        # These should not raise errors
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_convenience_functions(self):
        """Test convenience logging functions."""
        setup_logging(level="INFO", log_to_file=False)

        # These should not raise errors
        log_info("Info message")
        log_debug("Debug message")


@pytest.mark.unit
class TestLoggerConfiguration:
    """Test logger configuration options."""

    def test_setup_logging_with_different_levels(self):
        """Test setup with different log levels."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

        for level in levels:
            setup_logging(level=level, log_to_file=False)
            root_logger = logging.getLogger()

            expected_level = getattr(logging, level)
            assert root_logger.level == expected_level

    def test_get_logger_with_module_name(self):
        """Test getting logger with __name__ pattern."""
        logger1 = get_logger(__name__)
        logger2 = get_logger("src.logger")

        assert isinstance(logger1, logging.Logger)
        assert isinstance(logger2, logging.Logger)
        assert logger1.name == __name__
        assert logger2.name == "src.logger"
