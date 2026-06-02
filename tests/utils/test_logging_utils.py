"""
Tests for structured logging utilities.

Requirements: 7.4, 7.5, 7.6
"""

import logging
import pytest
from io import StringIO

from app.utils.logging_utils import (
    StructuredLogger,
    get_structured_logger,
)


class TestStructuredLogger:
    """Test StructuredLogger functionality."""
    
    def test_structured_logger_formats_with_context(self, caplog):
        """Test that structured logger includes context in messages."""
        logger = get_structured_logger(__name__)
        
        with caplog.at_level(logging.INFO):
            logger.info("Test message", file_size=1024, method="ai")
        
        # Verify log message contains context
        assert "Test message" in caplog.text
        assert "file_size=1024" in caplog.text
        assert "method=ai" in caplog.text
    
    def test_structured_logger_logs_duration(self, caplog):
        """Test that timed_operation logs duration."""
        logger = get_structured_logger(__name__)
        
        with caplog.at_level(logging.INFO):
            with logger.timed_operation("test operation", test_param="value"):
                pass
        
        # Verify start and completion messages
        assert "Starting test operation" in caplog.text
        assert "test operation completed" in caplog.text
        assert "duration" in caplog.text
        assert "test_param=value" in caplog.text
    
    def test_structured_logger_logs_errors_with_context(self, caplog):
        """Test that errors are logged with context and stack trace."""
        logger = get_structured_logger(__name__)
        
        test_error = ValueError("Test error")
        
        with caplog.at_level(logging.ERROR):
            logger.error(
                "Operation failed",
                error=test_error,
                file_size=2048,
                method="standard"
            )
        
        # Verify error message contains context
        assert "Operation failed" in caplog.text
        assert "ValueError: Test error" in caplog.text
        assert "file_size=2048" in caplog.text
        assert "method=standard" in caplog.text
    
    def test_structured_logger_timed_operation_logs_on_exception(self, caplog):
        """Test that timed_operation logs duration even on exception."""
        logger = get_structured_logger(__name__)
        
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                with logger.timed_operation("failing operation"):
                    raise ValueError("Test failure")
        
        # Verify error was logged with duration
        assert "failing operation failed" in caplog.text
        assert "duration" in caplog.text
        assert "ValueError: Test failure" in caplog.text
    
    def test_structured_logger_set_context(self, caplog):
        """Test that set_context persists across multiple log calls."""
        logger = get_structured_logger(__name__)
        
        logger.set_context(request_id="123", user="test")
        
        with caplog.at_level(logging.INFO):
            logger.info("First message")
            logger.info("Second message")
        
        # Both messages should have context
        log_lines = caplog.text.split('\n')
        assert any("request_id=123" in line and "First message" in line for line in log_lines)
        assert any("request_id=123" in line and "Second message" in line for line in log_lines)
    
    def test_structured_logger_clear_context(self, caplog):
        """Test that clear_context removes persistent context."""
        logger = get_structured_logger(__name__)
        
        logger.set_context(request_id="123")
        logger.clear_context()
        
        with caplog.at_level(logging.INFO):
            logger.info("Message after clear")
        
        # Context should not be present
        assert "request_id" not in caplog.text
        assert "Message after clear" in caplog.text
    
    def test_get_structured_logger_returns_structured_logger(self):
        """Test that get_structured_logger returns StructuredLogger instance."""
        logger = get_structured_logger(__name__)
        assert isinstance(logger, StructuredLogger)
    
    def test_structured_logger_warning_level(self, caplog):
        """Test that warning level works correctly."""
        logger = get_structured_logger(__name__)
        
        with caplog.at_level(logging.WARNING):
            logger.warning("Warning message", severity="high")
        
        assert "Warning message" in caplog.text
        assert "severity=high" in caplog.text


class TestLoggingPerformance:
    """Test that logging doesn't impact performance significantly."""
    
    def test_logging_overhead_is_minimal(self):
        """Test that structured logging has minimal overhead."""
        import time
        
        logger = get_structured_logger(__name__)
        
        # Measure time with logging
        start = time.time()
        for i in range(100):
            logger.info("Test message", iteration=i, data="value")
        duration_with_logging = time.time() - start
        
        # Logging 100 messages should take less than 100ms
        assert duration_with_logging < 0.1, f"Logging took {duration_with_logging}s, too slow"
    
    def test_timed_operation_overhead_is_minimal(self):
        """Test that timed_operation context manager has minimal overhead."""
        import time
        
        logger = get_structured_logger(__name__)
        
        # Measure overhead
        start = time.time()
        for i in range(50):
            with logger.timed_operation("test", iteration=i):
                pass  # No actual work
        duration = time.time() - start
        
        # 50 timed operations should take less than 50ms
        assert duration < 0.05, f"Timed operations took {duration}s, too slow"
