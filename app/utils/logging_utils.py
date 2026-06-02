"""
Structured logging utilities for AI Image Upscaling.

Provides structured logging with consistent context fields and performance tracking.
Requirements: 7.4, 7.5, 7.6
"""

import logging
import time
import json
from typing import Optional, Dict, Any
from contextlib import contextmanager
from functools import wraps


class StructuredLogger:
    """
    Wrapper around standard logger to provide structured logging.
    
    Adds consistent context fields:
    - timestamp
    - level
    - message
    - context (file_size, target_resolution, method, etc.)
    - duration (for timed operations)
    - error (for exceptions)
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs) -> None:
        """Set context fields for subsequent log messages."""
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context fields."""
        self._context.clear()
    
    def _format_message(
        self,
        message: str,
        extra_context: Optional[Dict[str, Any]] = None,
        duration: Optional[float] = None,
        error: Optional[Exception] = None
    ) -> str:
        """
        Format log message with structured data.
        
        Args:
            message: Log message
            extra_context: Additional context fields for this log entry
            duration: Processing duration in seconds
            error: Exception object if logging an error
        
        Returns:
            Formatted log message (human-readable with key=value pairs)
        """
        # Combine persistent context with extra context
        context = {**self._context}
        if extra_context:
            context.update(extra_context)
        
        # Build message parts
        parts = [message]
        
        # Add context fields
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            parts.append(f"[{context_str}]")
        
        # Add duration if provided
        if duration is not None:
            parts.append(f"duration={duration:.3f}s")
        
        # Add error details if provided
        if error:
            parts.append(f"error={type(error).__name__}: {str(error)}")
        
        return " ".join(parts)
    
    def debug(self, message: str, **extra_context) -> None:
        """Log debug message with context."""
        formatted = self._format_message(message, extra_context)
        self.logger.debug(formatted)
    
    def info(self, message: str, **extra_context) -> None:
        """Log info message with context."""
        formatted = self._format_message(message, extra_context)
        self.logger.info(formatted)
    
    def warning(self, message: str, **extra_context) -> None:
        """Log warning message with context."""
        formatted = self._format_message(message, extra_context)
        self.logger.warning(formatted)
    
    def error(
        self,
        message: str,
        error: Optional[Exception] = None,
        exc_info: bool = True,
        **extra_context
    ) -> None:
        """
        Log error message with context and stack trace.
        
        Args:
            message: Error message
            error: Exception object
            exc_info: Whether to include stack trace
            extra_context: Additional context fields
        """
        formatted = self._format_message(message, extra_context, error=error)
        self.logger.error(formatted, exc_info=exc_info)
    
    @contextmanager
    def timed_operation(self, operation_name: str, **extra_context):
        """
        Context manager to time an operation and log duration.
        
        Usage:
            with logger.timed_operation("AI upscaling", file_size=1024000):
                # ... operation code ...
                pass
        
        Args:
            operation_name: Name of the operation being timed
            extra_context: Additional context fields
        """
        start_time = time.time()
        self.info(f"Starting {operation_name}", **extra_context)
        
        try:
            yield
        except Exception as e:
            duration = time.time() - start_time
            self.error(
                f"{operation_name} failed",
                error=e,
                duration=duration,
                **extra_context
            )
            raise
        else:
            duration = time.time() - start_time
            self.info(
                f"{operation_name} completed",
                duration=duration,
                **extra_context
            )


def get_structured_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        StructuredLogger instance
    """
    logger = logging.getLogger(name)
    return StructuredLogger(logger)


def log_request_context(func):
    """
    Decorator to automatically log request context for processing functions.
    
    Extracts file_info and target_resolution from function arguments
    and logs them as context.
    
    Usage:
        @log_request_context
        def process(self, file_info, target_resolution, use_ai=True):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to extract context from arguments
        context = {}
        
        # Get file_info (usually first arg after self)
        if len(args) > 1:
            file_info = args[1]
            if hasattr(file_info, 'size'):
                context['file_size_bytes'] = file_info.size
                context['file_size_mb'] = round(file_info.size / (1024 * 1024), 2)
            if hasattr(file_info, 'filename'):
                context['filename'] = file_info.filename
        
        # Get target_resolution (usually second arg after self)
        if len(args) > 2:
            target_resolution = args[2]
            if hasattr(target_resolution, 'value'):
                context['target_resolution'] = target_resolution.value
        
        # Get use_ai flag
        use_ai = kwargs.get('use_ai', True)
        context['method'] = 'ai' if use_ai else 'standard'
        
        # Log context
        logger = logging.getLogger(func.__module__)
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            logger.info(f"Request context: {context_str}")
        
        return func(*args, **kwargs)
    
    return wrapper
