"""
Custom exceptions for the application.
"""


class AppException(Exception):
    """Base application exception."""
    status_code: int = 500
    detail: str = "Internal server error"

    def __init__(self, detail: str = None):
        if detail:
            self.detail = detail


class ValidationError(AppException):
    """Validation error exception."""
    status_code = 400
    detail = "Validation error"


class FileTooLargeError(ValidationError):
    """File size exceeds limit."""
    detail = "File too large"


class InvalidFileTypeError(ValidationError):
    """Invalid file type."""
    detail = "Invalid file type"


class ImageProcessingError(AppException):
    """Image processing error."""
    status_code = 500
    detail = "Failed to process image"


class ModelNotFoundError(AppException):
    """Model file not found."""
    status_code = 500
    detail = "AI model not found"


class ResolutionNotSupportedError(ValidationError):
    """Target resolution not supported."""
    detail = "Resolution not supported"
