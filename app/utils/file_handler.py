"""
File handling utilities.
"""

from app.models import UploadFileInfo
from app.core import ValidationError, settings
from fastapi import UploadFile


async def read_upload_file(file: UploadFile, max_size: int = None) -> UploadFileInfo:
    """
    Read and validate uploaded file with size limit enforcement.
    
    Reads file in chunks to prevent memory exhaustion from large uploads.

    Args:
        file: FastAPI UploadFile
        max_size: Maximum file size in bytes (defaults to settings.upload.max_file_size)

    Returns:
        UploadFileInfo with content and metadata
    
    Raises:
        ValidationError: If file size exceeds max_size
    """
    if max_size is None:
        max_size = settings.upload.max_file_size
    
    # Read file in chunks to enforce size limit
    chunks = []
    total_size = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        
        total_size += len(chunk)
        
        # Enforce size limit during read to prevent memory exhaustion
        if total_size > max_size:
            raise ValidationError(
                f"File size exceeds maximum allowed size of {max_size} bytes"
            )
        
        chunks.append(chunk)
    
    content = b''.join(chunks)

    return UploadFileInfo(
        filename=file.filename or "unknown",
        content_type=file.content_type or "application/octet-stream",
        size=total_size,
        content=content
    )
