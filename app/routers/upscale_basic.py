"""
Basic image upscaling endpoints (AI and Standard methods).
"""

import io
import logging
import time
from fastapi import APIRouter, File, Form, UploadFile, status, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from app.core import ValidationError, ImageProcessingError
from app.models import Resolution
from app.services import ImageProcessor
from app.utils import read_upload_file
from app.utils.logging_utils import get_structured_logger

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_synced_user
from app.db import get_db
from app.models.orm import User
from app.services.quota import check_and_consume


logger = get_structured_logger(__name__)
router = APIRouter(prefix="/upscale", tags=["Upscaling"])


def _get_resolution(resolution: str) -> Resolution:
    """
    Convert string to Resolution enum with strict validation.
    
    Args:
        resolution: Resolution string (e.g., "2k", "4k")
    
    Returns:
        Resolution enum
    
    Raises:
        ValueError: If resolution is not a valid enum value
    """
    try:
        return Resolution(resolution.lower())
    except ValueError:
        valid_values = ", ".join([r.value for r in Resolution])
        raise ValueError(f"Invalid resolution '{resolution}'. Valid values: {valid_values}")


@router.post(
    "/ai",
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
    summary="Upscale image using AI",
    description="Upscale an image to 2K or 4K using Real-ESRGAN AI model with optional face enhancement"
)
async def upscale_ai(
    file: UploadFile = File(..., description="Image file (JPG/PNG, max 10MB)"),
    target_resolution: str = Form("2k", description="Target resolution: 2k or 4k"),
    enhance_faces: bool = Form(True, description="Apply CodeFormer face enhancement")
,
    user: User = Depends(get_synced_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upscale image using AI (Real-ESRGAN) with optional face enhancement.

    - **file**: Image file (JPG/PNG)
    - **target_resolution**: Target resolution (2k or 4k)
    - **enhance_faces**: Enable CodeFormer face restoration (default: True)

    Returns the upscaled image as PNG.
    """
    start_time = time.time()
    processor = ImageProcessor()
    file_info = await read_upload_file(file)

    # Log request context (Requirement 7.4)
    logger.info(
        "AI upscale request received",
        filename=file_info.filename,
        file_size_bytes=file_info.size,
        file_size_mb=round(file_info.size / (1024 * 1024), 2),
        target_resolution=target_resolution,
        content_type=file_info.content_type,
        method="ai",
        enhance_faces=enhance_faces
    )

    try:
        resolution = _get_resolution(target_resolution)
        # Consume quota AFTER validation, TRƯỚC khi compute (tránh trừ quota trên input rác).
        await check_and_consume(db, user)
        result = processor.process(file_info, resolution, use_ai=True, enhance_faces=enhance_faces)

        buffer = result.to_bytes(quality=processor.config.quality)
        
        # Log request completion (Requirement 7.4)
        duration = time.time() - start_time
        logger.info(
            "AI upscale request completed",
            filename=file_info.filename,
            method="ai",
            target_resolution=resolution.value,
            output_width=result.final_width,
            output_height=result.final_height,
            scale_factor=round(result.scale_factor, 2),
            enhance_faces=enhance_faces,
            duration_seconds=round(duration, 3)
        )

        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=upscaled_{resolution.value}.png",
                "X-Image-Resolution": result.resolution,
                "X-Scale-Factor": f"{result.scale_factor:.2f}"
            }
        )
        
    except ValidationError as e:
        # Log validation error (Requirement 7.6)
        duration = time.time() - start_time
        logger.error(
            "AI upscale validation failed",
            error=e,
            filename=file_info.filename,
            file_size_bytes=file_info.size,
            target_resolution=target_resolution,
            enhance_faces=enhance_faces,
            duration_seconds=round(duration, 3),
            exc_info=False
        )
        raise HTTPException(status_code=400, detail=e.detail)
        
    except HTTPException:
        raise
    except Exception as e:
        # Log processing error (Requirement 7.6)
        duration = time.time() - start_time
        logger.error(
            "AI upscale request failed",
            error=e,
            filename=file_info.filename,
            file_size_bytes=file_info.size,
            target_resolution=target_resolution,
            duration_seconds=round(duration, 3),
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Image processing failed")


@router.post(
    "/standard",
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
    summary="Upscale image using standard method",
    description="Upscale an image using traditional resizing (faster, lower quality)"
)
async def upscale_standard(
    file: UploadFile = File(..., description="Image file (JPG/PNG, max 10MB)"),
    target_resolution: str = Form("2k", description="Target resolution: 2k or 4k")
,
    user: User = Depends(get_synced_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upscale image using standard resizing (LANCZOS).

    - **file**: Image file (JPG/PNG)
    - **target_resolution**: Target resolution (2k or 4k)

    Returns the upscaled image as PNG.
    
    Requirements: 1.1, 1.2, 1.6, 1.7, 2.1, 2.5, 2.6, 2.7, 7.1, 7.2, 7.3, 7.4, 10.5
    """
    start_time = time.time()
    processor = ImageProcessor()
    
    
    # Read uploaded file
    file_info = await read_upload_file(file)
    
    # Log request context (Req 8.1, 7.4)
    logger.info(
        "Standard upscale request received",
        filename=file_info.filename,
        file_size_bytes=file_info.size,
        file_size_mb=round(file_info.size / (1024 * 1024), 2),
        target_resolution=target_resolution,
        content_type=file_info.content_type,
        method="standard"
    )
    
    # Validate upload (reuse from image-upload-validation) (Req 2.1, 2.5)
    try:
        processor.validate_upload(file_info)
    except ValidationError as e:
        duration = time.time() - start_time
        logger.error(
            "Validation failed",
            error=e,
            filename=file_info.filename,
            file_size_bytes=file_info.size,
            duration_seconds=round(duration, 3),
            exc_info=False
        )
        raise HTTPException(status_code=400, detail=e.detail)
    
    # Validate target_resolution against RESOLUTION_MAP and supported_resolutions (Req 1.6, 1.7)
    # IMPORTANT: Reject invalid resolutions, do NOT default to 2k
    try:
        # First validate against supported_resolutions config
        processor.config.validate_resolution(target_resolution)
        # Then validate against RESOLUTION_MAP to ensure dimensions exist
        processor.config.get_dimensions(target_resolution)
        # Finally convert to enum (will raise if not in enum)
        resolution = _get_resolution(target_resolution)
    except ValueError as e:
        duration = time.time() - start_time
        logger.error(
            "Invalid resolution",
            error=e,
            target_resolution=target_resolution,
            duration_seconds=round(duration, 3),
            exc_info=False
        )
        raise HTTPException(status_code=400, detail=str(e))
    
    # Wrap image loading with try/except to catch corrupted images (Req 2.5, 2.6, 2.7)
    try:
        # This will trigger PIL to load and validate the image
        # DecompressionBombError will be raised here if image exceeds MAX_IMAGE_PIXELS
        image = file_info.to_image()
    except Image.DecompressionBombError as e:
        duration = time.time() - start_time
        logger.error(
            "Decompression bomb detected",
            error=e,
            filename=file_info.filename,
            file_size_bytes=file_info.size,
            duration_seconds=round(duration, 3),
            exc_info=True
        )
        raise HTTPException(
            status_code=400, 
            detail="Image too large. Maximum dimensions exceeded"
        )
    except (Image.UnidentifiedImageError, OSError) as e:
        duration = time.time() - start_time
        logger.error(
            "Invalid or corrupted image",
            error=e,
            filename=file_info.filename,
            file_size_bytes=file_info.size,
            duration_seconds=round(duration, 3),
            exc_info=True
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file"
        )
    
    # Consume quota AFTER mọi validation, TRƯỚC khi vào threadpool.
    await check_and_consume(db, user)

    # Offload CPU-bound processing to thread pool (Req 10.5)
    # This prevents blocking the async event loop
    # Pass the already-decoded image to avoid double decoding
    try:
        result = await run_in_threadpool(
            processor.process_from_image,
            image,
            resolution,
            use_ai=False
        )
    except ImageProcessingError as e:
        duration = time.time() - start_time
        logger.error(
            "Processing failed",
            error=e,
            filename=file_info.filename,
            target_resolution=resolution.value,
            duration_seconds=round(duration, 3),
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Image processing failed")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "Unexpected error processing image",
            error=e,
            filename=file_info.filename,
            target_resolution=resolution.value,
            duration_seconds=round(duration, 3),
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")
    
    # Log processing complete (Req 8.2, 8.3, 7.4)
    duration = time.time() - start_time
    logger.info(
        "Standard upscale request completed",
        filename=file_info.filename,
        method="standard",
        target_resolution=resolution.value,
        original_width=result.original_width,
        original_height=result.original_height,
        output_width=result.final_width,
        output_height=result.final_height,
        scale_factor=round(result.scale_factor, 2),
        duration_seconds=round(duration, 3)
    )
    
    # Encode to PNG with OUTPUT_QUALITY
    buffer = result.to_bytes(quality=processor.config.quality)

    # Return StreamingResponse with metadata headers (Req 5.1, 5.2, 5.3, 4.5)
    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers={
            "Content-Disposition": f"attachment; filename=upscaled_{resolution.value}.png",
            "X-Image-Resolution": result.resolution,
            "X-Scale-Factor": f"{result.scale_factor:.2f}"
        }
    )


@router.get(
    "/resolutions",
    status_code=status.HTTP_200_OK,
    summary="Get supported resolutions"
)
async def get_resolutions():
    """Get list of supported target resolutions."""
    processor = ImageProcessor()
    return {
        "resolutions": processor.get_supported_resolutions(),
        "default": processor.config.default_resolution
    }
