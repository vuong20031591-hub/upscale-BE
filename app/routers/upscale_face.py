"""
Face enhancement endpoint using CodeFormer.
"""

import io
import time
import asyncio
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from PIL import Image

from app.core import ValidationError
from app.services.face_enhancement_service import FaceEnhancementService
from app.validators.face_enhancement_validator import FaceEnhancementValidator
from app.utils import read_upload_file
from app.utils.logging_utils import get_structured_logger

from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_synced_user
from app.db import get_db
from app.models.orm import User
from app.services.quota import check_and_consume


logger = get_structured_logger(__name__)
router = APIRouter(prefix="/upscale", tags=["Face Enhancement"])


@router.post(
    "/enhance/face",
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
    summary="Enhance faces in image",
    description="Apply CodeFormer face enhancement with selectable mode (restoration/colorization/inpainting)"
)
async def enhance_face(
    file: UploadFile = File(..., description="Image file (JPG/PNG, max 10MB)"),
    mode: str = Form("restoration", description="Mode: restoration|colorization|inpainting"),
    weight: float = Form(None, description="Fidelity weight (0-1, restoration only)"),
    face_upsample: bool = Form(None, description="Enable face upsampling"),
    background_enhance: bool = Form(True, description="Enhance background with Real-ESRGAN"),
    bg_upscale: int = Form(2, description="Background upscale factor (1, 2, or 4)")
,
    user: User = Depends(get_synced_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Enhance faces using CodeFormer with specified mode.
    
    Modes:
    - **restoration**: Enhance facial details in old/low-quality photos
    - **colorization**: Add color to black-and-white photos (fixed w=0)
    - **inpainting**: Fill damaged/masked face regions (fixed w=1)
    
    Parameters:
    - **file**: Image file (JPG/PNG, max 10MB)
    - **mode**: Processing mode (restoration/colorization/inpainting)
    - **weight**: Fidelity weight for restoration (0-1, higher = more detail preserved)
                 Ignored for colorization/inpainting modes
    - **face_upsample**: Whether to upsample face regions after enhancement
    - **background_enhance**: Whether to enhance background with Real-ESRGAN
    - **bg_upscale**: Background upscale factor - 1, 2, or 4
    
    Returns:
        StreamingResponse with enhanced image and metadata headers:
        - X-Faces-Detected: Number of faces processed
        - X-Processing-Time: Processing duration in seconds
        - X-Mode-Used: Mode that was applied
        - X-Weight-Used: Weight value used (2 decimal places)
        - X-Background-Enhanced: Whether background was enhanced
        - X-BG-Upscale: Background upscale factor used
        - X-Warning: Warning message if no faces detected
    
    Error Responses:
        - 422: Invalid parameters (mode, weight, bg_upscale, file format)
        - 503: Model loading failed
        - 504: Processing timeout (>30 seconds)
    
    Requirements: 1.1-1.6, 2.1-2.6, 3.1-3.6, 4.1-4.7, 8.1-8.8, 11.1-11.11
    """
    start_time = time.time()

    # Read uploaded file with size validation (max 10MB)
    try:
        file_info = await read_upload_file(file, max_size=10485760)  # 10MB = 10,485,760 bytes
    except ValidationError as e:
        duration = time.time() - start_time
        logger.error(
            "Face enhancement file validation failed",
            error=str(e),
            filename=file.filename,
            duration_seconds=round(duration, 3),
            exc_info=False
        )
        raise HTTPException(status_code=422, detail=str(e))
    
    # Log request context
    logger.info(
        "Face enhancement request received",
        filename=file_info.filename,
        file_size_bytes=file_info.size,
        file_size_mb=round(file_info.size / (1024 * 1024), 2),
        mode=mode,
        weight=weight,
        face_upsample=face_upsample,
        background_enhance=background_enhance,
        bg_upscale=bg_upscale,
        content_type=file_info.content_type
    )
    
    # Initialize validator and service
    validator = FaceEnhancementValidator()
    service = FaceEnhancementService()
    
    try:
        # Validate mode parameter
        try:
            validated_mode = validator.validate_mode(mode)
        except ValidationError as e:
            duration = time.time() - start_time
            logger.error(
                "Invalid mode parameter",
                error=str(e),
                mode=mode,
                duration_seconds=round(duration, 3),
                exc_info=False
            )
            raise HTTPException(status_code=422, detail=str(e))
        
        # Validate weight and face_upsample parameters
        validated_weight = validator.validate_weight(weight, validated_mode)
        validated_face_upsample = validator.validate_face_upsample(face_upsample)
        
        # Validate bg_upscale (Requirement 11.11)
        try:
            validated_bg_upscale = validator.validate_bg_upscale(bg_upscale)
        except ValidationError as e:
            duration = time.time() - start_time
            logger.error(
                "Invalid bg_upscale parameter",
                error=str(e),
                bg_upscale=bg_upscale,
                duration_seconds=round(duration, 3),
                exc_info=False
            )
            raise HTTPException(status_code=422, detail=str(e))
        
        # Load image from file
        try:
            image = file_info.to_image()
        except (Image.UnidentifiedImageError, OSError) as e:
            duration = time.time() - start_time
            logger.error(
                "Invalid or corrupted image",
                error=str(e),
                filename=file_info.filename,
                duration_seconds=round(duration, 3),
                exc_info=True
            )
            raise HTTPException(
                status_code=422,
                detail="Invalid or corrupted image file"
            )
        
        # Consume quota AFTER mọi validation, TRƯỚC khi vào GPU.
        await check_and_consume(db, user)

        # Enhance faces with timeout protection (30 seconds)
        try:
            result = await asyncio.wait_for(
                service.enhance(
                    image=image,
                    mode=validated_mode,
                    weight=validated_weight,
                    face_upsample=validated_face_upsample,
                    background_enhance=background_enhance if background_enhance is not None else True,
                    bg_upscale=validated_bg_upscale,
                    timeout=30
                ),
                timeout=30
            )
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            logger.error(
                "Face enhancement timeout",
                filename=file_info.filename,
                mode=mode,
                timeout_seconds=30,
                duration_seconds=round(duration, 3)
            )
            raise HTTPException(
                status_code=504,
                detail="Face enhancement timed out after 30 seconds"
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "Face enhancement processing failed",
                error=str(e),
                filename=file_info.filename,
                mode=mode,
                duration_seconds=round(duration, 3),
                exc_info=True
            )
            # Check if it's a model loading error
            if "model" in str(e).lower() or "load" in str(e).lower():
                raise HTTPException(
                    status_code=503,
                    detail=f"Model loading failed: {str(e)}"
                )
            raise HTTPException(
                status_code=500,
                detail="Face enhancement failed"
            )
        
        # Convert result image to PNG bytes
        buffer = io.BytesIO()
        result.image.save(buffer, format="PNG", quality=95)
        buffer.seek(0)
        
        # Log completion
        duration = time.time() - start_time
        logger.info(
            "Face enhancement request completed",
            filename=file_info.filename,
            mode=result.mode_used.value,
            faces_detected=result.faces_detected,
            weight_used=result.weight_used,
            background_enhanced=result.background_enhanced,
            bg_upscale=result.bg_upscale,
            processing_time=round(result.processing_time, 3),
            total_duration=round(duration, 3),
            warning=result.warning
        )
        
        # Build response headers from result
        headers = result.to_response_headers()
        headers["Content-Disposition"] = f"attachment; filename=enhanced_face_{result.mode_used.value}.png"
        
        # Return StreamingResponse with enhanced image and metadata
        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers=headers
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch-all for unexpected errors
        duration = time.time() - start_time
        logger.error(
            "Unexpected error in face enhancement",
            error=str(e),
            filename=file_info.filename,
            mode=mode,
            duration_seconds=round(duration, 3),
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
