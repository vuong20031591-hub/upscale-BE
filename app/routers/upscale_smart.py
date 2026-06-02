"""
Smart auto-detection upscale endpoint.
"""

import io
import time
from fastapi import APIRouter, File, UploadFile, Form, status, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from app.core import ValidationError
from app.services.smart_processor import SmartProcessor
from app.utils import read_upload_file
from app.utils.logging_utils import get_structured_logger

logger = get_structured_logger(__name__)
router = APIRouter(prefix="/upscale", tags=["Smart Upscaling"])


@router.post(
    "/smart",
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
    summary="Smart auto-detection upscale",
    description="Automatically analyze image and apply appropriate enhancement (colorization/inpainting/upscaling/restoration)"
)
async def upscale_smart(
    file: UploadFile = File(..., description="Image file (JPG/PNG, max 10MB)"),
    bg_upscale: int = Form(2, description="Background upscale factor (2 or 4)")
):
    """
    Smart auto-detection endpoint - Backend Only.
    
    Automatically analyzes the uploaded image to detect issues (grayscale, white mask,
    low resolution, blur) and applies the most appropriate enhancement mode without
    requiring user interaction.
    
    Workflow:
        1. Upload image
        2. Analyze image (detect issues)
        3. Automatically select processing mode based on priority:
           - Grayscale (confidence > 0.9) → Colorization
           - White Mask (>30%) → Inpainting
           - Low Resolution (<512px) → Upscaling
           - Blur (variance < 100) → Restoration
           - Default → Restoration
        4. Process image with selected mode
        5. Return processed image with analysis metadata in headers
    
    Parameters:
        - file: Image file to analyze and process (JPG/PNG, max 10MB)
    
    Returns:
        StreamingResponse with processed image and analysis metadata headers:
        - X-Analysis-Grayscale: true/false
        - X-Analysis-Grayscale-Confidence: 0.0-1.0
        - X-Analysis-White-Mask: true/false
        - X-Analysis-White-Mask-Confidence: 0.0-1.0
        - X-Analysis-Low-Resolution: true/false
        - X-Analysis-Low-Resolution-Confidence: 0.0-1.0
        - X-Analysis-Blur: true/false
        - X-Analysis-Blur-Confidence: 0.0-1.0
        - X-Suggested-Mode: colorization|inpainting|upscaling|restoration
        - X-Mode-Used: colorization|inpainting|upscaling|restoration
        - X-Processing-Time: Total processing time in seconds
        - X-Analysis-Time: Analysis time in seconds
    
    Error Responses:
        - 400: Invalid file (size > 10MB, wrong format)
        - 500: Processing failed
    
    Requirements:
        - Requirements 3.1-3.6: Automatic processing workflow
        - Requirements 4.1-4.8: API endpoint design
        - Requirements 5.1-5.7: Transparent processing (backend only)
    
    Example:
        curl -X POST "http://localhost:8000/upscale/smart" \\
             -F "file=@photo.jpg" \\
             -o enhanced.png
    """
    start_time = time.time()
    
    # Read uploaded file (Requirement 4.2)
    file_info = await read_upload_file(file)
    
    # Log request context
    logger.info(
        "Smart upscale request received",
        filename=file_info.filename,
        file_size_bytes=file_info.size,
        file_size_mb=round(file_info.size / (1024 * 1024), 2),
        content_type=file_info.content_type,
        method="smart"
    )
    
    try:
        # Import analysis models for type hints
        from app.models.analysis import DetectionType
        
        # Load image from bytes
        image = Image.open(io.BytesIO(file_info.content))
        
        # Process image with smart processor (Requirement 3.1-3.6, 5.2)
        processor = SmartProcessor()
        result = processor.process_image(image, filename=file_info.filename, bg_upscale=bg_upscale)
        
        # Convert processed image to bytes
        buffer = result.to_bytes(format='PNG')
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Log completion
        logger.info(
            "Smart upscale request completed",
            filename=file_info.filename,
            mode_used=result.selected_mode.value,
            total_time_seconds=round(total_time, 3),
            analysis_time_seconds=round(result.analysis_time, 3)
        )
        
        # Return processed image with analysis metadata in headers (Requirement 4.5)
        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=smart_enhanced.png",
                # Analysis metadata headers
                "X-Analysis-Grayscale": str(result.analysis_result.detections[DetectionType.GRAYSCALE].detected).lower(),
                "X-Analysis-Grayscale-Confidence": f"{result.analysis_result.detections[DetectionType.GRAYSCALE].confidence:.2f}",
                "X-Analysis-White-Mask": str(result.analysis_result.detections[DetectionType.WHITE_MASK].detected).lower(),
                "X-Analysis-White-Mask-Confidence": f"{result.analysis_result.detections[DetectionType.WHITE_MASK].confidence:.2f}",
                "X-Analysis-Low-Resolution": str(result.analysis_result.detections[DetectionType.LOW_RESOLUTION].detected).lower(),
                "X-Analysis-Low-Resolution-Confidence": f"{result.analysis_result.detections[DetectionType.LOW_RESOLUTION].confidence:.2f}",
                "X-Analysis-Blur": str(result.analysis_result.detections[DetectionType.BLUR].detected).lower(),
                "X-Analysis-Blur-Confidence": f"{result.analysis_result.detections[DetectionType.BLUR].confidence:.2f}",
                "X-Suggested-Mode": result.selected_mode.value,
                "X-Mode-Used": result.selected_mode.value,
                "X-Processing-Time": f"{total_time:.2f}s",
                "X-Analysis-Time": f"{result.analysis_time:.2f}s",
                # Resolution metadata headers
                "X-Original-Width": str(result.original_width),
                "X-Original-Height": str(result.original_height),
                "X-Final-Width": str(result.final_width),
                "X-Final-Height": str(result.final_height)
            }
        )
        
    except ValidationError as e:
        # Log validation error (Requirement 4.6)
        duration = time.time() - start_time
        logger.error(
            "Smart upscale validation failed",
            error=str(e),
            filename=file_info.filename,
            file_size_bytes=file_info.size,
            duration_seconds=round(duration, 3),
            exc_info=False
        )
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        # Log processing error (Requirement 4.7)
        duration = time.time() - start_time
        logger.error(
            "Smart upscale request failed",
            error=str(e),
            filename=file_info.filename,
            file_size_bytes=file_info.size,
            duration_seconds=round(duration, 3),
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Image processing failed")
