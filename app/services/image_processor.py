"""
Image processing service.
"""

import logging
import time
from typing import Tuple

from PIL import Image

from app.core import settings, ValidationError, ImageProcessingError
from app.models import Resolution, ImageFormat, ProcessedImage, UploadFileInfo
from app.services.model_manager import ModelManager
from app.services.codeformer_manager import CodeFormerManager
from app.utils.logging_utils import get_structured_logger

logging.basicConfig(level=logging.INFO)
logger = get_structured_logger(__name__)


class ImageProcessor:
    """
    Service for processing and upscaling images.
    Supports both AI-based and traditional upscaling methods.
    """

    def __init__(self):
        """
        Initialize ImageProcessor with ModelManager singleton and output config.

        Sets up:
            - model_manager: Singleton instance for AI model operations
            - codeformer_manager: Singleton instance for face restoration
            - config: Output configuration from environment variables
        """
        self.model_manager = ModelManager()
        self.codeformer_manager = CodeFormerManager()
        self.config = settings.output

    def validate_upload(self, file_info: UploadFileInfo) -> None:
        """
        Validate uploaded file against size, type, and extension constraints.
        
        Performs three validation checks:
        1. File size must not exceed MAX_FILE_SIZE (default: 10 MiB)
        2. Content type must be in ALLOWED_CONTENT_TYPES (image/jpeg, image/png)
        3. File extension must be in ALLOWED_EXTENSIONS (jpg, jpeg, png)
        
        Args:
            file_info: Uploaded file information with metadata
        
        Raises:
            ValidationError: If any validation check fails, with descriptive message
        
        Requirements:
            - Requirement 1.1: Validate file size <= 10,485,760 bytes
            - Requirement 1.2: Validate content type in allowed list
            - Requirement 1.3: Validate extension in allowed list
            - Requirement 1.4: Return ValidationError with descriptive message
        
        Example:
            >>> processor = ImageProcessor()
            >>> file_info = UploadFileInfo(
            ...     filename="photo.jpg",
            ...     content_type="image/jpeg",
            ...     size=5_000_000,
            ...     content=b"..."
            ... )
            >>> processor.validate_upload(file_info)  # Passes
            >>> 
            >>> large_file = UploadFileInfo(..., size=20_000_000, ...)
            >>> processor.validate_upload(large_file)  # Raises ValidationError
        """
        # Check file size
        if file_info.size > settings.upload.max_file_size:
            max_mb = settings.upload.max_file_size / (1024 * 1024)
            raise ValidationError(f"File too large. Max size: {max_mb:.0f}MB")

        # Check content type
        if file_info.content_type not in settings.upload.allowed_content_types:
            allowed = ", ".join(settings.upload.allowed_content_types)
            raise ValidationError(f"Invalid file type. Allowed: {allowed}")

        # Check extension
        if file_info.extension not in settings.upload.allowed_extensions:
            allowed = ", ".join(settings.upload.allowed_extensions)
            raise ValidationError(f"Invalid extension. Allowed: {allowed}")

    def process(
        self,
        file_info: UploadFileInfo,
        target_resolution: Resolution,
        use_ai: bool = True,
        enhance_faces: bool = True
    ) -> ProcessedImage:
        """
        Process and upscale an image.

        Args:
            file_info: Uploaded file information
            target_resolution: Target resolution (2k, 4k, 8k)
            use_ai: Whether to use AI upscaling or traditional method
            enhance_faces: Whether to apply CodeFormer face enhancement (AI only)

        Returns:
            ProcessedImage with metadata
        """
        self.validate_upload(file_info)

        # Load source image
        source_image = file_info.to_image()

        return self.process_from_image(source_image, target_resolution, use_ai, enhance_faces)

    def process_from_image(
        self,
        source_image: Image.Image,
        target_resolution: Resolution,
        use_ai: bool = True,
        enhance_faces: bool = True
    ) -> ProcessedImage:
        """
        Process and upscale an already-loaded image.

        This method is useful when the image has already been decoded
        (e.g., for validation) to avoid double decoding.

        Args:
            source_image: PIL Image object (already loaded)
            target_resolution: Target resolution (2k, 4k, 8k)
            use_ai: Whether to use AI upscaling or traditional method
            enhance_faces: Whether to apply CodeFormer face enhancement (AI only)

        Returns:
            ProcessedImage with metadata
        """
        original_size = source_image.size
        method = "ai" if use_ai else "standard"

        # Log request context (Requirement 7.4)
        logger.info(
            "Processing image",
            original_width=original_size[0],
            original_height=original_size[1],
            target_resolution=target_resolution.value,
            method=method,
            enhance_faces=enhance_faces and use_ai
        )
        
        # Track processing time
        start_time = time.time()

        try:
            # Upscale
            if use_ai:
                # AI upscaling: upscale by scale factor (2x or 4x)
                # Get scale from target_resolution (2k=2x, 4k=4x)
                scale_factor = 2 if target_resolution.value == '2k' else 4
                final_image = self._upscale_ai(source_image, outscale=scale_factor, enhance_faces=enhance_faces)
                final_size = final_image.size

                # Calculate actual scale factor
                scale_w = final_size[0] / original_size[0]
                scale_h = final_size[1] / original_size[1]

                result = ProcessedImage(
                    image=final_image,
                    original_width=original_size[0],
                    original_height=original_size[1],
                    final_width=final_size[0],
                    final_height=final_size[1],
                    scale_factor=(scale_w + scale_h) / 2,
                    format=ImageFormat(self.config.format)
                )
            else:
                # Traditional upscaling: use dedicated method
                result = self._upscale_traditional(source_image, target_resolution)
            
            # Log processing metrics (Requirement 7.4)
            duration = time.time() - start_time
            logger.info(
                "Image processing completed",
                method=method,
                original_width=original_size[0],
                original_height=original_size[1],
                final_width=result.final_width,
                final_height=result.final_height,
                scale_factor=round(result.scale_factor, 2),
                duration_seconds=round(duration, 3)
            )
            
            return result
            
        except Exception as e:
            # Log error with context (Requirement 7.6)
            duration = time.time() - start_time
            logger.error(
                "Image processing failed",
                error=e,
                method=method,
                original_width=original_size[0],
                original_height=original_size[1],
                target_resolution=target_resolution.value,
                duration_seconds=round(duration, 3),
                exc_info=True
            )
            raise

    def _upscale_ai(
        self,
        image: Image.Image,
        outscale: int = 4,
        enhance_faces: bool = True
    ) -> Image.Image:
        """
        Upscale image using AI model (Real-ESRGAN) with optional CodeFormer face enhancement.

        Pipeline:
            1. Real-ESRGAN upscale with specified scale factor
            2. CodeFormer face enhancement (if enabled and faces detected)

        Args:
            image: PIL Image in RGB mode
            outscale: Scale factor (2 or 4)
            enhance_faces: Whether to apply CodeFormer face enhancement

        Returns:
            PIL Image: Upscaled image with enhanced faces (outscale x larger dimensions)

        Raises:
            ImageProcessingError: If AI inference fails

        Example:
            >>> processor = ImageProcessor()
            >>> input_img = Image.open("photo.jpg")  # 1920x1080
            >>> upscaled = processor._upscale_ai(input_img, outscale=4, enhance_faces=True)
            >>> upscaled.size  # (7680, 4320)
        """
        # Step 1: Real-ESRGAN upscale
        upscaled = self.model_manager.upscale(image, outscale=outscale)

        # Step 2: CodeFormer face enhancement (if enabled)
        if enhance_faces and self.codeformer_manager.is_enabled:
            try:
                upscaled = self.codeformer_manager.enhance_faces(
                    upscaled,
                    weight=settings.model.codeformer_weight,
                    face_upsample=settings.model.codeformer_face_upsample
                )
            except Exception as e:
                # Log error but return upscaled image without face enhancement
                logger.warning(
                    "Face enhancement failed, returning upscaled image without enhancement",
                    error=e,
                    exc_info=True
                )

        return upscaled

    def _upscale_traditional(
        self,
        image: Image.Image,
        target_resolution: Resolution
    ) -> ProcessedImage:
        """
        Upscale using LANCZOS algorithm.
        
        Args:
            image: PIL Image object (must be in RGB mode)
            target_resolution: Target resolution enum (2k, 4k, 8k)
        
        Returns:
            ProcessedImage with upscaled image and metadata
        
        Requirements: 3.1, 3.2, 4.1, 4.2, 4.3, 4.4, 8.1, 8.2, 8.3
        
        Note: RGB conversion is handled by UploadFileInfo.to_image() to avoid duplication
        """
        # Store original dimensions
        original_size = image.size
        start_time = time.time()
        
        # Log processing start with structured context (Requirement 8.1, 7.4)
        logger.info(
            "Starting LANCZOS upscaling",
            original_width=original_size[0],
            original_height=original_size[1],
            target_resolution=target_resolution.value,
            method="lanczos"
        )
        
        try:
            # Ensure RGB mode (defensive check, should already be RGB from to_image())
            if image.mode != 'RGB':
                logger.info(
                    "Converting image mode",
                    from_mode=image.mode,
                    to_mode="RGB"
                )
                image = image.convert('RGB')
            
            # Get target dimensions from RESOLUTION_MAP (Requirement 3.1)
            target_dims = self.config.get_dimensions(target_resolution.value)
            
            # Call _resize_to_target() with LANCZOS resampling (Requirement 3.1, 3.3)
            resized_image, scale = self._resize_to_target(image, target_dims)
            final_size = resized_image.size
            
            # Calculate scale_factor from original to final (Requirements 4.1, 4.2, 4.3)
            scale_w = final_size[0] / original_size[0]
            scale_h = final_size[1] / original_size[1]
            scale_factor = (scale_w + scale_h) / 2
            
            # Round scale_factor to 2 decimals using ROUND_HALF_UP (Requirement 4.4)
            from decimal import Decimal, ROUND_HALF_UP
            scale_factor_rounded = float(
                Decimal(str(scale_factor)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            )
            
            # Log processing complete with metrics (Requirements 8.2, 8.3, 7.4)
            duration = time.time() - start_time
            logger.info(
                "LANCZOS upscaling complete",
                method="lanczos",
                original_width=original_size[0],
                original_height=original_size[1],
                final_width=final_size[0],
                final_height=final_size[1],
                scale_factor=scale_factor_rounded,
                duration_seconds=round(duration, 3)
            )
            
            # Return ProcessedImage with metadata
            return ProcessedImage(
                image=resized_image,
                original_width=original_size[0],
                original_height=original_size[1],
                final_width=final_size[0],
                final_height=final_size[1],
                scale_factor=scale_factor_rounded,
                format=ImageFormat(self.config.format)
            )
            
        except Exception as e:
            # Log error with context (Requirement 7.6)
            duration = time.time() - start_time
            logger.error(
                "LANCZOS upscaling failed",
                error=e,
                method="lanczos",
                original_width=original_size[0],
                original_height=original_size[1],
                target_resolution=target_resolution.value,
                duration_seconds=round(duration, 3),
                exc_info=True
            )
            raise

    def _resize_to_target(
        self,
        image: Image.Image,
        target_dims: Tuple[int, int]
    ) -> Tuple[Image.Image, float]:
        """
        Resize image to fit within target dimensions while maintaining aspect ratio.
        
        Strategy: Fit-within maintaining aspect ratio
        - Calculate scale factors for width and height
        - Choose minimum scale to fit within target
        - Calculate output dimensions maintaining aspect ratio
        - Verify output fits within target (w_out <= w_target, h_out <= h_target)
        
        Args:
            image: PIL Image object
            target_dims: (target_width, target_height)
        
        Returns:
            Tuple of (resized_image, scale_factor)
        
        Raises:
            ImageProcessingError: If output dimensions exceed target
        
        Requirements: 3.3, 6.1, 6.2, 6.3
        """
        target_w, target_h = target_dims
        current_w, current_h = image.size

        # Calculate scale factors
        scale_w = target_w / current_w
        scale_h = target_h / current_h
        
        # Choose minimum scale to fit within target
        scale = min(scale_w, scale_h)

        # Calculate output dimensions maintaining aspect ratio
        new_w = int(current_w * scale)
        new_h = int(current_h * scale)
        
        # Verify output fits within target (explicit check instead of assert)
        if new_w > target_w:
            raise ImageProcessingError(
                f"Image resize failed: output width {new_w}px exceeds target {target_w}px "
                f"(original: {current_w}x{current_h}px, target: {target_w}x{target_h}px)"
            )
        if new_h > target_h:
            raise ImageProcessingError(
                f"Image resize failed: output height {new_h}px exceeds target {target_h}px "
                f"(original: {current_w}x{current_h}px, target: {target_w}x{target_h}px)"
            )

        # Resize with LANCZOS
        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        return resized_image, scale
    def get_supported_resolutions(self) -> list[str]:
        """
        Get list of supported target resolutions.
        
        Returns:
            list[str]: Supported resolution strings (e.g., ["2k", "4k"])
        
        Example:
            >>> processor = ImageProcessor()
            >>> processor.get_supported_resolutions()
            ['2k', '4k']
        """
        return self.config.supported_resolutions

    def get_max_file_size_mb(self) -> float:
        """
        Get maximum allowed file size in megabytes.
        
        Returns:
            float: Max file size in MB (e.g., 10.0 for 10 MiB)
        
        Example:
            >>> processor = ImageProcessor()
            >>> processor.get_max_file_size_mb()
            10.0
        """
        return settings.upload.max_file_size / (1024 * 1024)
