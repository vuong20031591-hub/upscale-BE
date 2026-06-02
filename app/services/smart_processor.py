"""
Smart image processing service.
Handles automatic mode selection and processing based on image analysis.

This service encapsulates the logic for:
- Automatic image analysis
- Mode selection based on detected issues
- Image processing with appropriate enhancement method

Requirements: 3.1-3.6, 5.1-5.4
"""

import io
import time
import logging
from typing import Tuple
from PIL import Image

from app.services.image_analyzer import ImageAnalyzer
from app.services.codeformer_manager import CodeFormerManager, CodeFormerMode
from app.services.model_manager import ModelManager
from app.models.analysis import ProcessingMode, AnalysisResult
from app.utils.logging_utils import get_structured_logger

logger = get_structured_logger(__name__)


class SmartProcessingResult:
    """
    Result of smart image processing.
    
    Attributes:
        processed_image: The enhanced PIL Image
        analysis_result: Analysis result with detected issues
        selected_mode: Processing mode that was applied
        analysis_time: Time spent on analysis (seconds)
        processing_time: Time spent on processing (seconds)
        total_time: Total time (seconds)
        original_width: Original image width in pixels
        original_height: Original image height in pixels
        final_width: Final image width in pixels
        final_height: Final image height in pixels
    """
    
    def __init__(
        self,
        processed_image: Image.Image,
        analysis_result: AnalysisResult,
        selected_mode: ProcessingMode,
        analysis_time: float,
        processing_time: float,
        original_width: int,
        original_height: int,
        final_width: int,
        final_height: int
    ):
        self.processed_image = processed_image
        self.analysis_result = analysis_result
        self.selected_mode = selected_mode
        self.analysis_time = analysis_time
        self.processing_time = processing_time
        self.total_time = analysis_time + processing_time
        self.original_width = original_width
        self.original_height = original_height
        self.final_width = final_width
        self.final_height = final_height
    
    def to_bytes(self, format: str = 'PNG') -> io.BytesIO:
        """
        Convert processed image to bytes buffer.
        
        Args:
            format: Image format (default: PNG)
        
        Returns:
            BytesIO buffer containing image data
        """
        buffer = io.BytesIO()
        self.processed_image.save(buffer, format=format)
        buffer.seek(0)
        return buffer


class SmartProcessor:
    """
    Service for smart automatic image processing.
    
    Processes images based on detected issues:
    - Colorization for grayscale images
    - Inpainting for white mask images
    - Upscaling for low resolution images
    - Restoration for blurry images
    
    Usage:
        processor = SmartProcessor()
        result = processor.process_image(image)
        buffer = result.to_bytes()
    """
    
    def __init__(self):
        """Initialize smart processor with required services."""
        self.analyzer = ImageAnalyzer()
        self.codeformer = CodeFormerManager()
        self.model_manager = ModelManager()
    
    def process_image(
        self,
        image: Image.Image,
        filename: str = "unknown",
        bg_upscale: int = 2
    ) -> SmartProcessingResult:
        """
        Process image with automatic mode selection.
        
        Workflow:
            1. Analyze image to detect issues
            2. Select appropriate processing mode
            3. Process image with selected mode
            4. Return result with metadata
        
        Args:
            image: PIL Image to process
            filename: Original filename (for logging)
            bg_upscale: Background upscale factor (2 or 4, default: 2)
        
        Returns:
            SmartProcessingResult with processed image and metadata
        
        Raises:
            Exception: If processing fails (analysis failures are handled with fallback)
        
        Requirements: 3.1-3.6, 5.2, 4.7, 4.8
        """
        # Step 1: Analyze image to detect issues (Requirement 3.1)
        # With fallback to restoration mode if analysis fails (Requirement 4.8)
        analysis_start = time.time()
        try:
            analysis_result = self.analyzer.analyze(image)
            analysis_time = time.time() - analysis_start
        except Exception as e:
            # Detection failure: fallback to restoration mode (Requirement 4.8)
            analysis_time = time.time() - analysis_start
            logger.error(
                "Image analysis failed, falling back to restoration mode",
                filename=filename,
                error=str(e),
                analysis_time_seconds=round(analysis_time, 3),
                exc_info=True
            )
            
            # Create fallback analysis result with restoration mode
            from app.models.analysis import DetectionType, DetectionResult
            analysis_result = AnalysisResult(
                detections={
                    DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                    DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                    DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                    DetectionType.BLUR: DetectionResult(False, 0.0, {})
                },
                suggested_mode=ProcessingMode.RESTORATION,
                alternative_modes=[],
                explanation="Analysis failed, using restoration mode as fallback"
            )
        
        # Log analysis results (Requirement 3.6)
        from app.models.analysis import DetectionType
        logger.info(
            "Image analysis complete",
            filename=filename,
            grayscale_detected=analysis_result.detections[DetectionType.GRAYSCALE].detected,
            grayscale_confidence=analysis_result.detections[DetectionType.GRAYSCALE].confidence,
            white_mask_detected=analysis_result.detections[DetectionType.WHITE_MASK].detected,
            white_mask_confidence=analysis_result.detections[DetectionType.WHITE_MASK].confidence,
            low_res_detected=analysis_result.detections[DetectionType.LOW_RESOLUTION].detected,
            low_res_confidence=analysis_result.detections[DetectionType.LOW_RESOLUTION].confidence,
            blur_detected=analysis_result.detections[DetectionType.BLUR].detected,
            blur_confidence=analysis_result.detections[DetectionType.BLUR].confidence,
            suggested_mode=analysis_result.suggested_mode.value,
            explanation=analysis_result.explanation,
            analysis_time_seconds=round(analysis_time, 3)
        )
        
        # Step 2: Automatically select mode (Requirement 3.2)
        selected_mode = analysis_result.suggested_mode
        
        # Step 3: Process image with selected mode (Requirement 3.3, 5.2)
        # Wrap in try-catch to provide better error context (Requirement 4.7)
        processing_start = time.time()
        try:
            processed_image = self._process_with_mode(image, selected_mode, filename, bg_upscale)
            processing_time = time.time() - processing_start
        except Exception as e:
            # Processing failure: log and re-raise (Requirement 4.7)
            processing_time = time.time() - processing_start
            logger.error(
                "Image processing failed",
                filename=filename,
                mode=selected_mode.value,
                processing_time_seconds=round(processing_time, 3),
                error=str(e),
                exc_info=True
            )
            raise  # Re-raise to be handled by endpoint
        
        # Log processing success
        logger.info(
            "Image processing complete",
            filename=filename,
            mode=selected_mode.value,
            processing_time_seconds=round(processing_time, 3),
            output_size=f"{processed_image.size[0]}x{processed_image.size[1]}"
        )
        
        # Get original and final dimensions
        original_width, original_height = image.size
        final_width, final_height = processed_image.size
        
        # Step 4: Return result with metadata
        return SmartProcessingResult(
            processed_image=processed_image,
            analysis_result=analysis_result,
            selected_mode=selected_mode,
            analysis_time=analysis_time,
            processing_time=processing_time,
            original_width=original_width,
            original_height=original_height,
            final_width=final_width,
            final_height=final_height
        )
    
    def _process_with_mode(
        self,
        image: Image.Image,
        mode: ProcessingMode,
        filename: str,
        bg_upscale: int = 2
    ) -> Image.Image:
        """
        Process image with specified mode.
        
        Args:
            image: PIL Image to process
            mode: ProcessingMode enum value
            filename: Original filename (for logging)
            bg_upscale: Background upscale factor (2 or 4, default: 2)
        
        Returns:
            Processed PIL Image
        
        Raises:
            Exception: If processing fails
        
        Requirements: 5.2
        """
        if mode == ProcessingMode.COLORIZATION:
            return self._process_colorization(image, filename, bg_upscale)
        
        elif mode == ProcessingMode.INPAINTING:
            return self._process_inpainting(image, filename, bg_upscale)
        
        elif mode == ProcessingMode.UPSCALING:
            return self._process_upscaling(image, filename, bg_upscale)
        
        elif mode == ProcessingMode.RESTORATION:
            return self._process_restoration(image, filename, bg_upscale)
        
        else:
            # Fallback: Should never reach here due to enum validation
            logger.warning(
                "Unknown processing mode, falling back to restoration",
                mode=mode.value,
                filename=filename
            )
            return self._process_restoration(image, filename, bg_upscale)
    
    def _process_colorization(self, image: Image.Image, filename: str, bg_upscale: int = 2) -> Image.Image:
        """
        Process image with colorization mode.
        
        Args:
            image: PIL Image to process
            filename: Original filename (for logging)
            bg_upscale: Background upscale factor (2 or 4, default: 2)
        
        Returns:
            Colorized PIL Image
        """
        logger.info("Processing with colorization mode", filename=filename, bg_upscale=bg_upscale)
        
        self.codeformer.load(CodeFormerMode.COLORIZATION)
        processed_image = self.codeformer.enhance_faces(
            image=image,
            mode=CodeFormerMode.COLORIZATION,
            weight=0,  # Fixed w=0 for colorization
            face_upsample=True,
            background_enhance=True,
            bg_upscale=bg_upscale
        )
        
        return processed_image
    
    def _process_inpainting(self, image: Image.Image, filename: str, bg_upscale: int = 2) -> Image.Image:
        """
        Process image with inpainting mode.
        
        Uses tile-based processing for large images (>512x512) to maintain quality.
        
        Args:
            image: PIL Image to process
            filename: Original filename (for logging)
            bg_upscale: Background upscale factor (2 or 4, default: 2)
        
        Returns:
            Inpainted PIL Image
        """
        logger.info("Processing with inpainting mode", filename=filename, bg_upscale=bg_upscale)
        
        self.codeformer.load(CodeFormerMode.INPAINTING)
        processed_image = self.codeformer.enhance_faces(
            image=image,
            mode=CodeFormerMode.INPAINTING,
            weight=1,  # Fixed w=1 for inpainting
            face_upsample=True,
            background_enhance=True,
            bg_upscale=bg_upscale
        )
        
        return processed_image
    
    def _process_upscaling(self, image: Image.Image, filename: str, bg_upscale: int = 2) -> Image.Image:
        """
        Process image with upscaling mode.
        
        Args:
            image: PIL Image to process
            filename: Original filename (for logging)
            bg_upscale: Background upscale factor (2 or 4, default: 2)
        
        Returns:
            Upscaled PIL Image
        """
        logger.info("Processing with upscaling mode", filename=filename, bg_upscale=bg_upscale)
        
        processed_image = self.model_manager.upscale(image, outscale=bg_upscale)
        
        return processed_image
    
    def _process_restoration(self, image: Image.Image, filename: str, bg_upscale: int = 2) -> Image.Image:
        """
        Process image with restoration mode.
        
        Args:
            image: PIL Image to process
            filename: Original filename (for logging)
            bg_upscale: Background upscale factor (2 or 4, default: 2)
        
        Returns:
            Restored PIL Image
        """
        logger.info("Processing with restoration mode", filename=filename, bg_upscale=bg_upscale)
        
        self.codeformer.load(CodeFormerMode.RESTORATION)
        processed_image = self.codeformer.enhance_faces(
            image=image,
            mode=CodeFormerMode.RESTORATION,
            weight=0.7,  # Default fidelity weight
            face_upsample=True,
            background_enhance=True,
            bg_upscale=bg_upscale
        )
        
        return processed_image
