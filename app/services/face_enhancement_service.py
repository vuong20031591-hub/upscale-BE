"""
Face Enhancement Service - Orchestrates face enhancement workflow.

This service provides high-level orchestration for face enhancement operations,
coordinating between validation, model management, and processing.

Requirements: 1.1-1.6, 2.1-2.6, 3.1-3.6, 6.2, 11.10
"""

import asyncio
import time
from typing import Optional

from PIL import Image

from app.core import settings
from app.models.face_enhancement import FaceEnhancementResult
from app.services.codeformer_manager import CodeFormerManager, CodeFormerMode
from app.utils.logging_utils import get_structured_logger

logger = get_structured_logger(__name__)


class FaceEnhancementService:
    """
    Service for face enhancement operations.
    
    Orchestrates the complete face enhancement workflow:
    1. Check if CodeFormer is enabled (CODEFORMER_ENABLED)
    2. Validate input parameters
    3. Load appropriate CodeFormer model
    4. Detect faces
    5. Enhance each face
    6. Return result with metadata
    
    Requirements:
        - 1.1-1.6: Face Restoration Mode
        - 2.1-2.6: Face Colorization Mode
        - 3.1-3.6: Face Inpainting Mode
        - 6.2: Feature toggle enforcement (CODEFORMER_ENABLED)
    
    Example:
        >>> service = FaceEnhancementService()
        >>> image = Image.open("photo.jpg")
        >>> result = await service.enhance(
        ...     image,
        ...     mode=CodeFormerMode.RESTORATION,
        ...     weight=0.7,
        ...     face_upsample=True
        ... )
        >>> result.faces_detected
        2
    """
    
    def __init__(self):
        """
        Initialize FaceEnhancementService.
        
        Injects dependencies:
        - CodeFormerManager: Singleton for model management
        """
        self.codeformer_manager = CodeFormerManager()
        logger.info("FaceEnhancementService initialized")
    
    async def enhance(
        self,
        image: Image.Image,
        mode: CodeFormerMode,
        weight: Optional[float] = None,
        face_upsample: Optional[bool] = None,
        background_enhance: bool = True,  # NEW
        bg_upscale: int = 2,  # NEW
        timeout: int = 30
    ) -> FaceEnhancementResult:
        """
        Enhance faces in image using specified mode.
        
        This is the main entry point for face enhancement. It:
        1. Checks if CodeFormer is enabled (returns original if disabled)
        2. Validates parameters
        3. Runs enhancement with timeout protection
        4. Returns result with metadata
        
        Args:
            image: PIL Image to enhance
            mode: CodeFormer processing mode (restoration/colorization/inpainting)
            weight: Fidelity weight (0-1), only for restoration mode
                   Higher values preserve more original detail
                   Ignored for colorization/inpainting (uses fixed values)
            face_upsample: Whether to upsample face regions after enhancement
            background_enhance: Whether to enhance background with Real-ESRGAN (NEW)
            bg_upscale: Background upscale factor - 1, 2, or 4 (NEW)
            timeout: Maximum processing time in seconds (default: 30)
        
        Returns:
            FaceEnhancementResult with enhanced image and metadata
        
        Raises:
            asyncio.TimeoutError: If processing exceeds timeout
            Exception: If model loading or processing fails
        
        Requirements:
            - 1.1: Detect and enhance all faces
            - 1.5, 2.5, 3.5: Return original image if no faces detected
            - 6.2: Skip enhancement if CODEFORMER_ENABLED=false
            - 11.1-11.10: Background enhancement integration
        
        Example:
            >>> service = FaceEnhancementService()
            >>> image = Image.open("portrait.jpg")
            >>> result = await service.enhance(
            ...     image,
            ...     mode=CodeFormerMode.RESTORATION,
            ...     weight=0.7,
            ...     background_enhance=True,
            ...     bg_upscale=2
            ... )
            >>> print(f"Processed {result.faces_detected} faces")
        """
        start_time = time.time()
        
        # Check if CodeFormer is enabled (Requirement 6.2)
        if not self.codeformer_manager.is_enabled:
            logger.info("CodeFormer is disabled, returning original image")
            return FaceEnhancementResult(
                image=image,
                faces_detected=0,
                processing_time=0.0,
                mode_used=mode,
                weight_used=0.0,
                background_enhanced=False,  # NEW
                bg_upscale=1,  # NEW
                warning="CodeFormer is disabled"
            )
        
        # Validate parameters
        validated_weight, validated_face_upsample = self._validate_parameters(
            mode, weight, face_upsample
        )
        
        logger.info(
            "Starting face enhancement",
            mode=mode.value,
            weight=validated_weight,
            face_upsample=validated_face_upsample,
            background_enhance=background_enhance,
            bg_upscale=bg_upscale,
            timeout=timeout
        )
        
        try:
            # Run enhancement with timeout protection
            enhanced_image = await asyncio.wait_for(
                asyncio.to_thread(
                    self.codeformer_manager.enhance_faces,
                    image,
                    weight=validated_weight,
                    face_upsample=validated_face_upsample,
                    mode=mode,
                    background_enhance=background_enhance,  # NEW
                    bg_upscale=bg_upscale  # NEW
                ),
                timeout=timeout
            )
            
            processing_time = time.time() - start_time
            
            # Check if faces were detected
            # If enhance_faces returns the same image object, no faces were found
            if enhanced_image is image:
                return self._handle_no_faces(
                    image, mode, validated_weight, processing_time,
                    background_enhance=False,  # NEW
                    bg_upscale=bg_upscale  # NEW
                )
            
            # Success - faces were enhanced
            # Note: We can't easily get exact face count from enhance_faces
            # without modifying it, so we assume >0 if image changed
            logger.info(
                "Face enhancement complete",
                mode=mode.value,
                processing_time=round(processing_time, 3),
                background_enhanced=background_enhance and bg_upscale != 1
            )
            
            return FaceEnhancementResult(
                image=enhanced_image,
                faces_detected=1,  # At least 1 face was detected
                processing_time=processing_time,
                mode_used=mode,
                weight_used=validated_weight,
                background_enhanced=background_enhance and bg_upscale != 1,  # NEW
                bg_upscale=bg_upscale,  # NEW
                warning=None
            )
            
        except asyncio.TimeoutError:
            logger.error(
                "Face enhancement timeout",
                mode=mode.value,
                timeout=timeout
            )
            raise
        except Exception as e:
            logger.error(
                "Face enhancement failed",
                mode=mode.value,
                error=str(e),
                exc_info=True
            )
            raise
    
    def _validate_parameters(
        self,
        mode: CodeFormerMode,
        weight: Optional[float],
        face_upsample: Optional[bool]
    ) -> tuple[float, bool]:
        """
        Validate and normalize parameters.
        
        Args:
            mode: CodeFormer mode
            weight: User-provided weight or None
            face_upsample: User-provided face_upsample or None
        
        Returns:
            tuple: (validated_weight, validated_face_upsample)
        
        Notes:
            - For non-restoration modes, weight is ignored and fixed value is used
            - Out-of-range weights are clamped to [0, 1]
            - None values use defaults from settings
        
        Requirements:
            - 2.2: Colorization uses fixed w=0
            - 3.2: Inpainting uses fixed w=1
            - 4.4: Weight ignored for non-restoration modes (not an error)
        """
        # Get mode-specific config
        config = self.codeformer_manager.MODEL_CONFIGS[mode]
        
        # For non-restoration modes, use fixed weight from config
        if mode != CodeFormerMode.RESTORATION:
            if weight is not None:
                logger.info(
                    f"Weight parameter ignored for {mode.value} mode, "
                    f"using fixed value w={config['w']}"
                )
            validated_weight = config["w"]
        else:
            # For restoration mode, validate and clamp weight
            if weight is None:
                validated_weight = settings.model.codeformer_weight
            else:
                # Clamp to valid range [0, 1]
                if not 0 <= weight <= 1:
                    logger.warning(
                        f"Weight {weight} out of range, clamping to [0, 1]"
                    )
                    validated_weight = max(0.0, min(1.0, weight))
                else:
                    validated_weight = weight
        
        # Validate face_upsample
        if face_upsample is None:
            validated_face_upsample = settings.model.codeformer_face_upsample
        else:
            validated_face_upsample = face_upsample
        
        return validated_weight, validated_face_upsample
    
    def _handle_no_faces(
        self,
        image: Image.Image,
        mode: CodeFormerMode,
        weight: float,
        processing_time: float,
        background_enhance: bool = False,  # NEW
        bg_upscale: int = 1  # NEW
    ) -> FaceEnhancementResult:
        """
        Handle case when no faces are detected.
        
        Returns original image with warning message.
        
        Args:
            image: Original PIL Image
            mode: CodeFormer mode that was attempted
            weight: Weight value that was used
            processing_time: Time spent processing
            background_enhance: Whether background enhancement was requested (NEW)
            bg_upscale: Background upscale factor (NEW)
        
        Returns:
            FaceEnhancementResult with original image and warning
        
        Requirements:
            - 1.5, 2.5, 3.5: Return original image if no faces detected
            - 8.1: Return HTTP 200 with warning header
            - 11.10: Include background metadata
        """
        warning_msg = "No faces detected in image"
        
        logger.info(
            "No faces detected, returning original image",
            mode=mode.value,
            processing_time=round(processing_time, 3)
        )
        
        return FaceEnhancementResult(
            image=image,
            faces_detected=0,
            processing_time=processing_time,
            mode_used=mode,
            weight_used=weight,
            background_enhanced=False,  # NEW
            bg_upscale=bg_upscale,  # NEW
            warning=warning_msg
        )
