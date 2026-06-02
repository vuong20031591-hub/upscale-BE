"""
Validator for face enhancement parameters.

This module provides validation logic for face enhancement requests,
ensuring parameters are valid and applying appropriate defaults.
"""

from typing import Optional
import logging

from app.services.codeformer_manager import CodeFormerManager, CodeFormerMode
from app.core import ValidationError, settings

logger = logging.getLogger(__name__)


class FaceEnhancementValidator:
    """
    Validator for face enhancement parameters.
    
    Validates mode, weight, and face_upsample parameters for face enhancement
    requests. Applies mode-specific rules and default values.
    
    Requirements: 4.2, 4.4, 6.3, 8.7, 8.8
    """
    
    VALID_MODES = {"restoration", "colorization", "inpainting"}
    
    def __init__(self):
        """
        Initialize validator with access to CodeFormerManager configurations.
        
        Accesses MODEL_CONFIGS from CodeFormerManager to validate parameters
        against mode-specific requirements.
        """
        # Access CodeFormerManager's MODEL_CONFIGS for validation
        self.model_configs = CodeFormerManager.MODEL_CONFIGS
    
    def validate_mode(self, mode: str) -> CodeFormerMode:
        """
        Validate and convert mode string to CodeFormerMode enum.
        
        Args:
            mode: Mode string (restoration|colorization|inpainting)
        
        Returns:
            CodeFormerMode enum value
        
        Raises:
            ValidationError: If mode is invalid
        
        Requirements: 4.2, 8.6
        
        Example:
            >>> validator = FaceEnhancementValidator()
            >>> mode = validator.validate_mode("restoration")
            >>> mode == CodeFormerMode.RESTORATION
            True
        """
        if mode not in self.VALID_MODES:
            raise ValidationError(
                f"Invalid mode '{mode}'. Must be one of: {', '.join(sorted(self.VALID_MODES))}"
            )
        return CodeFormerMode(mode)
    
    def validate_weight(self, weight: Optional[float], mode: CodeFormerMode) -> float:
        """
        Validate weight parameter with mode-specific rules.
        
        For non-restoration modes (colorization, inpainting), the weight parameter
        is ignored (not an error) and the fixed weight from model config is used.
        For restoration mode, weight is validated and clamped to [0, 1] range.
        
        Args:
            weight: Weight value or None
            mode: CodeFormer mode
        
        Returns:
            Validated weight value (clamped to 0-1 for restoration, fixed for others)
        
        Notes:
            - For colorization: Always returns w=0 (fixed)
            - For inpainting: Always returns w=1 (fixed)
            - For restoration: Validates and clamps to [0, 1]
            - Out-of-range weights are clamped with warning (not error)
            - User-provided weight is ignored for non-restoration modes (not error)
        
        Requirements: 4.4, 6.3, 8.7, 8.8
        
        Example:
            >>> validator = FaceEnhancementValidator()
            >>> # Restoration mode with valid weight
            >>> validator.validate_weight(0.5, CodeFormerMode.RESTORATION)
            0.5
            >>> # Restoration mode with out-of-range weight (clamped)
            >>> validator.validate_weight(1.5, CodeFormerMode.RESTORATION)
            1.0
            >>> # Colorization mode ignores weight parameter
            >>> validator.validate_weight(0.8, CodeFormerMode.COLORIZATION)
            0
        """
        # For non-restoration modes, use fixed weight from config
        # Weight parameter is ignored (not an error) per Requirements 4.4, 8.8
        if mode != CodeFormerMode.RESTORATION:
            if weight is not None:
                logger.info(
                    f"Weight parameter ignored for {mode.value} mode, "
                    f"using fixed value w={self.model_configs[mode]['w']}"
                )
            return self.model_configs[mode]["w"]
        
        # For restoration mode, validate and clamp weight
        if weight is None:
            # Use default from settings
            return settings.model.codeformer_weight
        
        # Clamp to valid range [0, 1] per Requirements 8.7
        # Out-of-range values are clamped with warning (not error)
        if not 0 <= weight <= 1:
            logger.warning(
                f"Weight {weight} out of range [0, 1], clamping to valid range"
            )
            weight = max(0.0, min(1.0, weight))
        
        return weight
    
    def validate_face_upsample(self, face_upsample: Optional[bool]) -> bool:
        """
        Validate face_upsample parameter with default fallback.
        
        Args:
            face_upsample: Face upsample flag or None
        
        Returns:
            Validated boolean value (uses default from settings if None)
        
        Requirements: 6.3
        
        Example:
            >>> validator = FaceEnhancementValidator()
            >>> validator.validate_face_upsample(True)
            True
            >>> validator.validate_face_upsample(None)  # Uses default from settings
            True
        """
        if face_upsample is None:
            return settings.model.codeformer_face_upsample
        return face_upsample
    
    def validate_bg_upscale(self, bg_upscale: Optional[int]) -> int:
        """
        Validate background upscale factor.
        
        Args:
            bg_upscale: Background upscale factor or None
        
        Returns:
            Validated bg_upscale (1, 2, or 4)
        
        Raises:
            ValidationError: If bg_upscale is not 1, 2, or 4
        
        Requirements: 11.3, 11.11 (validate bg_upscale values)
        """
        if bg_upscale is None:
            return 2  # Default value
        
        if bg_upscale not in [1, 2, 4]:
            raise ValidationError(
                f"Invalid bg_upscale '{bg_upscale}'. Must be 1, 2, or 4"
            )
        
        return bg_upscale
