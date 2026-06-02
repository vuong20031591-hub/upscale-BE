"""
Data models for face enhancement operations.

This module defines dataclasses for face enhancement requests and results,
supporting CodeFormer's 3 modes: restoration, colorization, and inpainting.

Requirements:
    - Requirement 4.1: API endpoint data structures
    - Requirement 4.5: Response metadata headers
"""

from dataclasses import dataclass
from typing import Optional

from PIL import Image
from fastapi import UploadFile

from app.services.codeformer_manager import CodeFormerMode


@dataclass
class FaceEnhancementRequest:
    """
    Raw request parameters for face enhancement.
    
    This dataclass represents the raw input from the API endpoint
    before validation. It contains the uploaded file and user-provided
    parameters that need to be validated.
    
    Attributes:
        file: Uploaded image file from multipart/form-data
        mode: Processing mode as string (will be validated to CodeFormerMode)
        weight: Optional fidelity weight (0-1), only for restoration mode
        face_upsample: Optional flag to enable face upsampling
    
    Requirements:
        - Requirement 4.1: API endpoint accepts multipart/form-data
        - Requirement 4.2: Required parameters (file, mode)
        - Requirement 4.3: Optional parameters (weight, face_upsample)
    
    Example:
        >>> request = FaceEnhancementRequest(
        ...     file=upload_file,
        ...     mode="restoration",
        ...     weight=0.7,
        ...     face_upsample=True
        ... )
        >>> validated = request.validate()
    """
    file: UploadFile
    mode: str  # Will be validated to CodeFormerMode
    weight: Optional[float] = None
    face_upsample: Optional[bool] = None
    
    def validate(self) -> "ValidatedFaceEnhancementRequest":
        """
        Validate and convert to validated request.
        
        This method should be called by the service layer to validate
        the raw request parameters and convert them to a validated request
        with proper types.
        
        Returns:
            ValidatedFaceEnhancementRequest with validated parameters
        
        Raises:
            ValidationError: If validation fails (handled by validator)
        
        Note:
            The actual validation logic is implemented in FaceEnhancementValidator.
            This method serves as a placeholder for the validation workflow.
        """
        # Validation logic will be implemented by FaceEnhancementValidator
        # This is just a placeholder to define the interface
        raise NotImplementedError(
            "Validation should be performed by FaceEnhancementValidator"
        )


@dataclass
class ValidatedFaceEnhancementRequest:
    """
    Validated face enhancement request with proper types.
    
    This dataclass represents a fully validated request with all parameters
    converted to their proper types and defaults applied. It's ready to be
    processed by the face enhancement service.
    
    Attributes:
        image: PIL Image object loaded from uploaded file
        mode: CodeFormerMode enum (validated)
        weight: Fidelity weight (always has value after validation)
        face_upsample: Face upsample flag (always has value after validation)
    
    Requirements:
        - Requirement 4.2: Validated mode parameter
        - Requirement 4.3: Validated weight and face_upsample with defaults
        - Requirement 4.4: Mode-specific weight enforcement
    
    Example:
        >>> validated = ValidatedFaceEnhancementRequest(
        ...     image=pil_image,
        ...     mode=CodeFormerMode.RESTORATION,
        ...     weight=0.7,
        ...     face_upsample=True
        ... )
    """
    image: Image.Image
    mode: CodeFormerMode
    weight: float  # Always has value after validation
    face_upsample: bool  # Always has value after validation


@dataclass
class FaceEnhancementResult:
    """
    Result of face enhancement operation with metadata.
    
    This dataclass contains the enhanced image along with metadata about
    the processing operation. It provides a method to convert metadata
    to HTTP response headers.
    
    Attributes:
        image: Enhanced PIL Image
        faces_detected: Number of faces detected and processed
        processing_time: Processing duration in seconds
        mode_used: CodeFormer mode that was applied
        weight_used: Fidelity weight value that was used
        background_enhanced: Whether background was enhanced (NEW)
        bg_upscale: Background upscale factor used (NEW)
        warning: Optional warning message (e.g., "No faces detected")
    
    Requirements:
        - Requirement 1.6: Metadata for restoration mode
        - Requirement 2.6: Metadata for colorization mode
        - Requirement 3.6: Metadata for inpainting mode
        - Requirement 4.5: Response headers with metadata
        - Requirement 8.1: Warning header when no faces detected
        - Requirement 11.10: Background enhancement metadata
    
    Example:
        >>> result = FaceEnhancementResult(
        ...     image=enhanced_image,
        ...     faces_detected=2,
        ...     processing_time=4.523,
        ...     mode_used=CodeFormerMode.RESTORATION,
        ...     weight_used=0.7,
        ...     background_enhanced=True,
        ...     bg_upscale=2,
        ...     warning=None
        ... )
        >>> headers = result.to_response_headers()
        >>> headers["X-Faces-Detected"]
        '2'
    """
    image: Image.Image
    faces_detected: int
    processing_time: float
    mode_used: CodeFormerMode
    weight_used: float
    background_enhanced: bool  # NEW
    bg_upscale: int  # NEW
    warning: Optional[str] = None
    
    def to_response_headers(self) -> dict:
        """
        Convert metadata to HTTP response headers.
        
        Converts the result metadata into a dictionary of HTTP headers
        that can be included in the StreamingResponse. All numeric values
        are formatted as strings with appropriate precision.
        
        Returns:
            Dictionary mapping header names to string values:
            - X-Faces-Detected: Number of faces (integer as string)
            - X-Processing-Time: Processing time in seconds (3 decimal places)
            - X-Mode-Used: Mode name (restoration/colorization/inpainting)
            - X-Weight-Used: Weight value (2 decimal places)
            - X-Background-Enhanced: Whether background was enhanced (NEW)
            - X-BG-Upscale: Background upscale factor (NEW)
            - X-Warning: Warning message (only if warning is not None)
        
        Requirements:
            - Requirement 4.5: Response metadata headers
            - Requirement 8.1: Warning header for no faces
            - Requirement 11.10: Add background headers
        
        Example:
            >>> result = FaceEnhancementResult(
            ...     image=img,
            ...     faces_detected=2,
            ...     processing_time=4.523456,
            ...     mode_used=CodeFormerMode.RESTORATION,
            ...     weight_used=0.7,
            ...     background_enhanced=True,
            ...     bg_upscale=2,
            ...     warning=None
            ... )
            >>> headers = result.to_response_headers()
            >>> headers
            {
                'X-Faces-Detected': '2',
                'X-Processing-Time': '4.523',
                'X-Mode-Used': 'restoration',
                'X-Weight-Used': '0.70',
                'X-Background-Enhanced': 'True',
                'X-BG-Upscale': '2'
            }
        """
        headers = {
            "X-Faces-Detected": str(self.faces_detected),
            "X-Processing-Time": f"{self.processing_time:.3f}",
            "X-Mode-Used": self.mode_used.value,
            "X-Weight-Used": f"{self.weight_used:.2f}",
            "X-Background-Enhanced": str(self.background_enhanced),  # NEW
            "X-BG-Upscale": str(self.bg_upscale)  # NEW
        }
        
        if self.warning:
            headers["X-Warning"] = self.warning
        
        return headers
