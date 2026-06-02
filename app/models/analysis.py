"""
Data models for smart auto-detection image analysis.

This module defines enums, dataclasses, and Pydantic models for the smart
auto-detection feature that analyzes images to detect issues (grayscale,
white mask, low resolution, blur) and suggests appropriate processing modes.

Requirements:
    - Requirements 1.1-1.5: Image analysis detection
    - Requirements 2.1-2.7: Mode suggestion logic
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any

from pydantic import BaseModel, Field


class DetectionType(str, Enum):
    """
    Types of image issues that can be detected.
    
    Attributes:
        GRAYSCALE: Image has identical RGB channels (grayscale)
        WHITE_MASK: Image has white regions (RGB >= 240) covering >30% area
        LOW_RESOLUTION: Image dimensions are below 512 pixels
        BLUR: Image has low Laplacian variance (< 100)
    
    Requirements:
        - Requirement 1.1: Grayscale detection
        - Requirement 1.2: White mask detection
        - Requirement 1.3: Low resolution detection
        - Requirement 1.4: Blur detection
    
    Example:
        >>> detection_type = DetectionType.GRAYSCALE
        >>> detection_type.value
        'grayscale'
    """
    GRAYSCALE = "grayscale"
    WHITE_MASK = "white_mask"
    LOW_RESOLUTION = "low_resolution"
    BLUR = "blur"


class ProcessingMode(str, Enum):
    """
    Available processing modes for image enhancement.
    
    Attributes:
        COLORIZATION: Convert grayscale images to color
        INPAINTING: Fill white mask regions with content
        UPSCALING: Increase image resolution
        RESTORATION: Enhance blurry or degraded images
    
    Requirements:
        - Requirement 2.1: Colorization mode suggestion
        - Requirement 2.2: Inpainting mode suggestion
        - Requirement 2.3: Upscaling mode suggestion
        - Requirement 2.4: Restoration mode suggestion
    
    Example:
        >>> mode = ProcessingMode.COLORIZATION
        >>> mode.value
        'colorization'
    """
    COLORIZATION = "colorization"
    INPAINTING = "inpainting"
    UPSCALING = "upscaling"
    RESTORATION = "restoration"


@dataclass
class DetectionResult:
    """
    Result of a single detection operation.
    
    Contains detection status, confidence score, and additional metadata
    for internal use within the ImageAnalyzer service.
    
    Attributes:
        detected: Whether the issue was detected
        confidence: Confidence score between 0.0 and 1.0
        metadata: Additional information specific to detection type
                 (e.g., white_percentage, variance, dimensions)
    
    Requirements:
        - Requirement 1.5: Confidence score between 0.0 and 1.0
    
    Example:
        >>> result = DetectionResult(
        ...     detected=True,
        ...     confidence=1.0,
        ...     metadata={"white_percentage": 35.5, "white_pixel_count": 50000}
        ... )
        >>> result.detected
        True
        >>> result.confidence
        1.0
    """
    detected: bool
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any]


@dataclass
class AnalysisResult:
    """
    Complete analysis result for an image.
    
    Contains all detection results, suggested processing mode, alternative
    modes, and explanation text. Used internally by ImageAnalyzer service.
    
    Attributes:
        detections: Dictionary mapping DetectionType to DetectionResult
        suggested_mode: Primary recommended processing mode
        alternative_modes: List of alternative processing modes
        explanation: Human-readable explanation for the suggestion
    
    Requirements:
        - Requirements 1.1-1.5: All detection results
        - Requirements 2.1-2.6: Mode suggestion logic
        - Requirement 2.7: Explanation text
    
    Example:
        >>> analysis = AnalysisResult(
        ...     detections={
        ...         DetectionType.GRAYSCALE: DetectionResult(True, 1.0, {}),
        ...         DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
        ...         DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
        ...         DetectionType.BLUR: DetectionResult(False, 0.0, {})
        ...     },
        ...     suggested_mode=ProcessingMode.COLORIZATION,
        ...     alternative_modes=[],
        ...     explanation="Image is grayscale with 100% confidence"
        ... )
    """
    detections: Dict[DetectionType, DetectionResult]
    suggested_mode: ProcessingMode
    alternative_modes: List[ProcessingMode]
    explanation: str


class DetectionResultModel(BaseModel):
    """
    Pydantic model for detection result in API responses.
    
    This model is used for JSON serialization in API responses.
    It represents a single detection result with validation.
    
    Attributes:
        detected: Whether the issue was detected
        confidence: Confidence score between 0.0 and 1.0 (validated)
        metadata: Additional information specific to detection type
    
    Requirements:
        - Requirement 1.5: Confidence score validation
        - Requirement 4.5: API response structure
    
    Example:
        >>> model = DetectionResultModel(
        ...     detected=True,
        ...     confidence=0.95,
        ...     metadata={"variance": 85.3}
        ... )
        >>> model.confidence
        0.95
    """
    detected: bool
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional detection metadata")


class AnalysisResponseModel(BaseModel):
    """
    Pydantic model for complete analysis API response.
    
    This model is used for JSON serialization when returning analysis
    results to the client (auto_process=false mode).
    
    Attributes:
        analysis: Dictionary mapping detection type to detection result
        suggested_mode: Primary recommended processing mode
        alternative_modes: List of alternative processing modes
        explanation: Human-readable explanation for the suggestion
        processing_time_ms: Time taken for analysis in milliseconds
    
    Requirements:
        - Requirements 1.1-1.6: Analysis detection and timing
        - Requirements 2.1-2.7: Mode suggestion and explanation
        - Requirement 4.5: API response structure
    
    Example:
        >>> response = AnalysisResponseModel(
        ...     analysis={
        ...         DetectionType.GRAYSCALE: DetectionResultModel(
        ...             detected=True,
        ...             confidence=1.0,
        ...             metadata={}
        ...         ),
        ...         DetectionType.WHITE_MASK: DetectionResultModel(
        ...             detected=False,
        ...             confidence=0.05,
        ...             metadata={"white_percentage": 5.2}
        ...         ),
        ...         DetectionType.LOW_RESOLUTION: DetectionResultModel(
        ...             detected=False,
        ...             confidence=0.0,
        ...             metadata={"width": 1920, "height": 1080}
        ...         ),
        ...         DetectionType.BLUR: DetectionResultModel(
        ...             detected=False,
        ...             confidence=0.0,
        ...             metadata={"variance": 250.5}
        ...         )
        ...     },
        ...     suggested_mode=ProcessingMode.COLORIZATION,
        ...     alternative_modes=[],
        ...     explanation="Image is grayscale with 100% confidence",
        ...     processing_time_ms=150.5
        ... )
        >>> response.suggested_mode
        <ProcessingMode.COLORIZATION: 'colorization'>
    """
    analysis: Dict[DetectionType, DetectionResultModel]
    suggested_mode: ProcessingMode
    alternative_modes: List[ProcessingMode]
    explanation: str
    processing_time_ms: float = Field(gt=0.0, description="Analysis processing time in milliseconds")
