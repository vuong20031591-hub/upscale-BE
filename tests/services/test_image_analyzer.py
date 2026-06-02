"""
Tests for ImageAnalyzer service.

This module tests the ImageAnalyzer singleton and detection algorithms:
- Singleton pattern implementation
- Grayscale detection
- White mask detection
- Low resolution detection
- Blur detection
- Mode suggestion logic
"""

import numpy as np
import pytest
from PIL import Image

from app.models.analysis import DetectionResult, ProcessingMode
from app.services.image_analyzer import ImageAnalyzer


class TestImageAnalyzerSingleton:
    """Tests for ImageAnalyzer Singleton pattern (Task 1.2, 1.3)."""
    
    def test_singleton_same_instance(self):
        """
        Test that multiple instantiations return the same instance.
        
        Validates: Requirements 1.1
        """
        # Create multiple instances
        analyzer1 = ImageAnalyzer()
        analyzer2 = ImageAnalyzer()
        analyzer3 = ImageAnalyzer()
        
        # All should be the same instance
        assert analyzer1 is analyzer2
        assert analyzer2 is analyzer3
        assert analyzer1 is analyzer3
    
    def test_singleton_same_id(self):
        """
        Test that multiple instances have the same id().
        
        Validates: Requirements 1.1
        """
        analyzer1 = ImageAnalyzer()
        analyzer2 = ImageAnalyzer()
        
        # Should have same id (same object in memory)
        assert id(analyzer1) == id(analyzer2)


class TestGrayscaleDetection:
    """Tests for grayscale detection algorithm (Task 2.1)."""
    
    def test_detect_grayscale_pure_gray_image(self):
        """
        Test grayscale detection on pure gray image (all channels identical).
        
        Validates: Requirements 1.1, 6.1, 6.5
        """
        # Create 100x100 gray image (all RGB channels = 128)
        gray_value = 128
        img_array = np.full((100, 100, 3), gray_value, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run detection
        analyzer = ImageAnalyzer()
        result = analyzer.detect_grayscale(image)
        
        # Verify result
        assert isinstance(result, DetectionResult)
        assert result.detected is True
        assert result.confidence == 1.0
        assert result.metadata == {}
    
    def test_detect_grayscale_black_image(self):
        """
        Test grayscale detection on pure black image.
        
        Validates: Requirements 1.1, 6.1, 6.5
        """
        # Create 50x50 black image (all RGB channels = 0)
        img_array = np.zeros((50, 50, 3), dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run detection
        analyzer = ImageAnalyzer()
        result = analyzer.detect_grayscale(image)
        
        # Verify result
        assert result.detected is True
        assert result.confidence == 1.0
    
    def test_detect_grayscale_white_image(self):
        """
        Test grayscale detection on pure white image.
        
        Validates: Requirements 1.1, 6.1, 6.5
        """
        # Create 50x50 white image (all RGB channels = 255)
        img_array = np.full((50, 50, 3), 255, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run detection
        analyzer = ImageAnalyzer()
        result = analyzer.detect_grayscale(image)
        
        # Verify result
        assert result.detected is True
        assert result.confidence == 1.0
    
    def test_detect_grayscale_color_image(self):
        """
        Test grayscale detection on color image (different RGB channels).
        
        Validates: Requirements 1.1, 6.1, 6.5
        """
        # Create 100x100 color image (R=255, G=0, B=0 - pure red)
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[:, :, 0] = 255  # Red channel
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run detection
        analyzer = ImageAnalyzer()
        result = analyzer.detect_grayscale(image)
        
        # Verify result
        assert result.detected is False
        assert result.confidence == 0.0
        assert result.metadata == {}
    
    def test_detect_grayscale_mixed_color_image(self):
        """
        Test grayscale detection on image with mixed colors.
        
        Validates: Requirements 1.1, 6.1, 6.5
        """
        # Create 100x100 image with different colors in each channel
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[:, :, 0] = 200  # Red
        img_array[:, :, 1] = 150  # Green
        img_array[:, :, 2] = 100  # Blue
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run detection
        analyzer = ImageAnalyzer()
        result = analyzer.detect_grayscale(image)
        
        # Verify result
        assert result.detected is False
        assert result.confidence == 0.0
    
    def test_detect_grayscale_rgba_mode(self):
        """
        Test grayscale detection handles RGBA mode images.
        
        Validates: Requirements 1.1, 6.1
        """
        # Create RGBA image (grayscale with alpha channel)
        img_array = np.full((100, 100, 4), 128, dtype=np.uint8)
        img_array[:, :, 3] = 255  # Alpha channel = opaque
        image = Image.fromarray(img_array, mode='RGBA')
        
        # Run detection (should convert to RGB internally)
        analyzer = ImageAnalyzer()
        result = analyzer.detect_grayscale(image)
        
        # Verify result
        assert result.detected is True
        assert result.confidence == 1.0
    
    def test_detect_grayscale_l_mode(self):
        """
        Test grayscale detection handles L mode (grayscale) images.
        
        Validates: Requirements 1.1, 6.1
        """
        # Create L mode image (native grayscale)
        img_array = np.full((100, 100), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='L')
        
        # Run detection (should convert to RGB internally)
        analyzer = ImageAnalyzer()
        result = analyzer.detect_grayscale(image)
        
        # Verify result
        assert result.detected is True
        assert result.confidence == 1.0
    
    def test_detect_grayscale_gradient_image(self):
        """
        Test grayscale detection on gradient image (grayscale gradient).
        
        Validates: Requirements 1.1, 6.1, 6.5
        """
        # Create 100x100 gradient image (0 to 255 horizontally, same for all channels)
        gradient = np.linspace(0, 255, 100, dtype=np.uint8)
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(3):
            img_array[:, :, i] = gradient[np.newaxis, :]
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run detection
        analyzer = ImageAnalyzer()
        result = analyzer.detect_grayscale(image)
        
        # Verify result
        assert result.detected is True
        assert result.confidence == 1.0
    
    def test_detect_grayscale_small_image(self):
        """
        Test grayscale detection on very small image.
        
        Validates: Requirements 1.1, 6.1
        """
        # Create 10x10 gray image
        img_array = np.full((10, 10, 3), 100, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run detection
        analyzer = ImageAnalyzer()
        result = analyzer.detect_grayscale(image)
        
        # Verify result
        assert result.detected is True
        assert result.confidence == 1.0
    
    def test_detect_grayscale_large_image(self):
        """
        Test grayscale detection on large image.
        
        Validates: Requirements 1.1, 6.1
        """
        # Create 1000x1000 gray image
        img_array = np.full((1000, 1000, 3), 150, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run detection
        analyzer = ImageAnalyzer()
        result = analyzer.detect_grayscale(image)
        
        # Verify result
        assert result.detected is True
        assert result.confidence == 1.0



class TestModeSuggestion:
    """Tests for mode suggestion logic (Task 3.3)."""
    
    def test_suggest_mode_grayscale_priority(self):
        """
        Test that grayscale detection (confidence > 0.9) suggests colorization.
        
        Priority 1: Grayscale → Colorization
        Validates: Requirements 2.1, 2.6
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with grayscale detected (confidence = 1.0)
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(True, 1.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.RESTORATION,  # Placeholder
            alternative_modes=[],
            explanation=""
        )
        
        # Run suggestion
        analyzer = ImageAnalyzer()
        mode = analyzer.suggest_mode(analysis)
        
        # Verify colorization is suggested
        assert mode == ProcessingMode.COLORIZATION
    
    def test_suggest_mode_grayscale_threshold_boundary(self):
        """
        Test grayscale detection at confidence threshold boundary (0.9).
        
        Validates: Requirements 2.1, 2.6
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Test confidence = 0.9 (at threshold, should NOT suggest colorization)
        analysis_at_threshold = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(True, 0.9, {}),
                DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.RESTORATION,
            alternative_modes=[],
            explanation=""
        )
        
        analyzer = ImageAnalyzer()
        mode_at = analyzer.suggest_mode(analysis_at_threshold)
        
        # At threshold (0.9), should NOT suggest colorization (needs > 0.9)
        assert mode_at != ProcessingMode.COLORIZATION
        
        # Test confidence = 0.91 (above threshold, should suggest colorization)
        analysis_above_threshold = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(True, 0.91, {}),
                DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.RESTORATION,
            alternative_modes=[],
            explanation=""
        )
        
        mode_above = analyzer.suggest_mode(analysis_above_threshold)
        assert mode_above == ProcessingMode.COLORIZATION
    
    def test_suggest_mode_white_mask_priority(self):
        """
        Test that white mask detection (>30%) suggests inpainting.
        
        Priority 2: White Mask → Inpainting
        Validates: Requirements 2.2, 2.6
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with white mask detected (>30%)
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(True, 0.35, {"white_percentage": 35.0}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.RESTORATION,
            alternative_modes=[],
            explanation=""
        )
        
        # Run suggestion
        analyzer = ImageAnalyzer()
        mode = analyzer.suggest_mode(analysis)
        
        # Verify inpainting is suggested
        assert mode == ProcessingMode.INPAINTING
    
    def test_suggest_mode_low_resolution_priority(self):
        """
        Test that low resolution detection suggests upscaling.
        
        Priority 3: Low Resolution → Upscaling
        Validates: Requirements 2.3, 2.6
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with low resolution detected
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                DetectionType.LOW_RESOLUTION: DetectionResult(True, 1.0, {"width": 480, "height": 320}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.RESTORATION,
            alternative_modes=[],
            explanation=""
        )
        
        # Run suggestion
        analyzer = ImageAnalyzer()
        mode = analyzer.suggest_mode(analysis)
        
        # Verify upscaling is suggested
        assert mode == ProcessingMode.UPSCALING
    
    def test_suggest_mode_blur_priority(self):
        """
        Test that blur detection suggests restoration.
        
        Priority 4: Blur → Restoration
        Validates: Requirements 2.4, 2.6
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with blur detected
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(True, 0.8, {"variance": 50.0})
            },
            suggested_mode=ProcessingMode.RESTORATION,
            alternative_modes=[],
            explanation=""
        )
        
        # Run suggestion
        analyzer = ImageAnalyzer()
        mode = analyzer.suggest_mode(analysis)
        
        # Verify restoration is suggested
        assert mode == ProcessingMode.RESTORATION
    
    def test_suggest_mode_default_no_issues(self):
        """
        Test that default mode (restoration) is suggested when no issues detected.
        
        Priority 5: Default → Restoration
        Validates: Requirements 2.5, 2.6
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with no issues detected
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.RESTORATION,
            alternative_modes=[],
            explanation=""
        )
        
        # Run suggestion
        analyzer = ImageAnalyzer()
        mode = analyzer.suggest_mode(analysis)
        
        # Verify restoration is suggested (default)
        assert mode == ProcessingMode.RESTORATION
    
    def test_suggest_mode_priority_order_grayscale_over_white_mask(self):
        """
        Test priority order: Grayscale (priority 1) takes precedence over White Mask (priority 2).
        
        Validates: Requirements 2.1, 2.2, 2.6
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with BOTH grayscale and white mask detected
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(True, 1.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(True, 0.35, {"white_percentage": 35.0}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.RESTORATION,
            alternative_modes=[],
            explanation=""
        )
        
        # Run suggestion
        analyzer = ImageAnalyzer()
        mode = analyzer.suggest_mode(analysis)
        
        # Verify colorization is suggested (grayscale has higher priority)
        assert mode == ProcessingMode.COLORIZATION
    
    def test_suggest_mode_priority_order_white_mask_over_low_res(self):
        """
        Test priority order: White Mask (priority 2) takes precedence over Low Resolution (priority 3).
        
        Validates: Requirements 2.2, 2.3, 2.6
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with BOTH white mask and low resolution detected
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(True, 0.40, {"white_percentage": 40.0}),
                DetectionType.LOW_RESOLUTION: DetectionResult(True, 1.0, {"width": 480, "height": 320}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.RESTORATION,
            alternative_modes=[],
            explanation=""
        )
        
        # Run suggestion
        analyzer = ImageAnalyzer()
        mode = analyzer.suggest_mode(analysis)
        
        # Verify inpainting is suggested (white mask has higher priority)
        assert mode == ProcessingMode.INPAINTING
    
    def test_suggest_mode_priority_order_low_res_over_blur(self):
        """
        Test priority order: Low Resolution (priority 3) takes precedence over Blur (priority 4).
        
        Validates: Requirements 2.3, 2.4, 2.6
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with BOTH low resolution and blur detected
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                DetectionType.LOW_RESOLUTION: DetectionResult(True, 1.0, {"width": 400, "height": 300}),
                DetectionType.BLUR: DetectionResult(True, 0.7, {"variance": 60.0})
            },
            suggested_mode=ProcessingMode.RESTORATION,
            alternative_modes=[],
            explanation=""
        )
        
        # Run suggestion
        analyzer = ImageAnalyzer()
        mode = analyzer.suggest_mode(analysis)
        
        # Verify upscaling is suggested (low resolution has higher priority)
        assert mode == ProcessingMode.UPSCALING
    
    def test_suggest_mode_priority_order_all_issues_detected(self):
        """
        Test priority order when ALL issues are detected.
        
        Should suggest colorization (highest priority).
        Validates: Requirements 2.1-2.6
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with ALL issues detected
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(True, 1.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(True, 0.35, {"white_percentage": 35.0}),
                DetectionType.LOW_RESOLUTION: DetectionResult(True, 1.0, {"width": 480, "height": 320}),
                DetectionType.BLUR: DetectionResult(True, 0.8, {"variance": 50.0})
            },
            suggested_mode=ProcessingMode.RESTORATION,
            alternative_modes=[],
            explanation=""
        )
        
        # Run suggestion
        analyzer = ImageAnalyzer()
        mode = analyzer.suggest_mode(analysis)
        
        # Verify colorization is suggested (highest priority)
        assert mode == ProcessingMode.COLORIZATION



class TestAlternativeModes:
    """Tests for get_alternative_modes() method (Task 3.5)."""
    
    def test_get_alternative_modes_no_alternatives(self):
        """
        Test that no alternatives are returned when only primary mode is detected.
        
        Validates: Requirements 2.7
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with only grayscale detected
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(True, 1.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.COLORIZATION,
            alternative_modes=[],
            explanation=""
        )
        
        # Get alternatives
        analyzer = ImageAnalyzer()
        alternatives = analyzer.get_alternative_modes(analysis, ProcessingMode.COLORIZATION)
        
        # Verify no alternatives (only grayscale detected, which is primary mode)
        assert alternatives == []
    
    def test_get_alternative_modes_single_alternative(self):
        """
        Test that single alternative is returned when two issues detected.
        
        Validates: Requirements 2.7
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with grayscale (primary) and low resolution (alternative)
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(True, 1.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                DetectionType.LOW_RESOLUTION: DetectionResult(True, 1.0, {"width": 480, "height": 320}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.COLORIZATION,
            alternative_modes=[],
            explanation=""
        )
        
        # Get alternatives
        analyzer = ImageAnalyzer()
        alternatives = analyzer.get_alternative_modes(analysis, ProcessingMode.COLORIZATION)
        
        # Verify upscaling is alternative (low resolution detected)
        assert len(alternatives) == 1
        assert ProcessingMode.UPSCALING in alternatives
        assert ProcessingMode.COLORIZATION not in alternatives  # Primary mode excluded
    
    def test_get_alternative_modes_multiple_alternatives(self):
        """
        Test that multiple alternatives are returned when multiple issues detected.
        
        Validates: Requirements 2.7
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with grayscale (primary), white mask, low resolution, and blur
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(True, 1.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(True, 0.35, {"white_percentage": 35.0}),
                DetectionType.LOW_RESOLUTION: DetectionResult(True, 1.0, {"width": 480, "height": 320}),
                DetectionType.BLUR: DetectionResult(True, 0.8, {"variance": 50.0})
            },
            suggested_mode=ProcessingMode.COLORIZATION,
            alternative_modes=[],
            explanation=""
        )
        
        # Get alternatives
        analyzer = ImageAnalyzer()
        alternatives = analyzer.get_alternative_modes(analysis, ProcessingMode.COLORIZATION)
        
        # Verify all other modes are alternatives
        assert len(alternatives) == 3
        assert ProcessingMode.INPAINTING in alternatives
        assert ProcessingMode.UPSCALING in alternatives
        assert ProcessingMode.RESTORATION in alternatives
        assert ProcessingMode.COLORIZATION not in alternatives  # Primary mode excluded
    
    def test_get_alternative_modes_excludes_primary_mode(self):
        """
        Test that primary mode is always excluded from alternatives.
        
        Validates: Requirements 2.7
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with white mask (primary) and low resolution
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(True, 0.40, {"white_percentage": 40.0}),
                DetectionType.LOW_RESOLUTION: DetectionResult(True, 1.0, {"width": 400, "height": 300}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.INPAINTING,
            alternative_modes=[],
            explanation=""
        )
        
        # Get alternatives
        analyzer = ImageAnalyzer()
        alternatives = analyzer.get_alternative_modes(analysis, ProcessingMode.INPAINTING)
        
        # Verify inpainting (primary) is excluded, upscaling is included
        assert ProcessingMode.INPAINTING not in alternatives
        assert ProcessingMode.UPSCALING in alternatives
    
    def test_get_alternative_modes_grayscale_threshold(self):
        """
        Test that grayscale is only alternative if confidence > 0.9.
        
        Validates: Requirements 2.7
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Test 1: Grayscale confidence = 0.9 (at threshold, should NOT be alternative)
        analysis_at_threshold = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(True, 0.9, {}),
                DetectionType.WHITE_MASK: DetectionResult(True, 0.35, {"white_percentage": 35.0}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.INPAINTING,
            alternative_modes=[],
            explanation=""
        )
        
        analyzer = ImageAnalyzer()
        alternatives_at = analyzer.get_alternative_modes(analysis_at_threshold, ProcessingMode.INPAINTING)
        
        # Grayscale confidence = 0.9 (not > 0.9), should NOT be alternative
        assert ProcessingMode.COLORIZATION not in alternatives_at
        
        # Test 2: Grayscale confidence = 0.91 (above threshold, should be alternative)
        analysis_above_threshold = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(True, 0.91, {}),
                DetectionType.WHITE_MASK: DetectionResult(True, 0.35, {"white_percentage": 35.0}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.INPAINTING,
            alternative_modes=[],
            explanation=""
        )
        
        alternatives_above = analyzer.get_alternative_modes(analysis_above_threshold, ProcessingMode.INPAINTING)
        
        # Grayscale confidence = 0.91 (> 0.9), should be alternative
        assert ProcessingMode.COLORIZATION in alternatives_above
    
    def test_get_alternative_modes_white_mask_threshold(self):
        """
        Test that white mask is only alternative if detected (>30%).
        
        Validates: Requirements 2.7
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Test 1: White mask = 30% (at threshold, detected=True)
        analysis_at_threshold = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(True, 1.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(False, 0.30, {"white_percentage": 30.0}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.COLORIZATION,
            alternative_modes=[],
            explanation=""
        )
        
        analyzer = ImageAnalyzer()
        alternatives_at = analyzer.get_alternative_modes(analysis_at_threshold, ProcessingMode.COLORIZATION)
        
        # White mask detected=False (30% is not > 30%), should NOT be alternative
        assert ProcessingMode.INPAINTING not in alternatives_at
        
        # Test 2: White mask = 31% (above threshold, detected=True)
        analysis_above_threshold = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(True, 1.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(True, 0.31, {"white_percentage": 31.0}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.COLORIZATION,
            alternative_modes=[],
            explanation=""
        )
        
        alternatives_above = analyzer.get_alternative_modes(analysis_above_threshold, ProcessingMode.COLORIZATION)
        
        # White mask detected=True (31% > 30%), should be alternative
        assert ProcessingMode.INPAINTING in alternatives_above
    
    def test_get_alternative_modes_order_matches_detection_order(self):
        """
        Test that alternatives are returned in detection order (grayscale, white mask, low res, blur).
        
        Validates: Requirements 2.7
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with all issues detected except primary (upscaling)
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(True, 1.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(True, 0.35, {"white_percentage": 35.0}),
                DetectionType.LOW_RESOLUTION: DetectionResult(True, 1.0, {"width": 480, "height": 320}),
                DetectionType.BLUR: DetectionResult(True, 0.8, {"variance": 50.0})
            },
            suggested_mode=ProcessingMode.UPSCALING,
            alternative_modes=[],
            explanation=""
        )
        
        # Get alternatives
        analyzer = ImageAnalyzer()
        alternatives = analyzer.get_alternative_modes(analysis, ProcessingMode.UPSCALING)
        
        # Verify order: COLORIZATION, INPAINTING, RESTORATION (excluding UPSCALING)
        assert len(alternatives) == 3
        assert alternatives[0] == ProcessingMode.COLORIZATION
        assert alternatives[1] == ProcessingMode.INPAINTING
        assert alternatives[2] == ProcessingMode.RESTORATION
    
    def test_get_alternative_modes_no_issues_detected(self):
        """
        Test that no alternatives are returned when no issues detected.
        
        Validates: Requirements 2.7
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with no issues detected
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.RESTORATION,
            alternative_modes=[],
            explanation=""
        )
        
        # Get alternatives
        analyzer = ImageAnalyzer()
        alternatives = analyzer.get_alternative_modes(analysis, ProcessingMode.RESTORATION)
        
        # Verify no alternatives (no issues detected)
        assert alternatives == []



class TestExplainSuggestion:
    """Tests for explain_suggestion() method (Task 3.6)."""
    
    def test_explain_suggestion_colorization_mode(self):
        """
        Test explanation for colorization mode (grayscale detected).
        
        Validates: Requirements 2.7
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with grayscale detected (confidence = 1.0)
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(True, 1.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.COLORIZATION,
            alternative_modes=[],
            explanation=""
        )
        
        # Generate explanation
        analyzer = ImageAnalyzer()
        explanation = analyzer.explain_suggestion(ProcessingMode.COLORIZATION, analysis)
        
        # Verify explanation format
        assert "grayscale" in explanation.lower()
        assert "100%" in explanation
        assert "colorization" in explanation.lower()
        assert explanation == "Image is grayscale with 100% confidence. Colorization recommended."
    
    def test_explain_suggestion_colorization_partial_confidence(self):
        """
        Test explanation for colorization with partial confidence.
        
        Validates: Requirements 2.7
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with grayscale detected (confidence = 0.95)
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(True, 0.95, {}),
                DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.COLORIZATION,
            alternative_modes=[],
            explanation=""
        )
        
        # Generate explanation
        analyzer = ImageAnalyzer()
        explanation = analyzer.explain_suggestion(ProcessingMode.COLORIZATION, analysis)
        
        # Verify explanation includes confidence percentage
        assert "95%" in explanation
        assert "colorization" in explanation.lower()
    
    def test_explain_suggestion_inpainting_mode(self):
        """
        Test explanation for inpainting mode (white mask detected).
        
        Validates: Requirements 2.7
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with white mask detected (35.2%)
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(True, 0.352, {"white_percentage": 35.2}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.INPAINTING,
            alternative_modes=[],
            explanation=""
        )
        
        # Generate explanation
        analyzer = ImageAnalyzer()
        explanation = analyzer.explain_suggestion(ProcessingMode.INPAINTING, analysis)
        
        # Verify explanation format
        assert "white mask" in explanation.lower()
        assert "35.2%" in explanation
        assert "inpainting" in explanation.lower()
        assert explanation == "Image has white mask regions (35.2% of image). Inpainting recommended."
    
    def test_explain_suggestion_upscaling_mode(self):
        """
        Test explanation for upscaling mode (low resolution detected).
        
        Validates: Requirements 2.7
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with low resolution detected (480x320)
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                DetectionType.LOW_RESOLUTION: DetectionResult(True, 1.0, {"width": 480, "height": 320}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.UPSCALING,
            alternative_modes=[],
            explanation=""
        )
        
        # Generate explanation
        analyzer = ImageAnalyzer()
        explanation = analyzer.explain_suggestion(ProcessingMode.UPSCALING, analysis)
        
        # Verify explanation format
        assert "low resolution" in explanation.lower()
        assert "480x320" in explanation
        assert "upscaling" in explanation.lower()
        assert explanation == "Image has low resolution (480x320). Upscaling recommended."
    
    def test_explain_suggestion_restoration_mode_blur_detected(self):
        """
        Test explanation for restoration mode when blur is detected.
        
        Validates: Requirements 2.7
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with blur detected (variance = 45.3)
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(True, 0.8, {"variance": 45.3})
            },
            suggested_mode=ProcessingMode.RESTORATION,
            alternative_modes=[],
            explanation=""
        )
        
        # Generate explanation
        analyzer = ImageAnalyzer()
        explanation = analyzer.explain_suggestion(ProcessingMode.RESTORATION, analysis)
        
        # Verify explanation format
        assert "blurry" in explanation.lower()
        assert "45.3" in explanation
        assert "restoration" in explanation.lower()
        assert explanation == "Image is blurry (variance: 45.3). Restoration recommended."
    
    def test_explain_suggestion_restoration_mode_default(self):
        """
        Test explanation for restoration mode when no issues detected (default).
        
        Validates: Requirements 2.7
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Create analysis with no issues detected
        analysis = AnalysisResult(
            detections={
                DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                DetectionType.BLUR: DetectionResult(False, 0.0, {})
            },
            suggested_mode=ProcessingMode.RESTORATION,
            alternative_modes=[],
            explanation=""
        )
        
        # Generate explanation
        analyzer = ImageAnalyzer()
        explanation = analyzer.explain_suggestion(ProcessingMode.RESTORATION, analysis)
        
        # Verify explanation format
        assert "no specific issues" in explanation.lower()
        assert "default" in explanation.lower()
        assert "restoration" in explanation.lower()
        assert explanation == "No specific issues detected. Restoration recommended as default."
    
    def test_explain_suggestion_non_empty_string(self):
        """
        Test that explanation is always non-empty string.
        
        Validates: Requirements 2.7 (Property 7)
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Test all four modes
        modes_to_test = [
            ProcessingMode.COLORIZATION,
            ProcessingMode.INPAINTING,
            ProcessingMode.UPSCALING,
            ProcessingMode.RESTORATION
        ]
        
        analyzer = ImageAnalyzer()
        
        for mode in modes_to_test:
            # Create minimal analysis
            analysis = AnalysisResult(
                detections={
                    DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                    DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                    DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                    DetectionType.BLUR: DetectionResult(False, 0.0, {})
                },
                suggested_mode=mode,
                alternative_modes=[],
                explanation=""
            )
            
            # Generate explanation
            explanation = analyzer.explain_suggestion(mode, analysis)
            
            # Verify explanation is non-empty
            assert explanation is not None
            assert isinstance(explanation, str)
            assert len(explanation) > 0
            assert explanation.strip() != ""
    
    def test_explain_suggestion_includes_mode_name(self):
        """
        Test that explanation always includes the mode name.
        
        Validates: Requirements 2.7
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Test all four modes
        test_cases = [
            (ProcessingMode.COLORIZATION, "colorization"),
            (ProcessingMode.INPAINTING, "inpainting"),
            (ProcessingMode.UPSCALING, "upscaling"),
            (ProcessingMode.RESTORATION, "restoration")
        ]
        
        analyzer = ImageAnalyzer()
        
        for mode, expected_keyword in test_cases:
            # Create minimal analysis
            analysis = AnalysisResult(
                detections={
                    DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                    DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                    DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                    DetectionType.BLUR: DetectionResult(False, 0.0, {})
                },
                suggested_mode=mode,
                alternative_modes=[],
                explanation=""
            )
            
            # Generate explanation
            explanation = analyzer.explain_suggestion(mode, analysis)
            
            # Verify mode name is in explanation
            assert expected_keyword in explanation.lower(), \
                f"Expected '{expected_keyword}' in explanation for {mode.value}, got: {explanation}"
    
    def test_explain_suggestion_variance_formatting(self):
        """
        Test that variance is formatted with one decimal place.
        
        Validates: Requirements 2.7
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Test with various variance values
        test_variances = [45.3, 50.0, 99.999, 10.5]
        
        analyzer = ImageAnalyzer()
        
        for variance in test_variances:
            # Create analysis with blur detected
            analysis = AnalysisResult(
                detections={
                    DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                    DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                    DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                    DetectionType.BLUR: DetectionResult(True, 0.8, {"variance": variance})
                },
                suggested_mode=ProcessingMode.RESTORATION,
                alternative_modes=[],
                explanation=""
            )
            
            # Generate explanation
            explanation = analyzer.explain_suggestion(ProcessingMode.RESTORATION, analysis)
            
            # Verify variance is formatted with one decimal place
            expected_variance_str = f"{variance:.1f}"
            assert expected_variance_str in explanation, \
                f"Expected '{expected_variance_str}' in explanation, got: {explanation}"
    
    def test_explain_suggestion_white_percentage_formatting(self):
        """
        Test that white percentage is formatted with one decimal place.
        
        Validates: Requirements 2.7
        """
        from app.models.analysis import AnalysisResult, DetectionType
        
        # Test with various white percentages
        test_percentages = [35.2, 40.0, 50.999, 31.5]
        
        analyzer = ImageAnalyzer()
        
        for percentage in test_percentages:
            # Create analysis with white mask detected
            analysis = AnalysisResult(
                detections={
                    DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                    DetectionType.WHITE_MASK: DetectionResult(True, percentage / 100.0, {"white_percentage": percentage}),
                    DetectionType.LOW_RESOLUTION: DetectionResult(False, 0.0, {}),
                    DetectionType.BLUR: DetectionResult(False, 0.0, {})
                },
                suggested_mode=ProcessingMode.INPAINTING,
                alternative_modes=[],
                explanation=""
            )
            
            # Generate explanation
            explanation = analyzer.explain_suggestion(ProcessingMode.INPAINTING, analysis)
            
            # Verify percentage is formatted with one decimal place
            expected_percentage_str = f"{percentage:.1f}%"
            assert expected_percentage_str in explanation, \
                f"Expected '{expected_percentage_str}' in explanation, got: {explanation}"
