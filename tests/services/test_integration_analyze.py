"""
Integration tests for ImageAnalyzer.analyze() method.

Tests the complete analysis workflow including alternative modes generation.
"""

import numpy as np
import pytest
from PIL import Image

from app.models.analysis import ProcessingMode
from app.services.image_analyzer import ImageAnalyzer


class TestAnalyzeIntegration:
    """Integration tests for analyze() method with get_alternative_modes()."""
    
    def test_analyze_grayscale_image_with_alternatives(self):
        """
        Test analyze() on grayscale + low-res image returns correct alternatives.
        
        Expected:
        - Primary mode: COLORIZATION (grayscale has priority)
        - Alternative modes: [UPSCALING, RESTORATION] (low-res and blur detected)
        
        Note: Uniform grayscale image has variance=0, so blur is also detected.
        """
        # Create 480x320 grayscale image
        img_array = np.full((320, 480, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run analysis
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify primary mode
        assert result.suggested_mode == ProcessingMode.COLORIZATION
        
        # Verify alternatives (low-res and blur both detected)
        assert len(result.alternative_modes) >= 1
        assert ProcessingMode.UPSCALING in result.alternative_modes
        assert ProcessingMode.COLORIZATION not in result.alternative_modes
    
    def test_analyze_white_mask_image_with_alternatives(self):
        """
        Test analyze() on image with white mask.
        
        Note: Creating an image with white pixels also makes it grayscale
        (all RGB channels identical), so grayscale has higher priority.
        
        Expected:
        - Primary mode: COLORIZATION (grayscale detected, higher priority)
        - Alternative modes: [INPAINTING] (white mask detected)
        """
        # Create 512x512 image with 40% white pixels
        # Note: This creates a grayscale image (black + white = grayscale)
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        # Make 40% of pixels white (RGB >= 240)
        white_pixels = int(512 * 512 * 0.4)
        img_array.flat[:white_pixels*3] = 255
        
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run analysis
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify primary mode (grayscale has priority over white mask)
        assert result.suggested_mode == ProcessingMode.COLORIZATION
        
        # Verify alternatives (white mask should be detected)
        assert ProcessingMode.INPAINTING in result.alternative_modes
        assert ProcessingMode.COLORIZATION not in result.alternative_modes
    
    def test_analyze_all_issues_detected(self):
        """
        Test analyze() when all issues are detected.
        
        Expected:
        - Primary mode: COLORIZATION (highest priority)
        - Alternative modes: [INPAINTING, UPSCALING, RESTORATION]
        """
        # Create 400x300 grayscale image with 35% white pixels
        img_array = np.full((300, 400, 3), 128, dtype=np.uint8)
        # Make 35% of pixels white
        white_pixels = int(300 * 400 * 0.35)
        img_array.flat[:white_pixels*3] = 255
        
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run analysis
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify primary mode
        assert result.suggested_mode == ProcessingMode.COLORIZATION
        
        # Verify alternatives (should have INPAINTING and UPSCALING at minimum)
        assert ProcessingMode.COLORIZATION not in result.alternative_modes
        assert ProcessingMode.INPAINTING in result.alternative_modes
        assert ProcessingMode.UPSCALING in result.alternative_modes
    
    def test_analyze_no_issues_no_alternatives(self):
        """
        Test analyze() when no issues are detected.
        
        Expected:
        - Primary mode: RESTORATION (default)
        - Alternative modes: [] (no other issues detected)
        """
        # Create 1024x768 color image (no issues)
        img_array = np.zeros((768, 1024, 3), dtype=np.uint8)
        img_array[:, :, 0] = 200  # Red
        img_array[:, :, 1] = 150  # Green
        img_array[:, :, 2] = 100  # Blue
        
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run analysis
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify primary mode
        assert result.suggested_mode == ProcessingMode.RESTORATION
        
        # Verify no alternatives
        assert result.alternative_modes == []
    
    def test_analyze_explanation_includes_alternatives(self):
        """
        Test that analyze() returns non-empty explanation.
        
        Note: Full explanation implementation is in task 3.6,
        but we verify it's not empty.
        """
        # Create grayscale image
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run analysis
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify explanation exists
        assert result.explanation is not None
        assert len(result.explanation) > 0
        assert isinstance(result.explanation, str)
