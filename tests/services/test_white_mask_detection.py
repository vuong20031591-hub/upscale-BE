"""
Unit tests for white mask detection (Task 2.3).

Tests the detect_white_mask() method to ensure it correctly:
- Counts pixels with all RGB >= 240
- Calculates white percentage
- Returns correct confidence score
- Detects white mask when percentage > 30%
"""

import numpy as np
import pytest
from PIL import Image

from app.models.analysis import DetectionResult
from app.services.image_analyzer import ImageAnalyzer


class TestWhiteMaskDetection:
    """Tests for white mask detection algorithm (Task 2.3)."""
    
    def test_detect_white_mask_no_white_pixels(self):
        """
        Test white mask detection on image with no white pixels.
        
        Validates: Requirements 1.2, 6.2, 6.6
        """
        # Create 100x100 black image (RGB = 0)
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run detection
        analyzer = ImageAnalyzer()
        result = analyzer.detect_white_mask(image)
        
        # Verify result
        assert isinstance(result, DetectionResult)
        assert result.detected is False
        assert result.confidence == 0.0
        assert result.metadata["white_percentage"] == 0.0
        assert result.metadata["white_pixel_count"] == 0
    
    def test_detect_white_mask_all_white_pixels(self):
        """
        Test white mask detection on pure white image (100% white).
        
        Validates: Requirements 1.2, 6.2, 6.6
        """
        # Create 100x100 white image (RGB = 255)
        img_array = np.full((100, 100, 3), 255, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run detection
        analyzer = ImageAnalyzer()
        result = analyzer.detect_white_mask(image)
        
        # Verify result
        assert result.detected is True  # 100% > 30%
        assert result.confidence == 1.0  # 100 / 100 = 1.0
        assert result.metadata["white_percentage"] == 100.0
        assert result.metadata["white_pixel_count"] == 10000  # 100x100
    
    def test_detect_white_mask_exactly_30_percent(self):
        """
        Test white mask detection at exactly 30% threshold.
        
        Validates: Requirements 1.2, 6.2, 6.6
        """
        # Create 100x100 image with exactly 30% white pixels
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        # Set first 30 rows to white (3000 pixels = 30%)
        img_array[:30, :, :] = 255
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run detection
        analyzer = ImageAnalyzer()
        result = analyzer.detect_white_mask(image)
        
        # Verify result
        # 30% is NOT > 30%, so should be False
        assert result.detected is False
        assert result.confidence == 0.30
        assert result.metadata["white_percentage"] == 30.0
        assert result.metadata["white_pixel_count"] == 3000
    
    def test_detect_white_mask_31_percent(self):
        """
        Test white mask detection at 31% (just above threshold).
        
        Validates: Requirements 1.2, 6.2, 6.6
        """
        # Create 100x100 image with 31% white pixels
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        # Set first 31 rows to white (3100 pixels = 31%)
        img_array[:31, :, :] = 255
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run detection
        analyzer = ImageAnalyzer()
        result = analyzer.detect_white_mask(image)
        
        # Verify result
        assert result.detected is True  # 31% > 30%
        assert result.confidence == 0.31
        assert result.metadata["white_percentage"] == 31.0
        assert result.metadata["white_pixel_count"] == 3100
    
    def test_detect_white_mask_threshold_240(self):
        """
        Test that only pixels with RGB >= 240 are counted as white.
        
        Validates: Requirements 1.2, 6.2, 6.6
        """
        # Create 100x100 image with different brightness levels
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # First 25 rows: RGB = 239 (NOT white, below threshold)
        img_array[:25, :, :] = 239
        
        # Next 25 rows: RGB = 240 (white, at threshold)
        img_array[25:50, :, :] = 240
        
        # Next 25 rows: RGB = 250 (white, above threshold)
        img_array[50:75, :, :] = 250
        
        # Last 25 rows: RGB = 0 (black)
        # Already 0 from initialization
        
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run detection
        analyzer = ImageAnalyzer()
        result = analyzer.detect_white_mask(image)
        
        # Verify result
        # Only rows 25-75 should be counted (50 rows = 5000 pixels = 50%)
        assert result.detected is True  # 50% > 30%
        assert result.confidence == 0.50
        assert result.metadata["white_percentage"] == 50.0
        assert result.metadata["white_pixel_count"] == 5000
    
    def test_detect_white_mask_partial_white_channels(self):
        """
        Test that ALL RGB channels must be >= 240 to count as white.
        
        Validates: Requirements 1.2, 6.2, 6.6
        """
        # Create 100x100 image
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # First 50 rows: R=255, G=255, B=239 (NOT white, B < 240)
        img_array[:50, :, 0] = 255  # R
        img_array[:50, :, 1] = 255  # G
        img_array[:50, :, 2] = 239  # B (below threshold)
        
        # Last 50 rows: R=255, G=255, B=255 (white)
        img_array[50:, :, :] = 255
        
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run detection
        analyzer = ImageAnalyzer()
        result = analyzer.detect_white_mask(image)
        
        # Verify result
        # Only last 50 rows should be counted (5000 pixels = 50%)
        assert result.detected is True  # 50% > 30%
        assert result.confidence == 0.50
        assert result.metadata["white_percentage"] == 50.0
        assert result.metadata["white_pixel_count"] == 5000
    
    def test_detect_white_mask_rgba_mode(self):
        """
        Test white mask detection handles RGBA mode images.
        
        Validates: Requirements 1.2, 6.2
        """
        # Create RGBA image with 50% white pixels
        img_array = np.zeros((100, 100, 4), dtype=np.uint8)
        img_array[:50, :, :3] = 255  # First 50 rows white (RGB)
        img_array[:, :, 3] = 255  # Alpha channel = opaque
        image = Image.fromarray(img_array, mode='RGBA')
        
        # Run detection (should convert to RGB internally)
        analyzer = ImageAnalyzer()
        result = analyzer.detect_white_mask(image)
        
        # Verify result
        assert result.detected is True  # 50% > 30%
        assert result.confidence == 0.50
        assert result.metadata["white_percentage"] == 50.0
    
    def test_detect_white_mask_small_image(self):
        """
        Test white mask detection on very small image.
        
        Validates: Requirements 1.2, 6.2
        """
        # Create 10x10 image with 40% white pixels
        img_array = np.zeros((10, 10, 3), dtype=np.uint8)
        img_array[:4, :, :] = 255  # First 4 rows white (40 pixels = 40%)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run detection
        analyzer = ImageAnalyzer()
        result = analyzer.detect_white_mask(image)
        
        # Verify result
        assert result.detected is True  # 40% > 30%
        assert result.confidence == 0.40
        assert result.metadata["white_percentage"] == 40.0
        assert result.metadata["white_pixel_count"] == 40
    
    def test_detect_white_mask_scattered_white_pixels(self):
        """
        Test white mask detection with scattered white pixels.
        
        Validates: Requirements 1.2, 6.2, 6.6
        """
        # Create 100x100 image with scattered white pixels
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create checkerboard pattern of white pixels
        # Every other pixel in every other row = 2500 pixels = 25%
        img_array[::2, ::2, :] = 255
        
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run detection
        analyzer = ImageAnalyzer()
        result = analyzer.detect_white_mask(image)
        
        # Verify result
        assert result.detected is False  # 25% < 30%
        assert result.confidence == 0.25
        assert result.metadata["white_percentage"] == 25.0
        assert result.metadata["white_pixel_count"] == 2500
    
    def test_detect_white_mask_confidence_calculation(self):
        """
        Test that confidence = white_percentage / 100.0.
        
        Validates: Requirements 6.6
        """
        # Create image with 45% white pixels
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[:45, :, :] = 255  # 4500 pixels = 45%
        image = Image.fromarray(img_array, mode='RGB')
        
        # Run detection
        analyzer = ImageAnalyzer()
        result = analyzer.detect_white_mask(image)
        
        # Verify confidence calculation
        expected_confidence = 45.0 / 100.0
        assert result.confidence == expected_confidence
        assert result.metadata["white_percentage"] == 45.0
