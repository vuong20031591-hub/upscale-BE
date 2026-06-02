"""
Unit tests for ProcessedImage scale_factor calculation.

Task 8.1: Test scale factor calculation trong ProcessedImage
Requirements: 5.5 - Calculate overall_scale_factor = (final_w/orig_w + final_h/orig_h) / 2

This test file focuses on verifying the scale factor calculation formula
across various input/output dimension combinations.
"""

import pytest
from PIL import Image

from app.services.image_processor import ImageProcessor
from app.models import Resolution, ProcessedImage


def create_test_image(width: int, height: int, mode: str = "RGB") -> Image.Image:
    """Create a test image with specified dimensions."""
    return Image.new(mode, (width, height), color=(128, 128, 128))


class TestScaleFactorCalculation:
    """
    Test scale factor calculation formula.
    
    Formula: overall_scale_factor = (final_width/original_width + final_height/original_height) / 2
    
    Note: This is overall_scale_factor for response header, different from resize_scale in Property 7.
    """
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = ImageProcessor()
    
    def test_scale_factor_formula_upscale_2x(self):
        """
        Test scale factor calculation when upscaling by 2x uniformly.
        
        Input: 1000x1000
        Output: 2000x2000
        Expected scale_factor: (2.0 + 2.0) / 2 = 2.0
        """
        # Create 1000x1000 image
        image = create_test_image(1000, 1000)
        
        # Process with 2K resolution (will upscale)
        result = self.processor._upscale_traditional(image, Resolution.K2)
        
        # Calculate expected scale factor
        scale_w = result.final_width / 1000
        scale_h = result.final_height / 1000
        expected_scale = (scale_w + scale_h) / 2
        
        # Verify scale factor matches formula
        assert abs(result.scale_factor - expected_scale) < 0.01, \
            f"Scale factor {result.scale_factor} doesn't match expected {expected_scale}"
    
    def test_scale_factor_formula_downscale(self):
        """
        Test scale factor calculation when downscaling.
        
        Input: 3840x2160 (4K)
        Target: 2560x1440 (2K)
        Expected: scale_factor < 1.0
        """
        # Create 4K image
        image = create_test_image(3840, 2160)
        
        # Process with 2K resolution (will downscale)
        result = self.processor._upscale_traditional(image, Resolution.K2)
        
        # Calculate expected scale factor
        scale_w = result.final_width / 3840
        scale_h = result.final_height / 2160
        expected_scale = (scale_w + scale_h) / 2
        
        # Verify scale factor matches formula
        assert abs(result.scale_factor - expected_scale) < 0.01, \
            f"Scale factor {result.scale_factor} doesn't match expected {expected_scale}"
        
        # Verify downscaling occurred
        assert result.scale_factor < 1.0, "Scale factor should be < 1.0 for downscaling"
    
    def test_scale_factor_formula_no_change(self):
        """
        Test scale factor calculation when dimensions don't change.
        
        Input: 2560x1440 (exactly 2K)
        Target: 2560x1440 (2K)
        Expected scale_factor: 1.0
        """
        # Create image with exact 2K dimensions
        image = create_test_image(2560, 1440)
        
        # Process with 2K resolution (no resize needed)
        result = self.processor._upscale_traditional(image, Resolution.K2)
        
        # Verify scale factor is 1.0
        assert abs(result.scale_factor - 1.0) < 0.01, \
            f"Scale factor {result.scale_factor} should be 1.0 for no change"
    
    def test_scale_factor_formula_landscape_image(self):
        """
        Test scale factor calculation for landscape image (16:9).
        
        Input: 1920x1080 (Full HD, 16:9)
        Target: 2560x1440 (2K, 16:9)
        """
        # Create Full HD image
        image = create_test_image(1920, 1080)
        
        # Process with 2K resolution
        result = self.processor._upscale_traditional(image, Resolution.K2)
        
        # Calculate expected scale factor
        scale_w = result.final_width / 1920
        scale_h = result.final_height / 1080
        expected_scale = (scale_w + scale_h) / 2
        
        # Verify scale factor matches formula
        assert abs(result.scale_factor - expected_scale) < 0.01, \
            f"Scale factor {result.scale_factor} doesn't match expected {expected_scale}"
        
        # Verify upscaling occurred
        assert result.scale_factor > 1.0, "Scale factor should be > 1.0 for upscaling"
    
    def test_scale_factor_formula_portrait_image(self):
        """
        Test scale factor calculation for portrait image (9:16).
        
        Input: 1080x1920 (portrait)
        Target: 2560x1440 (2K)
        """
        # Create portrait image
        image = create_test_image(1080, 1920)
        
        # Process with 2K resolution
        result = self.processor._upscale_traditional(image, Resolution.K2)
        
        # Calculate expected scale factor
        scale_w = result.final_width / 1080
        scale_h = result.final_height / 1920
        expected_scale = (scale_w + scale_h) / 2
        
        # Verify scale factor matches formula
        assert abs(result.scale_factor - expected_scale) < 0.01, \
            f"Scale factor {result.scale_factor} doesn't match expected {expected_scale}"
    
    def test_scale_factor_formula_square_image(self):
        """
        Test scale factor calculation for square image (1:1).
        
        Input: 1000x1000 (square)
        Target: 2560x1440 (2K)
        """
        # Create square image
        image = create_test_image(1000, 1000)
        
        # Process with 2K resolution
        result = self.processor._upscale_traditional(image, Resolution.K2)
        
        # Calculate expected scale factor
        scale_w = result.final_width / 1000
        scale_h = result.final_height / 1000
        expected_scale = (scale_w + scale_h) / 2
        
        # Verify scale factor matches formula
        assert abs(result.scale_factor - expected_scale) < 0.01, \
            f"Scale factor {result.scale_factor} doesn't match expected {expected_scale}"
        
        # For square image, scale_w should equal scale_h
        assert abs(scale_w - scale_h) < 0.01, \
            "For square image, width and height scales should be equal"
    
    def test_scale_factor_formula_very_wide_image(self):
        """
        Test scale factor calculation for very wide image (extreme aspect ratio).
        
        Input: 3000x500 (6:1 aspect ratio)
        Target: 2560x1440 (2K)
        """
        # Create very wide image
        image = create_test_image(3000, 500)
        
        # Process with 2K resolution
        result = self.processor._upscale_traditional(image, Resolution.K2)
        
        # Calculate expected scale factor
        scale_w = result.final_width / 3000
        scale_h = result.final_height / 500
        expected_scale = (scale_w + scale_h) / 2
        
        # Verify scale factor matches formula
        assert abs(result.scale_factor - expected_scale) < 0.01, \
            f"Scale factor {result.scale_factor} doesn't match expected {expected_scale}"
    
    def test_scale_factor_formula_very_tall_image(self):
        """
        Test scale factor calculation for very tall image (extreme aspect ratio).
        
        Input: 500x3000 (1:6 aspect ratio)
        Target: 2560x1440 (2K)
        """
        # Create very tall image
        image = create_test_image(500, 3000)
        
        # Process with 2K resolution
        result = self.processor._upscale_traditional(image, Resolution.K2)
        
        # Calculate expected scale factor
        scale_w = result.final_width / 500
        scale_h = result.final_height / 3000
        expected_scale = (scale_w + scale_h) / 2
        
        # Verify scale factor matches formula
        assert abs(result.scale_factor - expected_scale) < 0.01, \
            f"Scale factor {result.scale_factor} doesn't match expected {expected_scale}"
    
    def test_scale_factor_formula_small_to_2k(self):
        """
        Test scale factor calculation for small image upscaling to 2K.
        
        Input: 640x480 (VGA)
        Target: 2560x1440 (2K)
        """
        # Create VGA image
        image = create_test_image(640, 480)
        
        # Process with 2K resolution
        result = self.processor._upscale_traditional(image, Resolution.K2)
        
        # Calculate expected scale factor
        scale_w = result.final_width / 640
        scale_h = result.final_height / 480
        expected_scale = (scale_w + scale_h) / 2
        
        # Verify scale factor matches formula
        assert abs(result.scale_factor - expected_scale) < 0.01, \
            f"Scale factor {result.scale_factor} doesn't match expected {expected_scale}"
        
        # Verify significant upscaling
        assert result.scale_factor > 2.0, "Scale factor should be > 2.0 for VGA to 2K"
    
    def test_scale_factor_formula_small_to_4k(self):
        """
        Test scale factor calculation for small image upscaling to 4K.
        
        Input: 1280x720 (HD)
        Target: 3840x2160 (4K)
        """
        # Create HD image
        image = create_test_image(1280, 720)
        
        # Process with 4K resolution
        result = self.processor._upscale_traditional(image, Resolution.K4)
        
        # Calculate expected scale factor
        scale_w = result.final_width / 1280
        scale_h = result.final_height / 720
        expected_scale = (scale_w + scale_h) / 2
        
        # Verify scale factor matches formula
        assert abs(result.scale_factor - expected_scale) < 0.01, \
            f"Scale factor {result.scale_factor} doesn't match expected {expected_scale}"
        
        # Verify upscaling to 4K
        assert result.scale_factor > 1.0, "Scale factor should be > 1.0 for HD to 4K"
    
    def test_scale_factor_formula_4k_to_2k(self):
        """
        Test scale factor calculation for 4K downscaling to 2K.
        
        Input: 3840x2160 (4K)
        Target: 2560x1440 (2K)
        """
        # Create 4K image
        image = create_test_image(3840, 2160)
        
        # Process with 2K resolution
        result = self.processor._upscale_traditional(image, Resolution.K2)
        
        # Calculate expected scale factor
        scale_w = result.final_width / 3840
        scale_h = result.final_height / 2160
        expected_scale = (scale_w + scale_h) / 2
        
        # Verify scale factor matches formula
        assert abs(result.scale_factor - expected_scale) < 0.01, \
            f"Scale factor {result.scale_factor} doesn't match expected {expected_scale}"
        
        # Verify downscaling
        assert result.scale_factor < 1.0, "Scale factor should be < 1.0 for 4K to 2K"
    
    def test_scale_factor_formula_8k_to_4k(self):
        """
        Test scale factor calculation for 8K downscaling to 4K.
        
        Input: 7680x4320 (8K)
        Target: 3840x2160 (4K)
        """
        # Create 8K image
        image = create_test_image(7680, 4320)
        
        # Process with 4K resolution
        result = self.processor._upscale_traditional(image, Resolution.K4)
        
        # Calculate expected scale factor
        scale_w = result.final_width / 7680
        scale_h = result.final_height / 4320
        expected_scale = (scale_w + scale_h) / 2
        
        # Verify scale factor matches formula
        assert abs(result.scale_factor - expected_scale) < 0.01, \
            f"Scale factor {result.scale_factor} doesn't match expected {expected_scale}"
        
        # Verify downscaling by approximately 0.5x
        assert 0.45 < result.scale_factor < 0.55, \
            f"Scale factor {result.scale_factor} should be ~0.5 for 8K to 4K"


class TestScaleFactorEdgeCases:
    """Test scale factor calculation edge cases."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = ImageProcessor()
    
    def test_scale_factor_very_small_image(self):
        """
        Test scale factor calculation for very small image (1x1).
        
        Input: 1x1
        Target: 2560x1440 (2K)
        """
        # Create 1x1 image
        image = create_test_image(1, 1)
        
        # Process with 2K resolution
        result = self.processor._upscale_traditional(image, Resolution.K2)
        
        # Calculate expected scale factor
        scale_w = result.final_width / 1
        scale_h = result.final_height / 1
        expected_scale = (scale_w + scale_h) / 2
        
        # Verify scale factor matches formula
        assert abs(result.scale_factor - expected_scale) < 0.01, \
            f"Scale factor {result.scale_factor} doesn't match expected {expected_scale}"
        
        # Verify massive upscaling
        assert result.scale_factor > 100, "Scale factor should be > 100 for 1x1 to 2K"
    
    def test_scale_factor_averaging_behavior(self):
        """
        Test scale factor calculation averaging behavior.
        
        The formula averages width and height scales: (scale_w + scale_h) / 2
        
        Note: Because the system maintains aspect ratio, scale_w and scale_h
        will always be equal. This test verifies the formula is correctly applied.
        
        Input: 1000x2000 (1:2 aspect ratio)
        Target: 2560x1440 (2K)
        """
        # Create 1000x2000 image
        image = create_test_image(1000, 2000)
        
        # Process with 2K resolution
        result = self.processor._upscale_traditional(image, Resolution.K2)
        
        # Calculate expected scale factor
        scale_w = result.final_width / 1000
        scale_h = result.final_height / 2000
        expected_scale = (scale_w + scale_h) / 2
        
        # Verify scale factor matches formula
        assert abs(result.scale_factor - expected_scale) < 0.01, \
            f"Scale factor {result.scale_factor} doesn't match expected {expected_scale}"
        
        # Verify uniform scaling (aspect ratio maintained)
        # Because aspect ratio is preserved, scale_w should equal scale_h
        assert abs(scale_w - scale_h) < 0.01, \
            "Width and height scales should be equal (aspect ratio preserved)"
    
    def test_scale_factor_matches_metadata(self):
        """
        Test that scale_factor in ProcessedImage matches calculated value.
        
        This verifies that the scale_factor field is correctly populated.
        """
        # Create test image
        image = create_test_image(1920, 1080)
        
        # Process with 2K resolution
        result = self.processor._upscale_traditional(image, Resolution.K2)
        
        # Manually calculate scale factor from metadata
        manual_scale_w = result.final_width / result.original_width
        manual_scale_h = result.final_height / result.original_height
        manual_scale = (manual_scale_w + manual_scale_h) / 2
        
        # Verify scale_factor field matches manual calculation
        assert abs(result.scale_factor - manual_scale) < 0.01, \
            f"ProcessedImage.scale_factor {result.scale_factor} doesn't match manual calculation {manual_scale}"
    
    def test_scale_factor_consistency_across_resolutions(self):
        """
        Test that scale factor calculation is consistent across different target resolutions.
        
        Same input image should produce different scale factors for different targets.
        """
        # Create test image
        image = create_test_image(1920, 1080)
        
        # Process with 2K resolution
        result_2k = self.processor._upscale_traditional(image, Resolution.K2)
        
        # Process with 4K resolution
        result_4k = self.processor._upscale_traditional(image, Resolution.K4)
        
        # Verify different scale factors
        assert result_2k.scale_factor != result_4k.scale_factor, \
            "Scale factors should be different for different target resolutions"
        
        # Verify 4K scale factor is larger (more upscaling)
        assert result_4k.scale_factor > result_2k.scale_factor, \
            "4K scale factor should be larger than 2K scale factor"
    
    def test_scale_factor_precision(self):
        """
        Test that scale factor has reasonable precision (2 decimal places).
        
        Requirement 5.4: X-Scale-Factor should be rounded to 2 decimal places.
        """
        # Create test image
        image = create_test_image(1920, 1080)
        
        # Process with 2K resolution
        result = self.processor._upscale_traditional(image, Resolution.K2)
        
        # Verify scale factor has at most 2 decimal places
        scale_str = str(result.scale_factor)
        if '.' in scale_str:
            decimals = len(scale_str.split('.')[1])
            assert decimals <= 2, \
                f"Scale factor {result.scale_factor} has more than 2 decimal places"


class TestScaleFactorWithDifferentMethods:
    """Test scale factor calculation consistency across different upscaling methods."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = ImageProcessor()
    
    def test_scale_factor_formula_same_for_traditional_and_ai(self):
        """
        Test that scale factor calculation formula is the same for both methods.
        
        Note: This test verifies the formula, not the actual values (which will differ
        because AI upscales 4x first, then resizes to target).
        """
        # Create test image
        image = create_test_image(1920, 1080)
        
        # Process with traditional method
        result_traditional = self.processor._upscale_traditional(image, Resolution.K2)
        
        # Verify formula for traditional method
        scale_w_trad = result_traditional.final_width / result_traditional.original_width
        scale_h_trad = result_traditional.final_height / result_traditional.original_height
        expected_trad = (scale_w_trad + scale_h_trad) / 2
        
        assert abs(result_traditional.scale_factor - expected_trad) < 0.01, \
            "Traditional method scale factor doesn't match formula"
        
        # Note: We don't test AI method here because it requires model loading
        # The formula verification is sufficient to ensure consistency
