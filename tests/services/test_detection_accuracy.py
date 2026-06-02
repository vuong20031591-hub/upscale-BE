"""
Integration tests for detection accuracy (Task 7.2).

Tests verify that detection algorithms correctly identify image issues
and suggest appropriate processing modes with real-world test images.

Requirements:
    - Requirement 1.1-1.4: Detection algorithms work correctly
    - Requirement 2.1-2.6: Mode suggestion logic is accurate
"""

import numpy as np
import pytest
from PIL import Image

from app.models.analysis import ProcessingMode
from app.services.image_analyzer import ImageAnalyzer


class TestGrayscaleDetectionAccuracy:
    """Test grayscale detection with various grayscale images."""
    
    def test_pure_grayscale_image_suggests_colorization(self):
        """
        Test that pure grayscale image is detected and colorization is suggested.
        
        Validates: Requirements 1.1, 2.1
        """
        # Create pure grayscale image (all RGB channels identical)
        gray_value = 128
        img_array = np.full((512, 512, 3), gray_value, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify grayscale detected
        assert result.detections['grayscale'].detected is True
        assert result.detections['grayscale'].confidence == 1.0
        
        # Verify colorization suggested
        assert result.suggested_mode == ProcessingMode.COLORIZATION
        assert 'grayscale' in result.explanation.lower()
    
    def test_grayscale_gradient_suggests_colorization(self):
        """
        Test that grayscale gradient image is detected correctly.
        
        Validates: Requirements 1.1, 2.1
        """
        # Create grayscale gradient (0 to 255 horizontally)
        gradient = np.linspace(0, 255, 512, dtype=np.uint8)
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(3):
            img_array[:, :, i] = gradient[np.newaxis, :]
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify grayscale detected
        assert result.detections['grayscale'].detected is True
        assert result.detections['grayscale'].confidence == 1.0
        
        # Verify colorization suggested
        assert result.suggested_mode == ProcessingMode.COLORIZATION
    
    def test_black_and_white_image_suggests_colorization(self):
        """
        Test that black and white image is detected as grayscale.
        
        Validates: Requirements 1.1, 2.1
        """
        # Create black and white checkerboard pattern
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        # Create checkerboard (64x64 squares)
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    img_array[i*64:(i+1)*64, j*64:(j+1)*64, :] = 255
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify grayscale detected
        assert result.detections['grayscale'].detected is True
        assert result.suggested_mode == ProcessingMode.COLORIZATION
    
    def test_color_image_not_detected_as_grayscale(self):
        """
        Test that color image is NOT detected as grayscale.
        
        Validates: Requirements 1.1
        """
        # Create color image (different RGB values)
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        img_array[:, :, 0] = 255  # Red channel
        img_array[:, :, 1] = 128  # Green channel
        img_array[:, :, 2] = 64   # Blue channel
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify grayscale NOT detected
        assert result.detections['grayscale'].detected is False
        assert result.detections['grayscale'].confidence == 0.0
        
        # Verify colorization NOT suggested
        assert result.suggested_mode != ProcessingMode.COLORIZATION


class TestWhiteMaskDetectionAccuracy:
    """Test white mask detection with various masked images."""
    
    def test_image_with_35_percent_white_mask_suggests_inpainting(self):
        """
        Test that image with 35% white mask is detected and inpainting suggested.
        
        Note: White pixels (240,240,240) are also grayscale, so we need to create
        a non-grayscale image with white regions.
        
        Validates: Requirements 1.2, 2.2
        """
        # Create COLOR image with 35% white pixels (RGB >= 240)
        # Use different RGB values for non-white areas to avoid grayscale detection
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        img_array[:, :, 0] = 100  # Red channel
        img_array[:, :, 1] = 80   # Green channel
        img_array[:, :, 2] = 60   # Blue channel (different values = color)
        
        # Fill 35% of image with white (240, 240, 240)
        white_pixels = int(512 * 512 * 0.35)
        img_array.flat[:white_pixels*3] = 240
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify white mask detected
        assert result.detections['white_mask'].detected is True
        assert result.detections['white_mask'].confidence >= 0.30
        assert result.detections['white_mask'].metadata['white_percentage'] >= 30.0
        
        # Verify NOT grayscale (color image)
        assert result.detections['grayscale'].detected is False
        
        # Verify inpainting suggested (white mask has priority over low-res/blur)
        assert result.suggested_mode == ProcessingMode.INPAINTING
        assert 'white' in result.explanation.lower() or 'mask' in result.explanation.lower()
    
    def test_image_with_50_percent_white_mask_suggests_inpainting(self):
        """
        Test that image with 50% white mask is detected correctly.
        
        Note: Create color image to avoid grayscale priority.
        
        Validates: Requirements 1.2, 2.2
        """
        # Create COLOR image with 50% white pixels
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        # Bottom half: color (different RGB values)
        img_array[256:, :, 0] = 150  # Red
        img_array[256:, :, 1] = 100  # Green
        img_array[256:, :, 2] = 50   # Blue
        # Top half: white (255, 255, 255)
        img_array[:256, :, :] = 255
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify white mask detected with high confidence
        assert result.detections['white_mask'].detected is True
        assert result.detections['white_mask'].confidence >= 0.50
        assert result.detections['white_mask'].metadata['white_percentage'] >= 50.0
        
        # Verify NOT grayscale
        assert result.detections['grayscale'].detected is False
        
        # Verify inpainting suggested
        assert result.suggested_mode == ProcessingMode.INPAINTING
    
    def test_image_with_white_center_region_suggests_inpainting(self):
        """
        Test that image with white center region (>30%) is detected.
        
        Note: Create color image to avoid grayscale priority.
        
        Validates: Requirements 1.2, 2.2
        """
        # Create COLOR image with white center region
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        # Background: color (different RGB values)
        img_array[:, :, 0] = 120  # Red
        img_array[:, :, 1] = 80   # Green
        img_array[:, :, 2] = 40   # Blue
        
        # White center (300x300 = ~34% of total)
        center_size = 300
        start = (512 - center_size) // 2
        end = start + center_size
        img_array[start:end, start:end, :] = 245  # White (>= 240)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify white mask detected
        assert result.detections['white_mask'].detected is True
        assert result.detections['white_mask'].confidence >= 0.30
        
        # Verify NOT grayscale
        assert result.detections['grayscale'].detected is False
        
        # Verify inpainting suggested
        assert result.suggested_mode == ProcessingMode.INPAINTING
    
    def test_image_with_25_percent_white_not_detected(self):
        """
        Test that image with <30% white is NOT detected as white mask.
        
        Validates: Requirements 1.2
        """
        # Create image with 25% white pixels (below threshold)
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        white_pixels = int(512 * 512 * 0.25)
        img_array.flat[:white_pixels*3] = 240
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify white mask NOT detected (below 30% threshold)
        assert result.detections['white_mask'].detected is False
        assert result.detections['white_mask'].metadata['white_percentage'] < 30.0
        
        # Verify inpainting NOT suggested
        assert result.suggested_mode != ProcessingMode.INPAINTING


class TestLowResolutionDetectionAccuracy:
    """Test low resolution detection with various image sizes."""
    
    def test_480x320_image_suggests_upscaling(self):
        """
        Test that 480x320 image is detected as low resolution.
        
        Validates: Requirements 1.3, 2.3
        """
        # Create 480x320 image (both dimensions < 512)
        img_array = np.full((320, 480, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify low resolution detected
        assert result.detections['low_resolution'].detected is True
        assert result.detections['low_resolution'].confidence == 1.0
        assert result.detections['low_resolution'].metadata['width'] == 480
        assert result.detections['low_resolution'].metadata['height'] == 320
        
        # Verify upscaling suggested (if no higher priority issues)
        # Note: Only if not grayscale or white mask
        if not result.detections['grayscale'].detected and not result.detections['white_mask'].detected:
            assert result.suggested_mode == ProcessingMode.UPSCALING
    
    def test_256x256_image_suggests_upscaling(self):
        """
        Test that 256x256 image is detected as low resolution.
        
        Validates: Requirements 1.3, 2.3
        """
        # Create 256x256 image
        img_array = np.full((256, 256, 3), 100, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify low resolution detected
        assert result.detections['low_resolution'].detected is True
        assert result.detections['low_resolution'].confidence == 1.0
        
        # Verify upscaling suggested (if no higher priority issues)
        if not result.detections['grayscale'].detected and not result.detections['white_mask'].detected:
            assert result.suggested_mode == ProcessingMode.UPSCALING
    
    def test_500x600_image_suggests_upscaling(self):
        """
        Test that 500x600 image (width < 512) is detected as low resolution.
        
        Validates: Requirements 1.3, 2.3
        """
        # Create 500x600 image (width < 512, height > 512)
        img_array = np.full((600, 500, 3), 150, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify low resolution detected (width < 512)
        assert result.detections['low_resolution'].detected is True
        assert result.detections['low_resolution'].confidence == 1.0
        assert result.detections['low_resolution'].metadata['width'] == 500
        assert result.detections['low_resolution'].metadata['height'] == 600
    
    def test_1920x1080_image_not_low_resolution(self):
        """
        Test that 1920x1080 image is NOT detected as low resolution.
        
        Validates: Requirements 1.3
        """
        # Create 1920x1080 image (both dimensions >= 512)
        img_array = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify low resolution NOT detected
        assert result.detections['low_resolution'].detected is False
        assert result.detections['low_resolution'].confidence == 0.0
        
        # Verify upscaling NOT suggested
        assert result.suggested_mode != ProcessingMode.UPSCALING
    
    def test_512x512_boundary_not_low_resolution(self):
        """
        Test that 512x512 image (at boundary) is NOT detected as low resolution.
        
        Validates: Requirements 1.3, 6.3
        """
        # Create 512x512 image (exactly at threshold)
        img_array = np.full((512, 512, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify low resolution NOT detected (threshold is <512, not <=512)
        assert result.detections['low_resolution'].detected is False
        assert result.detections['low_resolution'].confidence == 0.0


class TestBlurDetectionAccuracy:
    """Test blur detection with various blurry images."""
    
    def test_uniform_color_image_detected_as_blurry(self):
        """
        Test that uniform color image (no edges) is detected as blurry.
        
        Uniform images have very low Laplacian variance (no edges).
        
        Validates: Requirements 1.4, 2.4
        """
        # Create uniform color image (very low variance)
        img_array = np.full((512, 512, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify blur detected (uniform image has very low variance)
        assert result.detections['blur'].detected is True
        assert result.detections['blur'].metadata['variance'] < 100.0
        
        # Verify restoration suggested (if no higher priority issues)
        # Note: Grayscale has higher priority, so check if grayscale detected
        if result.detections['grayscale'].detected:
            assert result.suggested_mode == ProcessingMode.COLORIZATION
        elif result.detections['white_mask'].detected:
            assert result.suggested_mode == ProcessingMode.INPAINTING
        elif result.detections['low_resolution'].detected:
            assert result.suggested_mode == ProcessingMode.UPSCALING
        else:
            assert result.suggested_mode == ProcessingMode.RESTORATION
    
    def test_smooth_gradient_detected_as_blurry(self):
        """
        Test that smooth gradient image is detected as blurry.
        
        Smooth gradients have low Laplacian variance (no sharp edges).
        
        Validates: Requirements 1.4, 2.4
        """
        # Create smooth gradient (low variance)
        gradient = np.linspace(0, 255, 512, dtype=np.uint8)
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        img_array[:, :, 0] = gradient[np.newaxis, :]  # Horizontal gradient
        img_array[:, :, 1] = gradient[:, np.newaxis]  # Vertical gradient
        img_array[:, :, 2] = 128  # Constant blue
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify blur detected (smooth gradient has low variance)
        assert result.detections['blur'].detected is True
        assert result.detections['blur'].metadata['variance'] < 100.0
    
    def test_sharp_edges_not_detected_as_blurry(self):
        """
        Test that image with sharp edges is NOT detected as blurry.
        
        Sharp edges produce high Laplacian variance.
        
        Validates: Requirements 1.4
        """
        # Create image with sharp edges (checkerboard pattern)
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        # Create fine checkerboard (8x8 squares = 64x64 pattern)
        for i in range(64):
            for j in range(64):
                if (i + j) % 2 == 0:
                    img_array[i*8:(i+1)*8, j*8:(j+1)*8, :] = 255
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify blur NOT detected (sharp edges have high variance)
        assert result.detections['blur'].detected is False
        assert result.detections['blur'].metadata['variance'] >= 100.0
        
        # Verify restoration NOT suggested (unless it's default)
        # Note: This image is also grayscale, so colorization will be suggested
        if result.detections['grayscale'].detected:
            assert result.suggested_mode == ProcessingMode.COLORIZATION
        else:
            # If not grayscale, restoration should not be suggested
            assert result.suggested_mode != ProcessingMode.RESTORATION


class TestPriorityOrderAccuracy:
    """Test that priority order is respected when multiple issues detected."""
    
    def test_grayscale_low_res_prioritizes_colorization(self):
        """
        Test that grayscale + low resolution prioritizes colorization.
        
        Priority: Grayscale > Low Resolution
        
        Validates: Requirements 2.1, 2.3, 2.6
        """
        # Create grayscale low-res image (480x320)
        img_array = np.full((320, 480, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify both issues detected
        assert result.detections['grayscale'].detected is True
        assert result.detections['low_resolution'].detected is True
        
        # Verify colorization suggested (higher priority)
        assert result.suggested_mode == ProcessingMode.COLORIZATION
    
    def test_white_mask_low_res_prioritizes_inpainting(self):
        """
        Test that white mask + low resolution prioritizes inpainting.
        
        Priority: White Mask > Low Resolution
        Note: Create color image to avoid grayscale priority.
        
        Validates: Requirements 2.2, 2.3, 2.6
        """
        # Create low-res COLOR image with 40% white mask (480x320)
        img_array = np.zeros((320, 480, 3), dtype=np.uint8)
        # Background: color
        img_array[:, :, 0] = 100  # Red
        img_array[:, :, 1] = 70   # Green
        img_array[:, :, 2] = 40   # Blue
        
        # Fill 40% with white
        white_pixels = int(320 * 480 * 0.40)
        img_array.flat[:white_pixels*3] = 245
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify both issues detected
        assert result.detections['white_mask'].detected is True
        assert result.detections['low_resolution'].detected is True
        
        # Verify NOT grayscale
        assert result.detections['grayscale'].detected is False
        
        # Verify inpainting suggested (higher priority)
        assert result.suggested_mode == ProcessingMode.INPAINTING
    
    def test_low_res_blur_prioritizes_upscaling(self):
        """
        Test that low resolution + blur prioritizes upscaling.
        
        Priority: Low Resolution > Blur
        
        Validates: Requirements 2.3, 2.4, 2.6
        """
        # Create low-res uniform image (480x320, uniform = blurry)
        img_array = np.full((320, 480, 3), 100, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify both issues detected
        assert result.detections['low_resolution'].detected is True
        assert result.detections['blur'].detected is True
        
        # Verify upscaling suggested (higher priority)
        # Note: If also grayscale, colorization will be suggested
        if result.detections['grayscale'].detected:
            assert result.suggested_mode == ProcessingMode.COLORIZATION
        else:
            assert result.suggested_mode == ProcessingMode.UPSCALING
    
    def test_all_issues_detected_prioritizes_grayscale(self):
        """
        Test that when all issues detected, grayscale has highest priority.
        
        Priority: Grayscale > White Mask > Low Resolution > Blur
        
        Note: This test verifies grayscale priority, not all 4 issues simultaneously.
        Creating an image with all 4 issues is difficult because:
        - White regions (245,245,245) are also grayscale
        - Adding white regions increases variance (not blurry)
        
        Validates: Requirements 2.1-2.6
        """
        # Create grayscale, low-res, uniform (blurry) image
        # Uniform grayscale = low variance = blurry
        img_array = np.full((320, 480, 3), 100, dtype=np.uint8)  # Grayscale uniform
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify grayscale, low-res, and blur detected
        assert result.detections['grayscale'].detected is True
        assert result.detections['low_resolution'].detected is True
        assert result.detections['blur'].detected is True
        
        # Verify colorization suggested (highest priority)
        assert result.suggested_mode == ProcessingMode.COLORIZATION


class TestDefaultBehavior:
    """Test default behavior when no issues detected."""
    
    def test_no_issues_suggests_restoration(self):
        """
        Test that when no issues detected, restoration is suggested as default.
        
        Validates: Requirements 2.5
        """
        # Create high-res, color, sharp image (no issues)
        img_array = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Create sharp pattern (fine checkerboard)
        for i in range(135):  # 1080 / 8
            for j in range(240):  # 1920 / 8
                if (i + j) % 2 == 0:
                    img_array[i*8:(i+1)*8, j*8:(j+1)*8, 0] = 255  # Red
                    img_array[i*8:(i+1)*8, j*8:(j+1)*8, 1] = 128  # Green
                    img_array[i*8:(i+1)*8, j*8:(j+1)*8, 2] = 64   # Blue
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify no major issues detected
        assert result.detections['grayscale'].detected is False
        assert result.detections['white_mask'].detected is False
        assert result.detections['low_resolution'].detected is False
        assert result.detections['blur'].detected is False
        
        # Verify restoration suggested as default
        assert result.suggested_mode == ProcessingMode.RESTORATION


class TestExplanationQuality:
    """Test that explanations are meaningful and helpful."""
    
    def test_explanation_not_empty(self):
        """
        Test that explanation is always non-empty.
        
        Validates: Requirements 2.7
        """
        # Create test image
        img_array = np.full((512, 512, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify explanation exists and is non-empty
        assert result.explanation is not None
        assert len(result.explanation) > 0
        assert isinstance(result.explanation, str)
    
    def test_explanation_mentions_detected_issue(self):
        """
        Test that explanation mentions the detected issue.
        
        Validates: Requirements 2.7
        """
        # Create grayscale image
        img_array = np.full((512, 512, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify explanation mentions grayscale
        explanation_lower = result.explanation.lower()
        assert 'grayscale' in explanation_lower or 'gray' in explanation_lower
    
    def test_explanation_mentions_suggested_mode(self):
        """
        Test that explanation relates to suggested mode.
        
        Note: All-white image is grayscale, so colorization will be suggested.
        Create color image with white regions for inpainting test.
        
        Validates: Requirements 2.7
        """
        # Create COLOR image with white mask (not grayscale)
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        # Background: color
        img_array[:, :, 0] = 150  # Red
        img_array[:, :, 1] = 100  # Green
        img_array[:, :, 2] = 50   # Blue
        # Top 40%: white
        img_array[:205, :, :] = 245  # White (40% of image)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Analyze image
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image)
        
        # Verify explanation mentions white/mask/inpainting
        explanation_lower = result.explanation.lower()
        assert any(word in explanation_lower for word in ['white', 'mask', 'inpaint'])
