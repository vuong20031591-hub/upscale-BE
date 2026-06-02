"""
Quick test for low resolution detection implementation.
"""

import numpy as np
from PIL import Image

from app.services.image_analyzer import ImageAnalyzer


def test_detect_low_resolution_below_threshold():
    """Test low resolution detection for image below 512px threshold."""
    # Create 480x320 image (both dimensions below 512)
    img_array = np.zeros((320, 480, 3), dtype=np.uint8)
    image = Image.fromarray(img_array, mode='RGB')
    
    analyzer = ImageAnalyzer()
    result = analyzer.detect_low_resolution(image)
    
    # Should be detected
    assert result.detected is True
    assert result.confidence == 1.0
    assert result.metadata["width"] == 480
    assert result.metadata["height"] == 320
    print("✓ Test passed: Low resolution detected for 480x320 image")


def test_detect_low_resolution_width_below_threshold():
    """Test low resolution detection for image with only width below 512px."""
    # Create 400x600 image (width < 512, height > 512)
    img_array = np.zeros((600, 400, 3), dtype=np.uint8)
    image = Image.fromarray(img_array, mode='RGB')
    
    analyzer = ImageAnalyzer()
    result = analyzer.detect_low_resolution(image)
    
    # Should be detected (width < 512)
    assert result.detected is True
    assert result.confidence == 1.0
    assert result.metadata["width"] == 400
    assert result.metadata["height"] == 600
    print("✓ Test passed: Low resolution detected for 400x600 image")


def test_detect_low_resolution_height_below_threshold():
    """Test low resolution detection for image with only height below 512px."""
    # Create 800x300 image (width > 512, height < 512)
    img_array = np.zeros((300, 800, 3), dtype=np.uint8)
    image = Image.fromarray(img_array, mode='RGB')
    
    analyzer = ImageAnalyzer()
    result = analyzer.detect_low_resolution(image)
    
    # Should be detected (height < 512)
    assert result.detected is True
    assert result.confidence == 1.0
    assert result.metadata["width"] == 800
    assert result.metadata["height"] == 300
    print("✓ Test passed: Low resolution detected for 800x300 image")


def test_detect_low_resolution_at_threshold():
    """Test low resolution detection at exactly 512x512 (boundary case)."""
    # Create 512x512 image (at threshold, should NOT be detected)
    img_array = np.zeros((512, 512, 3), dtype=np.uint8)
    image = Image.fromarray(img_array, mode='RGB')
    
    analyzer = ImageAnalyzer()
    result = analyzer.detect_low_resolution(image)
    
    # Should NOT be detected (not below threshold)
    assert result.detected is False
    assert result.confidence == 0.0
    assert result.metadata["width"] == 512
    assert result.metadata["height"] == 512
    print("✓ Test passed: Low resolution NOT detected for 512x512 image")


def test_detect_low_resolution_above_threshold():
    """Test low resolution detection for image above 512px threshold."""
    # Create 1024x768 image (both dimensions above 512)
    img_array = np.zeros((768, 1024, 3), dtype=np.uint8)
    image = Image.fromarray(img_array, mode='RGB')
    
    analyzer = ImageAnalyzer()
    result = analyzer.detect_low_resolution(image)
    
    # Should NOT be detected
    assert result.detected is False
    assert result.confidence == 0.0
    assert result.metadata["width"] == 1024
    assert result.metadata["height"] == 768
    print("✓ Test passed: Low resolution NOT detected for 1024x768 image")


if __name__ == "__main__":
    test_detect_low_resolution_below_threshold()
    test_detect_low_resolution_width_below_threshold()
    test_detect_low_resolution_height_below_threshold()
    test_detect_low_resolution_at_threshold()
    test_detect_low_resolution_above_threshold()
    print("\n✅ All tests passed!")
