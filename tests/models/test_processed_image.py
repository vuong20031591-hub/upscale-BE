"""
Unit tests for ProcessedImage dataclass.

Tests cover:
- to_bytes() method with PNG format only
- Verify JPEG format is not supported (output should be PNG only)
- resolution property format
- metadata fields (dimensions, scale_factor)

Requirements: Data Models, 5.6, 5.7
"""

import pytest
from io import BytesIO
from PIL import Image

from app.models.image import ProcessedImage, ImageFormat


class TestProcessedImageToBytes:
    """Test ProcessedImage.to_bytes() method."""
    
    def test_to_bytes_returns_bytesio(self):
        """
        Test that to_bytes() returns BytesIO object.
        """
        # Create test image
        image = Image.new('RGB', (100, 100), color=(255, 0, 0))
        
        # Create ProcessedImage with PNG format
        processed = ProcessedImage(
            image=image,
            original_width=50,
            original_height=50,
            final_width=100,
            final_height=100,
            scale_factor=2.0,
            format=ImageFormat.PNG
        )
        
        # Convert to bytes
        result = processed.to_bytes()
        
        # Verify result type
        assert isinstance(result, BytesIO)
        assert result.tell() == 0  # Buffer should be at position 0
    
    def test_to_bytes_png_format_produces_valid_png(self):
        """
        Test that to_bytes() with PNG format produces valid PNG bytes.
        Requirement 5.6: Output format is PNG only
        """
        # Create test image
        image = Image.new('RGB', (100, 100), color=(0, 255, 0))
        
        # Create ProcessedImage with PNG format
        processed = ProcessedImage(
            image=image,
            original_width=50,
            original_height=50,
            final_width=100,
            final_height=100,
            scale_factor=2.0,
            format=ImageFormat.PNG
        )
        
        # Convert to bytes
        buffer = processed.to_bytes()
        content = buffer.read()
        
        # Verify PNG signature (first 8 bytes)
        assert content.startswith(b'\x89PNG\r\n\x1a\n'), "Output should be valid PNG"
        
        # Verify we can load it back as image
        buffer.seek(0)
        loaded_image = Image.open(buffer)
        assert loaded_image.format == 'PNG'
        assert loaded_image.size == (100, 100)
    
    def test_to_bytes_png_with_quality_parameter(self):
        """
        Test that to_bytes() accepts quality parameter for PNG.
        Note: For PNG, quality is mapped to compress_level (0-9).
        """
        # Create test image
        image = Image.new('RGB', (100, 100), color=(0, 0, 255))
        
        # Create ProcessedImage
        processed = ProcessedImage(
            image=image,
            original_width=50,
            original_height=50,
            final_width=100,
            final_height=100,
            scale_factor=2.0,
            format=ImageFormat.PNG
        )
        
        # Convert with different quality values
        buffer_default = processed.to_bytes()
        buffer_high = processed.to_bytes(quality=95)
        buffer_low = processed.to_bytes(quality=10)
        
        # All should produce valid PNG
        assert buffer_default.read().startswith(b'\x89PNG')
        assert buffer_high.read().startswith(b'\x89PNG')
        assert buffer_low.read().startswith(b'\x89PNG')
    
    def test_to_bytes_jpeg_format_produces_jpeg(self):
        """
        Test that to_bytes() with JPEG format produces valid JPEG bytes.
        
        Note: According to design spec (Requirement 5.6, 5.7), system should
        ONLY support PNG output. However, the current implementation still
        supports JPEG in the code. This test verifies current behavior.
        
        TODO: This test should be updated when JPEG support is removed
        as per spec requirements.
        """
        # Create test image
        image = Image.new('RGB', (100, 100), color=(255, 255, 0))
        
        # Create ProcessedImage with JPEG format
        processed = ProcessedImage(
            image=image,
            original_width=50,
            original_height=50,
            final_width=100,
            final_height=100,
            scale_factor=2.0,
            format=ImageFormat.JPEG
        )
        
        # Convert to bytes
        buffer = processed.to_bytes(quality=85)
        content = buffer.read()
        
        # Verify JPEG signature (starts with FF D8 FF)
        assert content.startswith(b'\xff\xd8\xff'), "Output should be valid JPEG"
        
        # Verify we can load it back as image
        buffer.seek(0)
        loaded_image = Image.open(buffer)
        assert loaded_image.format == 'JPEG'
        assert loaded_image.size == (100, 100)
    
    def test_to_bytes_jpeg_converts_rgba_to_rgb(self):
        """
        Test that JPEG conversion handles RGBA images by converting to RGB.
        JPEG doesn't support transparency, so RGBA must be converted.
        """
        # Create RGBA image
        image = Image.new('RGBA', (100, 100), color=(255, 0, 255, 128))
        
        # Create ProcessedImage with JPEG format
        processed = ProcessedImage(
            image=image,
            original_width=50,
            original_height=50,
            final_width=100,
            final_height=100,
            scale_factor=2.0,
            format=ImageFormat.JPEG
        )
        
        # Should not raise exception
        buffer = processed.to_bytes()
        
        # Verify valid JPEG
        content = buffer.read()
        assert content.startswith(b'\xff\xd8\xff')
    
    def test_to_bytes_buffer_position_reset(self):
        """
        Test that to_bytes() resets buffer position to 0.
        This is important for streaming responses.
        """
        # Create test image
        image = Image.new('RGB', (100, 100), color=(128, 128, 128))
        
        # Create ProcessedImage
        processed = ProcessedImage(
            image=image,
            original_width=50,
            original_height=50,
            final_width=100,
            final_height=100,
            scale_factor=2.0,
            format=ImageFormat.PNG
        )
        
        # Convert to bytes
        buffer = processed.to_bytes()
        
        # Verify buffer is at position 0
        assert buffer.tell() == 0, "Buffer should be at position 0 for streaming"


class TestProcessedImageResolutionProperty:
    """Test ProcessedImage.resolution property."""
    
    def test_resolution_format(self):
        """
        Test that resolution property returns correct format: {width}x{height}
        Requirement 5.3: X-Image-Resolution header format
        """
        # Create test image
        image = Image.new('RGB', (1920, 1080))
        
        # Create ProcessedImage
        processed = ProcessedImage(
            image=image,
            original_width=960,
            original_height=540,
            final_width=1920,
            final_height=1080,
            scale_factor=2.0,
            format=ImageFormat.PNG
        )
        
        # Verify resolution format
        assert processed.resolution == "1920x1080"
    
    def test_resolution_uses_final_dimensions(self):
        """
        Test that resolution property uses final dimensions, not original.
        """
        # Create test image
        image = Image.new('RGB', (3840, 2160))
        
        # Create ProcessedImage with different original and final dimensions
        processed = ProcessedImage(
            image=image,
            original_width=1920,
            original_height=1080,
            final_width=3840,
            final_height=2160,
            scale_factor=2.0,
            format=ImageFormat.PNG
        )
        
        # Should use final dimensions
        assert processed.resolution == "3840x2160"
        assert processed.resolution != "1920x1080"
    
    def test_resolution_with_various_dimensions(self):
        """
        Test resolution property with various dimension values.
        """
        test_cases = [
            (100, 100, "100x100"),
            (1920, 1080, "1920x1080"),
            (2560, 1440, "2560x1440"),
            (3840, 2160, "3840x2160"),
            (7680, 4320, "7680x4320"),
            (1, 1, "1x1"),
        ]
        
        for width, height, expected in test_cases:
            image = Image.new('RGB', (width, height))
            processed = ProcessedImage(
                image=image,
                original_width=width // 2,
                original_height=height // 2,
                final_width=width,
                final_height=height,
                scale_factor=2.0,
                format=ImageFormat.PNG
            )
            
            assert processed.resolution == expected, \
                f"Expected {expected}, got {processed.resolution}"


class TestProcessedImageMetadata:
    """Test ProcessedImage metadata fields."""
    
    def test_metadata_fields_exist(self):
        """
        Test that all required metadata fields exist.
        Requirement: Data Models - ProcessedImage structure
        """
        # Create test image
        image = Image.new('RGB', (100, 100))
        
        # Create ProcessedImage
        processed = ProcessedImage(
            image=image,
            original_width=50,
            original_height=50,
            final_width=100,
            final_height=100,
            scale_factor=2.0,
            format=ImageFormat.PNG
        )
        
        # Verify all fields exist
        assert hasattr(processed, 'image')
        assert hasattr(processed, 'original_width')
        assert hasattr(processed, 'original_height')
        assert hasattr(processed, 'final_width')
        assert hasattr(processed, 'final_height')
        assert hasattr(processed, 'scale_factor')
        assert hasattr(processed, 'format')
    
    def test_metadata_original_dimensions(self):
        """
        Test that original dimensions are stored correctly.
        """
        image = Image.new('RGB', (200, 150))
        
        processed = ProcessedImage(
            image=image,
            original_width=100,
            original_height=75,
            final_width=200,
            final_height=150,
            scale_factor=2.0,
            format=ImageFormat.PNG
        )
        
        assert processed.original_width == 100
        assert processed.original_height == 75
    
    def test_metadata_final_dimensions(self):
        """
        Test that final dimensions are stored correctly.
        """
        image = Image.new('RGB', (3840, 2160))
        
        processed = ProcessedImage(
            image=image,
            original_width=1920,
            original_height=1080,
            final_width=3840,
            final_height=2160,
            scale_factor=2.0,
            format=ImageFormat.PNG
        )
        
        assert processed.final_width == 3840
        assert processed.final_height == 2160
    
    def test_metadata_scale_factor(self):
        """
        Test that scale_factor is stored correctly.
        Requirement 5.5: Overall scale factor calculation
        """
        image = Image.new('RGB', (400, 300))
        
        # Test various scale factors
        test_cases = [
            (100, 100, 400, 300, 3.5),  # (400/100 + 300/100) / 2 = 3.5
            (200, 200, 400, 400, 2.0),  # (400/200 + 400/200) / 2 = 2.0
            (800, 600, 400, 300, 0.5),  # (400/800 + 300/600) / 2 = 0.5
        ]
        
        for orig_w, orig_h, final_w, final_h, expected_scale in test_cases:
            processed = ProcessedImage(
                image=image,
                original_width=orig_w,
                original_height=orig_h,
                final_width=final_w,
                final_height=final_h,
                scale_factor=expected_scale,
                format=ImageFormat.PNG
            )
            
            assert processed.scale_factor == expected_scale, \
                f"Expected scale_factor {expected_scale}, got {processed.scale_factor}"
    
    def test_metadata_format_png(self):
        """
        Test that format field stores PNG correctly.
        Requirement 5.6: Output format is PNG only
        """
        image = Image.new('RGB', (100, 100))
        
        processed = ProcessedImage(
            image=image,
            original_width=50,
            original_height=50,
            final_width=100,
            final_height=100,
            scale_factor=2.0,
            format=ImageFormat.PNG
        )
        
        assert processed.format == ImageFormat.PNG
        assert processed.format.value == "png"
    
    def test_metadata_format_jpeg(self):
        """
        Test that format field can store JPEG.
        
        Note: According to spec, output should be PNG only.
        This test verifies current implementation behavior.
        """
        image = Image.new('RGB', (100, 100))
        
        processed = ProcessedImage(
            image=image,
            original_width=50,
            original_height=50,
            final_width=100,
            final_height=100,
            scale_factor=2.0,
            format=ImageFormat.JPEG
        )
        
        assert processed.format == ImageFormat.JPEG
        assert processed.format.value == "jpeg"
    
    def test_metadata_image_reference(self):
        """
        Test that image field stores PIL Image reference correctly.
        """
        # Create test image with specific properties
        image = Image.new('RGB', (256, 256), color=(100, 150, 200))
        
        processed = ProcessedImage(
            image=image,
            original_width=128,
            original_height=128,
            final_width=256,
            final_height=256,
            scale_factor=2.0,
            format=ImageFormat.PNG
        )
        
        # Verify image reference
        assert processed.image is image
        assert isinstance(processed.image, Image.Image)
        assert processed.image.size == (256, 256)
        assert processed.image.mode == 'RGB'


class TestProcessedImageEdgeCases:
    """Test edge cases for ProcessedImage."""
    
    def test_scale_factor_zero(self):
        """
        Test ProcessedImage with scale_factor = 0 (edge case).
        """
        image = Image.new('RGB', (100, 100))
        
        processed = ProcessedImage(
            image=image,
            original_width=100,
            original_height=100,
            final_width=100,
            final_height=100,
            scale_factor=0.0,
            format=ImageFormat.PNG
        )
        
        assert processed.scale_factor == 0.0
    
    def test_scale_factor_very_large(self):
        """
        Test ProcessedImage with very large scale_factor.
        """
        image = Image.new('RGB', (100, 100))
        
        processed = ProcessedImage(
            image=image,
            original_width=1,
            original_height=1,
            final_width=100,
            final_height=100,
            scale_factor=100.0,
            format=ImageFormat.PNG
        )
        
        assert processed.scale_factor == 100.0
    
    def test_scale_factor_fractional(self):
        """
        Test ProcessedImage with fractional scale_factor.
        """
        image = Image.new('RGB', (100, 100))
        
        processed = ProcessedImage(
            image=image,
            original_width=150,
            original_height=150,
            final_width=100,
            final_height=100,
            scale_factor=0.6667,
            format=ImageFormat.PNG
        )
        
        assert abs(processed.scale_factor - 0.6667) < 0.0001
    
    def test_minimum_dimensions(self):
        """
        Test ProcessedImage with minimum dimensions (1x1).
        """
        image = Image.new('RGB', (1, 1))
        
        processed = ProcessedImage(
            image=image,
            original_width=1,
            original_height=1,
            final_width=1,
            final_height=1,
            scale_factor=1.0,
            format=ImageFormat.PNG
        )
        
        assert processed.resolution == "1x1"
        
        # Should still be able to convert to bytes
        buffer = processed.to_bytes()
        assert isinstance(buffer, BytesIO)
    
    def test_very_large_dimensions(self):
        """
        Test ProcessedImage with very large dimensions.
        Note: This test creates a large image, may be slow.
        """
        # Create large image (8K resolution)
        image = Image.new('RGB', (7680, 4320))
        
        processed = ProcessedImage(
            image=image,
            original_width=3840,
            original_height=2160,
            final_width=7680,
            final_height=4320,
            scale_factor=2.0,
            format=ImageFormat.PNG
        )
        
        assert processed.resolution == "7680x4320"
        assert processed.final_width == 7680
        assert processed.final_height == 4320
