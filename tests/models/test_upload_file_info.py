"""
Unit tests for UploadFileInfo dataclass.

Tests cover:
- Extension property extraction
- to_image() conversion with various modes
- RGB conversion behavior

Requirements: 1.5
"""

import pytest
from io import BytesIO
from PIL import Image

from app.models.image import UploadFileInfo


class TestUploadFileInfoExtension:
    """Test extension property extraction."""
    
    @pytest.mark.parametrize("filename,expected_extension", [
        # Standard cases
        ("test.jpg", "jpg"),
        ("test.jpeg", "jpeg"),
        ("test.png", "png"),
        
        # Case normalization
        ("TEST.JPG", "jpg"),
        ("Test.PNG", "png"),
        ("IMAGE.JPEG", "jpeg"),
        
        # Multiple dots - should extract last part
        ("test.tar.gz", "gz"),
        ("archive.backup.zip", "zip"),
        ("file.name.with.dots.png", "png"),
        
        # No extension
        ("test", ""),
        ("noextension", ""),
        
        # Hidden files
        (".gitignore", "gitignore"),
        (".htaccess", "htaccess"),
        
        # Edge cases
        (".", ""),
        ("..", ""),
        ("file.", ""),
    ])
    def test_extension_extraction(self, filename, expected_extension):
        """
        Test extension property extracts correct extension.
        
        Validates:
        - Lowercase normalization
        - Last part extraction (after last dot)
        - Empty string for no extension
        """
        file_info = UploadFileInfo(
            filename=filename,
            content_type="image/jpeg",
            size=100,
            content=b"test content"
        )
        
        assert file_info.extension == expected_extension


class TestUploadFileInfoToImage:
    """Test to_image() conversion with various modes."""
    
    def test_to_image_rgb_mode(self, create_test_image_bytes):
        """Test to_image() with RGB image - should remain RGB."""
        content = create_test_image_bytes(100, 100, 'RGB', 'PNG')
        
        file_info = UploadFileInfo(
            filename="test.png",
            content_type="image/png",
            size=len(content),
            content=content
        )
        
        image = file_info.to_image()
        
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
        assert image.size == (100, 100)
    
    def test_to_image_rgba_mode_converts_to_rgb(self, create_test_image_bytes):
        """Test to_image() with RGBA image - should convert to RGB."""
        content = create_test_image_bytes(100, 100, 'RGBA', 'PNG')
        
        file_info = UploadFileInfo(
            filename="test.png",
            content_type="image/png",
            size=len(content),
            content=content
        )
        
        image = file_info.to_image()
        
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'  # Converted from RGBA
        assert image.size == (100, 100)
    
    def test_to_image_grayscale_mode_converts_to_rgb(self, create_test_image_bytes):
        """Test to_image() with grayscale (L) image - should convert to RGB."""
        content = create_test_image_bytes(100, 100, 'L', 'PNG')
        
        file_info = UploadFileInfo(
            filename="test.png",
            content_type="image/png",
            size=len(content),
            content=content
        )
        
        image = file_info.to_image()
        
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'  # Converted from L
        assert image.size == (100, 100)
    
    def test_to_image_cmyk_mode_converts_to_rgb(self, create_test_image_bytes):
        """Test to_image() with CMYK image - should convert to RGB."""
        content = create_test_image_bytes(100, 100, 'CMYK', 'JPEG')
        
        file_info = UploadFileInfo(
            filename="test.jpg",
            content_type="image/jpeg",
            size=len(content),
            content=content
        )
        
        image = file_info.to_image()
        
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'  # Converted from CMYK
        assert image.size == (100, 100)
    
    def test_to_image_palette_mode_converts_to_rgb(self):
        """Test to_image() with palette (P) mode - should convert to RGB."""
        # Create palette mode image
        image_p = Image.new('P', (100, 100))
        buffer = BytesIO()
        image_p.save(buffer, format='PNG')
        content = buffer.getvalue()
        
        file_info = UploadFileInfo(
            filename="test.png",
            content_type="image/png",
            size=len(content),
            content=content
        )
        
        image = file_info.to_image()
        
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'  # Converted from P
        assert image.size == (100, 100)
    
    def test_to_image_with_jpeg_format(self, create_test_image_bytes):
        """Test to_image() with JPEG format."""
        content = create_test_image_bytes(200, 150, 'RGB', 'JPEG')
        
        file_info = UploadFileInfo(
            filename="test.jpg",
            content_type="image/jpeg",
            size=len(content),
            content=content
        )
        
        image = file_info.to_image()
        
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
        assert image.size == (200, 150)
    
    def test_to_image_with_png_format(self, create_test_image_bytes):
        """Test to_image() with PNG format."""
        content = create_test_image_bytes(200, 150, 'RGB', 'PNG')
        
        file_info = UploadFileInfo(
            filename="test.png",
            content_type="image/png",
            size=len(content),
            content=content
        )
        
        image = file_info.to_image()
        
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
        assert image.size == (200, 150)


class TestUploadFileInfoRGBConversion:
    """Test RGB conversion behavior specifically."""
    
    def test_rgb_conversion_idempotence(self, create_test_image_bytes):
        """
        Test that calling to_image() multiple times returns RGB consistently.
        
        Validates: Requirements 1.5 - RGB conversion idempotence
        """
        content = create_test_image_bytes(100, 100, 'RGBA', 'PNG')
        
        file_info = UploadFileInfo(
            filename="test.png",
            content_type="image/png",
            size=len(content),
            content=content
        )
        
        # Call to_image() multiple times
        image1 = file_info.to_image()
        image2 = file_info.to_image()
        image3 = file_info.to_image()
        
        # All should be RGB
        assert image1.mode == 'RGB'
        assert image2.mode == 'RGB'
        assert image3.mode == 'RGB'
    
    @pytest.mark.parametrize("mode", ['RGB', 'RGBA', 'L', 'CMYK', 'P'])
    def test_all_modes_convert_to_rgb(self, mode):
        """
        Test that all common image modes convert to RGB.
        
        Validates: Requirements 1.5 - All modes should convert to RGB
        """
        # Create image with specific mode
        if mode == 'RGB':
            image = Image.new(mode, (100, 100), (255, 0, 0))
        elif mode == 'RGBA':
            image = Image.new(mode, (100, 100), (255, 0, 0, 255))
        elif mode == 'L':
            image = Image.new(mode, (100, 100), 255)
        elif mode == 'CMYK':
            image = Image.new(mode, (100, 100), (0, 100, 100, 0))
        elif mode == 'P':
            image = Image.new(mode, (100, 100))
        
        # Convert to bytes
        buffer = BytesIO()
        format_type = 'JPEG' if mode == 'CMYK' else 'PNG'
        image.save(buffer, format=format_type)
        content = buffer.getvalue()
        
        file_info = UploadFileInfo(
            filename=f"test.{format_type.lower()}",
            content_type=f"image/{format_type.lower()}",
            size=len(content),
            content=content
        )
        
        result = file_info.to_image()
        
        assert result.mode == 'RGB', f"Mode {mode} should convert to RGB"
    
    def test_rgb_conversion_preserves_dimensions(self, create_test_image_bytes):
        """Test that RGB conversion preserves image dimensions."""
        original_width, original_height = 300, 200
        content = create_test_image_bytes(original_width, original_height, 'RGBA', 'PNG')
        
        file_info = UploadFileInfo(
            filename="test.png",
            content_type="image/png",
            size=len(content),
            content=content
        )
        
        image = file_info.to_image()
        
        assert image.size == (original_width, original_height)
        assert image.mode == 'RGB'


class TestUploadFileInfoInvalidCases:
    """Test error handling for invalid inputs."""
    
    def test_to_image_with_invalid_content_raises_exception(self):
        """Test that invalid image content raises appropriate exception."""
        file_info = UploadFileInfo(
            filename="test.png",
            content_type="image/png",
            size=10,
            content=b"not an image"
        )
        
        with pytest.raises(Exception):  # PIL will raise various exceptions
            file_info.to_image()
    
    def test_to_image_with_empty_content_raises_exception(self):
        """Test that empty content raises appropriate exception."""
        file_info = UploadFileInfo(
            filename="test.png",
            content_type="image/png",
            size=0,
            content=b""
        )
        
        with pytest.raises(Exception):  # PIL will raise exception
            file_info.to_image()


class TestUploadFileInfoMetadata:
    """Test metadata fields of UploadFileInfo."""
    
    def test_size_matches_content_length(self, create_test_image_bytes):
        """Test that size field matches actual content length."""
        content = create_test_image_bytes(100, 100, 'RGB', 'PNG')
        
        file_info = UploadFileInfo(
            filename="test.png",
            content_type="image/png",
            size=len(content),
            content=content
        )
        
        assert file_info.size == len(file_info.content)
    
    def test_filename_preserved(self, create_test_image_bytes):
        """Test that filename is preserved correctly."""
        content = create_test_image_bytes(100, 100, 'RGB', 'PNG')
        filename = "my_test_image.png"
        
        file_info = UploadFileInfo(
            filename=filename,
            content_type="image/png",
            size=len(content),
            content=content
        )
        
        assert file_info.filename == filename
    
    def test_content_type_preserved(self, create_test_image_bytes):
        """Test that content_type is preserved correctly."""
        content = create_test_image_bytes(100, 100, 'RGB', 'PNG')
        content_type = "image/png"
        
        file_info = UploadFileInfo(
            filename="test.png",
            content_type=content_type,
            size=len(content),
            content=content
        )
        
        assert file_info.content_type == content_type
