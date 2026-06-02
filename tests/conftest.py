"""
Pytest configuration and shared fixtures for AI Image Upscaling tests.
"""

import io
import os
from pathlib import Path
from typing import BinaryIO
from unittest.mock import Mock, MagicMock

import pytest
from PIL import Image
from fastapi.testclient import TestClient

# Set high rate limit for tests to avoid rate limiting
os.environ['RATE_LIMIT_PER_MINUTE'] = '1000'

from app.main import app
from app.services.model_manager import ModelManager
from app.models.image import UploadFileInfo


# ============================================================================
# Test Client Fixtures
# ============================================================================

@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


# ============================================================================
# Image Generation Fixtures
# ============================================================================

@pytest.fixture
def create_test_image():
    """
    Factory fixture to create test images with specified dimensions and mode.
    
    Usage:
        image = create_test_image(width=100, height=100, mode='RGB')
    """
    def _create(width: int = 100, height: int = 100, mode: str = 'RGB', color: tuple = None) -> Image.Image:
        """
        Create a test PIL Image.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            mode: PIL image mode (RGB, RGBA, L, CMYK, etc.)
            color: Color tuple (default: red for RGB, white for L)
        
        Returns:
            PIL Image object
        """
        if color is None:
            if mode == 'RGB':
                color = (255, 0, 0)  # Red
            elif mode == 'RGBA':
                color = (255, 0, 0, 255)  # Red with full opacity
            elif mode == 'L':
                color = 255  # White
            elif mode == 'CMYK':
                color = (0, 100, 100, 0)  # Red in CMYK
            else:
                color = (255, 0, 0)  # Default red
        
        return Image.new(mode, (width, height), color)
    
    return _create


@pytest.fixture
def create_test_image_bytes():
    """
    Factory fixture to create test image bytes for upload simulation.
    
    Usage:
        image_bytes = create_test_image_bytes(width=100, height=100, format='PNG')
    """
    def _create(
        width: int = 100,
        height: int = 100,
        mode: str = 'RGB',
        format: str = 'PNG',
        color: tuple = None
    ) -> bytes:
        """
        Create test image as bytes.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            mode: PIL image mode
            format: Output format (PNG, JPEG)
            color: Color tuple
        
        Returns:
            Image bytes
        """
        if color is None:
            color = (255, 0, 0) if mode == 'RGB' else 255
        
        image = Image.new(mode, (width, height), color)
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        return buffer.read()
    
    return _create


@pytest.fixture
def sample_image_small(create_test_image):
    """Small test image (100x100 RGB)."""
    return create_test_image(100, 100, 'RGB')


@pytest.fixture
def sample_image_medium(create_test_image):
    """Medium test image (1920x1080 RGB)."""
    return create_test_image(1920, 1080, 'RGB')


@pytest.fixture
def sample_image_large(create_test_image):
    """Large test image (3840x2160 RGB) - 4K resolution."""
    return create_test_image(3840, 2160, 'RGB')


@pytest.fixture
def sample_image_rgba(create_test_image):
    """Test image with alpha channel (RGBA mode)."""
    return create_test_image(100, 100, 'RGBA')


@pytest.fixture
def sample_image_grayscale(create_test_image):
    """Grayscale test image (L mode)."""
    return create_test_image(100, 100, 'L')


@pytest.fixture
def sample_image_cmyk(create_test_image):
    """CMYK test image."""
    return create_test_image(100, 100, 'CMYK')


# ============================================================================
# UploadFileInfo Fixtures
# ============================================================================

@pytest.fixture
def create_upload_file_info(create_test_image_bytes):
    """
    Factory fixture to create UploadFileInfo objects.
    
    Usage:
        file_info = create_upload_file_info(
            filename='test.jpg',
            content_type='image/jpeg',
            width=100,
            height=100
        )
    """
    def _create(
        filename: str = 'test.png',
        content_type: str = 'image/png',
        width: int = 100,
        height: int = 100,
        mode: str = 'RGB',
        format: str = 'PNG'
    ) -> UploadFileInfo:
        """
        Create UploadFileInfo for testing.
        
        Args:
            filename: File name
            content_type: MIME type
            width: Image width
            height: Image height
            mode: PIL image mode
            format: Image format
        
        Returns:
            UploadFileInfo object
        """
        content = create_test_image_bytes(width, height, mode, format)
        return UploadFileInfo(
            filename=filename,
            content_type=content_type,
            size=len(content),
            content=content
        )
    
    return _create


@pytest.fixture
def valid_upload_file_info(create_upload_file_info):
    """Valid UploadFileInfo for testing (100x100 PNG)."""
    return create_upload_file_info(
        filename='test.png',
        content_type='image/png',
        width=100,
        height=100
    )


# ============================================================================
# ModelManager Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_model_manager(monkeypatch):
    """
    Mock ModelManager to avoid loading real AI model.
    
    This fixture mocks the upscale method to return a 4x scaled image
    without actually loading the Real-ESRGAN model.
    """
    def mock_upscale(self, image: Image.Image, outscale: int = 4) -> Image.Image:
        """Mock upscale that simply resizes by outscale factor."""
        new_size = (image.width * outscale, image.height * outscale)
        return image.resize(new_size, Image.Resampling.LANCZOS)
    
    def mock_load(self):
        """Mock load that does nothing."""
        pass
    
    # Mock the methods
    monkeypatch.setattr(ModelManager, 'upscale', mock_upscale)
    monkeypatch.setattr(ModelManager, 'load', mock_load)
    monkeypatch.setattr(ModelManager, 'is_loaded', property(lambda self: True))
    
    return ModelManager()


@pytest.fixture
def mock_model_manager_not_loaded(monkeypatch):
    """
    Mock ModelManager in not-loaded state.
    
    Useful for testing lazy loading behavior.
    """
    def mock_load(self):
        """Mock load that sets is_loaded to True."""
        self._model = MagicMock()
    
    monkeypatch.setattr(ModelManager, 'load', mock_load)
    monkeypatch.setattr(ModelManager, '_model', None)
    
    return ModelManager()


@pytest.fixture
def mock_model_download_failure(monkeypatch):
    """
    Mock ModelManager with download failure.
    
    Useful for testing error handling when model cannot be downloaded.
    """
    def mock_download_if_needed(self):
        """Mock download that raises exception."""
        from app.core.exceptions import ModelNotFoundError
        raise ModelNotFoundError("Failed to download model from GitHub")
    
    monkeypatch.setattr(ModelManager, '_download_if_needed', mock_download_if_needed)
    
    return ModelManager()


# ============================================================================
# File Size Fixtures
# ============================================================================

@pytest.fixture
def create_file_with_size():
    """
    Factory fixture to create files with specific byte sizes.
    
    Usage:
        file_bytes = create_file_with_size(size_bytes=1024)
    """
    def _create(size_bytes: int) -> bytes:
        """
        Create a byte array of specified size.
        
        Args:
            size_bytes: Size in bytes
        
        Returns:
            Bytes of specified size
        """
        return b'x' * size_bytes
    
    return _create


@pytest.fixture
def file_exactly_10mb(create_file_with_size):
    """File exactly 10 MiB (10,485,760 bytes)."""
    return create_file_with_size(10 * 1024 * 1024)


@pytest.fixture
def file_over_10mb(create_file_with_size):
    """File over 10 MiB (10,485,761 bytes)."""
    return create_file_with_size(10 * 1024 * 1024 + 1)


@pytest.fixture
def file_under_10mb(create_file_with_size):
    """File under 10 MiB (5 MiB)."""
    return create_file_with_size(5 * 1024 * 1024)


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_model_manager_singleton():
    """
    Reset ModelManager singleton between tests.
    
    This ensures each test starts with a fresh ModelManager instance.
    """
    yield
    # Reset singleton instance after each test
    ModelManager._instance = None
    ModelManager._model = None
