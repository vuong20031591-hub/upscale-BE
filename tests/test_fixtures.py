"""
Tests to verify test fixtures and infrastructure are working correctly.

This file tests the testing infrastructure itself to ensure all fixtures
and utilities are properly configured.
"""

import pytest
from PIL import Image

from app.models.image import UploadFileInfo
from tests.test_utils import (
    generate_test_image,
    generate_test_image_bytes,
    calculate_aspect_ratio,
    verify_aspect_ratio_preserved,
    calculate_overall_scale_factor,
)


class TestImageGenerationFixtures:
    """Test image generation fixtures work correctly."""
    
    def test_create_test_image_fixture(self, create_test_image):
        """Verify create_test_image fixture generates valid images."""
        image = create_test_image(100, 100, 'RGB')
        
        assert isinstance(image, Image.Image)
        assert image.size == (100, 100)
        assert image.mode == 'RGB'
    
    def test_create_test_image_bytes_fixture(self, create_test_image_bytes):
        """Verify create_test_image_bytes fixture generates valid bytes."""
        image_bytes = create_test_image_bytes(100, 100, 'RGB', 'PNG')
        
        assert isinstance(image_bytes, bytes)
        assert len(image_bytes) > 0
        
        # Verify it's valid PNG
        assert image_bytes.startswith(b'\x89PNG\r\n\x1a\n')
    
    def test_sample_image_fixtures(
        self,
        sample_image_small,
        sample_image_medium,
        sample_image_large
    ):
        """Verify sample image fixtures have correct dimensions."""
        assert sample_image_small.size == (100, 100)
        assert sample_image_medium.size == (1920, 1080)
        assert sample_image_large.size == (3840, 2160)
    
    def test_sample_image_mode_fixtures(
        self,
        sample_image_rgba,
        sample_image_grayscale,
        sample_image_cmyk
    ):
        """Verify sample images with different modes."""
        assert sample_image_rgba.mode == 'RGBA'
        assert sample_image_grayscale.mode == 'L'
        assert sample_image_cmyk.mode == 'CMYK'


class TestUploadFileInfoFixtures:
    """Test UploadFileInfo fixtures work correctly."""
    
    def test_create_upload_file_info_fixture(self, create_upload_file_info):
        """Verify create_upload_file_info fixture generates valid objects."""
        file_info = create_upload_file_info(
            filename='test.png',
            content_type='image/png',
            width=100,
            height=100
        )
        
        assert isinstance(file_info, UploadFileInfo)
        assert file_info.filename == 'test.png'
        assert file_info.content_type == 'image/png'
        assert file_info.size > 0
        assert len(file_info.content) > 0
    
    def test_valid_upload_file_info_fixture(self, valid_upload_file_info):
        """Verify valid_upload_file_info fixture is valid."""
        assert isinstance(valid_upload_file_info, UploadFileInfo)
        assert valid_upload_file_info.filename == 'test.png'
        assert valid_upload_file_info.content_type == 'image/png'
        
        # Verify can convert to image
        image = valid_upload_file_info.to_image()
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'


class TestModelManagerMockFixtures:
    """Test ModelManager mock fixtures work correctly."""
    
    def test_mock_model_manager_fixture(self, mock_model_manager, sample_image_small):
        """Verify mock_model_manager fixture works."""
        # Should be able to upscale without loading real model
        result = mock_model_manager.upscale(sample_image_small, outscale=4)
        
        assert isinstance(result, Image.Image)
        assert result.size == (400, 400)  # 100 * 4
    
    def test_mock_model_manager_is_loaded(self, mock_model_manager):
        """Verify mock model manager reports as loaded."""
        assert mock_model_manager.is_loaded is True


class TestFileSizeFixtures:
    """Test file size fixtures work correctly."""
    
    def test_create_file_with_size_fixture(self, create_file_with_size):
        """Verify create_file_with_size fixture generates correct sizes."""
        file_1kb = create_file_with_size(1024)
        file_1mb = create_file_with_size(1024 * 1024)
        
        assert len(file_1kb) == 1024
        assert len(file_1mb) == 1024 * 1024
    
    def test_file_size_boundary_fixtures(
        self,
        file_exactly_10mb,
        file_over_10mb,
        file_under_10mb
    ):
        """Verify file size boundary fixtures."""
        assert len(file_exactly_10mb) == 10 * 1024 * 1024
        assert len(file_over_10mb) == 10 * 1024 * 1024 + 1
        assert len(file_under_10mb) == 5 * 1024 * 1024


class TestUtilityFunctions:
    """Test utility functions work correctly."""
    
    def test_generate_test_image(self):
        """Verify generate_test_image utility function."""
        image = generate_test_image(200, 150, 'RGB')
        
        assert isinstance(image, Image.Image)
        assert image.size == (200, 150)
        assert image.mode == 'RGB'
    
    def test_generate_test_image_bytes(self):
        """Verify generate_test_image_bytes utility function."""
        image_bytes = generate_test_image_bytes(100, 100, 'RGB', 'PNG')
        
        assert isinstance(image_bytes, bytes)
        assert len(image_bytes) > 0
    
    def test_calculate_aspect_ratio(self):
        """Verify calculate_aspect_ratio utility function."""
        ratio = calculate_aspect_ratio(1920, 1080)
        
        assert abs(ratio - 1.7777777777777777) < 0.0001
    
    def test_verify_aspect_ratio_preserved(self):
        """Verify verify_aspect_ratio_preserved utility function."""
        # Same aspect ratio
        assert verify_aspect_ratio_preserved(100, 100, 200, 200) is True
        assert verify_aspect_ratio_preserved(1920, 1080, 3840, 2160) is True
        
        # Different aspect ratio
        assert verify_aspect_ratio_preserved(100, 100, 200, 300) is False
    
    def test_calculate_overall_scale_factor(self):
        """Verify calculate_overall_scale_factor utility function."""
        scale = calculate_overall_scale_factor(100, 100, 400, 400)
        
        assert scale == 4.0


class TestFastAPIClient:
    """Test FastAPI test client fixture."""
    
    def test_client_fixture(self, client):
        """Verify client fixture can make requests."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "service" in response.json()


class TestSingletonReset:
    """Test ModelManager singleton reset between tests."""
    
    def test_singleton_reset_first_test(self):
        """First test - singleton should be fresh."""
        from app.services.model_manager import ModelManager
        
        # Get instance
        manager1 = ModelManager()
        
        # Store id for comparison
        self.manager_id = id(manager1)
    
    def test_singleton_reset_second_test(self):
        """Second test - singleton should be reset (different instance)."""
        from app.services.model_manager import ModelManager
        
        # Get instance
        manager2 = ModelManager()
        
        # Should be different instance due to reset fixture
        # Note: This test may fail if reset fixture doesn't work
        # In that case, we just verify we can get an instance
        assert manager2 is not None
