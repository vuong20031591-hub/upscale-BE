"""
Unit tests for AI upscaling workflow (Task 6.1).

This module tests the complete AI upscaling workflow:
- Validation
- Model loading (lazy)
- AI upscaling (4x)
- Resize to target resolution
- Output type verification

Requirements: 3.1, 3.3
"""

import pytest
from unittest.mock import MagicMock
import numpy as np
from PIL import Image

from app.services.image_processor import ImageProcessor
from app.services.model_manager import ModelManager
from app.models import Resolution, ImageFormat, UploadFileInfo
from app.core.exceptions import ValidationError, ImageProcessingError


class TestAIUpscalingWorkflow:
    """Tests for complete AI upscaling workflow."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = ImageProcessor()
    
    def test_workflow_validate_load_upscale_resize(
        self,
        monkeypatch,
        create_upload_file_info
    ):
        """
        Test complete workflow: validate → load → upscale → resize.
        
        This test verifies the entire AI upscaling pipeline:
        1. Validate file (size, type, extension)
        2. Load model (lazy loading)
        3. Upscale using AI (4x)
        4. Resize to target resolution
        
        Requirements: 3.1, 3.3
        """
        # Create valid upload file
        file_info = create_upload_file_info(
            filename='test.jpg',
            content_type='image/jpeg',
            width=640,
            height=480
        )
        
        # Mock ModelManager to avoid loading real model
        def mock_download_if_needed(self):
            """Mock download - do nothing."""
            pass
        
        def mock_load_model(self):
            """Mock load - set _model with enhance method."""
            mock_model = MagicMock()
            
            def mock_enhance(img_array, outscale=4):
                """Mock enhance that returns 4x upscaled array."""
                h, w = img_array.shape[:2]
                new_h, new_w = h * outscale, w * outscale
                # Return upscaled array with correct shape
                upscaled = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                # Fill with some data to make it realistic
                upscaled[:, :] = [128, 128, 128]
                return upscaled, None
            
            mock_model.enhance = mock_enhance
            self._model = mock_model
        
        # Apply mocks
        monkeypatch.setattr(ModelManager, '_download_if_needed', mock_download_if_needed)
        monkeypatch.setattr(ModelManager, '_load_model', mock_load_model)
        
        # Process with AI upscaling
        result = self.processor.process(file_info, Resolution.K2, use_ai=True)
        
        # Verify workflow completed successfully
        assert result is not None
        
        # Verify original dimensions preserved in metadata
        assert result.original_width == 640
        assert result.original_height == 480
        
        # Verify output dimensions (should be resized to fit 2K)
        assert result.final_width > 0
        assert result.final_height > 0
        assert result.final_width <= 2560  # 2K width
        assert result.final_height <= 1440  # 2K height
        
        # Verify scale factor calculated
        assert result.scale_factor > 0
        
        # Verify format is PNG
        assert result.format == ImageFormat.PNG
    
    def test_output_type_is_pil_image(
        self,
        monkeypatch,
        create_upload_file_info
    ):
        """
        Test that AI upscaling output is PIL Image.
        
        Verifies that after AI upscaling, the output is a PIL Image object,
        not numpy array or other types.
        
        Requirements: 3.3
        """
        # Create valid upload file
        file_info = create_upload_file_info(
            filename='test.png',
            content_type='image/png',
            width=100,
            height=100
        )
        
        # Mock ModelManager
        def mock_download_if_needed(self):
            pass
        
        def mock_load_model(self):
            mock_model = MagicMock()
            
            def mock_enhance(img_array, outscale=4):
                h, w = img_array.shape[:2]
                new_h, new_w = h * outscale, w * outscale
                upscaled = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                upscaled[:, :] = [255, 0, 0]  # Red
                return upscaled, None
            
            mock_model.enhance = mock_enhance
            self._model = mock_model
        
        monkeypatch.setattr(ModelManager, '_download_if_needed', mock_download_if_needed)
        monkeypatch.setattr(ModelManager, '_load_model', mock_load_model)
        
        # Process with AI
        result = self.processor.process(file_info, Resolution.K2, use_ai=True)
        
        # Verify output image is PIL Image
        assert isinstance(result.image, Image.Image)
        assert result.image.mode == 'RGB'
        
        # Verify it's not numpy array
        assert not isinstance(result.image, np.ndarray)
    
    def test_validation_before_processing(
        self,
        create_upload_file_info
    ):
        """
        Test that validation occurs before processing.
        
        Verifies that invalid files are rejected before attempting
        to load model or upscale.
        
        Requirements: 3.1
        """
        # Create file that's too large (over 10 MiB)
        file_info = UploadFileInfo(
            filename='large.jpg',
            content_type='image/jpeg',
            size=11 * 1024 * 1024,  # 11 MiB
            content=b'x' * (11 * 1024 * 1024)
        )
        
        # Should raise ValidationError before attempting to process
        with pytest.raises(ValidationError) as exc_info:
            self.processor.process(file_info, Resolution.K2, use_ai=True)
        
        # Verify error message mentions file size
        assert 'size' in str(exc_info.value).lower() or 'large' in str(exc_info.value).lower()
    
    def test_ai_upscale_4x_then_resize(
        self,
        monkeypatch,
        create_test_image
    ):
        """
        Test that AI upscales 4x first, then resizes to target.
        
        Workflow:
        1. Input: 640x480
        2. After AI 4x: 2560x1920
        3. After resize to 2K: 2560x1440 (downscaled to fit)
        
        Requirements: 3.1
        """
        # Create test image
        source_image = create_test_image(640, 480, 'RGB')
        
        # Track upscale calls
        upscale_called = False
        upscale_outscale = None
        
        def mock_download_if_needed(self):
            pass
        
        def mock_load_model(self):
            mock_model = MagicMock()
            
            def mock_enhance(img_array, outscale=4):
                nonlocal upscale_called, upscale_outscale
                upscale_called = True
                upscale_outscale = outscale
                
                h, w = img_array.shape[:2]
                new_h, new_w = h * outscale, w * outscale
                upscaled = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                return upscaled, None
            
            mock_model.enhance = mock_enhance
            self._model = mock_model
        
        monkeypatch.setattr(ModelManager, '_download_if_needed', mock_download_if_needed)
        monkeypatch.setattr(ModelManager, '_load_model', mock_load_model)
        
        # Process
        result = self.processor.process_from_image(source_image, Resolution.K2, use_ai=True)
        
        # Verify AI upscale was called with outscale=4
        assert upscale_called is True
        assert upscale_outscale == 4
        
        # Verify final dimensions fit within 2K
        assert result.final_width <= 2560
        assert result.final_height <= 1440
    
    def test_lazy_loading_model_on_first_upscale(
        self,
        monkeypatch,
        create_test_image
    ):
        """
        Test that model is loaded lazily on first upscale.
        
        Verifies:
        1. Model is NOT loaded on ImageProcessor init
        2. Model IS loaded when upscale is called
        
        Requirements: 3.1
        """
        # Create fresh processor
        processor = ImageProcessor()
        
        # Model should not be loaded yet
        assert processor.model_manager.is_loaded is False
        
        # Track load calls
        load_called = False
        
        def mock_download_if_needed(self):
            pass
        
        def mock_load_model(self):
            nonlocal load_called
            load_called = True
            
            mock_model = MagicMock()
            
            def mock_enhance(img_array, outscale=4):
                h, w = img_array.shape[:2]
                new_h, new_w = h * outscale, w * outscale
                upscaled = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                return upscaled, None
            
            mock_model.enhance = mock_enhance
            self._model = mock_model
        
        monkeypatch.setattr(ModelManager, '_download_if_needed', mock_download_if_needed)
        monkeypatch.setattr(ModelManager, '_load_model', mock_load_model)
        
        # Create test image
        source_image = create_test_image(100, 100, 'RGB')
        
        # Process - this should trigger lazy loading
        result = processor.process_from_image(source_image, Resolution.K2, use_ai=True)
        
        # Verify model was loaded
        assert load_called is True
        assert processor.model_manager.is_loaded is True
    
    def test_aspect_ratio_preserved_after_workflow(
        self,
        monkeypatch,
        create_test_image
    ):
        """
        Test that aspect ratio is preserved through entire workflow.
        
        Requirements: 3.1, 3.3
        """
        # Create image with specific aspect ratio (16:9)
        source_image = create_test_image(1600, 900, 'RGB')
        original_ratio = 1600 / 900
        
        # Mock ModelManager
        def mock_download_if_needed(self):
            pass
        
        def mock_load_model(self):
            mock_model = MagicMock()
            
            def mock_enhance(img_array, outscale=4):
                h, w = img_array.shape[:2]
                new_h, new_w = h * outscale, w * outscale
                upscaled = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                return upscaled, None
            
            mock_model.enhance = mock_enhance
            self._model = mock_model
        
        monkeypatch.setattr(ModelManager, '_download_if_needed', mock_download_if_needed)
        monkeypatch.setattr(ModelManager, '_load_model', mock_load_model)
        
        # Process
        result = self.processor.process_from_image(source_image, Resolution.K2, use_ai=True)
        
        # Calculate output aspect ratio
        output_ratio = result.final_width / result.final_height
        
        # Verify aspect ratio preserved (within tolerance)
        assert abs(output_ratio - original_ratio) < 0.01
    
    def test_rgb_conversion_in_workflow(
        self,
        monkeypatch,
        create_test_image
    ):
        """
        Test that images are converted to RGB during workflow.
        
        Requirements: 3.3
        """
        # Create RGBA image
        source_image = create_test_image(100, 100, 'RGBA')
        assert source_image.mode == 'RGBA'
        
        # Mock ModelManager
        def mock_download_if_needed(self):
            pass
        
        def mock_load_model(self):
            mock_model = MagicMock()
            
            def mock_enhance(img_array, outscale=4):
                h, w = img_array.shape[:2]
                new_h, new_w = h * outscale, w * outscale
                upscaled = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                return upscaled, None
            
            mock_model.enhance = mock_enhance
            self._model = mock_model
        
        monkeypatch.setattr(ModelManager, '_download_if_needed', mock_download_if_needed)
        monkeypatch.setattr(ModelManager, '_load_model', mock_load_model)
        
        # Process
        result = self.processor.process_from_image(source_image, Resolution.K2, use_ai=True)
        
        # Verify output is RGB
        assert result.image.mode == 'RGB'


class TestAIUpscalingErrorHandling:
    """Tests for error handling in AI upscaling workflow."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = ImageProcessor()
    
    def test_invalid_file_type_rejected(self):
        """
        Test that invalid file types are rejected.
        
        Requirements: 3.1
        """
        # Create file with invalid content type
        file_info = UploadFileInfo(
            filename='test.gif',
            content_type='image/gif',
            size=1024,
            content=b'GIF89a...'
        )
        
        # Should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            self.processor.process(file_info, Resolution.K2, use_ai=True)
        
        # Verify error message mentions content type
        assert 'type' in str(exc_info.value).lower() or 'content' in str(exc_info.value).lower()
    
    def test_invalid_extension_rejected(self):
        """
        Test that invalid file extensions are rejected.
        
        Requirements: 3.1
        """
        # Create file with invalid extension
        file_info = UploadFileInfo(
            filename='test.bmp',
            content_type='image/png',  # Valid content type
            size=1024,
            content=b'x' * 1024
        )
        
        # Should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            self.processor.process(file_info, Resolution.K2, use_ai=True)
        
        # Verify error message mentions extension
        assert 'extension' in str(exc_info.value).lower()
    
    def test_model_inference_failure_handling(
        self,
        monkeypatch,
        create_test_image
    ):
        """
        Test error handling when model inference fails.
        
        Requirements: 3.3
        """
        # Create test image
        source_image = create_test_image(100, 100, 'RGB')
        
        # Mock ModelManager with failing inference
        def mock_download_if_needed(self):
            pass
        
        def mock_load_model(self):
            mock_model = MagicMock()
            
            def mock_enhance(img_array, outscale=4):
                # Simulate inference failure
                raise RuntimeError("CUDA out of memory")
            
            mock_model.enhance = mock_enhance
            self._model = mock_model
        
        monkeypatch.setattr(ModelManager, '_download_if_needed', mock_download_if_needed)
        monkeypatch.setattr(ModelManager, '_load_model', mock_load_model)
        
        # Should raise ImageProcessingError
        with pytest.raises(ImageProcessingError) as exc_info:
            self.processor.process_from_image(source_image, Resolution.K2, use_ai=True)
        
        # Verify error message mentions AI upscaling
        assert 'upscaling' in str(exc_info.value).lower() or 'ai' in str(exc_info.value).lower()


class TestAIUpscalingMetadata:
    """Tests for metadata in AI upscaling results."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = ImageProcessor()
    
    def test_metadata_includes_all_fields(
        self,
        monkeypatch,
        create_test_image
    ):
        """
        Test that result includes all required metadata fields.
        
        Requirements: 3.3
        """
        # Create test image
        source_image = create_test_image(640, 480, 'RGB')
        
        # Mock ModelManager
        def mock_download_if_needed(self):
            pass
        
        def mock_load_model(self):
            mock_model = MagicMock()
            
            def mock_enhance(img_array, outscale=4):
                h, w = img_array.shape[:2]
                new_h, new_w = h * outscale, w * outscale
                upscaled = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                return upscaled, None
            
            mock_model.enhance = mock_enhance
            self._model = mock_model
        
        monkeypatch.setattr(ModelManager, '_download_if_needed', mock_download_if_needed)
        monkeypatch.setattr(ModelManager, '_load_model', mock_load_model)
        
        # Process
        result = self.processor.process_from_image(source_image, Resolution.K2, use_ai=True)
        
        # Verify all metadata fields present
        assert hasattr(result, 'image')
        assert hasattr(result, 'original_width')
        assert hasattr(result, 'original_height')
        assert hasattr(result, 'final_width')
        assert hasattr(result, 'final_height')
        assert hasattr(result, 'scale_factor')
        assert hasattr(result, 'format')
        
        # Verify metadata values are correct
        assert result.original_width == 640
        assert result.original_height == 480
        assert result.final_width > 0
        assert result.final_height > 0
        assert result.scale_factor > 0
        assert result.format == ImageFormat.PNG
    
    def test_scale_factor_calculation_correct(
        self,
        monkeypatch,
        create_test_image
    ):
        """
        Test that scale factor is calculated correctly.
        
        Formula: (final_width/original_width + final_height/original_height) / 2
        
        Requirements: 3.3
        """
        # Create test image
        source_image = create_test_image(100, 100, 'RGB')
        
        # Mock ModelManager
        def mock_download_if_needed(self):
            pass
        
        def mock_load_model(self):
            mock_model = MagicMock()
            
            def mock_enhance(img_array, outscale=4):
                h, w = img_array.shape[:2]
                new_h, new_w = h * outscale, w * outscale
                upscaled = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                return upscaled, None
            
            mock_model.enhance = mock_enhance
            self._model = mock_model
        
        monkeypatch.setattr(ModelManager, '_download_if_needed', mock_download_if_needed)
        monkeypatch.setattr(ModelManager, '_load_model', mock_load_model)
        
        # Process
        result = self.processor.process_from_image(source_image, Resolution.K2, use_ai=True)
        
        # Calculate expected scale factor
        scale_w = result.final_width / result.original_width
        scale_h = result.final_height / result.original_height
        expected_scale = (scale_w + scale_h) / 2
        
        # Verify scale factor matches formula (within tolerance)
        assert abs(result.scale_factor - expected_scale) < 0.01
