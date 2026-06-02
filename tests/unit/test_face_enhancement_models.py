"""
Unit tests for face enhancement data models.

Tests the dataclasses defined in app.models.face_enhancement to ensure
they correctly represent request/response data structures.

Requirements:
    - Requirement 4.1: API endpoint data structures
    - Requirement 4.5: Response metadata headers
"""

import pytest
from PIL import Image
from fastapi import UploadFile
from io import BytesIO

from app.models.face_enhancement import (
    FaceEnhancementRequest,
    ValidatedFaceEnhancementRequest,
    FaceEnhancementResult
)
from app.services.codeformer_manager import CodeFormerMode


class TestFaceEnhancementRequest:
    """Test FaceEnhancementRequest dataclass."""
    
    def test_create_request_with_all_parameters(self):
        """Test creating request with all parameters."""
        # Create mock UploadFile with BytesIO
        file_content = BytesIO(b"fake image content")
        mock_file = UploadFile(filename="test.jpg", file=file_content)
        
        request = FaceEnhancementRequest(
            file=mock_file,
            mode="restoration",
            weight=0.7,
            face_upsample=True
        )
        
        assert request.file == mock_file
        assert request.mode == "restoration"
        assert request.weight == 0.7
        assert request.face_upsample is True
    
    def test_create_request_with_optional_parameters_none(self):
        """Test creating request with optional parameters as None."""
        file_content = BytesIO(b"fake image content")
        mock_file = UploadFile(filename="test.jpg", file=file_content)
        
        request = FaceEnhancementRequest(
            file=mock_file,
            mode="colorization"
        )
        
        assert request.file == mock_file
        assert request.mode == "colorization"
        assert request.weight is None
        assert request.face_upsample is None
    
    def test_validate_method_raises_not_implemented(self):
        """Test that validate() method raises NotImplementedError."""
        file_content = BytesIO(b"fake image content")
        mock_file = UploadFile(filename="test.jpg", file=file_content)
        request = FaceEnhancementRequest(
            file=mock_file,
            mode="restoration"
        )
        
        with pytest.raises(NotImplementedError) as exc_info:
            request.validate()
        
        assert "FaceEnhancementValidator" in str(exc_info.value)


class TestValidatedFaceEnhancementRequest:
    """Test ValidatedFaceEnhancementRequest dataclass."""
    
    def test_create_validated_request(self):
        """Test creating validated request with all fields."""
        # Create a simple test image
        image = Image.new('RGB', (100, 100), color='red')
        
        validated = ValidatedFaceEnhancementRequest(
            image=image,
            mode=CodeFormerMode.RESTORATION,
            weight=0.7,
            face_upsample=True
        )
        
        assert validated.image == image
        assert validated.mode == CodeFormerMode.RESTORATION
        assert validated.weight == 0.7
        assert validated.face_upsample is True
    
    def test_validated_request_requires_all_fields(self):
        """Test that all fields are required (no defaults)."""
        image = Image.new('RGB', (100, 100), color='red')
        
        # Should work with all fields
        validated = ValidatedFaceEnhancementRequest(
            image=image,
            mode=CodeFormerMode.COLORIZATION,
            weight=0.0,
            face_upsample=False
        )
        
        assert validated.weight == 0.0  # Even 0 is valid
        assert validated.face_upsample is False  # Even False is valid


class TestFaceEnhancementResult:
    """Test FaceEnhancementResult dataclass."""
    
    def test_create_result_without_warning(self):
        """Test creating result without warning."""
        image = Image.new('RGB', (100, 100), color='blue')
        
        result = FaceEnhancementResult(
            image=image,
            faces_detected=2,
            processing_time=4.523456,
            mode_used=CodeFormerMode.RESTORATION,
            weight_used=0.7,
            background_enhanced=True,
            bg_upscale=2
        )
        
        assert result.image == image
        assert result.faces_detected == 2
        assert result.processing_time == 4.523456
        assert result.mode_used == CodeFormerMode.RESTORATION
        assert result.weight_used == 0.7
        assert result.background_enhanced is True
        assert result.bg_upscale == 2
        assert result.warning is None
    
    def test_create_result_with_warning(self):
        """Test creating result with warning message."""
        image = Image.new('RGB', (100, 100), color='blue')
        
        result = FaceEnhancementResult(
            image=image,
            faces_detected=0,
            processing_time=1.234,
            mode_used=CodeFormerMode.RESTORATION,
            weight_used=0.7,
            background_enhanced=False,
            bg_upscale=1,
            warning="No faces detected"
        )
        
        assert result.faces_detected == 0
        assert result.background_enhanced is False
        assert result.bg_upscale == 1
        assert result.warning == "No faces detected"
    
    def test_to_response_headers_without_warning(self):
        """Test converting result to response headers without warning."""
        image = Image.new('RGB', (100, 100), color='blue')
        
        result = FaceEnhancementResult(
            image=image,
            faces_detected=2,
            processing_time=4.523456,
            mode_used=CodeFormerMode.RESTORATION,
            weight_used=0.7,
            background_enhanced=True,
            bg_upscale=2
        )
        
        headers = result.to_response_headers()
        
        assert headers["X-Faces-Detected"] == "2"
        assert headers["X-Processing-Time"] == "4.523"  # 3 decimal places
        assert headers["X-Mode-Used"] == "restoration"
        assert headers["X-Weight-Used"] == "0.70"  # 2 decimal places
        assert headers["X-Background-Enhanced"] == "True"
        assert headers["X-BG-Upscale"] == "2"
        assert "X-Warning" not in headers
    
    def test_to_response_headers_with_warning(self):
        """Test converting result to response headers with warning."""
        image = Image.new('RGB', (100, 100), color='blue')
        
        result = FaceEnhancementResult(
            image=image,
            faces_detected=0,
            processing_time=1.234567,
            mode_used=CodeFormerMode.COLORIZATION,
            weight_used=0.0,
            background_enhanced=False,
            bg_upscale=1,
            warning="No faces detected in image"
        )
        
        headers = result.to_response_headers()
        
        assert headers["X-Faces-Detected"] == "0"
        assert headers["X-Processing-Time"] == "1.235"  # Rounded to 3 decimals
        assert headers["X-Mode-Used"] == "colorization"
        assert headers["X-Weight-Used"] == "0.00"
        assert headers["X-Background-Enhanced"] == "False"
        assert headers["X-BG-Upscale"] == "1"
        assert headers["X-Warning"] == "No faces detected in image"
    
    def test_to_response_headers_formatting(self):
        """Test that headers are formatted correctly for all modes."""
        image = Image.new('RGB', (100, 100), color='blue')
        
        # Test restoration mode
        result_restoration = FaceEnhancementResult(
            image=image,
            faces_detected=5,
            processing_time=10.999999,
            mode_used=CodeFormerMode.RESTORATION,
            weight_used=0.75,
            background_enhanced=True,
            bg_upscale=4
        )
        headers = result_restoration.to_response_headers()
        assert headers["X-Mode-Used"] == "restoration"
        assert headers["X-Weight-Used"] == "0.75"
        assert headers["X-Processing-Time"] == "11.000"  # Rounded up
        assert headers["X-Background-Enhanced"] == "True"
        assert headers["X-BG-Upscale"] == "4"
        
        # Test colorization mode
        result_colorization = FaceEnhancementResult(
            image=image,
            faces_detected=1,
            processing_time=3.1415,
            mode_used=CodeFormerMode.COLORIZATION,
            weight_used=0.0,
            background_enhanced=True,
            bg_upscale=2
        )
        headers = result_colorization.to_response_headers()
        assert headers["X-Mode-Used"] == "colorization"
        assert headers["X-Weight-Used"] == "0.00"
        assert headers["X-Background-Enhanced"] == "True"
        assert headers["X-BG-Upscale"] == "2"
        
        # Test inpainting mode
        result_inpainting = FaceEnhancementResult(
            image=image,
            faces_detected=3,
            processing_time=7.5,
            mode_used=CodeFormerMode.INPAINTING,
            weight_used=1.0,
            background_enhanced=False,
            bg_upscale=1
        )
        headers = result_inpainting.to_response_headers()
        assert headers["X-Mode-Used"] == "inpainting"
        assert headers["X-Weight-Used"] == "1.00"
        assert headers["X-Background-Enhanced"] == "False"
        assert headers["X-BG-Upscale"] == "1"
    
    def test_to_response_headers_all_values_are_strings(self):
        """Test that all header values are strings."""
        image = Image.new('RGB', (100, 100), color='blue')
        
        result = FaceEnhancementResult(
            image=image,
            faces_detected=2,
            processing_time=4.5,
            mode_used=CodeFormerMode.RESTORATION,
            weight_used=0.7,
            background_enhanced=True,
            bg_upscale=2,
            warning="Test warning"
        )
        
        headers = result.to_response_headers()
        
        # All values must be strings for HTTP headers
        for key, value in headers.items():
            assert isinstance(value, str), f"Header {key} value is not a string: {type(value)}"
    
    def test_background_enhancement_fields(self):
        """Test background enhancement fields are properly stored and returned."""
        image = Image.new('RGB', (100, 100), color='blue')
        
        # Test with background enhancement enabled
        result_enhanced = FaceEnhancementResult(
            image=image,
            faces_detected=1,
            processing_time=5.0,
            mode_used=CodeFormerMode.RESTORATION,
            weight_used=0.7,
            background_enhanced=True,
            bg_upscale=4
        )
        
        assert result_enhanced.background_enhanced is True
        assert result_enhanced.bg_upscale == 4
        
        headers = result_enhanced.to_response_headers()
        assert headers["X-Background-Enhanced"] == "True"
        assert headers["X-BG-Upscale"] == "4"
        
        # Test with background enhancement disabled
        result_no_enhance = FaceEnhancementResult(
            image=image,
            faces_detected=1,
            processing_time=3.0,
            mode_used=CodeFormerMode.COLORIZATION,
            weight_used=0.0,
            background_enhanced=False,
            bg_upscale=1
        )
        
        assert result_no_enhance.background_enhanced is False
        assert result_no_enhance.bg_upscale == 1
        
        headers = result_no_enhance.to_response_headers()
        assert headers["X-Background-Enhanced"] == "False"
        assert headers["X-BG-Upscale"] == "1"
    
    def test_background_upscale_values(self):
        """Test different bg_upscale values (1, 2, 4)."""
        image = Image.new('RGB', (100, 100), color='blue')
        
        for upscale_value in [1, 2, 4]:
            result = FaceEnhancementResult(
                image=image,
                faces_detected=1,
                processing_time=3.0,
                mode_used=CodeFormerMode.RESTORATION,
                weight_used=0.7,
                background_enhanced=(upscale_value != 1),
                bg_upscale=upscale_value
            )
            
            assert result.bg_upscale == upscale_value
            headers = result.to_response_headers()
            assert headers["X-BG-Upscale"] == str(upscale_value)
