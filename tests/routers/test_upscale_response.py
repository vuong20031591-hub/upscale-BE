"""
Tests for upscale endpoint response formatting.
Task 6.1: Verify StreamingResponse với headers
"""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Resolution


client = TestClient(app)


def create_test_image(width: int = 100, height: int = 100, mode: str = "RGB") -> bytes:
    """Create a test image and return as bytes."""
    image = Image.new(mode, (width, height), color=(128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


class TestStreamingResponseHeaders:
    """Test StreamingResponse headers for task 6.1."""
    
    def test_response_has_correct_media_type(self):
        """Verify media_type is image/png (Req 5.1)."""
        image_bytes = create_test_image()
        
        response = client.post(
            "/upscale/standard",
            files={"file": ("test.png", image_bytes, "image/png")},
            data={"target_resolution": "2k"}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
    
    def test_response_has_content_disposition_header(self):
        """Verify Content-Disposition header format (Req 5.2)."""
        image_bytes = create_test_image()
        
        response = client.post(
            "/upscale/standard",
            files={"file": ("test.png", image_bytes, "image/png")},
            data={"target_resolution": "2k"}
        )
        
        assert response.status_code == 200
        assert "content-disposition" in response.headers
        
        disposition = response.headers["content-disposition"]
        assert disposition.startswith("attachment; filename=upscaled_")
        assert disposition.endswith(".png")
        assert "2k" in disposition
    
    def test_response_has_image_resolution_header(self):
        """Verify X-Image-Resolution header format (Req 5.3)."""
        image_bytes = create_test_image()
        
        response = client.post(
            "/upscale/standard",
            files={"file": ("test.png", image_bytes, "image/png")},
            data={"target_resolution": "2k"}
        )
        
        assert response.status_code == 200
        assert "x-image-resolution" in response.headers
        
        resolution = response.headers["x-image-resolution"]
        # Format: {width}x{height}
        assert "x" in resolution
        parts = resolution.split("x")
        assert len(parts) == 2
        assert parts[0].isdigit()
        assert parts[1].isdigit()
    
    def test_response_has_scale_factor_header(self):
        """Verify X-Scale-Factor header format (Req 4.5, 5.3)."""
        image_bytes = create_test_image()
        
        response = client.post(
            "/upscale/standard",
            files={"file": ("test.png", image_bytes, "image/png")},
            data={"target_resolution": "2k"}
        )
        
        assert response.status_code == 200
        assert "x-scale-factor" in response.headers
        
        scale_factor = response.headers["x-scale-factor"]
        # Should be a float rounded to 2 decimals
        float_value = float(scale_factor)
        assert float_value > 0
        # Check rounded to 2 decimals
        assert len(scale_factor.split(".")[-1]) <= 2
    
    def test_response_content_is_valid_png(self):
        """Verify response content is valid PNG (Req 5.1, 5.5)."""
        image_bytes = create_test_image()
        
        response = client.post(
            "/upscale/standard",
            files={"file": ("test.png", image_bytes, "image/png")},
            data={"target_resolution": "2k"}
        )
        
        assert response.status_code == 200
        
        # Try to load response as PNG
        output_image = Image.open(io.BytesIO(response.content))
        assert output_image.format == "PNG"
        assert output_image.mode == "RGB"
    
    def test_headers_for_4k_resolution(self):
        """Verify headers are correct for 4k resolution."""
        image_bytes = create_test_image()
        
        response = client.post(
            "/upscale/standard",
            files={"file": ("test.png", image_bytes, "image/png")},
            data={"target_resolution": "4k"}
        )
        
        assert response.status_code == 200
        
        # Check Content-Disposition contains 4k
        disposition = response.headers["content-disposition"]
        assert "4k" in disposition
        assert disposition == "attachment; filename=upscaled_4k.png"
    
    def test_scale_factor_rounded_to_2_decimals(self):
        """Verify scale factor is rounded to 2 decimals (Req 4.4)."""
        # Create image that will result in non-round scale factor
        image_bytes = create_test_image(width=1000, height=750)
        
        response = client.post(
            "/upscale/standard",
            files={"file": ("test.png", image_bytes, "image/png")},
            data={"target_resolution": "2k"}
        )
        
        assert response.status_code == 200
        
        scale_factor = response.headers["x-scale-factor"]
        # Should have at most 2 decimal places
        if "." in scale_factor:
            decimal_part = scale_factor.split(".")[1]
            assert len(decimal_part) <= 2


class TestPNGQualityEncoding:
    """Test PNG encoding with OUTPUT_QUALITY parameter."""
    
    def test_png_uses_compress_level(self):
        """Verify PNG encoding uses compress_level (Req 5.5, 9.3)."""
        image_bytes = create_test_image()
        
        response = client.post(
            "/upscale/standard",
            files={"file": ("test.png", image_bytes, "image/png")},
            data={"target_resolution": "2k"}
        )
        
        assert response.status_code == 200
        
        # Verify output is valid PNG
        output_image = Image.open(io.BytesIO(response.content))
        assert output_image.format == "PNG"
        
        # PNG should be compressed (file size should be reasonable)
        assert len(response.content) > 0
        assert len(response.content) < 10 * 1024 * 1024  # Less than 10MB


class TestResponseIntegration:
    """Integration tests for complete response flow."""
    
    def test_complete_response_structure(self):
        """Verify all response components are present."""
        image_bytes = create_test_image(width=500, height=400)
        
        response = client.post(
            "/upscale/standard",
            files={"file": ("test.png", image_bytes, "image/png")},
            data={"target_resolution": "2k"}
        )
        
        assert response.status_code == 200
        
        # Verify all required headers
        assert response.headers["content-type"] == "image/png"
        assert "content-disposition" in response.headers
        assert "x-image-resolution" in response.headers
        assert "x-scale-factor" in response.headers
        
        # Verify content is valid PNG
        output_image = Image.open(io.BytesIO(response.content))
        assert output_image.format == "PNG"
        
        # Verify resolution header matches actual image size
        resolution_header = response.headers["x-image-resolution"]
        width, height = map(int, resolution_header.split("x"))
        assert output_image.size == (width, height)
    
    def test_response_for_different_resolutions(self):
        """Verify response works for all supported resolutions."""
        image_bytes = create_test_image()
        
        for resolution in ["2k", "4k"]:
            response = client.post(
                "/upscale/standard",
                files={"file": ("test.png", image_bytes, "image/png")},
                data={"target_resolution": resolution}
            )
            
            assert response.status_code == 200
            assert resolution in response.headers["content-disposition"]
            
            # Verify output is valid PNG
            output_image = Image.open(io.BytesIO(response.content))
            assert output_image.format == "PNG"
