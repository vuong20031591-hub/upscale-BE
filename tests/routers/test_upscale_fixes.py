"""
Tests to verify critical fixes for standard upscaling endpoint.

These tests verify:
1. Resolution validation mismatch fix (no silent default to 2k)
2. Double image decoding fix (performance improvement)
3. Assert statements replaced with explicit checks
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.routers.upscale_basic import _get_resolution
from app.models import Resolution


client = TestClient(app)


def create_test_image(width: int = 100, height: int = 100, mode: str = "RGB") -> bytes:
    """Create a test image and return as bytes."""
    image = Image.new(mode, (width, height), color=(128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


class TestResolutionValidationFix:
    """
    Test fix for critical bug: Resolution validation mismatch.
    
    Previously, _get_resolution() would default to 2k for invalid resolutions,
    causing silent failures where valid config values (e.g., "6k") would be
    silently downgraded to 2k.
    
    Fix: _get_resolution() now raises ValueError for invalid resolutions.
    """
    
    def test_get_resolution_raises_on_invalid_resolution(self):
        """Verify _get_resolution raises ValueError instead of defaulting to 2k."""
        with pytest.raises(ValueError) as exc_info:
            _get_resolution("invalid")
        
        assert "Invalid resolution" in str(exc_info.value)
        assert "Valid values:" in str(exc_info.value)
    
    def test_get_resolution_raises_on_unsupported_resolution(self):
        """Verify _get_resolution raises ValueError for unsupported but valid-looking resolutions."""
        with pytest.raises(ValueError) as exc_info:
            _get_resolution("6k")
        
        assert "Invalid resolution '6k'" in str(exc_info.value)
    
    def test_get_resolution_accepts_valid_resolutions(self):
        """Verify _get_resolution accepts valid resolutions."""
        assert _get_resolution("2k") == Resolution.K2
        assert _get_resolution("4k") == Resolution.K4
        assert _get_resolution("8k") == Resolution.K8
    
    def test_get_resolution_case_insensitive(self):
        """Verify _get_resolution is case-insensitive."""
        assert _get_resolution("2K") == Resolution.K2
        assert _get_resolution("4K") == Resolution.K4
    
    def test_endpoint_rejects_invalid_resolution_with_400(self):
        """Verify endpoint returns 400 for invalid resolutions instead of silently defaulting."""
        image_bytes = create_test_image()
        
        response = client.post(
            "/upscale/standard",
            files={"file": ("test.png", image_bytes, "image/png")},
            data={"target_resolution": "invalid"}
        )
        
        assert response.status_code == 400
        detail = response.json()["detail"]
        # Should mention the invalid resolution
        assert "invalid" in detail.lower()
        assert ("not found in RESOLUTION_MAP" in detail or "Invalid resolution" in detail)
    
    def test_endpoint_rejects_unsupported_resolution_with_400(self):
        """Verify endpoint returns 400 for unsupported resolutions (e.g., 6k)."""
        image_bytes = create_test_image()
        
        response = client.post(
            "/upscale/standard",
            files={"file": ("test.png", image_bytes, "image/png")},
            data={"target_resolution": "6k"}
        )
        
        assert response.status_code == 400
        # Should fail at either get_dimensions or _get_resolution
        detail = response.json()["detail"]
        assert "6k" in detail.lower()
        assert ("not found in RESOLUTION_MAP" in detail or "Invalid resolution" in detail)
    
    def test_endpoint_validates_against_resolution_map(self):
        """Verify endpoint validates resolution against RESOLUTION_MAP, not just supported_resolutions."""
        image_bytes = create_test_image()
        
        # Test with valid RESOLUTION_MAP values
        for resolution in ["2k", "4k"]:
            response = client.post(
                "/upscale/standard",
                files={"file": ("test.png", image_bytes, "image/png")},
                data={"target_resolution": resolution}
            )
            assert response.status_code == 200, f"Failed for resolution {resolution}"


class TestDoubleDecodingFix:
    """
    Test fix for performance issue: Double image decoding.
    
    Previously, image was decoded twice:
    1. In endpoint: file_info.to_image() for validation
    2. In processor.process(): file_info.to_image() again
    
    Fix: Pass already-decoded image to _upscale_traditional() directly.
    
    Note: This is a behavioral test - we verify the endpoint still works correctly
    after the refactoring. Performance improvement is implicit.
    """
    
    def test_endpoint_processes_image_correctly_after_refactor(self):
        """Verify endpoint still works correctly after removing double decode."""
        image_bytes = create_test_image(width=500, height=400)
        
        response = client.post(
            "/upscale/standard",
            files={"file": ("test.png", image_bytes, "image/png")},
            data={"target_resolution": "2k"}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        
        # Verify output is valid PNG
        output_image = Image.open(io.BytesIO(response.content))
        assert output_image.format == "PNG"
    
    def test_endpoint_handles_different_color_modes(self):
        """Verify RGB conversion still works after refactoring."""
        # Create RGBA image
        image = Image.new("RGBA", (100, 100), color=(128, 128, 128, 255))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        response = client.post(
            "/upscale/standard",
            files={"file": ("test.png", image_bytes, "image/png")},
            data={"target_resolution": "2k"}
        )
        
        assert response.status_code == 200
        
        # Output should be RGB
        output_image = Image.open(io.BytesIO(response.content))
        assert output_image.mode == "RGB"


class TestAssertReplacementFix:
    """
    Test fix for production safety: Assert statements replaced with explicit checks.
    
    Previously, _resize_to_target() used assert statements to verify output dimensions.
    These are stripped when running Python with -O flag, causing checks to be skipped.
    
    Fix: Replace assert with explicit if checks that raise ImageProcessingError.
    
    Note: This is difficult to test directly without mocking, but we verify the
    behavior is correct and errors are raised when expected.
    """
    
    def test_resize_to_target_validates_output_dimensions(self):
        """Verify _resize_to_target still validates output dimensions after assert removal."""
        # This is an indirect test - we verify the endpoint works correctly
        # and would fail if dimension validation was broken
        image_bytes = create_test_image(width=3840, height=2160)  # 4k image
        
        # Downscale to 2k - should work
        response = client.post(
            "/upscale/standard",
            files={"file": ("test.png", image_bytes, "image/png")},
            data={"target_resolution": "2k"}
        )
        
        assert response.status_code == 200
        
        # Verify output fits within 2k (2560x1440)
        output_image = Image.open(io.BytesIO(response.content))
        assert output_image.width <= 2560
        assert output_image.height <= 1440


class TestIntegrationAfterFixes:
    """Integration tests to verify all fixes work together correctly."""
    
    def test_complete_flow_with_valid_input(self):
        """Verify complete flow works correctly after all fixes."""
        image_bytes = create_test_image(width=1920, height=1080)
        
        response = client.post(
            "/upscale/standard",
            files={"file": ("test.png", image_bytes, "image/png")},
            data={"target_resolution": "2k"}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert "x-image-resolution" in response.headers
        assert "x-scale-factor" in response.headers
        
        # Verify output
        output_image = Image.open(io.BytesIO(response.content))
        assert output_image.format == "PNG"
        assert output_image.mode == "RGB"
    
    def test_error_handling_still_works(self):
        """Verify error handling still works correctly after fixes."""
        # Test invalid resolution
        image_bytes = create_test_image()
        
        response = client.post(
            "/upscale/standard",
            files={"file": ("test.png", image_bytes, "image/png")},
            data={"target_resolution": "invalid"}
        )
        
        assert response.status_code == 400
        assert "detail" in response.json()
    
    def test_different_resolutions_after_fixes(self):
        """Verify all valid resolutions work correctly after fixes."""
        image_bytes = create_test_image()
        
        for resolution in ["2k", "4k"]:
            response = client.post(
                "/upscale/standard",
                files={"file": ("test.png", image_bytes, "image/png")},
                data={"target_resolution": resolution}
            )
            
            assert response.status_code == 200
            assert resolution in response.headers["content-disposition"]
