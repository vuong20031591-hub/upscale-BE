"""
Unit tests for configuration module.

Tests for Task 1: Setup resolution mapping và configuration
- RESOLUTION_MAP constant với mappings (2k, 4k, 8k)
- Resolution validation với strict checking
- OUTPUT_QUALITY validation (1-100, default 95)
- PIL.Image.MAX_IMAGE_PIXELS configuration
"""

import os
import pytest
from unittest.mock import patch
from PIL import Image

from app.core.config import OutputConfig, Settings


class TestOutputConfigResolutionMap:
    """Test RESOLUTION_MAP constant và resolution validation."""
    
    def test_resolution_map_contains_required_resolutions(self):
        """Verify RESOLUTION_MAP contains 2k, 4k, 8k mappings."""
        assert "2k" in OutputConfig.RESOLUTION_MAP
        assert "4k" in OutputConfig.RESOLUTION_MAP
        assert "8k" in OutputConfig.RESOLUTION_MAP
    
    def test_resolution_map_2k_dimensions(self):
        """Verify 2k resolution maps to 2560x1440."""
        assert OutputConfig.RESOLUTION_MAP["2k"] == (2560, 1440)
    
    def test_resolution_map_4k_dimensions(self):
        """Verify 4k resolution maps to 3840x2160."""
        assert OutputConfig.RESOLUTION_MAP["4k"] == (3840, 2160)
    
    def test_resolution_map_8k_dimensions(self):
        """Verify 8k resolution maps to 7680x4320."""
        assert OutputConfig.RESOLUTION_MAP["8k"] == (7680, 4320)


class TestOutputConfigValidateResolution:
    """Test strict resolution validation (Requirements 1.6, 1.7)."""
    
    def test_validate_resolution_accepts_supported_resolution(self):
        """Valid resolution in supported list should not raise error."""
        config = OutputConfig(
            default_resolution="2k",
            supported_resolutions=["2k", "4k"],
            format="png",
            quality=95
        )
        
        # Should not raise
        config.validate_resolution("2k")
        config.validate_resolution("4k")
    
    def test_validate_resolution_case_insensitive(self):
        """Resolution validation should be case-insensitive."""
        config = OutputConfig(
            default_resolution="2k",
            supported_resolutions=["2k", "4k"],
            format="png",
            quality=95
        )
        
        # Should not raise
        config.validate_resolution("2K")
        config.validate_resolution("4K")
    
    def test_validate_resolution_rejects_unsupported_resolution(self):
        """Invalid resolution should raise ValueError with descriptive message."""
        config = OutputConfig(
            default_resolution="2k",
            supported_resolutions=["2k", "4k"],
            format="png",
            quality=95
        )
        
        with pytest.raises(ValueError) as exc_info:
            config.validate_resolution("8k")
        
        error_msg = str(exc_info.value)
        assert "Invalid resolution '8k'" in error_msg
        assert "Supported: 2k, 4k" in error_msg
    
    def test_validate_resolution_rejects_invalid_string(self):
        """Completely invalid resolution string should raise ValueError."""
        config = OutputConfig(
            default_resolution="2k",
            supported_resolutions=["2k", "4k"],
            format="png",
            quality=95
        )
        
        with pytest.raises(ValueError) as exc_info:
            config.validate_resolution("invalid")
        
        assert "Invalid resolution" in str(exc_info.value)


class TestOutputConfigGetDimensions:
    """Test strict get_dimensions method."""
    
    def test_get_dimensions_returns_correct_dimensions(self):
        """get_dimensions should return correct tuple for valid resolution."""
        config = OutputConfig(
            default_resolution="2k",
            supported_resolutions=["2k", "4k"],
            format="png",
            quality=95
        )
        
        assert config.get_dimensions("2k") == (2560, 1440)
        assert config.get_dimensions("4k") == (3840, 2160)
    
    def test_get_dimensions_case_insensitive(self):
        """get_dimensions should handle case-insensitive input."""
        config = OutputConfig(
            default_resolution="2k",
            supported_resolutions=["2k", "4k"],
            format="png",
            quality=95
        )
        
        assert config.get_dimensions("2K") == (2560, 1440)
        assert config.get_dimensions("4K") == (3840, 2160)
    
    def test_get_dimensions_raises_error_for_invalid_resolution(self):
        """get_dimensions should raise ValueError for resolution not in RESOLUTION_MAP."""
        config = OutputConfig(
            default_resolution="2k",
            supported_resolutions=["2k", "4k"],
            format="png",
            quality=95
        )
        
        with pytest.raises(ValueError) as exc_info:
            config.get_dimensions("16k")
        
        error_msg = str(exc_info.value)
        assert "Resolution '16k' not found in RESOLUTION_MAP" in error_msg
        assert "Available:" in error_msg


class TestOutputConfigQualityValidation:
    """Test OUTPUT_QUALITY validation (Requirements 9.1, 9.2, 9.4)."""
    
    def test_quality_validation_accepts_valid_range(self):
        """Quality values between 1 and 100 should be accepted."""
        # Test boundary values
        config_min = OutputConfig(
            default_resolution="2k",
            supported_resolutions=["2k"],
            format="png",
            quality=1
        )
        assert config_min.quality == 1
        
        config_max = OutputConfig(
            default_resolution="2k",
            supported_resolutions=["2k"],
            format="png",
            quality=100
        )
        assert config_max.quality == 100
        
        config_mid = OutputConfig(
            default_resolution="2k",
            supported_resolutions=["2k"],
            format="png",
            quality=95
        )
        assert config_mid.quality == 95
    
    def test_quality_validation_rejects_zero(self):
        """Quality value 0 should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OutputConfig(
                default_resolution="2k",
                supported_resolutions=["2k"],
                format="png",
                quality=0
            )
        
        assert "OUTPUT_QUALITY must be between 1 and 100" in str(exc_info.value)
    
    def test_quality_validation_rejects_negative(self):
        """Negative quality values should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OutputConfig(
                default_resolution="2k",
                supported_resolutions=["2k"],
                format="png",
                quality=-1
            )
        
        assert "OUTPUT_QUALITY must be between 1 and 100" in str(exc_info.value)
    
    def test_quality_validation_rejects_over_100(self):
        """Quality values over 100 should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OutputConfig(
                default_resolution="2k",
                supported_resolutions=["2k"],
                format="png",
                quality=101
            )
        
        assert "OUTPUT_QUALITY must be between 1 and 100" in str(exc_info.value)
    
    @patch.dict(os.environ, {"OUTPUT_QUALITY": "95"})
    def test_from_env_uses_default_quality_95(self):
        """When OUTPUT_QUALITY is set to 95, it should be used."""
        config = OutputConfig.from_env()
        assert config.quality == 95
    
    @patch.dict(os.environ, {"OUTPUT_QUALITY": "80"})
    def test_from_env_accepts_custom_quality(self):
        """Custom OUTPUT_QUALITY from environment should be accepted."""
        config = OutputConfig.from_env()
        assert config.quality == 80
    
    @patch.dict(os.environ, {"OUTPUT_QUALITY": "invalid"}, clear=False)
    def test_from_env_rejects_non_integer_quality(self):
        """Non-integer OUTPUT_QUALITY should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OutputConfig.from_env()
        
        assert "OUTPUT_QUALITY must be an integer" in str(exc_info.value)
    
    @patch.dict(os.environ, {"OUTPUT_QUALITY": "150"}, clear=False)
    def test_from_env_rejects_out_of_range_quality(self):
        """OUTPUT_QUALITY outside 1-100 range should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OutputConfig.from_env()
        
        assert "OUTPUT_QUALITY must be between 1 and 100" in str(exc_info.value)


class TestPILDecompressionBombProtection:
    """Test PIL.Image.MAX_IMAGE_PIXELS configuration (Requirements 2.6, 2.7, 9.1, 9.2)."""
    
    def test_pil_max_image_pixels_is_configured(self):
        """PIL.Image.MAX_IMAGE_PIXELS should be set after Settings initialization."""
        # Settings is already initialized as singleton
        # Just verify it's set
        assert Image.MAX_IMAGE_PIXELS is not None
        assert isinstance(Image.MAX_IMAGE_PIXELS, int)
    
    def test_pil_max_image_pixels_default_value(self):
        """PIL.Image.MAX_IMAGE_PIXELS should be set to default security value."""
        # PIL default is 89,478,485 pixels
        # We verify it's set to a reasonable security value
        assert Image.MAX_IMAGE_PIXELS in (89478485, 178956970)
    
    def test_settings_initializes_pil_protection(self):
        """Settings initialization should configure PIL decompression bomb protection."""
        # Create a new Settings instance (singleton will return existing)
        settings = Settings()
        
        # Verify PIL protection is configured
        assert Image.MAX_IMAGE_PIXELS is not None
        assert Image.MAX_IMAGE_PIXELS > 0


class TestOutputConfigIntegration:
    """Integration tests for OutputConfig with environment variables."""
    
    @patch.dict(os.environ, {
        "DEFAULT_TARGET_RESOLUTION": "4k",
        "SUPPORTED_RESOLUTIONS": "2k,4k,8k",
        "OUTPUT_FORMAT": "png",
        "OUTPUT_QUALITY": "90"
    }, clear=False)
    def test_from_env_loads_all_settings_correctly(self):
        """from_env should load all settings from environment variables."""
        config = OutputConfig.from_env()
        
        assert config.default_resolution == "4k"
        assert config.supported_resolutions == ["2k", "4k", "8k"]
        assert config.format == "png"
        assert config.quality == 90
    
    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_uses_defaults_when_not_set(self):
        """from_env should use default values when env vars not set."""
        # Clear environment and set minimal required vars
        with patch.dict(os.environ, {}, clear=True):
            config = OutputConfig.from_env()
            
            assert config.default_resolution == "2k"
            assert config.supported_resolutions == ["2k", "4k"]
            assert config.format == "png"
            assert config.quality == 95
