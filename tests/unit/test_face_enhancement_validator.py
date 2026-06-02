"""
Unit tests for FaceEnhancementValidator.

Tests validation logic for face enhancement parameters including:
- Mode validation (restoration/colorization/inpainting)
- Weight validation with clamping and mode-specific rules
- Face upsample validation with defaults
- Background upscale validation (bg_upscale: 1, 2, 4)

Requirements: 4.2, 4.4, 6.3, 8.7, 8.8, 11.3, 11.11
"""

import pytest
from app.validators.face_enhancement_validator import FaceEnhancementValidator
from app.services.codeformer_manager import CodeFormerMode
from app.core import ValidationError


class TestFaceEnhancementValidator:
    """Test suite for FaceEnhancementValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance for tests."""
        return FaceEnhancementValidator()
    
    # ==================== Mode Validation Tests ====================
    
    def test_validate_mode_restoration(self, validator):
        """Test validation of restoration mode."""
        mode = validator.validate_mode("restoration")
        assert mode == CodeFormerMode.RESTORATION
    
    def test_validate_mode_colorization(self, validator):
        """Test validation of colorization mode."""
        mode = validator.validate_mode("colorization")
        assert mode == CodeFormerMode.COLORIZATION
    
    def test_validate_mode_inpainting(self, validator):
        """Test validation of inpainting mode."""
        mode = validator.validate_mode("inpainting")
        assert mode == CodeFormerMode.INPAINTING
    
    def test_validate_mode_invalid(self, validator):
        """Test that invalid mode raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_mode("invalid_mode")
        
        error_msg = str(exc_info.value)
        assert "Invalid mode 'invalid_mode'" in error_msg
        assert "restoration" in error_msg
        assert "colorization" in error_msg
        assert "inpainting" in error_msg
    
    def test_validate_mode_case_sensitive(self, validator):
        """Test that mode validation is case-sensitive."""
        with pytest.raises(ValidationError):
            validator.validate_mode("RESTORATION")
        
        with pytest.raises(ValidationError):
            validator.validate_mode("Colorization")
    
    # ==================== Weight Validation Tests ====================
    
    def test_validate_weight_restoration_valid(self, validator):
        """Test weight validation for restoration mode with valid values."""
        # Test various valid weights
        assert validator.validate_weight(0.0, CodeFormerMode.RESTORATION) == 0.0
        assert validator.validate_weight(0.5, CodeFormerMode.RESTORATION) == 0.5
        assert validator.validate_weight(0.7, CodeFormerMode.RESTORATION) == 0.7
        assert validator.validate_weight(1.0, CodeFormerMode.RESTORATION) == 1.0
    
    def test_validate_weight_restoration_none_uses_default(self, validator):
        """Test that None weight uses default from settings for restoration."""
        weight = validator.validate_weight(None, CodeFormerMode.RESTORATION)
        # Should return default from settings (typically 0.7)
        assert isinstance(weight, float)
        assert 0.0 <= weight <= 1.0
    
    def test_validate_weight_restoration_clamping_above(self, validator):
        """Test that weight > 1.0 is clamped to 1.0 for restoration."""
        # Requirement 8.7: Out-of-range weights are clamped with warning
        assert validator.validate_weight(1.5, CodeFormerMode.RESTORATION) == 1.0
        assert validator.validate_weight(2.0, CodeFormerMode.RESTORATION) == 1.0
        assert validator.validate_weight(100.0, CodeFormerMode.RESTORATION) == 1.0
    
    def test_validate_weight_restoration_clamping_below(self, validator):
        """Test that weight < 0.0 is clamped to 0.0 for restoration."""
        # Requirement 8.7: Out-of-range weights are clamped with warning
        assert validator.validate_weight(-0.5, CodeFormerMode.RESTORATION) == 0.0
        assert validator.validate_weight(-1.0, CodeFormerMode.RESTORATION) == 0.0
    
    def test_validate_weight_colorization_fixed(self, validator):
        """Test that colorization mode uses fixed weight w=0."""
        # Requirement 4.4, 8.8: Weight ignored for non-restoration modes
        assert validator.validate_weight(None, CodeFormerMode.COLORIZATION) == 0
        assert validator.validate_weight(0.5, CodeFormerMode.COLORIZATION) == 0
        assert validator.validate_weight(0.9, CodeFormerMode.COLORIZATION) == 0
    
    def test_validate_weight_inpainting_fixed(self, validator):
        """Test that inpainting mode uses fixed weight w=1."""
        # Requirement 4.4, 8.8: Weight ignored for non-restoration modes
        assert validator.validate_weight(None, CodeFormerMode.INPAINTING) == 1
        assert validator.validate_weight(0.5, CodeFormerMode.INPAINTING) == 1
        assert validator.validate_weight(0.0, CodeFormerMode.INPAINTING) == 1
    
    # ==================== Face Upsample Validation Tests ====================
    
    def test_validate_face_upsample_true(self, validator):
        """Test face_upsample validation with True value."""
        assert validator.validate_face_upsample(True) is True
    
    def test_validate_face_upsample_false(self, validator):
        """Test face_upsample validation with False value."""
        assert validator.validate_face_upsample(False) is False
    
    def test_validate_face_upsample_none_uses_default(self, validator):
        """Test that None face_upsample uses default from settings."""
        # Requirement 6.3: Use default from settings if None
        result = validator.validate_face_upsample(None)
        assert isinstance(result, bool)
    
    # ==================== Background Upscale Validation Tests ====================
    
    def test_validate_bg_upscale_valid_1(self, validator):
        """Test bg_upscale validation with valid value 1."""
        # Requirement 11.3: Support bg_upscale values 1, 2, 4
        assert validator.validate_bg_upscale(1) == 1
    
    def test_validate_bg_upscale_valid_2(self, validator):
        """Test bg_upscale validation with valid value 2."""
        # Requirement 11.3: Support bg_upscale values 1, 2, 4
        assert validator.validate_bg_upscale(2) == 2
    
    def test_validate_bg_upscale_valid_4(self, validator):
        """Test bg_upscale validation with valid value 4."""
        # Requirement 11.3: Support bg_upscale values 1, 2, 4
        assert validator.validate_bg_upscale(4) == 4
    
    def test_validate_bg_upscale_none_uses_default(self, validator):
        """Test that None bg_upscale returns default value 2."""
        # Default value should be 2
        assert validator.validate_bg_upscale(None) == 2
    
    def test_validate_bg_upscale_invalid_zero(self, validator):
        """Test that bg_upscale=0 raises ValidationError."""
        # Requirement 11.11: Invalid bg_upscale should raise HTTP 422
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_bg_upscale(0)
        
        error_msg = str(exc_info.value)
        assert "Invalid bg_upscale '0'" in error_msg
        assert "Must be 1, 2, or 4" in error_msg
    
    def test_validate_bg_upscale_invalid_3(self, validator):
        """Test that bg_upscale=3 raises ValidationError."""
        # Requirement 11.11: Only 1, 2, 4 are valid
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_bg_upscale(3)
        
        error_msg = str(exc_info.value)
        assert "Invalid bg_upscale '3'" in error_msg
        assert "Must be 1, 2, or 4" in error_msg
    
    def test_validate_bg_upscale_invalid_5(self, validator):
        """Test that bg_upscale=5 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_bg_upscale(5)
        
        error_msg = str(exc_info.value)
        assert "Invalid bg_upscale '5'" in error_msg
    
    def test_validate_bg_upscale_invalid_negative(self, validator):
        """Test that negative bg_upscale raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_bg_upscale(-1)
        
        error_msg = str(exc_info.value)
        assert "Invalid bg_upscale '-1'" in error_msg
    
    def test_validate_bg_upscale_invalid_large(self, validator):
        """Test that large bg_upscale values raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_bg_upscale(10)
        
        error_msg = str(exc_info.value)
        assert "Invalid bg_upscale '10'" in error_msg
    
    @pytest.mark.parametrize("invalid_value", [0, 3, 5, 6, 8, 10, -1, -5, 100])
    def test_validate_bg_upscale_parametrized_invalid(self, validator, invalid_value):
        """Test multiple invalid bg_upscale values using parametrization."""
        # Requirement 11.11: All values except 1, 2, 4 should raise error
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_bg_upscale(invalid_value)
        
        error_msg = str(exc_info.value)
        assert f"Invalid bg_upscale '{invalid_value}'" in error_msg
        assert "Must be 1, 2, or 4" in error_msg
    
    @pytest.mark.parametrize("valid_value", [1, 2, 4])
    def test_validate_bg_upscale_parametrized_valid(self, validator, valid_value):
        """Test all valid bg_upscale values using parametrization."""
        # Requirement 11.3: Only 1, 2, 4 are valid
        result = validator.validate_bg_upscale(valid_value)
        assert result == valid_value
    
    # ==================== Integration Tests ====================
    
    def test_validator_has_model_configs(self, validator):
        """Test that validator has access to MODEL_CONFIGS."""
        assert hasattr(validator, 'model_configs')
        assert validator.model_configs is not None
        assert CodeFormerMode.RESTORATION in validator.model_configs
        assert CodeFormerMode.COLORIZATION in validator.model_configs
        assert CodeFormerMode.INPAINTING in validator.model_configs
    
    def test_validator_valid_modes_constant(self, validator):
        """Test that VALID_MODES constant is correctly defined."""
        assert validator.VALID_MODES == {"restoration", "colorization", "inpainting"}
        assert len(validator.VALID_MODES) == 3
