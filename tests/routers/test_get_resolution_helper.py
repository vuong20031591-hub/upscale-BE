"""
Unit tests for _get_resolution() helper function.

Task 11.1: Test _get_resolution() helper function
- Test với valid resolutions ("2k", "4k")
- Test với invalid resolutions (raise ValueError, không fallback)
- Test case sensitivity
- Requirements: 8.6

Note: Implementation hiện tại raise ValueError thay vì fallback to 2k.
Đây là kết quả của Critical Fix #1 (FIXES_SUMMARY.md) để tránh silent failures.
"""

import pytest
from app.routers.upscale_basic import _get_resolution
from app.models import Resolution


class TestGetResolutionHelper:
    """
    Unit tests for _get_resolution() helper function.
    
    Validates: Requirements 8.6
    Task: 11.1
    """
    
    def test_valid_resolution_2k_lowercase(self):
        """Test _get_resolution() với valid resolution "2k" (lowercase)."""
        result = _get_resolution("2k")
        assert result == Resolution.K2
        assert result.value == "2k"
    
    def test_valid_resolution_4k_lowercase(self):
        """Test _get_resolution() với valid resolution "4k" (lowercase)."""
        result = _get_resolution("4k")
        assert result == Resolution.K4
        assert result.value == "4k"
    
    def test_valid_resolution_8k_lowercase(self):
        """Test _get_resolution() với valid resolution "8k" (lowercase)."""
        result = _get_resolution("8k")
        assert result == Resolution.K8
        assert result.value == "8k"
    
    def test_case_insensitive_2k_uppercase(self):
        """Test case sensitivity - "2K" (uppercase) should work."""
        result = _get_resolution("2K")
        assert result == Resolution.K2
    
    def test_case_insensitive_4k_uppercase(self):
        """Test case sensitivity - "4K" (uppercase) should work."""
        result = _get_resolution("4K")
        assert result == Resolution.K4
    
    def test_case_insensitive_mixed_case(self):
        """Test case sensitivity - mixed case "2K", "4k" should work."""
        assert _get_resolution("2K") == Resolution.K2
        assert _get_resolution("4k") == Resolution.K4
        assert _get_resolution("8K") == Resolution.K8
    
    def test_invalid_resolution_raises_value_error(self):
        """Test invalid resolution raises ValueError (không fallback to 2k)."""
        with pytest.raises(ValueError) as exc_info:
            _get_resolution("invalid")
        
        error_message = str(exc_info.value)
        assert "Invalid resolution" in error_message
        assert "invalid" in error_message
        assert "Valid values:" in error_message
    
    def test_invalid_resolution_6k_raises_value_error(self):
        """Test unsupported resolution "6k" raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _get_resolution("6k")
        
        error_message = str(exc_info.value)
        assert "Invalid resolution '6k'" in error_message
        assert "Valid values:" in error_message
    
    def test_invalid_resolution_empty_string(self):
        """Test empty string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _get_resolution("")
        
        assert "Invalid resolution" in str(exc_info.value)
    
    def test_invalid_resolution_random_string(self):
        """Test random invalid strings raise ValueError."""
        invalid_resolutions = ["1k", "3k", "5k", "10k", "hd", "fullhd", "uhd", "random"]
        
        for invalid_res in invalid_resolutions:
            with pytest.raises(ValueError) as exc_info:
                _get_resolution(invalid_res)
            
            error_message = str(exc_info.value)
            assert "Invalid resolution" in error_message
            assert invalid_res in error_message
    
    def test_error_message_includes_valid_values(self):
        """Test error message includes list of valid values."""
        with pytest.raises(ValueError) as exc_info:
            _get_resolution("invalid")
        
        error_message = str(exc_info.value)
        # Should list all valid enum values
        assert "2k" in error_message
        assert "4k" in error_message
        assert "8k" in error_message
    
    def test_return_type_is_resolution_enum(self):
        """Test return type is Resolution enum, not plain string."""
        result = _get_resolution("2k")
        assert isinstance(result, Resolution)
        # Note: Resolution inherits from str, so isinstance(result, str) is True
        # We verify it's specifically a Resolution enum, not just any string
        assert type(result).__name__ == "Resolution"
    
    def test_resolution_enum_properties(self):
        """Test Resolution enum has correct properties."""
        result = _get_resolution("4k")
        assert result.value == "4k"
        assert result.name == "K4"
        assert isinstance(result, str)  # Resolution inherits from str


class TestGetResolutionEdgeCases:
    """Edge cases for _get_resolution() helper function."""
    
    def test_whitespace_in_resolution(self):
        """Test resolution with whitespace raises ValueError."""
        with pytest.raises(ValueError):
            _get_resolution(" 2k ")
        
        with pytest.raises(ValueError):
            _get_resolution("2k ")
        
        with pytest.raises(ValueError):
            _get_resolution(" 4k")
    
    def test_numeric_input(self):
        """Test numeric values raise ValueError."""
        with pytest.raises((ValueError, AttributeError)):
            # May raise AttributeError if .lower() is called on int
            _get_resolution(2)  # type: ignore
    
    def test_none_input(self):
        """Test None input raises appropriate error."""
        with pytest.raises((ValueError, AttributeError)):
            _get_resolution(None)  # type: ignore
    
    def test_resolution_with_special_characters(self):
        """Test resolutions with special characters raise ValueError."""
        invalid_inputs = ["2k!", "4k?", "2-k", "4_k", "2.k"]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError):
                _get_resolution(invalid_input)


class TestGetResolutionBehaviorConsistency:
    """Test consistency of _get_resolution() behavior."""
    
    def test_same_input_returns_same_enum(self):
        """Test calling with same input returns same enum value."""
        result1 = _get_resolution("2k")
        result2 = _get_resolution("2k")
        assert result1 == result2
        assert result1 is result2  # Should be same enum instance
    
    def test_case_variations_return_same_enum(self):
        """Test different case variations return same enum value."""
        result_lower = _get_resolution("2k")
        result_upper = _get_resolution("2K")
        assert result_lower == result_upper
        assert result_lower is result_upper
    
    def test_all_valid_resolutions_are_testable(self):
        """Test all Resolution enum values can be obtained via _get_resolution()."""
        # Get all enum values
        all_resolutions = [r.value for r in Resolution]
        
        # Verify each can be obtained via _get_resolution()
        for res_value in all_resolutions:
            result = _get_resolution(res_value)
            assert result.value == res_value
