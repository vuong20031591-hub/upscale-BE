"""
Unit tests for CodeFormerManager helper methods.

Tests the get_mode_info() and get_all_modes_info() methods added to
CodeFormerManager to support mode information retrieval.

Requirements:
    - Requirement 5.1: Model management with lazy loading
    - Requirement 5.2: Mode-specific model loading
"""

import pytest
from app.services.codeformer_manager import CodeFormerManager, CodeFormerMode


class TestCodeFormerManagerHelpers:
    """Test helper methods in CodeFormerManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh CodeFormerManager instance for each test."""
        # Reset singleton instance to ensure clean state
        CodeFormerManager._instance = None
        CodeFormerManager._models = {}
        CodeFormerManager._detection_model = None
        return CodeFormerManager()
    
    def test_get_mode_info_restoration(self, manager):
        """Test get_mode_info returns correct config for restoration mode."""
        info = manager.get_mode_info(CodeFormerMode.RESTORATION)
        
        # Verify all expected keys are present
        assert "url" in info
        assert "codebook_size" in info
        assert "connect_list" in info
        assert "w" in info
        assert "adain" in info
        assert "loaded" in info
        
        # Verify restoration-specific values
        assert info["codebook_size"] == 1024
        assert info["connect_list"] == ['32', '64', '128', '256']
        assert info["w"] == 0.7
        assert info["adain"] is True
        assert info["loaded"] is False  # Not loaded initially
        assert "codeformer.pth" in info["url"]
    
    def test_get_mode_info_colorization(self, manager):
        """Test get_mode_info returns correct config for colorization mode."""
        info = manager.get_mode_info(CodeFormerMode.COLORIZATION)
        
        # Verify colorization-specific values
        assert info["codebook_size"] == 1024
        assert info["connect_list"] == ['32', '64', '128']
        assert info["w"] == 0  # Fixed w=0 for colorization
        assert info["adain"] is True
        assert info["loaded"] is False
        assert "codeformer_colorization.pth" in info["url"]
    
    def test_get_mode_info_inpainting(self, manager):
        """Test get_mode_info returns correct config for inpainting mode."""
        info = manager.get_mode_info(CodeFormerMode.INPAINTING)
        
        # Verify inpainting-specific values
        assert info["codebook_size"] == 512
        assert info["connect_list"] == ['32', '64', '128']
        assert info["w"] == 1  # Fixed w=1 for inpainting
        assert info["adain"] is False
        assert info["loaded"] is False
        assert "codeformer_inpainting.pth" in info["url"]
    
    def test_get_mode_info_loaded_status_false_initially(self, manager):
        """Test that loaded status is False before model is loaded."""
        for mode in CodeFormerMode:
            info = manager.get_mode_info(mode)
            assert info["loaded"] is False, f"{mode.value} should not be loaded initially"
    
    def test_get_mode_info_loaded_status_true_after_mock_load(self, manager):
        """Test that loaded status is True after model is added to _models."""
        # Simulate model loading by adding to _models dict
        manager._models[CodeFormerMode.RESTORATION] = "mock_model"
        
        info_restoration = manager.get_mode_info(CodeFormerMode.RESTORATION)
        info_colorization = manager.get_mode_info(CodeFormerMode.COLORIZATION)
        
        assert info_restoration["loaded"] is True
        assert info_colorization["loaded"] is False
    
    def test_get_all_modes_info_returns_all_modes(self, manager):
        """Test get_all_modes_info returns info for all three modes."""
        all_info = manager.get_all_modes_info()
        
        # Verify all modes are present
        assert "restoration" in all_info
        assert "colorization" in all_info
        assert "inpainting" in all_info
        assert len(all_info) == 3
    
    def test_get_all_modes_info_structure(self, manager):
        """Test that get_all_modes_info returns correct structure."""
        all_info = manager.get_all_modes_info()
        
        # Each mode should have the same structure as get_mode_info
        for mode_name, mode_info in all_info.items():
            assert isinstance(mode_name, str)
            assert isinstance(mode_info, dict)
            assert "url" in mode_info
            assert "codebook_size" in mode_info
            assert "connect_list" in mode_info
            assert "w" in mode_info
            assert "adain" in mode_info
            assert "loaded" in mode_info
    
    def test_get_all_modes_info_values_match_individual_calls(self, manager):
        """Test that get_all_modes_info values match individual get_mode_info calls."""
        all_info = manager.get_all_modes_info()
        
        for mode in CodeFormerMode:
            individual_info = manager.get_mode_info(mode)
            all_info_entry = all_info[mode.value]
            
            # Compare all fields
            assert individual_info == all_info_entry, \
                f"Mismatch for {mode.value}: {individual_info} != {all_info_entry}"
    
    def test_get_all_modes_info_reflects_loaded_status(self, manager):
        """Test that get_all_modes_info reflects current loaded status."""
        # Initially all should be not loaded
        all_info = manager.get_all_modes_info()
        assert all_info["restoration"]["loaded"] is False
        assert all_info["colorization"]["loaded"] is False
        assert all_info["inpainting"]["loaded"] is False
        
        # Simulate loading restoration model
        manager._models[CodeFormerMode.RESTORATION] = "mock_model"
        
        # Check updated status
        all_info = manager.get_all_modes_info()
        assert all_info["restoration"]["loaded"] is True
        assert all_info["colorization"]["loaded"] is False
        assert all_info["inpainting"]["loaded"] is False
    
    def test_get_mode_info_does_not_modify_original_config(self, manager):
        """Test that get_mode_info doesn't modify MODEL_CONFIGS."""
        original_config = manager.MODEL_CONFIGS[CodeFormerMode.RESTORATION].copy()
        
        # Call get_mode_info
        info = manager.get_mode_info(CodeFormerMode.RESTORATION)
        
        # Modify returned info
        info["loaded"] = True
        info["w"] = 0.5
        
        # Verify original config is unchanged
        current_config = manager.MODEL_CONFIGS[CodeFormerMode.RESTORATION]
        assert current_config == original_config
        assert "loaded" not in current_config  # loaded is added, not in original
    
    def test_get_mode_info_with_all_modes(self, manager):
        """Test get_mode_info works for all CodeFormerMode enum values."""
        for mode in CodeFormerMode:
            info = manager.get_mode_info(mode)
            
            # Should not raise exception
            assert info is not None
            assert isinstance(info, dict)
            assert info["loaded"] is False  # Initially not loaded
    
    def test_mode_info_url_format(self, manager):
        """Test that all mode URLs follow expected format."""
        all_info = manager.get_all_modes_info()
        
        for mode_name, mode_info in all_info.items():
            url = mode_info["url"]
            
            # Verify URL format
            assert url.startswith("https://github.com/sczhou/CodeFormer/releases/")
            assert url.endswith(".pth")
            assert "v0.1.0" in url
    
    def test_mode_info_codebook_sizes(self, manager):
        """Test that codebook sizes are as expected per design."""
        all_info = manager.get_all_modes_info()
        
        # Per design document
        assert all_info["restoration"]["codebook_size"] == 1024
        assert all_info["colorization"]["codebook_size"] == 1024
        assert all_info["inpainting"]["codebook_size"] == 512
    
    def test_mode_info_fixed_weights(self, manager):
        """Test that fixed weights are correct for each mode."""
        all_info = manager.get_all_modes_info()
        
        # Per design document
        assert all_info["restoration"]["w"] == 0.7  # Default, can be overridden
        assert all_info["colorization"]["w"] == 0  # Fixed
        assert all_info["inpainting"]["w"] == 1  # Fixed
    
    def test_mode_info_adain_settings(self, manager):
        """Test that AdaIN settings are correct for each mode."""
        all_info = manager.get_all_modes_info()
        
        # Per design document
        assert all_info["restoration"]["adain"] is True
        assert all_info["colorization"]["adain"] is True
        assert all_info["inpainting"]["adain"] is False
