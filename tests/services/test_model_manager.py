"""
Tests for ModelManager service.

This module tests the ModelManager singleton, including:
- Singleton pattern implementation
- Lazy loading behavior
- Model loading idempotence
"""

import pytest
from PIL import Image

from app.services.model_manager import ModelManager


class TestModelManagerSingleton:
    """Tests for ModelManager Singleton pattern (Task 4.1)."""
    
    def test_singleton_same_instance(self):
        """
        Test that multiple instantiations return the same instance.
        
        Validates: Requirements 2.1
        """
        # Create multiple instances
        manager1 = ModelManager()
        manager2 = ModelManager()
        manager3 = ModelManager()
        
        # All should be the same instance
        assert manager1 is manager2
        assert manager2 is manager3
        assert manager1 is manager3
    
    def test_singleton_same_id(self):
        """
        Test that multiple instances have the same id().
        
        Validates: Requirements 2.1
        """
        # Create multiple instances
        manager1 = ModelManager()
        manager2 = ModelManager()
        
        # Should have same id (same object in memory)
        assert id(manager1) == id(manager2)
    
    def test_singleton_multiple_calls(self):
        """
        Test singleton pattern with many instantiations.
        
        Validates: Requirements 2.1
        """
        # Create 10 instances
        instances = [ModelManager() for _ in range(10)]
        
        # All should be the same instance
        first_id = id(instances[0])
        for instance in instances:
            assert id(instance) == first_id
    
    def test_singleton_state_shared(self):
        """
        Test that singleton instances share state.
        
        Validates: Requirements 2.1
        """
        # Get first instance
        manager1 = ModelManager()
        initial_loaded = manager1.is_loaded
        
        # Get second instance
        manager2 = ModelManager()
        
        # Should have same state
        assert manager2.is_loaded == initial_loaded
        
        # Modifying internal state on one should affect the other
        # (This is just to verify they're truly the same object)
        manager1._test_attribute = "test_value"
        assert hasattr(manager2, '_test_attribute')
        assert manager2._test_attribute == "test_value"
        
        # Cleanup
        delattr(manager1, '_test_attribute')


class TestModelManagerLazyLoading:
    """Tests for ModelManager lazy loading behavior (Task 4.2)."""
    
    def test_not_loaded_on_init(self):
        """
        Test that model is not loaded on initialization.
        
        Validates: Requirements 2.2, 2.3
        """
        manager = ModelManager()
        
        # Model should not be loaded yet (lazy loading)
        assert manager.is_loaded is False
    
    def test_is_loaded_property_no_side_effects(self):
        """
        Test that is_loaded property doesn't trigger loading.
        
        Validates: Requirements 2.2
        """
        manager = ModelManager()
        
        # Check is_loaded multiple times
        for _ in range(5):
            loaded = manager.is_loaded
            assert loaded is False
        
        # Model should still not be loaded
        assert manager._model is None
    
    def test_loaded_after_first_upscale(self, monkeypatch, sample_image_small):
        """
        Test that model is loaded after first upscale() call.
        
        This test verifies the lazy loading behavior:
        1. Model is NOT loaded on initialization
        2. Model IS loaded after first upscale() call
        
        Validates: Requirements 2.2, 2.3
        """
        from unittest.mock import MagicMock
        import numpy as np
        
        # Create a fresh manager instance
        manager = ModelManager()
        
        # Verify model is not loaded initially
        assert manager.is_loaded is False
        assert manager._model is None
        
        # Mock the internal methods to avoid loading real model
        def mock_download_if_needed(self):
            """Mock download - do nothing."""
            pass
        
        def mock_load_model(self):
            """Mock load - set _model to a mock object with enhance method."""
            mock_model = MagicMock()
            
            # Configure enhance method to return proper tuple (output_array, None)
            def mock_enhance(img_array, outscale=4):
                """Mock enhance that returns (upscaled_array, None)."""
                h, w = img_array.shape[:2]
                new_h, new_w = h * outscale, w * outscale
                # Return simple upscaled array with correct shape
                upscaled = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                return upscaled, None
            
            mock_model.enhance = mock_enhance
            self._model = mock_model
        
        # Apply mocks
        monkeypatch.setattr(ModelManager, '_download_if_needed', mock_download_if_needed)
        monkeypatch.setattr(ModelManager, '_load_model', mock_load_model)
        
        # Call upscale() - this should trigger lazy loading
        result = manager.upscale(sample_image_small, outscale=4)
        
        # Verify model is now loaded
        assert manager.is_loaded is True
        assert manager._model is not None
        
        # Verify result is PIL Image
        assert isinstance(result, Image.Image)
        # Verify dimensions are scaled correctly (100x100 -> 400x400)
        assert result.size == (400, 400)
    
    def test_upscale_triggers_load_only_once(self, monkeypatch, sample_image_small):
        """
        Test that upscale() only loads model once (idempotent).
        
        This test verifies that calling upscale() multiple times
        doesn't reload the model each time.
        
        Validates: Requirements 2.3, 2.8
        """
        from unittest.mock import MagicMock
        import numpy as np
        
        manager = ModelManager()
        
        # Track how many times _load_model is called
        load_call_count = 0
        
        def mock_download_if_needed(self):
            """Mock download - do nothing."""
            pass
        
        def mock_load_model(self):
            """Mock load - track calls and set _model with enhance method."""
            nonlocal load_call_count
            load_call_count += 1
            
            mock_model = MagicMock()
            
            # Configure enhance method to return proper tuple
            def mock_enhance(img_array, outscale=4):
                """Mock enhance that returns (upscaled_array, None)."""
                h, w = img_array.shape[:2]
                new_h, new_w = h * outscale, w * outscale
                upscaled = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                return upscaled, None
            
            mock_model.enhance = mock_enhance
            self._model = mock_model
        
        # Apply mocks
        monkeypatch.setattr(ModelManager, '_download_if_needed', mock_download_if_needed)
        monkeypatch.setattr(ModelManager, '_load_model', mock_load_model)
        
        # Call upscale() multiple times
        for i in range(3):
            result = manager.upscale(sample_image_small, outscale=4)
            
            # Model should be loaded after first call
            assert manager.is_loaded is True
            
            # _load_model should only be called once
            assert load_call_count == 1
            
            # Verify result is valid
            assert isinstance(result, Image.Image)
            assert result.size == (400, 400)


class TestModelManagerMocked:
    """Tests for ModelManager with mocked model (to avoid loading real model)."""
    
    def test_upscale_with_mock(self, mock_model_manager, sample_image_small):
        """
        Test upscale method with mocked model.
        
        This test uses mock to avoid loading the real AI model.
        """
        result = mock_model_manager.upscale(sample_image_small, outscale=4)
        
        assert isinstance(result, Image.Image)
        assert result.size == (400, 400)  # 100 * 4
    
    def test_mock_model_manager_is_loaded(self, mock_model_manager):
        """Test that mocked model manager reports as loaded."""
        assert mock_model_manager.is_loaded is True
    
    def test_get_info_structure(self):
        """
        Test get_info() returns correct structure.
        
        Validates: Requirements 2.6, 2.7
        """
        manager = ModelManager()
        info = manager.get_info()
        
        # Verify structure
        assert isinstance(info, dict)
        assert "name" in info
        assert "scale" in info
        assert "loaded" in info
        assert "half_precision" in info
        assert "model_file" in info
        
        # Verify types
        assert isinstance(info["name"], str)
        assert isinstance(info["scale"], int)
        assert isinstance(info["loaded"], bool)
        assert isinstance(info["half_precision"], bool)
        assert isinstance(info["model_file"], str)
        
        # Verify model_file is filename only, not full path
        assert "/" not in info["model_file"]
        assert "\\" not in info["model_file"]
