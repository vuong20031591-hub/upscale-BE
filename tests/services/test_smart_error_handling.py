"""
Tests for error handling in smart auto-detection feature.

This module tests error scenarios:
- Detection failures with fallback to restoration
- Processing failures with proper logging
- File validation errors
- Error response formats

Requirements: 4.7, 4.8, 7.2
"""

import io
import numpy as np
import pytest
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from app.services.smart_processor import SmartProcessor
from app.models.analysis import ProcessingMode, DetectionType


class TestDetectionFailureFallback:
    """Tests for detection failure fallback to restoration mode (Requirement 4.8)."""
    
    def test_detection_failure_fallback_to_restoration(self):
        """
        Test that detection failure falls back to restoration mode.
        
        When ImageAnalyzer.analyze() raises an exception, the system should:
        1. Log the error with context
        2. Create fallback analysis result with restoration mode
        3. Continue processing with restoration mode
        4. Return successful result
        
        Validates: Requirements 4.8, 7.2
        """
        # Create test image
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Create processor
        processor = SmartProcessor()
        
        # Mock analyzer to raise exception
        with patch.object(processor.analyzer, 'analyze') as mock_analyze:
            mock_analyze.side_effect = Exception("OpenCV error: invalid image")
            
            # Mock restoration processing to return test image
            with patch.object(processor, '_process_restoration') as mock_restoration:
                mock_restoration.return_value = image
                
                # Process image (should not raise exception)
                result = processor.process_image(image, filename="test.jpg")
        
        # Verify fallback to restoration mode
        assert result.selected_mode == ProcessingMode.RESTORATION
        assert result.analysis_result.suggested_mode == ProcessingMode.RESTORATION
        assert "fallback" in result.analysis_result.explanation.lower()
        
        # Verify all detections are False (fallback state)
        assert result.analysis_result.detections[DetectionType.GRAYSCALE].detected is False
        assert result.analysis_result.detections[DetectionType.WHITE_MASK].detected is False
        assert result.analysis_result.detections[DetectionType.LOW_RESOLUTION].detected is False
        assert result.analysis_result.detections[DetectionType.BLUR].detected is False
        
        # Verify restoration was called
        mock_restoration.assert_called_once()
    
    def test_detection_failure_logs_error(self):
        """
        Test that detection failure logs error with full context.
        
        Validates: Requirements 4.8, 7.2
        """
        # Create test image
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Create processor
        processor = SmartProcessor()
        
        # Mock analyzer to raise exception
        with patch.object(processor.analyzer, 'analyze') as mock_analyze:
            mock_analyze.side_effect = ValueError("Invalid image format")
            
            # Mock restoration processing
            with patch.object(processor, '_process_restoration') as mock_restoration:
                mock_restoration.return_value = image
                
                # Mock logger to capture log calls
                with patch('app.services.smart_processor.logger') as mock_logger:
                    # Process image
                    result = processor.process_image(image, filename="error_test.jpg")
                    
                    # Verify error was logged
                    mock_logger.error.assert_called_once()
                    call_args = mock_logger.error.call_args
                    
                    # Check log message
                    assert "analysis failed" in call_args[0][0].lower()
                    
                    # Check log context
                    assert call_args[1]['filename'] == "error_test.jpg"
                    assert 'error' in call_args[1]
                    assert 'analysis_time_seconds' in call_args[1]
                    assert call_args[1]['exc_info'] is True
    
    def test_detection_failure_numpy_error(self):
        """
        Test detection failure with NumPy error (common scenario).
        
        Validates: Requirements 4.8
        """
        # Create test image
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Create processor
        processor = SmartProcessor()
        
        # Mock analyzer to raise generic NumPy error (ValueError is common)
        with patch.object(processor.analyzer, 'analyze') as mock_analyze:
            mock_analyze.side_effect = ValueError("NumPy: invalid array shape")
            
            # Mock restoration processing
            with patch.object(processor, '_process_restoration') as mock_restoration:
                mock_restoration.return_value = image
                
                # Process image (should handle gracefully)
                result = processor.process_image(image, filename="numpy_error.jpg")
        
        # Verify fallback worked
        assert result.selected_mode == ProcessingMode.RESTORATION
        assert result.processed_image is not None
    
    def test_detection_failure_opencv_error(self):
        """
        Test detection failure with OpenCV error (common scenario).
        
        Validates: Requirements 4.8
        """
        # Create test image
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Create processor
        processor = SmartProcessor()
        
        # Mock analyzer to raise OpenCV-like error
        with patch.object(processor.analyzer, 'analyze') as mock_analyze:
            mock_analyze.side_effect = RuntimeError("cv2.error: invalid image dimensions")
            
            # Mock restoration processing
            with patch.object(processor, '_process_restoration') as mock_restoration:
                mock_restoration.return_value = image
                
                # Process image (should handle gracefully)
                result = processor.process_image(image, filename="opencv_error.jpg")
        
        # Verify fallback worked
        assert result.selected_mode == ProcessingMode.RESTORATION


class TestProcessingFailureHandling:
    """Tests for processing failure handling (Requirement 4.7)."""
    
    def test_processing_failure_raises_exception(self):
        """
        Test that processing failure raises exception (no fallback for processing).
        
        When processing fails, the system should:
        1. Log the error with context
        2. Re-raise the exception to be handled by endpoint
        
        Validates: Requirements 4.7, 7.2
        """
        # Create test image
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Create processor
        processor = SmartProcessor()
        
        # Mock restoration to raise exception
        with patch.object(processor, '_process_restoration') as mock_restoration:
            mock_restoration.side_effect = RuntimeError("CUDA out of memory")
            
            # Process image (should raise exception)
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                processor.process_image(image, filename="cuda_error.jpg")
    
    def test_processing_failure_logs_error(self):
        """
        Test that processing failure logs error with full context.
        
        Validates: Requirements 4.7, 7.2
        """
        # Create test image
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Create processor
        processor = SmartProcessor()
        
        # Mock restoration to raise exception
        with patch.object(processor, '_process_restoration') as mock_restoration:
            mock_restoration.side_effect = RuntimeError("Model inference failed")
            
            # Mock logger to capture log calls
            with patch('app.services.smart_processor.logger') as mock_logger:
                # Process image (should raise exception)
                with pytest.raises(RuntimeError):
                    processor.process_image(image, filename="inference_error.jpg")
                
                # Verify error was logged
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args
                
                # Check log message
                assert "processing failed" in call_args[0][0].lower()
                
                # Check log context
                assert call_args[1]['filename'] == "inference_error.jpg"
                assert call_args[1]['mode'] == ProcessingMode.RESTORATION.value
                assert 'processing_time_seconds' in call_args[1]
                assert 'error' in call_args[1]
                assert call_args[1]['exc_info'] is True
    
    def test_processing_failure_colorization_mode(self):
        """
        Test processing failure in colorization mode.
        
        Validates: Requirements 4.7
        """
        # Create grayscale test image
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Create processor
        processor = SmartProcessor()
        
        # Mock colorization to raise exception
        with patch.object(processor, '_process_colorization') as mock_colorization:
            mock_colorization.side_effect = Exception("CodeFormer colorization failed")
            
            # Process image (should raise exception)
            with pytest.raises(Exception, match="colorization failed"):
                processor.process_image(image, filename="colorization_error.jpg")
    
    def test_processing_failure_upscaling_mode(self):
        """
        Test processing failure in upscaling mode.
        
        Validates: Requirements 4.7
        """
        # Create low-res test image
        img_array = np.full((400, 300, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        # Create processor
        processor = SmartProcessor()
        
        # Mock upscaling to raise exception
        with patch.object(processor, '_process_upscaling') as mock_upscaling:
            mock_upscaling.side_effect = RuntimeError("Real-ESRGAN upscaling failed")
            
            # Mock analysis to suggest upscaling
            from app.models.analysis import AnalysisResult, DetectionResult
            mock_analysis = AnalysisResult(
                detections={
                    DetectionType.GRAYSCALE: DetectionResult(False, 0.0, {}),
                    DetectionType.WHITE_MASK: DetectionResult(False, 0.0, {}),
                    DetectionType.LOW_RESOLUTION: DetectionResult(True, 1.0, {"width": 400, "height": 300}),
                    DetectionType.BLUR: DetectionResult(False, 0.0, {})
                },
                suggested_mode=ProcessingMode.UPSCALING,
                alternative_modes=[],
                explanation="Low resolution detected"
            )
            
            with patch.object(processor.analyzer, 'analyze', return_value=mock_analysis):
                # Process image (should raise exception)
                with pytest.raises(RuntimeError, match="upscaling failed"):
                    processor.process_image(image, filename="upscaling_error.jpg")


class TestErrorContextLogging:
    """Tests for error logging with proper context (Requirement 7.2)."""
    
    def test_detection_error_includes_filename(self):
        """
        Test that detection error logs include filename.
        
        Validates: Requirements 7.2
        """
        # Create test image
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        processor = SmartProcessor()
        
        with patch.object(processor.analyzer, 'analyze') as mock_analyze:
            mock_analyze.side_effect = Exception("Test error")
            
            with patch.object(processor, '_process_restoration', return_value=image):
                with patch('app.services.smart_processor.logger') as mock_logger:
                    processor.process_image(image, filename="context_test.jpg")
                    
                    # Verify filename in log context
                    call_args = mock_logger.error.call_args
                    assert call_args[1]['filename'] == "context_test.jpg"
    
    def test_processing_error_includes_mode(self):
        """
        Test that processing error logs include selected mode.
        
        Validates: Requirements 7.2
        """
        # Create test image
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        processor = SmartProcessor()
        
        with patch.object(processor, '_process_restoration') as mock_restoration:
            mock_restoration.side_effect = Exception("Test processing error")
            
            with patch('app.services.smart_processor.logger') as mock_logger:
                with pytest.raises(Exception):
                    processor.process_image(image, filename="mode_test.jpg")
                
                # Verify mode in log context
                call_args = mock_logger.error.call_args
                assert call_args[1]['mode'] == ProcessingMode.RESTORATION.value
    
    def test_error_logs_include_timing(self):
        """
        Test that error logs include timing information.
        
        Validates: Requirements 7.2
        """
        # Create test image
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        processor = SmartProcessor()
        
        # Test detection error timing
        with patch.object(processor.analyzer, 'analyze') as mock_analyze:
            mock_analyze.side_effect = Exception("Detection error")
            
            with patch.object(processor, '_process_restoration', return_value=image):
                with patch('app.services.smart_processor.logger') as mock_logger:
                    processor.process_image(image, filename="timing_test.jpg")
                    
                    # Verify timing in log context
                    call_args = mock_logger.error.call_args
                    assert 'analysis_time_seconds' in call_args[1]
                    assert isinstance(call_args[1]['analysis_time_seconds'], (int, float))
        
        # Test processing error timing
        with patch.object(processor, '_process_restoration') as mock_restoration:
            mock_restoration.side_effect = Exception("Processing error")
            
            with patch('app.services.smart_processor.logger') as mock_logger:
                with pytest.raises(Exception):
                    processor.process_image(image, filename="timing_test2.jpg")
                
                # Verify timing in log context
                call_args = mock_logger.error.call_args
                assert 'processing_time_seconds' in call_args[1]
                assert isinstance(call_args[1]['processing_time_seconds'], (int, float))
    
    def test_error_logs_include_exc_info(self):
        """
        Test that error logs include full stack trace (exc_info=True).
        
        Validates: Requirements 7.2
        """
        # Create test image
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        processor = SmartProcessor()
        
        # Test detection error
        with patch.object(processor.analyzer, 'analyze') as mock_analyze:
            mock_analyze.side_effect = Exception("Detection error")
            
            with patch.object(processor, '_process_restoration', return_value=image):
                with patch('app.services.smart_processor.logger') as mock_logger:
                    processor.process_image(image, filename="exc_info_test.jpg")
                    
                    # Verify exc_info=True
                    call_args = mock_logger.error.call_args
                    assert call_args[1]['exc_info'] is True
        
        # Test processing error
        with patch.object(processor, '_process_restoration') as mock_restoration:
            mock_restoration.side_effect = Exception("Processing error")
            
            with patch('app.services.smart_processor.logger') as mock_logger:
                with pytest.raises(Exception):
                    processor.process_image(image, filename="exc_info_test2.jpg")
                
                # Verify exc_info=True
                call_args = mock_logger.error.call_args
                assert call_args[1]['exc_info'] is True


class TestGracefulDegradation:
    """Tests for graceful degradation strategies (Requirement 4.8)."""
    
    def test_fallback_analysis_has_all_detections(self):
        """
        Test that fallback analysis result has all detection types.
        
        Validates: Requirements 4.8
        """
        # Create test image
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        processor = SmartProcessor()
        
        with patch.object(processor.analyzer, 'analyze') as mock_analyze:
            mock_analyze.side_effect = Exception("Analysis failed")
            
            with patch.object(processor, '_process_restoration', return_value=image):
                result = processor.process_image(image, filename="fallback_test.jpg")
        
        # Verify all detection types present
        assert DetectionType.GRAYSCALE in result.analysis_result.detections
        assert DetectionType.WHITE_MASK in result.analysis_result.detections
        assert DetectionType.LOW_RESOLUTION in result.analysis_result.detections
        assert DetectionType.BLUR in result.analysis_result.detections
    
    def test_fallback_analysis_all_detections_false(self):
        """
        Test that fallback analysis has all detections set to False.
        
        Validates: Requirements 4.8
        """
        # Create test image
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        processor = SmartProcessor()
        
        with patch.object(processor.analyzer, 'analyze') as mock_analyze:
            mock_analyze.side_effect = Exception("Analysis failed")
            
            with patch.object(processor, '_process_restoration', return_value=image):
                result = processor.process_image(image, filename="fallback_test.jpg")
        
        # Verify all detections are False
        for detection_type, detection_result in result.analysis_result.detections.items():
            assert detection_result.detected is False
            assert detection_result.confidence == 0.0
    
    def test_fallback_explanation_indicates_failure(self):
        """
        Test that fallback analysis explanation indicates failure.
        
        Validates: Requirements 4.8
        """
        # Create test image
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        processor = SmartProcessor()
        
        with patch.object(processor.analyzer, 'analyze') as mock_analyze:
            mock_analyze.side_effect = Exception("Analysis failed")
            
            with patch.object(processor, '_process_restoration', return_value=image):
                result = processor.process_image(image, filename="fallback_test.jpg")
        
        # Verify explanation indicates fallback
        explanation = result.analysis_result.explanation.lower()
        assert "failed" in explanation or "fallback" in explanation
        assert "restoration" in explanation
