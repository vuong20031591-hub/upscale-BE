"""
Unit tests for ImageProcessor validation logic.

Task 3.1: Test validation logic trong ImageProcessor
- Test file size validation (boundary: 10MB)
- Test content type validation
- Test extension validation
- Verify ValidationError messages

Requirements: 1.1, 1.2, 1.3, 1.4
Feature: ai-image-upscaling
"""

import pytest
from io import BytesIO
from PIL import Image

from app.services.image_processor import ImageProcessor
from app.models.image import UploadFileInfo
from app.core import ValidationError


# ============================================================================
# Test Constants
# ============================================================================

MAX_FILE_SIZE = 10_485_760  # 10 MiB = 10 × 1024 × 1024 bytes
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png"]
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]


# ============================================================================
# Helper Functions
# ============================================================================

def create_upload_file_info(
    filename: str = "test.jpg",
    content_type: str = "image/jpeg",
    size: int = 1000,
    content: bytes = None
) -> UploadFileInfo:
    """Create UploadFileInfo for testing."""
    if content is None:
        content = b"x" * min(size, 1000)  # Limit actual content for memory
    
    return UploadFileInfo(
        filename=filename,
        content_type=content_type,
        size=size,
        content=content
    )


# ============================================================================
# File Size Validation Tests (Requirement 1.1)
# ============================================================================

class TestFileSizeValidation:
    """Test file size validation logic."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = ImageProcessor()
    
    def test_accepts_file_at_max_size(self):
        """
        Test that file exactly at MAX_FILE_SIZE is accepted.
        Requirement 1.1: Validate file size không vượt quá 10,485,760 bytes
        """
        file_info = create_upload_file_info(
            filename="test.jpg",
            content_type="image/jpeg",
            size=MAX_FILE_SIZE,
            content=b"x" * 100
        )
        
        # Should not raise ValidationError
        try:
            self.processor.validate_upload(file_info)
        except ValidationError as e:
            # If it fails, it should not be due to size
            assert "File too large" not in str(e)
    
    def test_rejects_file_over_max_size(self):
        """
        Test that file over MAX_FILE_SIZE is rejected.
        Requirement 1.1: Validate file size không vượt quá 10,485,760 bytes
        """
        file_info = create_upload_file_info(
            filename="test.jpg",
            content_type="image/jpeg",
            size=MAX_FILE_SIZE + 1,
            content=b"x" * 100
        )
        
        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)
        
        # Verify error message
        assert "File too large" in str(exc_info.value)
    
    def test_accepts_small_file(self):
        """
        Test that small file is accepted.
        Requirement 1.1: Validate file size không vượt quá 10,485,760 bytes
        """
        file_info = create_upload_file_info(
            filename="test.jpg",
            content_type="image/jpeg",
            size=1000,
            content=b"x" * 100
        )
        
        # Should not raise ValidationError
        try:
            self.processor.validate_upload(file_info)
        except ValidationError as e:
            # If it fails, it should not be due to size
            assert "File too large" not in str(e)
    
    def test_size_error_message_format(self):
        """
        Test that size error message includes MB value.
        Requirement 1.4: ValidationError với message mô tả cụ thể
        """
        file_info = create_upload_file_info(
            filename="test.jpg",
            content_type="image/jpeg",
            size=MAX_FILE_SIZE + 1000,
            content=b"x" * 100
        )
        
        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)
        
        error_msg = str(exc_info.value)
        assert "File too large" in error_msg
        assert "10MB" in error_msg or "10" in error_msg
    
    def test_boundary_values(self):
        """
        Test boundary values around MAX_FILE_SIZE.
        Requirement 1.1: Validate file size không vượt quá 10,485,760 bytes
        """
        # Test MAX_FILE_SIZE - 1 (should pass)
        file_info = create_upload_file_info(
            filename="test.jpg",
            content_type="image/jpeg",
            size=MAX_FILE_SIZE - 1,
            content=b"x" * 100
        )
        
        try:
            self.processor.validate_upload(file_info)
        except ValidationError as e:
            assert "File too large" not in str(e)
        
        # Test MAX_FILE_SIZE + 1 (should fail)
        file_info = create_upload_file_info(
            filename="test.jpg",
            content_type="image/jpeg",
            size=MAX_FILE_SIZE + 1,
            content=b"x" * 100
        )
        
        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)
        
        assert "File too large" in str(exc_info.value)


# ============================================================================
# Content Type Validation Tests (Requirement 1.2)
# ============================================================================

class TestContentTypeValidation:
    """Test content type validation logic."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = ImageProcessor()
    
    def test_accepts_image_jpeg(self):
        """
        Test that image/jpeg content type is accepted.
        Requirement 1.2: Validate content type thuộc danh sách cho phép
        """
        file_info = create_upload_file_info(
            filename="test.jpg",
            content_type="image/jpeg",
            size=1000
        )
        
        # Should not raise ValidationError
        try:
            self.processor.validate_upload(file_info)
        except ValidationError as e:
            # If it fails, it should not be due to content type
            assert "Invalid file type" not in str(e)
    
    def test_accepts_image_png(self):
        """
        Test that image/png content type is accepted.
        Requirement 1.2: Validate content type thuộc danh sách cho phép
        """
        file_info = create_upload_file_info(
            filename="test.png",
            content_type="image/png",
            size=1000
        )
        
        # Should not raise ValidationError
        try:
            self.processor.validate_upload(file_info)
        except ValidationError as e:
            # If it fails, it should not be due to content type
            assert "Invalid file type" not in str(e)
    
    def test_rejects_invalid_content_type(self):
        """
        Test that invalid content type is rejected.
        Requirement 1.2: Validate content type thuộc danh sách cho phép
        """
        invalid_types = [
            "image/gif",
            "image/webp",
            "application/pdf",
            "text/plain",
            "application/octet-stream",
            "video/mp4",
            "audio/mpeg"
        ]
        
        for invalid_type in invalid_types:
            file_info = create_upload_file_info(
                filename="test.jpg",
                content_type=invalid_type,
                size=1000
            )
            
            with pytest.raises(ValidationError) as exc_info:
                self.processor.validate_upload(file_info)
            
            # Verify error message
            assert "Invalid file type" in str(exc_info.value)
    
    def test_content_type_error_message_format(self):
        """
        Test that content type error message lists allowed types.
        Requirement 1.4: ValidationError với message mô tả cụ thể
        """
        file_info = create_upload_file_info(
            filename="test.gif",
            content_type="image/gif",
            size=1000
        )
        
        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)
        
        error_msg = str(exc_info.value)
        assert "Invalid file type" in error_msg
        assert "Allowed:" in error_msg
        # Should mention at least one allowed type
        assert "image/jpeg" in error_msg or "image/png" in error_msg


# ============================================================================
# Extension Validation Tests (Requirement 1.3)
# ============================================================================

class TestExtensionValidation:
    """Test extension validation logic."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = ImageProcessor()
    
    def test_accepts_jpg_extension(self):
        """
        Test that .jpg extension is accepted.
        Requirement 1.3: Validate extension thuộc danh sách cho phép
        """
        file_info = create_upload_file_info(
            filename="test.jpg",
            content_type="image/jpeg",
            size=1000
        )
        
        # Should not raise ValidationError
        try:
            self.processor.validate_upload(file_info)
        except ValidationError as e:
            # If it fails, it should not be due to extension
            assert "Invalid extension" not in str(e)
    
    def test_accepts_jpeg_extension(self):
        """
        Test that .jpeg extension is accepted.
        Requirement 1.3: Validate extension thuộc danh sách cho phép
        """
        file_info = create_upload_file_info(
            filename="test.jpeg",
            content_type="image/jpeg",
            size=1000
        )
        
        # Should not raise ValidationError
        try:
            self.processor.validate_upload(file_info)
        except ValidationError as e:
            # If it fails, it should not be due to extension
            assert "Invalid extension" not in str(e)
    
    def test_accepts_png_extension(self):
        """
        Test that .png extension is accepted.
        Requirement 1.3: Validate extension thuộc danh sách cho phép
        """
        file_info = create_upload_file_info(
            filename="test.png",
            content_type="image/png",
            size=1000
        )
        
        # Should not raise ValidationError
        try:
            self.processor.validate_upload(file_info)
        except ValidationError as e:
            # If it fails, it should not be due to extension
            assert "Invalid extension" not in str(e)
    
    def test_rejects_invalid_extension(self):
        """
        Test that invalid extensions are rejected.
        Requirement 1.3: Validate extension thuộc danh sách cho phép
        """
        invalid_extensions = [
            "test.gif",
            "test.bmp",
            "test.webp",
            "test.tiff",
            "test.pdf",
            "test.txt",
            "test.exe",
            "test.php"
        ]
        
        for filename in invalid_extensions:
            file_info = create_upload_file_info(
                filename=filename,
                content_type="image/jpeg",
                size=1000
            )
            
            with pytest.raises(ValidationError) as exc_info:
                self.processor.validate_upload(file_info)
            
            # Verify error message
            assert "Invalid extension" in str(exc_info.value)
    
    def test_extension_case_insensitive(self):
        """
        Test that extension validation is case-insensitive.
        Requirement 1.3: Validate extension thuộc danh sách cho phép
        """
        # Test uppercase extensions
        uppercase_files = ["test.JPG", "test.JPEG", "test.PNG"]
        
        for filename in uppercase_files:
            file_info = create_upload_file_info(
                filename=filename,
                content_type="image/jpeg",
                size=1000
            )
            
            # Should not raise ValidationError
            try:
                self.processor.validate_upload(file_info)
            except ValidationError as e:
                # If it fails, it should not be due to extension
                assert "Invalid extension" not in str(e)
    
    def test_extension_error_message_format(self):
        """
        Test that extension error message lists allowed extensions.
        Requirement 1.4: ValidationError với message mô tả cụ thể
        """
        file_info = create_upload_file_info(
            filename="test.gif",
            content_type="image/jpeg",
            size=1000
        )
        
        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)
        
        error_msg = str(exc_info.value)
        assert "Invalid extension" in error_msg
        assert "Allowed:" in error_msg
        # Should mention at least one allowed extension
        assert "jpg" in error_msg or "jpeg" in error_msg or "png" in error_msg
    
    def test_no_extension(self):
        """
        Test that file without extension is rejected.
        Requirement 1.3: Validate extension thuộc danh sách cho phép
        """
        file_info = create_upload_file_info(
            filename="test",
            content_type="image/jpeg",
            size=1000
        )
        
        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)
        
        # Verify error message
        assert "Invalid extension" in str(exc_info.value)


# ============================================================================
# Validation Order Tests (Requirement 1.4)
# ============================================================================

class TestValidationOrder:
    """Test validation order and fail-fast behavior."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = ImageProcessor()
    
    def test_size_checked_first(self):
        """
        Test that size is checked first (fail-fast).
        Requirement 1.4: ValidationError với message mô tả cụ thể
        """
        # Create file with oversized AND invalid type AND invalid extension
        file_info = create_upload_file_info(
            filename="test.gif",
            content_type="image/gif",
            size=MAX_FILE_SIZE + 1000,
            content=b"x" * 100
        )
        
        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)
        
        # Should fail with size error (first check)
        assert "File too large" in str(exc_info.value)
    
    def test_content_type_checked_second(self):
        """
        Test that content type is checked after size.
        Requirement 1.4: ValidationError với message mô tả cụ thể
        """
        # Create file with valid size, invalid type, invalid extension
        file_info = create_upload_file_info(
            filename="test.gif",
            content_type="image/gif",
            size=1000
        )
        
        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)
        
        # Should fail with content type error (second check)
        assert "Invalid file type" in str(exc_info.value)
    
    def test_extension_checked_third(self):
        """
        Test that extension is checked after size and content type.
        Requirement 1.4: ValidationError với message mô tả cụ thể
        """
        # Create file with valid size, valid type, invalid extension
        file_info = create_upload_file_info(
            filename="test.gif",
            content_type="image/jpeg",
            size=1000
        )
        
        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)
        
        # Should fail with extension error (third check)
        assert "Invalid extension" in str(exc_info.value)


# ============================================================================
# Integration Tests
# ============================================================================

class TestValidationIntegration:
    """Integration tests for validation logic."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = ImageProcessor()
    
    def test_valid_file_passes_all_checks(self):
        """
        Test that valid file passes all validation checks.
        Requirements: 1.1, 1.2, 1.3
        """
        valid_files = [
            ("test.jpg", "image/jpeg", 1000),
            ("test.jpeg", "image/jpeg", 5000),
            ("test.png", "image/png", 10000),
            ("image.JPG", "image/jpeg", MAX_FILE_SIZE),
            ("photo.PNG", "image/png", MAX_FILE_SIZE - 1)
        ]
        
        for filename, content_type, size in valid_files:
            file_info = create_upload_file_info(
                filename=filename,
                content_type=content_type,
                size=size
            )
            
            # Should not raise ValidationError
            try:
                self.processor.validate_upload(file_info)
            except ValidationError as e:
                pytest.fail(f"Valid file {filename} failed validation: {e}")
    
    def test_multiple_invalid_fields(self):
        """
        Test validation with multiple invalid fields.
        Requirement 1.4: ValidationError với message mô tả cụ thể
        """
        # File with all invalid fields
        file_info = create_upload_file_info(
            filename="test.exe",
            content_type="application/octet-stream",
            size=MAX_FILE_SIZE + 1000
        )
        
        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)
        
        # Should fail with first error (size)
        error_msg = str(exc_info.value)
        assert "File too large" in error_msg or "Invalid file type" in error_msg or "Invalid extension" in error_msg
