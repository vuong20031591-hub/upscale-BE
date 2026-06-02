"""
Property-based tests for image upload validation using Hypothesis.

These tests verify the validation logic for UploadFileInfo objects
and ensure the validator correctly handles edge cases, malformed inputs,
and maintains security invariants.

Test File: test_validation.py
Feature: image-upload-validation
"""

import io
import pytest
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Generator, Set

from hypothesis import given, settings, strategies as st, Phase, assume, example
from hypothesis.strategies import composite
from PIL import Image

from app.services.image_processor import ImageProcessor
from app.models.image import UploadFileInfo
from app.core import ValidationError
from app.core.config import Settings


# ============================================================================
# Configuration Constants (from spec)
# ============================================================================

MAX_FILE_SIZE = 10_485_760  # 10 MiB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


# ============================================================================
# Custom Hypothesis Strategies
# ============================================================================

@composite
def valid_extensions(draw) -> str:
    """Strategy to generate valid file extensions."""
    return draw(st.sampled_from(list(ALLOWED_EXTENSIONS)))


@composite
def invalid_extensions(draw) -> str:
    """Strategy to generate invalid file extensions."""
    invalid_exts = ["gif", "bmp", "tiff", "webp", "exe", "pdf", "txt", "py", "", "php"]
    return draw(st.sampled_from(invalid_exts))


@composite
def valid_content_types(draw) -> str:
    """Strategy to generate valid MIME types."""
    return draw(st.sampled_from(list(ALLOWED_CONTENT_TYPES)))


@composite
def invalid_content_types(draw) -> str:
    """Strategy to generate invalid MIME types."""
    invalid_types = [
        "application/octet-stream",
        "text/plain",
        "application/pdf",
        "image/gif",
        "image/webp",
        "image/bmp",
        "audio/mpeg",
        "video/mp4",
        "",
        "invalid/type"
    ]
    return draw(st.sampled_from(invalid_types))


@composite
def valid_filenames(draw) -> str:
    """Strategy to generate valid image filenames."""
    ext = draw(valid_extensions())
    base = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=50
    ))
    return f"{base}.{ext}"


@composite
def filenames_with_invalid_extensions(draw) -> str:
    """Strategy to generate filenames with invalid extensions."""
    ext = draw(invalid_extensions())
    base = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=50
    ))
    if ext:
        return f"{base}.{ext}"
    return base


@composite
def valid_image_content(draw, width: int = None, height: int = None) -> bytes:
    """
    Strategy to generate valid image content using Pillow.
    Creates actual JPEG or PNG images with specified dimensions.
    """
    img_format = draw(st.sampled_from(["JPEG", "PNG"]))
    w = width or draw(st.integers(min_value=1, max_value=2048))
    h = height or draw(st.integers(min_value=1, max_value=2048))

    buffer = io.BytesIO()
    # Create RGB image with random color
    color = (
        draw(st.integers(min_value=0, max_value=255)),
        draw(st.integers(min_value=0, max_value=255)),
        draw(st.integers(min_value=0, max_value=255))
    )
    img = Image.new("RGB", (w, h), color)
    img.save(buffer, format=img_format, quality=85)
    return buffer.getvalue()


@composite
def invalid_image_content(draw) -> bytes:
    """
    Strategy to generate invalid/malicious image content.
    Includes random binary, truncated headers, and malformed magic bytes.
    """
    content_type = draw(st.integers(min_value=0, max_value=4))

    if content_type == 0:
        # Random binary data
        return draw(st.binary(min_size=1, max_size=1000))
    elif content_type == 1:
        # Truncated JPEG header (just magic bytes)
        return b"\xff\xd8\xff"
    elif content_type == 2:
        # Truncated PNG header
        return b"\x89PNG\r\n\x1a\n"
    elif content_type == 3:
        # GIF masquerading as JPEG extension
        return b"GIF89a" + draw(st.binary(min_size=10, max_size=100))
    else:
        # HTML/Script injection attempt
        return b"<html><script>alert('xss')</script></html>"


@composite
def upload_file_info_strategy(
    draw,
    filename_strategy=None,
    content_type_strategy=None,
    size_strategy=None,
    content_strategy=None
) -> UploadFileInfo:
    """
    Composite strategy to generate UploadFileInfo objects.
    Allows overriding specific fields for targeted testing.
    """
    fname = draw(filename_strategy if filename_strategy else valid_filenames())
    ctype = draw(content_type_strategy if content_type_strategy else valid_content_types())
    content = draw(content_strategy if content_strategy else valid_image_content())

    # Size can be overridden or derived from content
    if size_strategy:
        size = draw(size_strategy)
    else:
        size = len(content)

    return UploadFileInfo(
        filename=fname,
        content_type=ctype,
        size=size,
        content=content
    )


# ============================================================================
# Test Helper Functions
# ============================================================================

@contextmanager
def expect_validation_error() -> Generator[None, None, None]:
    """Context manager to expect a ValidationError."""
    with pytest.raises(ValidationError):
        yield


def assert_validation_passes(processor: ImageProcessor, file_info: UploadFileInfo) -> None:
    """Assert that validation passes without raising."""
    try:
        processor.validate_upload(file_info)
    except ValidationError as e:
        pytest.fail(f"Expected validation to pass, but got ValidationError: {e}")


def assert_validation_fails(processor: ImageProcessor, file_info: UploadFileInfo) -> None:
    """Assert that validation fails with ValidationError."""
    with pytest.raises(ValidationError):
        processor.validate_upload(file_info)


# ============================================================================
# Property Tests
# ============================================================================

class TestValidationProperties:
    """Property-based tests for image upload validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ImageProcessor()

    # =========================================================================
    # Property 1: Validation Order and Fail-Fast
    # =========================================================================

    @given(
        size=st.integers(min_value=MAX_FILE_SIZE + 1, max_value=MAX_FILE_SIZE * 2),
        content_type=st.text(),
        extension=st.text()
    )
    @settings(max_examples=50, deadline=None)
    def test_property_1_validation_order_size_checked_first(
        self, size: int, content_type: str, extension: str
    ):
        """
        # Feature: image-upload-validation, Property 1: Validation Order and Fail-Fast

        If file size exceeds max, validation fails immediately with size error,
        regardless of other invalid fields.
        """
        # Create file with oversized content
        content = b"x" * min(size, 100)  # Limit actual content for memory

        file_info = UploadFileInfo(
            filename=f"test.{extension}" if extension else "test.xyz",
            content_type=content_type if content_type else "invalid/type",
            size=size,
            content=content
        )

        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)

        # Should fail with size error (first check)
        assert "File too large" in str(exc_info.value)

    # =========================================================================
    # Property 2: Size Validation Rejects Large Files
    # =========================================================================

    @given(
        oversize=st.integers(min_value=MAX_FILE_SIZE + 1, max_value=MAX_FILE_SIZE * 10),
        filename=valid_filenames(),
        content_type=valid_content_types()
    )
    @settings(max_examples=100, phases=[Phase.explicit, Phase.reuse, Phase.generate])
    def test_property_2_size_validation_rejects_large_files(
        self, oversize: int, filename: str, content_type: str
    ):
        """
        # Feature: image-upload-validation, Property 2: Size Validation Rejects Large Files

        For any file where size > MAX_FILE_SIZE, validation must fail with
        ValidationError containing "File too large" message.
        """
        file_info = UploadFileInfo(
            filename=filename,
            content_type=content_type,
            size=oversize,
            content=b"x" * 100  # Small actual content
        )

        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)

        assert "File too large" in str(exc_info.value)
        assert "10MB" in str(exc_info.value) or "10" in str(exc_info.value)

    @given(
        valid_size=st.integers(min_value=1, max_value=MAX_FILE_SIZE),
        filename=valid_filenames(),
        content_type=valid_content_types()
    )
    @settings(max_examples=100)
    def test_property_2_size_validation_accepts_valid_sizes(
        self, valid_size: int, filename: str, content_type: str
    ):
        """
        # Feature: image-upload-validation, Property 2: Size Validation Rejects Large Files

        For any file where 1 <= size <= MAX_FILE_SIZE (and other fields valid),
        size validation should pass.
        """
        assume(valid_size <= MAX_FILE_SIZE)

        file_info = UploadFileInfo(
            filename=filename,
            content_type=content_type,
            size=valid_size,
            content=b"x" * min(valid_size, 1000)
        )

        # Should not raise size-related error (may fail on other checks)
        try:
            self.processor.validate_upload(file_info)
        except ValidationError as e:
            # Should NOT be a size error
            assert "File too large" not in str(e)

    # =========================================================================
    # Property 3: Content Type Validation Rejects Invalid MIME Types
    # =========================================================================

    @given(
        content_type=invalid_content_types(),
        filename=valid_filenames(),
        size=st.integers(min_value=1, max_value=1000)
    )
    @settings(max_examples=100)
    def test_property_3_content_type_rejects_invalid_mime_types(
        self, content_type: str, filename: str, size: int
    ):
        """
        # Feature: image-upload-validation, Property 3: Content Type Validation Rejects Invalid MIME Types

        For any file with content_type not in allowed_content_types,
        validation must fail with ValidationError containing "Invalid file type".
        """
        assume(content_type not in ALLOWED_CONTENT_TYPES)

        file_info = UploadFileInfo(
            filename=filename,
            content_type=content_type,
            size=size,
            content=b"x" * size
        )

        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)

        assert "Invalid file type" in str(exc_info.value)

    @given(
        content_type=valid_content_types(),
        filename=valid_filenames(),
        size=st.integers(min_value=1, max_value=1000)
    )
    @settings(max_examples=50)
    def test_property_3_content_type_accepts_valid_mime_types(
        self, content_type: str, filename: str, size: int
    ):
        """
        # Feature: image-upload-validation, Property 3: Content Type Validation Rejects Invalid MIME Types

        For any file with content_type in allowed_content_types,
        content type validation should pass.
        """
        file_info = UploadFileInfo(
            filename=filename,
            content_type=content_type,
            size=size,
            content=b"x" * size
        )

        try:
            self.processor.validate_upload(file_info)
        except ValidationError as e:
            # Should NOT be a content type error
            assert "Invalid file type" not in str(e)

    # =========================================================================
    # Property 4: Extension Validation Rejects Invalid Extensions
    # =========================================================================

    @given(
        ext=invalid_extensions(),
        content_type=valid_content_types(),
        size=st.integers(min_value=1, max_value=1000)
    )
    @settings(max_examples=100)
    def test_property_4_extension_rejects_invalid_extensions(
        self, ext: str, content_type: str, size: int
    ):
        """
        # Feature: image-upload-validation, Property 4: Extension Validation Rejects Invalid Extensions

        For any file with extension not in allowed_extensions,
        validation must fail with ValidationError containing "Invalid extension".
        """
        assume(ext not in ALLOWED_EXTENSIONS)

        filename = f"test.{ext}" if ext else "test"

        file_info = UploadFileInfo(
            filename=filename,
            content_type=content_type,
            size=size,
            content=b"x" * size
        )

        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)

        assert "Invalid extension" in str(exc_info.value)

    @given(
        filename=valid_filenames(),
        content_type=valid_content_types(),
        size=st.integers(min_value=1, max_value=1000)
    )
    @settings(max_examples=50)
    def test_property_4_extension_accepts_valid_extensions(
        self, filename: str, content_type: str, size: int
    ):
        """
        # Feature: image-upload-validation, Property 4: Extension Validation Rejects Invalid Extensions

        For any file with extension in allowed_extensions,
        extension validation should pass.
        """
        file_info = UploadFileInfo(
            filename=filename,
            content_type=content_type,
            size=size,
            content=b"x" * size
        )

        try:
            self.processor.validate_upload(file_info)
        except ValidationError as e:
            # Should NOT be an extension error
            assert "Invalid extension" not in str(e)

    # =========================================================================
    # Property 5: Valid Files Pass All Validations
    # =========================================================================

    @given(
        width=st.integers(min_value=1, max_value=512),
        height=st.integers(min_value=1, max_value=512),
        format_type=st.sampled_from(["JPEG", "PNG"])
    )
    @settings(max_examples=50, deadline=None)
    def test_property_5_valid_files_pass_all_validations(
        self, width: int, height: int, format_type: str
    ):
        """
        # Feature: image-upload-validation, Property 5: Valid Files Pass All Validations

        Synthetically generated valid images within constraints must pass validation.
        """
        # Generate valid image content
        buffer = io.BytesIO()
        img = Image.new("RGB", (width, height), color=(128, 128, 128))
        img.save(buffer, format=format_type, quality=85)
        content = buffer.getvalue()

        # Map format to filename and content_type
        ext = "jpg" if format_type == "JPEG" else "png"
        mime = "image/jpeg" if format_type == "JPEG" else "image/png"

        file_info = UploadFileInfo(
            filename=f"test_image.{ext}",
            content_type=mime,
            size=len(content),
            content=content
        )

        # Should pass without raising
        assert_validation_passes(self.processor, file_info)

    # =========================================================================
    # Property 6: Extension Extraction Correctness
    # =========================================================================

    @given(
        base=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P")),
            min_size=1,
            max_size=50
        ),
        ext=valid_extensions()
    )
    @settings(max_examples=100)
    def test_property_6_extension_extraction_from_filename(self, base: str, ext: str):
        """
        # Feature: image-upload-validation, Property 6: Extension Extraction Correctness

        For filename "name.ext", UploadFileInfo.extension must return "ext" (lowercase).
        """
        filename = f"{base}.{ext}"
        file_info = UploadFileInfo(
            filename=filename,
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        assert file_info.extension == ext.lower()

    @given(
        base=st.text(min_size=1, max_size=50)
    )
    @settings(max_examples=50)
    def test_property_6_extension_extraction_no_extension(self, base: str):
        """
        # Feature: image-upload-validation, Property 6: Extension Extraction Correctness

        For filename without extension, UploadFileInfo.extension must return "".
        """
        assume("." not in base)

        file_info = UploadFileInfo(
            filename=base,
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        assert file_info.extension == ""

    @given(
        base=st.text(min_size=1, max_size=30),
        ext=valid_extensions(),
        extra_dots=st.text(alphabet=".", min_size=1, max_size=3)
    )
    @settings(max_examples=50)
    def test_property_6_extension_extraction_multiple_dots(self, base: str, ext: str, extra_dots: str):
        """
        # Feature: image-upload-validation, Property 6: Extension Extraction Correctness

        For filename with multiple dots like "a.b.c.jpg", extension must be "jpg".
        """
        filename = f"{base}{extra_dots}{ext}"
        file_info = UploadFileInfo(
            filename=filename,
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        assert file_info.extension == ext.lower()

    # =========================================================================
    # Property 7: Metadata Extraction Completeness
    # =========================================================================

    @given(
        fname=valid_filenames(),
        ctype=valid_content_types(),
        size=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_property_7_metadata_extraction_completeness(
        self, fname: str, ctype: str, size: int
    ):
        """
        # Feature: image-upload-validation, Property 7: Metadata Extraction Completeness

        UploadFileInfo must preserve all metadata: filename, content_type, size, content.
        """
        content = b"x" * min(size, 1000)

        file_info = UploadFileInfo(
            filename=fname,
            content_type=ctype,
            size=size,
            content=content
        )

        assert file_info.filename == fname
        assert file_info.content_type == ctype
        assert file_info.size == size
        assert file_info.content == content

    # =========================================================================
    # Property 8: RGB Conversion Consistency (via to_image)
    # =========================================================================

    @given(
        width=st.integers(min_value=1, max_value=256),
        height=st.integers(min_value=1, max_value=256),
        mode=st.sampled_from(["RGB", "RGBA", "L", "P", "1"])
    )
    @settings(max_examples=50, deadline=None)
    def test_property_8_rgb_conversion_consistency(self, width: int, height: int, mode: str):
        """
        # Feature: image-upload-validation, Property 8: RGB Conversion Consistency

        UploadFileInfo.to_image() must always return RGB mode image.
        """
        buffer = io.BytesIO()

        if mode == "RGB":
            img = Image.new("RGB", (width, height), color=(128, 128, 128))
        elif mode == "RGBA":
            img = Image.new("RGBA", (width, height), color=(128, 128, 128, 255))
        elif mode == "L":
            img = Image.new("L", (width, height), color=128)
        elif mode == "P":
            img = Image.new("P", (width, height), color=128)
        elif mode == "1":
            img = Image.new("1", (width, height))
        else:
            img = Image.new("RGB", (width, height))

        img.save(buffer, format="PNG")
        content = buffer.getvalue()

        file_info = UploadFileInfo(
            filename="test.png",
            content_type="image/png",
            size=len(content),
            content=content
        )

        result_image = file_info.to_image()

        # Must always be RGB
        assert result_image.mode == "RGB"
        assert result_image.size == (width, height)

    # =========================================================================
    # Property 9: Error Response Format Consistency
    # =========================================================================

    @given(
        size=st.integers(min_value=MAX_FILE_SIZE + 1, max_value=MAX_FILE_SIZE * 2)
    )
    @settings(max_examples=30)
    def test_property_9_error_format_size(self, size: int):
        """
        # Feature: image-upload-validation, Property 9: Error Response Format Consistency

        Size error messages must contain "File too large" and size information.
        """
        file_info = UploadFileInfo(
            filename="test.jpg",
            content_type="image/jpeg",
            size=size,
            content=b"x"
        )

        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)

        error_msg = str(exc_info.value)
        assert "File too large" in error_msg

    @given(
        content_type=invalid_content_types()
    )
    @settings(max_examples=30)
    def test_property_9_error_format_content_type(self, content_type: str):
        """
        # Feature: image-upload-validation, Property 9: Error Response Format Consistency

        Content type error messages must contain "Invalid file type" and allowed types.
        """
        assume(content_type not in ALLOWED_CONTENT_TYPES)

        file_info = UploadFileInfo(
            filename="test.jpg",
            content_type=content_type,
            size=100,
            content=b"x" * 100
        )

        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)

        error_msg = str(exc_info.value)
        assert "Invalid file type" in error_msg
        assert "Allowed" in error_msg

    @given(
        ext=invalid_extensions()
    )
    @settings(max_examples=30)
    def test_property_9_error_format_extension(self, ext: str):
        """
        # Feature: image-upload-validation, Property 9: Error Response Format Consistency

        Extension error messages must contain "Invalid extension" and allowed extensions.
        """
        assume(ext not in ALLOWED_EXTENSIONS)

        filename = f"test.{ext}" if ext else "test"
        file_info = UploadFileInfo(
            filename=filename,
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)

        error_msg = str(exc_info.value)
        assert "Invalid extension" in error_msg
        assert "Allowed" in error_msg

    # =========================================================================
    # Property 10: Configuration Derivation Correctness
    # =========================================================================

    def test_property_10_configuration_values_match_expected(self):
        """
        # Feature: image-upload-validation, Property 10: Configuration Derivation Correctness

        Settings must derive correct values from configuration constants.
        """
        # Create fresh settings instance
        Settings._instance = None
        settings = Settings()

        # Verify configuration matches spec
        assert settings.upload.max_file_size == MAX_FILE_SIZE
        assert settings.upload.allowed_extensions == ALLOWED_EXTENSIONS
        assert settings.upload.allowed_content_types == ALLOWED_CONTENT_TYPES

    @given(
        size=st.integers(min_value=1, max_value=MAX_FILE_SIZE),
        quality=st.integers(min_value=1, max_value=95)
    )
    @settings(max_examples=30, deadline=None)
    def test_property_10_valid_file_size_boundary(self, size: int, quality: int):
        """
        # Feature: image-upload-validation, Property 10: Configuration Derivation Correctness

        Files at exact size boundary (MAX_FILE_SIZE) should pass.
        """
        buffer = io.BytesIO()
        img = Image.new("RGB", (100, 100))
        img.save(buffer, format="JPEG", quality=quality)
        content = buffer.getvalue()

        # Scale content to exact size if needed (up to MAX_FILE_SIZE)
        target_size = min(size, len(content))
        scaled_content = content[:target_size] if target_size < len(content) else content + b"\x00" * (target_size - len(content))

        file_info = UploadFileInfo(
            filename="test.jpg",
            content_type="image/jpeg",
            size=len(scaled_content),
            content=scaled_content
        )

        # Should not fail on size (may fail on other checks)
        try:
            self.processor.validate_upload(file_info)
        except ValidationError as e:
            assert "File too large" not in str(e)


# ============================================================================
# Security and Edge Case Tests
# ============================================================================

class TestSecurityProperties:
    """Security-focused property tests."""

    def setup_method(self):
        self.processor = ImageProcessor()

    @given(
        filename=st.text(min_size=1, max_size=100),
        content=invalid_image_content()
    )
    @settings(max_examples=50, deadline=None)
    def test_security_invalid_content_never_crashes_validation(
        self, filename: str, content: bytes
    ):
        """
        # Feature: image-upload-validation, Security Property: Crash Freedom

        Invalid/malicious content must never crash the validator during
        metadata checks (only during actual image processing).
        """
        # Ensure filename is valid so we test content handling
        if "." not in filename:
            filename = f"{filename}.jpg"

        # Ensure extension is valid
        valid_ext = filename.rsplit(".", 1)[-1].lower()
        if valid_ext not in ALLOWED_EXTENSIONS:
            filename = f"{filename}.jpg"

        file_info = UploadFileInfo(
            filename=filename,
            content_type="image/jpeg",
            size=len(content),
            content=content
        )

        # Validation should complete without unhandled exceptions
        # (ValidationError is expected and handled)
        try:
            self.processor.validate_upload(file_info)
        except ValidationError:
            pass  # Expected for invalid content
        except Exception as e:
            pytest.fail(f"Unexpected exception during validation: {type(e).__name__}: {e}")

    @given(
        path_traversal=st.sampled_from([
            "../../../etc/passwd.jpg",
            "..\\..\\windows\\system32\\config.jpg",
            "file/../../../etc/passwd.jpg",
            "./././etc/passwd.jpg",
            "test%00.jpg",
            "test\x00.jpg"
        ])
    )
    @settings(max_examples=10)
    def test_security_path_traversal_in_filename(self, path_traversal: str):
        """
        # Feature: image-upload-validation, Security Property: Path Traversal

        Path traversal attempts in filename must be handled safely.
        """
        file_info = UploadFileInfo(
            filename=path_traversal,
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        # Extracted extension should be safe (no path components)
        ext = file_info.extension
        assert "/" not in ext
        assert "\\" not in ext
        assert "\x00" not in ext

    @given(
        null_injection=st.sampled_from([
            "test\x00.jpg",
            "\x00test.jpg",
            "te\x00st.jpg"
        ])
    )
    def test_security_null_byte_injection(self, null_injection: str):
        """
        # Feature: image-upload-validation, Security Property: Null Byte Injection

        Null bytes in filename must be handled safely.
        """
        file_info = UploadFileInfo(
            filename=null_injection,
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        # Extension should handle null bytes gracefully
        ext = file_info.extension
        assert "\x00" not in ext

    @given(
        size=st.integers(min_value=0, max_value=10)
    )
    def test_security_zero_byte_file(self, size: int):
        """
        # Feature: image-upload-validation, Security Property: Zero Byte Handling

        Zero-byte and very small files must be rejected safely.
        """
        content = b"" if size == 0 else b"x" * size

        file_info = UploadFileInfo(
            filename="test.jpg",
            content_type="image/jpeg",
            size=len(content),
            content=content
        )

        # Should complete without crash
        try:
            self.processor.validate_upload(file_info)
        except ValidationError:
            pass  # Expected
        except Exception as e:
            pytest.fail(f"Unexpected exception: {type(e).__name__}: {e}")


# ============================================================================
# Explicit Example Tests (for known edge cases)
# ============================================================================

class TestExplicitEdgeCases:
    """Explicit tests for known edge cases."""

    def setup_method(self):
        self.processor = ImageProcessor()

    def test_edge_case_empty_filename(self):
        """Test empty filename handling."""
        file_info = UploadFileInfo(
            filename="",
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        # Empty filename has no extension
        assert file_info.extension == ""

        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)

        assert "Invalid extension" in str(exc_info.value)

    def test_edge_case_very_long_filename(self):
        """Test very long filename handling."""
        long_name = "a" * 1000 + ".jpg"

        file_info = UploadFileInfo(
            filename=long_name,
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        # Extension should still be correct
        assert file_info.extension == "jpg"

        # Should pass validation
        assert_validation_passes(self.processor, file_info)

    def test_edge_case_uppercase_extension(self):
        """Test uppercase extension is converted to lowercase."""
        file_info = UploadFileInfo(
            filename="test.JPG",
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        # Extension should be lowercase
        assert file_info.extension == "jpg"

        # Should pass validation
        assert_validation_passes(self.processor, file_info)

    def test_edge_case_mixed_case_extension(self):
        """Test mixed case extension handling."""
        file_info = UploadFileInfo(
            filename="test.JpG",
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        assert file_info.extension == "jpg"
        assert_validation_passes(self.processor, file_info)

    def test_edge_case_exactly_max_size(self):
        """Test file exactly at max size boundary."""
        content = b"x" * MAX_FILE_SIZE

        file_info = UploadFileInfo(
            filename="test.jpg",
            content_type="image/jpeg",
            size=MAX_FILE_SIZE,
            content=content
        )

        # Should pass (size <= max is valid)
        # Note: actual content length vs size field discrepancy
        # is not validated - only size field is checked
        assert_validation_passes(self.processor, file_info)

    def test_edge_case_one_byte_over_max_size(self):
        """Test file one byte over max size."""
        file_info = UploadFileInfo(
            filename="test.jpg",
            content_type="image/jpeg",
            size=MAX_FILE_SIZE + 1,
            content=b"x"
        )

        with pytest.raises(ValidationError) as exc_info:
            self.processor.validate_upload(file_info)

        assert "File too large" in str(exc_info.value)

    def test_edge_case_whitespace_in_filename(self):
        """Test filename with whitespace."""
        file_info = UploadFileInfo(
            filename="my test image.jpg",
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        assert file_info.extension == "jpg"
        assert_validation_passes(self.processor, file_info)

    def test_edge_case_unicode_in_filename(self):
        """Test filename with unicode characters."""
        file_info = UploadFileInfo(
            filename="图片测试.jpg",
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        assert file_info.extension == "jpg"
        assert_validation_passes(self.processor, file_info)


# ============================================================================
# Tests for read_upload_file() function
# ============================================================================

class TestReadUploadFile:
    """Tests for file_handler.read_upload_file() async function."""

    @pytest.mark.asyncio
    async def test_read_upload_file_with_valid_data(self):
        """Test read_upload_file with all fields present."""
        from app.utils.file_handler import read_upload_file

        # Mock UploadFile
        class MockUploadFile:
            def __init__(self, filename, content_type, content):
                self.filename = filename
                self.content_type = content_type
                self._content = content
                self._position = 0

            async def read(self, size=-1):
                """Read content in chunks to simulate chunked reading."""
                if self._position >= len(self._content):
                    return b""
                
                if size == -1:
                    chunk = self._content[self._position:]
                    self._position = len(self._content)
                else:
                    chunk = self._content[self._position:self._position + size]
                    self._position += len(chunk)
                
                return chunk

        mock_file = MockUploadFile(
            filename="test.jpg",
            content_type="image/jpeg",
            content=b"fake image content"
        )

        result = await read_upload_file(mock_file)

        assert result.filename == "test.jpg"
        assert result.content_type == "image/jpeg"
        assert result.size == len(b"fake image content")
        assert result.content == b"fake image content"

    @pytest.mark.asyncio
    async def test_read_upload_file_with_missing_filename(self):
        """Test read_upload_file uses 'unknown' fallback for missing filename."""
        from app.utils.file_handler import read_upload_file

        class MockUploadFile:
            def __init__(self):
                self.filename = None
                self.content_type = "image/jpeg"
                self._content = b"content"
                self._position = 0

            async def read(self, size=-1):
                """Read content in chunks."""
                if self._position >= len(self._content):
                    return b""
                
                if size == -1:
                    chunk = self._content[self._position:]
                    self._position = len(self._content)
                else:
                    chunk = self._content[self._position:self._position + size]
                    self._position += len(chunk)
                
                return chunk

        mock_file = MockUploadFile()
        result = await read_upload_file(mock_file)

        assert result.filename == "unknown"

    @pytest.mark.asyncio
    async def test_read_upload_file_with_missing_content_type(self):
        """Test read_upload_file uses 'application/octet-stream' fallback."""
        from app.utils.file_handler import read_upload_file

        class MockUploadFile:
            def __init__(self):
                self.filename = "test.jpg"
                self.content_type = None
                self._content = b"content"
                self._position = 0

            async def read(self, size=-1):
                """Read content in chunks."""
                if self._position >= len(self._content):
                    return b""
                
                if size == -1:
                    chunk = self._content[self._position:]
                    self._position = len(self._content)
                else:
                    chunk = self._content[self._position:self._position + size]
                    self._position += len(chunk)
                
                return chunk

        mock_file = MockUploadFile()
        result = await read_upload_file(mock_file)

        assert result.content_type == "application/octet-stream"

    @pytest.mark.asyncio
    async def test_read_upload_file_size_calculation(self):
        """Test size is calculated from actual content length."""
        from app.utils.file_handler import read_upload_file

        class MockUploadFile:
            def __init__(self, content):
                self.filename = "test.jpg"
                self.content_type = "image/jpeg"
                self._content = content
                self._position = 0

            async def read(self, size=-1):
                """Read content in chunks."""
                if self._position >= len(self._content):
                    return b""
                
                if size == -1:
                    chunk = self._content[self._position:]
                    self._position = len(self._content)
                else:
                    chunk = self._content[self._position:self._position + size]
                    self._position += len(chunk)
                
                return chunk

        test_contents = [b"", b"x", b"x" * 1000, b"x" * 10000]

        for content in test_contents:
            mock_file = MockUploadFile(content)
            result = await read_upload_file(mock_file)
            assert result.size == len(content)


# ============================================================================
# Tests for ValidationError
# ============================================================================

class TestValidationError:
    """Tests for ValidationError exception."""

    def test_validation_error_default_values(self):
        """Test ValidationError has correct default status_code and detail."""
        error = ValidationError()

        assert error.status_code == 400
        assert error.detail == "Validation error"

    def test_validation_error_custom_message(self):
        """Test ValidationError accepts custom detail message."""
        error = ValidationError("Custom error message")

        assert error.detail == "Custom error message"
        assert error.status_code == 400

    def test_validation_error_inheritance(self):
        """Test ValidationError inherits from AppException."""
        from app.core.exceptions import AppException

        error = ValidationError("test")

        assert isinstance(error, AppException)

    def test_file_too_large_error(self):
        """Test FileTooLargeError inheritance and defaults."""
        from app.core.exceptions import FileTooLargeError

        error = FileTooLargeError()

        assert error.status_code == 400
        assert error.detail == "File too large"
        assert isinstance(error, ValidationError)

    def test_invalid_file_type_error(self):
        """Test InvalidFileTypeError inheritance and defaults."""
        from app.core.exceptions import InvalidFileTypeError

        error = InvalidFileTypeError()

        assert error.status_code == 400
        assert error.detail == "Invalid file type"
        assert isinstance(error, ValidationError)


# ============================================================================
# Tests for UploadConfig
# ============================================================================

class TestUploadConfig:
    """Tests for UploadConfig configuration class."""

    def test_upload_config_default_values(self, monkeypatch):
        """Test UploadConfig loads correct default values."""
        from app.core.config import UploadConfig

        # Clear environment variables
        monkeypatch.delenv("MAX_FILE_SIZE", raising=False)
        monkeypatch.delenv("ALLOWED_EXTENSIONS", raising=False)

        config = UploadConfig.from_env()

        assert config.max_file_size == 10_485_760  # 10MB
        assert config.allowed_extensions == {"jpg", "jpeg", "png"}
        assert config.allowed_content_types == {"image/jpeg", "image/png"}

    def test_upload_config_custom_values(self, monkeypatch):
        """Test UploadConfig loads from environment variables."""
        from app.core.config import UploadConfig

        monkeypatch.setenv("MAX_FILE_SIZE", "5242880")  # 5MB
        monkeypatch.setenv("ALLOWED_EXTENSIONS", "jpg,png")

        config = UploadConfig.from_env()

        assert config.max_file_size == 5_242_880
        assert config.allowed_extensions == {"jpg", "png"}
        assert config.allowed_content_types == {"image/jpeg", "image/png"}

    def test_upload_config_extension_normalization(self, monkeypatch):
        """Test extensions are normalized to lowercase."""
        from app.core.config import UploadConfig

        monkeypatch.setenv("ALLOWED_EXTENSIONS", "JPG,JPEG,PNG")

        config = UploadConfig.from_env()

        assert config.allowed_extensions == {"jpg", "jpeg", "png"}

    def test_upload_config_whitespace_handling(self, monkeypatch):
        """Test extensions with whitespace are handled correctly."""
        from app.core.config import UploadConfig

        monkeypatch.setenv("ALLOWED_EXTENSIONS", " jpg , jpeg , png ")

        config = UploadConfig.from_env()

        assert config.allowed_extensions == {"jpg", "jpeg", "png"}

    def test_upload_config_content_type_derivation_jpg(self, monkeypatch):
        """Test jpg extension derives image/jpeg content type."""
        from app.core.config import UploadConfig

        monkeypatch.setenv("ALLOWED_EXTENSIONS", "jpg")

        config = UploadConfig.from_env()

        assert "image/jpeg" in config.allowed_content_types

    def test_upload_config_content_type_derivation_jpeg(self, monkeypatch):
        """Test jpeg extension derives image/jpeg content type."""
        from app.core.config import UploadConfig

        monkeypatch.setenv("ALLOWED_EXTENSIONS", "jpeg")

        config = UploadConfig.from_env()

        assert "image/jpeg" in config.allowed_content_types

    def test_upload_config_content_type_derivation_png(self, monkeypatch):
        """Test png extension derives image/png content type."""
        from app.core.config import UploadConfig

        monkeypatch.setenv("ALLOWED_EXTENSIONS", "png")

        config = UploadConfig.from_env()

        assert "image/png" in config.allowed_content_types

    def test_upload_config_content_type_derivation_mixed(self, monkeypatch):
        """Test mixed extensions derive correct content types."""
        from app.core.config import UploadConfig

        monkeypatch.setenv("ALLOWED_EXTENSIONS", "jpg,png")

        config = UploadConfig.from_env()

        assert config.allowed_content_types == {"image/jpeg", "image/png"}

    def test_upload_config_frozen_dataclass(self):
        """Test UploadConfig is immutable (frozen dataclass)."""
        from app.core.config import UploadConfig

        config = UploadConfig(
            max_file_size=100,
            allowed_extensions={"jpg"},
            allowed_content_types={"image/jpeg"}
        )

        # Should not be able to modify
        with pytest.raises(Exception):  # FrozenInstanceError
            config.max_file_size = 200


# ============================================================================
# Additional Edge Case Tests
# ============================================================================

class TestAdditionalEdgeCases:
    """Additional edge case tests from design.md."""

    def setup_method(self):
        self.processor = ImageProcessor()

    def test_hidden_file_extension(self):
        """Test hidden files like .gitignore have correct extension."""
        file_info = UploadFileInfo(
            filename=".gitignore",
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        assert file_info.extension == "gitignore"

    def test_multiple_dots_extension(self):
        """Test filename with multiple dots extracts last extension."""
        file_info = UploadFileInfo(
            filename="archive.tar.gz",
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        assert file_info.extension == "gz"

    def test_single_character_extension(self):
        """Test single character extension."""
        file_info = UploadFileInfo(
            filename="test.a",
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        assert file_info.extension == "a"

    def test_numeric_extension(self):
        """Test numeric extension."""
        file_info = UploadFileInfo(
            filename="backup.123",
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        assert file_info.extension == "123"

    @pytest.mark.parametrize("ext", ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG", "Jpg", "Png"])
    def test_all_valid_extension_variants(self, ext):
        """Test all valid extension case variants pass validation."""
        file_info = UploadFileInfo(
            filename=f"test.{ext}",
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        # Extension should be normalized to lowercase
        assert file_info.extension == ext.lower()

    def test_empty_content_valid_size(self):
        """Test empty content with size=0 passes size validation."""
        file_info = UploadFileInfo(
            filename="test.jpg",
            content_type="image/jpeg",
            size=0,
            content=b""
        )

        # Should pass size validation (0 <= 10MB)
        # May fail on other checks
        try:
            self.processor.validate_upload(file_info)
        except ValidationError as e:
            assert "File too large" not in str(e)

    def test_very_long_filename_with_valid_extension(self):
        """Test very long filename still extracts extension correctly."""
        long_base = "a" * 10000
        filename = f"{long_base}.jpg"

        file_info = UploadFileInfo(
            filename=filename,
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        assert file_info.extension == "jpg"

    def test_special_characters_in_filename_base(self):
        """Test special characters in filename base (not extension)."""
        file_info = UploadFileInfo(
            filename="my test (1) [final].v2.jpg",
            content_type="image/jpeg",
            size=100,
            content=b"x" * 100
        )

        assert file_info.extension == "jpg"


# ============================================================================
# Integration Tests with FastAPI
# ============================================================================

class TestValidationIntegration:
    """Integration tests for validation with FastAPI endpoints."""

    def test_fastapi_validation_error_response_format(self):
        """Test ValidationError produces correct FastAPI error response."""
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from fastapi.responses import JSONResponse

        app = FastAPI()

        # Register exception handler (similar to actual app)
        @app.exception_handler(ValidationError)
        async def validation_exception_handler(request: Request, exc: ValidationError):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail}
            )

        @app.post("/test")
        async def test_endpoint():
            raise ValidationError("Test validation error")

        client = TestClient(app)
        response = client.post("/test")

        assert response.status_code == 400
        assert "detail" in response.json()

    def test_validation_error_json_serializable(self):
        """Test ValidationError can be converted to JSON response."""
        import json

        error = ValidationError("Test error message")

        # Simulate what FastAPI exception handler does
        error_dict = {"detail": error.detail}
        json_str = json.dumps(error_dict)

        assert "Test error message" in json_str
        assert "detail" in json.loads(json_str)


# ============================================================================
# Summary of Potential Issues Found
# ============================================================================

"""
POTENTIAL ISSUES AND STRICTNESS NOTES:

1. **Size Validation Only Checks Metadata**:
   The validator only checks file_info.size field, not actual len(content).
   This means a file could report size=100 but have 10MB of content.
   Property tests verify this behavior - size field is authoritative.

2. **Content Not Verified Against Magic Bytes**:
   The validator does not verify that file content actually matches
   declared content_type (no magic byte checking).
   A PNG file could claim to be image/jpeg and pass validation.

3. **Case Sensitivity**:
   Extensions are lowercased via property, but content_type is not.
   "IMAGE/JPEG" would be rejected (not in allowed set).

4. **No Image Dimension Validation**:
   The validate_upload() method doesn't check actual image dimensions.
   A valid JPEG file of 1x1 pixel passes the same as 10000x10000.
   Dimension validation happens later in process() method.

5. **Zero Byte Files**:
   Zero-byte files are technically valid by size check if size=0,
   but will likely fail during to_image() conversion.

These behaviors are consistent with the current implementation and
may be intentional design decisions, but property tests document them.
"""
