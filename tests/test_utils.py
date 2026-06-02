"""
Test utilities for AI Image Upscaling tests.

This module provides:
- Image generation utilities
- Hypothesis strategies for property-based testing
- Helper functions for test assertions
"""

import io
from typing import Tuple, Optional

from PIL import Image
from hypothesis import strategies as st


# ============================================================================
# Image Generation Utilities
# ============================================================================

def generate_test_image(
    width: int,
    height: int,
    mode: str = 'RGB',
    color: Optional[Tuple] = None
) -> Image.Image:
    """
    Generate a test PIL Image with specified parameters.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        mode: PIL image mode (RGB, RGBA, L, CMYK, P, etc.)
        color: Color tuple (auto-selected if None)
    
    Returns:
        PIL Image object
    
    Examples:
        >>> img = generate_test_image(100, 100, 'RGB')
        >>> img.size
        (100, 100)
        >>> img.mode
        'RGB'
    """
    if color is None:
        color = get_default_color_for_mode(mode)
    
    return Image.new(mode, (width, height), color)


def generate_test_image_bytes(
    width: int,
    height: int,
    mode: str = 'RGB',
    format: str = 'PNG',
    color: Optional[Tuple] = None
) -> bytes:
    """
    Generate test image as bytes.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        mode: PIL image mode
        format: Output format (PNG, JPEG)
        color: Color tuple
    
    Returns:
        Image bytes
    
    Examples:
        >>> img_bytes = generate_test_image_bytes(100, 100, 'RGB', 'PNG')
        >>> len(img_bytes) > 0
        True
    """
    image = generate_test_image(width, height, mode, color)
    buffer = io.BytesIO()
    
    # Convert to RGB if saving as JPEG and mode is not compatible
    if format.upper() == 'JPEG' and mode not in ('RGB', 'L'):
        image = image.convert('RGB')
    
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer.read()


def get_default_color_for_mode(mode: str) -> Tuple:
    """
    Get default color for a given PIL image mode.
    
    Args:
        mode: PIL image mode
    
    Returns:
        Color tuple appropriate for the mode
    
    Examples:
        >>> get_default_color_for_mode('RGB')
        (255, 0, 0)
        >>> get_default_color_for_mode('L')
        255
    """
    color_map = {
        'RGB': (255, 0, 0),      # Red
        'RGBA': (255, 0, 0, 255), # Red with full opacity
        'L': 255,                 # White
        'CMYK': (0, 100, 100, 0), # Red in CMYK
        'P': 255,                 # White (palette mode)
        '1': 1,                   # White (binary)
    }
    return color_map.get(mode, (255, 0, 0))


def image_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
    """
    Convert PIL Image to bytes.
    
    Args:
        image: PIL Image object
        format: Output format (PNG, JPEG)
    
    Returns:
        Image bytes
    """
    buffer = io.BytesIO()
    
    # Convert to RGB if saving as JPEG and mode is not compatible
    if format.upper() == 'JPEG' and image.mode not in ('RGB', 'L'):
        image = image.convert('RGB')
    
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer.read()


# ============================================================================
# Aspect Ratio Utilities
# ============================================================================

def calculate_aspect_ratio(width: int, height: int) -> float:
    """
    Calculate aspect ratio (width / height).
    
    Args:
        width: Image width
        height: Image height
    
    Returns:
        Aspect ratio as float
    
    Examples:
        >>> calculate_aspect_ratio(1920, 1080)
        1.7777777777777777
        >>> calculate_aspect_ratio(100, 100)
        1.0
    """
    return width / height


def aspect_ratios_equal(ratio1: float, ratio2: float, tolerance: float = 0.01) -> bool:
    """
    Check if two aspect ratios are equal within tolerance.
    
    Args:
        ratio1: First aspect ratio
        ratio2: Second aspect ratio
        tolerance: Acceptable difference (default: 0.01)
    
    Returns:
        True if ratios are equal within tolerance
    
    Examples:
        >>> aspect_ratios_equal(1.777, 1.778, 0.01)
        True
        >>> aspect_ratios_equal(1.5, 2.0, 0.01)
        False
    """
    return abs(ratio1 - ratio2) < tolerance


def verify_aspect_ratio_preserved(
    original_width: int,
    original_height: int,
    final_width: int,
    final_height: int,
    tolerance: float = 0.01
) -> bool:
    """
    Verify that aspect ratio is preserved between original and final dimensions.
    
    Args:
        original_width: Original image width
        original_height: Original image height
        final_width: Final image width
        final_height: Final image height
        tolerance: Acceptable difference (default: 0.01)
    
    Returns:
        True if aspect ratio is preserved
    
    Examples:
        >>> verify_aspect_ratio_preserved(100, 100, 200, 200)
        True
        >>> verify_aspect_ratio_preserved(1920, 1080, 3840, 2160)
        True
        >>> verify_aspect_ratio_preserved(100, 100, 200, 300)
        False
    """
    original_ratio = calculate_aspect_ratio(original_width, original_height)
    final_ratio = calculate_aspect_ratio(final_width, final_height)
    return aspect_ratios_equal(original_ratio, final_ratio, tolerance)


# ============================================================================
# Scale Factor Utilities
# ============================================================================

def calculate_overall_scale_factor(
    original_width: int,
    original_height: int,
    final_width: int,
    final_height: int
) -> float:
    """
    Calculate overall scale factor as average of width and height scales.
    
    Formula: (final_width/original_width + final_height/original_height) / 2
    
    Args:
        original_width: Original image width
        original_height: Original image height
        final_width: Final image width
        final_height: Final image height
    
    Returns:
        Overall scale factor
    
    Examples:
        >>> calculate_overall_scale_factor(100, 100, 400, 400)
        4.0
        >>> calculate_overall_scale_factor(100, 100, 200, 200)
        2.0
    """
    scale_w = final_width / original_width
    scale_h = final_height / original_height
    return (scale_w + scale_h) / 2


def calculate_resize_scale(
    current_width: int,
    current_height: int,
    target_width: int,
    target_height: int
) -> float:
    """
    Calculate resize scale for fit-within operation.
    
    Formula: min(target_width/current_width, target_height/current_height)
    
    Args:
        current_width: Current image width
        current_height: Current image height
        target_width: Target width
        target_height: Target height
    
    Returns:
        Resize scale factor
    
    Examples:
        >>> calculate_resize_scale(1000, 1000, 2560, 1440)
        1.44
        >>> calculate_resize_scale(4000, 3000, 2560, 1440)
        0.48
    """
    scale_w = target_width / current_width
    scale_h = target_height / current_height
    return min(scale_w, scale_h)


# ============================================================================
# Hypothesis Strategies
# ============================================================================

# Image dimension strategies
image_width_strategy = st.integers(min_value=100, max_value=4000)
image_height_strategy = st.integers(min_value=100, max_value=4000)

# Small dimensions for faster tests
small_width_strategy = st.integers(min_value=50, max_value=500)
small_height_strategy = st.integers(min_value=50, max_value=500)

# Large dimensions (larger than target resolutions)
large_width_strategy = st.integers(min_value=3000, max_value=8000)
large_height_strategy = st.integers(min_value=2000, max_value=5000)

# PIL image mode strategy
image_mode_strategy = st.sampled_from(['RGB', 'RGBA', 'L', 'CMYK', 'P'])

# Common image modes (excluding problematic ones)
common_mode_strategy = st.sampled_from(['RGB', 'RGBA', 'L'])

# Image format strategy
image_format_strategy = st.sampled_from(['PNG', 'JPEG'])

# Resolution strategy
resolution_strategy = st.sampled_from(['2k', '4k'])

# Invalid resolution strategy (for testing fallback)
invalid_resolution_strategy = st.text(
    alphabet=st.characters(blacklist_categories=('Cs',)),
    min_size=1,
    max_size=20
).filter(lambda x: x.lower() not in ['2k', '4k'])

# File size strategy
file_size_strategy = st.integers(min_value=1, max_value=15 * 1024 * 1024)  # 1 byte to 15 MB

# Valid file size strategy (under 10 MiB)
valid_file_size_strategy = st.integers(min_value=1, max_value=10 * 1024 * 1024)

# Invalid file size strategy (over 10 MiB)
invalid_file_size_strategy = st.integers(min_value=10 * 1024 * 1024 + 1, max_value=50 * 1024 * 1024)

# Content type strategy
valid_content_type_strategy = st.sampled_from(['image/jpeg', 'image/png'])
invalid_content_type_strategy = st.sampled_from([
    'image/gif',
    'image/bmp',
    'image/webp',
    'application/pdf',
    'text/plain',
    'video/mp4'
])

# File extension strategy
valid_extension_strategy = st.sampled_from(['jpg', 'jpeg', 'png'])
invalid_extension_strategy = st.sampled_from([
    'gif',
    'bmp',
    'webp',
    'pdf',
    'txt',
    'mp4',
    'svg'
])

# Filename strategy
filename_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_-'),
    min_size=1,
    max_size=50
)


# ============================================================================
# Composite Strategies
# ============================================================================

@st.composite
def image_dimensions_strategy(draw):
    """
    Composite strategy for generating image dimensions.
    
    Returns:
        Tuple of (width, height)
    """
    width = draw(image_width_strategy)
    height = draw(image_height_strategy)
    return (width, height)


@st.composite
def small_image_dimensions_strategy(draw):
    """
    Composite strategy for generating small image dimensions.
    
    Returns:
        Tuple of (width, height) smaller than 2K resolution
    """
    width = draw(st.integers(min_value=100, max_value=2000))
    height = draw(st.integers(min_value=100, max_value=1200))
    return (width, height)


@st.composite
def large_image_dimensions_strategy(draw):
    """
    Composite strategy for generating large image dimensions.
    
    Returns:
        Tuple of (width, height) larger than 4K resolution
    """
    width = draw(st.integers(min_value=4000, max_value=8000))
    height = draw(st.integers(min_value=2500, max_value=5000))
    return (width, height)


@st.composite
def pil_image_strategy(draw):
    """
    Composite strategy for generating test PIL Images.
    
    Returns:
        PIL Image object
    """
    width = draw(small_width_strategy)
    height = draw(small_height_strategy)
    mode = draw(common_mode_strategy)
    
    return generate_test_image(width, height, mode)


@st.composite
def upload_file_info_strategy(draw):
    """
    Composite strategy for generating UploadFileInfo objects.
    
    Returns:
        UploadFileInfo object
    """
    from app.models.image import UploadFileInfo
    
    width = draw(small_width_strategy)
    height = draw(small_height_strategy)
    mode = draw(common_mode_strategy)
    format = draw(image_format_strategy)
    
    # Generate filename
    name = draw(filename_strategy)
    ext = 'png' if format == 'PNG' else 'jpg'
    filename = f"{name}.{ext}"
    
    # Generate content
    content = generate_test_image_bytes(width, height, mode, format)
    
    # Determine content type
    content_type = 'image/png' if format == 'PNG' else 'image/jpeg'
    
    return UploadFileInfo(
        filename=filename,
        content_type=content_type,
        size=len(content),
        content=content
    )


# ============================================================================
# Assertion Helpers
# ============================================================================

def assert_valid_png_bytes(data: bytes) -> None:
    """
    Assert that bytes represent a valid PNG image.
    
    Args:
        data: Image bytes to validate
    
    Raises:
        AssertionError: If data is not valid PNG
    """
    assert data.startswith(b'\x89PNG\r\n\x1a\n'), "Data is not a valid PNG"
    
    # Try to load as PIL Image
    buffer = io.BytesIO(data)
    image = Image.open(buffer)
    assert image.format == 'PNG', f"Image format is {image.format}, expected PNG"


def assert_valid_image_dimensions(width: int, height: int) -> None:
    """
    Assert that image dimensions are valid (positive integers).
    
    Args:
        width: Image width
        height: Image height
    
    Raises:
        AssertionError: If dimensions are invalid
    """
    assert isinstance(width, int), f"Width must be int, got {type(width)}"
    assert isinstance(height, int), f"Height must be int, got {type(height)}"
    assert width > 0, f"Width must be positive, got {width}"
    assert height > 0, f"Height must be positive, got {height}"


def assert_dimensions_within_target(
    width: int,
    height: int,
    target_width: int,
    target_height: int
) -> None:
    """
    Assert that dimensions are within target resolution.
    
    Args:
        width: Image width
        height: Image height
        target_width: Target width
        target_height: Target height
    
    Raises:
        AssertionError: If dimensions exceed target
    """
    assert width <= target_width, f"Width {width} exceeds target {target_width}"
    assert height <= target_height, f"Height {height} exceeds target {target_height}"
