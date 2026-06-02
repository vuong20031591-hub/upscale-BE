"""
Image data models.
"""

from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Optional
from PIL import Image


class Resolution(str, Enum):
    """
    Supported output resolutions for image upscaling.
    
    Attributes:
        K2: 2K resolution (2560x1440 pixels)
        K4: 4K resolution (3840x2160 pixels)
        K8: 8K resolution (7680x4320 pixels) - Reserved for future
    
    Note:
        8K resolution is defined but not currently supported in the API.
    """
    K2 = "2k"
    K4 = "4k"
    K8 = "8k"


class ImageFormat(str, Enum):
    """
    Supported image output formats.
    
    Attributes:
        PNG: PNG format (lossless, supports transparency)
        JPEG: JPEG format (lossy compression)
    
    Note:
        Current implementation only supports PNG output.
        JPEG is defined but not used in the API.
    """
    PNG = "png"
    JPEG = "jpeg"


@dataclass
class ProcessedImage:
    """
    Result of image processing with metadata.
    
    Contains the processed image along with dimensional information
    and scale factor for response headers.
    
    Attributes:
        image: Processed PIL Image object
        original_width: Original image width in pixels
        original_height: Original image height in pixels
        final_width: Final image width in pixels after processing
        final_height: Final image height in pixels after processing
        scale_factor: Overall scale factor (average of width and height scales)
        format: Output image format (PNG or JPEG)
    
    Requirements:
        - Requirement 5.5: Calculate overall_scale_factor as average
        - Requirement 5.6: Output format is PNG only
    
    Example:
        >>> result = ProcessedImage(
        ...     image=processed_img,
        ...     original_width=1920,
        ...     original_height=1080,
        ...     final_width=2560,
        ...     final_height=1440,
        ...     scale_factor=1.33,
        ...     format=ImageFormat.PNG
        ... )
        >>> result.resolution
        '2560x1440'
    """
    image: Image.Image
    original_width: int
    original_height: int
    final_width: int
    final_height: int
    scale_factor: float
    format: ImageFormat

    def to_bytes(self, quality: int = 95) -> BytesIO:
        """
        Convert image to bytes buffer.
        
        Args:
            quality: For JPEG: quality level (1-100)
                    For PNG: mapped to compress_level (0-9)
        
        Returns:
            BytesIO buffer with encoded image
        """
        buffer = BytesIO()

        if self.format == ImageFormat.JPEG:
            rgb_image = self.image.convert('RGB')
            rgb_image.save(buffer, format='JPEG', quality=quality, optimize=True)
        else:
            # PNG doesn't use 'quality' parameter, use 'compress_level' instead
            # Map quality (1-100) to compress_level (0-9)
            # quality 95 (default) -> compress_level 9 (best compression)
            compress_level = min(9, max(0, quality // 10))
            self.image.save(buffer, format='PNG', compress_level=compress_level, optimize=True)

        buffer.seek(0)
        return buffer

    @property
    def resolution(self) -> str:
        """
        Get resolution string in format 'widthxheight'.
        
        Returns:
            str: Resolution string (e.g., "2560x1440")
        
        Requirements:
            - Requirement 5.3: X-Image-Resolution header format
        
        Example:
            >>> result.final_width = 2560
            >>> result.final_height = 1440
            >>> result.resolution
            '2560x1440'
        """
        return f"{self.final_width}x{self.final_height}"


@dataclass
class UploadFileInfo:
    """
    Metadata and content of an uploaded file.
    
    Contains file information extracted from FastAPI UploadFile
    for validation and processing.
    
    Attributes:
        filename: Original filename from upload
        content_type: MIME type (e.g., "image/jpeg", "image/png")
        size: File size in bytes
        content: Raw file content as bytes
    
    Requirements:
        - Requirement 1.1: File size validation
        - Requirement 1.2: Content type validation
        - Requirement 1.3: Extension validation
        - Requirement 1.5: RGB conversion
    
    Example:
        >>> file_info = UploadFileInfo(
        ...     filename="photo.jpg",
        ...     content_type="image/jpeg",
        ...     size=5_000_000,
        ...     content=b"\\xff\\xd8\\xff..."
        ... )
        >>> file_info.extension
        'jpg'
        >>> image = file_info.to_image()
        >>> image.mode
        'RGB'
    """
    filename: str
    content_type: str
    size: int
    content: bytes

    @property
    def extension(self) -> str:
        """
        Extract file extension from filename.
        
        Returns:
            str: Lowercase file extension without dot (e.g., "jpg", "png")
                 Empty string if no extension found
        
        Requirements:
            - Requirement 1.3: Extension validation
        
        Example:
            >>> file_info.filename = "photo.JPG"
            >>> file_info.extension
            'jpg'
            >>> file_info.filename = "noext"
            >>> file_info.extension
            ''
        """
        return self.filename.split('.')[-1].lower() if '.' in self.filename else ""

    def to_image(self) -> Image.Image:
        """
        Convert uploaded file content to PIL Image in RGB mode.
        
        Opens the image from bytes and ensures it's in RGB mode.
        Non-RGB images (RGBA, L, CMYK, P, etc.) are automatically converted.
        
        Returns:
            Image.Image: PIL Image in RGB mode
        
        Requirements:
            - Requirement 1.5: Convert to RGB mode if not already
        
        Raises:
            PIL.UnidentifiedImageError: If content is not a valid image
        
        Example:
            >>> file_info = UploadFileInfo(
            ...     filename="photo.png",
            ...     content_type="image/png",
            ...     size=1000,
            ...     content=png_bytes
            ... )
            >>> image = file_info.to_image()
            >>> image.mode
            'RGB'
            >>> isinstance(image, Image.Image)
            True
        """
        image = Image.open(BytesIO(self.content))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image
