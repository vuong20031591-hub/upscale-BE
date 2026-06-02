"""
Application configuration loaded from environment variables.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set


@dataclass(frozen=True)
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    rate_limit_per_minute: int = 10
    trusted_proxies: List[str] = None  # NEW: Trusted proxy IPs for X-Forwarded-For

    @classmethod
    def from_env(cls) -> "APIConfig":
        # Parse rate limit with validation
        rate_limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
        if rate_limit < 1:
            raise ValueError(f"RATE_LIMIT_PER_MINUTE must be >= 1, got {rate_limit}")
        
        # Parse trusted proxies
        proxies_str = os.getenv("TRUSTED_PROXIES", "")
        trusted_proxies = [p.strip() for p in proxies_str.split(",") if p.strip()] if proxies_str else []
        
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            debug=os.getenv("API_DEBUG", "false").lower() == "true",
            rate_limit_per_minute=rate_limit,
            trusted_proxies=trusted_proxies
        )


@dataclass(frozen=True)
class CORSConfig:
    """CORS configuration."""
    origins: List[str]

    @classmethod
    def from_env(cls) -> "CORSConfig":
        origins_str = os.getenv("CORS_ORIGINS", "http://localhost:3000")
        return cls(origins=[o.strip() for o in origins_str.split(",")])


@dataclass(frozen=True)
class UploadConfig:
    """File upload configuration."""
    max_file_size: int
    allowed_extensions: Set[str]
    allowed_content_types: Set[str]

    @classmethod
    def from_env(cls) -> "UploadConfig":
        ext_str = os.getenv("ALLOWED_EXTENSIONS", "jpg,jpeg,png")
        extensions = {e.strip().lower() for e in ext_str.split(",")}

        content_types = set()
        for ext in extensions:
            if ext in ("jpg", "jpeg"):
                content_types.add("image/jpeg")
            elif ext == "png":
                content_types.add("image/png")

        max_size = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))

        return cls(
            max_file_size=max_size,
            allowed_extensions=extensions,
            allowed_content_types=content_types
        )


@dataclass(frozen=True)
class ModelConfig:
    """
    AI Model configuration loaded from environment variables.
    
    Immutable configuration for Real-ESRGAN model settings.
    
    Attributes:
        name: Model name (default: "RealESRGAN_x4plus")
        scale: Scale factor for upscaling (default: 4)
        path: Directory path for model weights (default: "weights/")
        tile_size: Tile size for processing large images (0 = no tiling)
        half_precision: Enable FP16 mode for GPU memory optimization
        codeformer_enabled: Enable CodeFormer face restoration
        codeformer_weight: CodeFormer fidelity weight (0-1, higher = more original detail)
        codeformer_face_upsample: Enable face upsampling in CodeFormer
        gpu_cache_clear_frequency: Clear GPU cache every N faces (1 = after each face, 0 = never)

    Environment Variables:
        - MODEL_NAME: Model name
        - MODEL_SCALE: Scale factor (integer)
        - MODEL_PATH: Weights directory path
        - TILE_SIZE: Tile size for processing (0 = disabled)
        - HALF_PRECISION: "true" or "false" for FP16 mode
        - CODEFORMER_ENABLED: "true" or "false" to enable face restoration
        - CODEFORMER_WEIGHT: Fidelity weight (default: 0.7)
        - CODEFORMER_FACE_UPSAMPLE: "true" or "false" for face upsampling
        - GPU_CACHE_CLEAR_FREQUENCY: Clear GPU cache every N faces (default: 1)
    
    Requirements:
        - Requirement 2.6: RRDBNet architecture parameters
        - Requirement 2.7: Half precision configuration
        - Requirement 6.2: Model configuration from env
    
    Example:
        >>> config = ModelConfig.from_env()
        >>> config.name
        'RealESRGAN_x4plus'
        >>> config.model_file
        PosixPath('weights/RealESRGAN_x4plus.pth')
        >>> config.model_url
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    """
    name: str
    scale: int
    path: Path
    tile_size: int
    half_precision: bool
    codeformer_enabled: bool
    codeformer_weight: float
    codeformer_face_upsample: bool
    gpu_cache_clear_frequency: int

    @classmethod
    def from_env(cls) -> "ModelConfig":
        path = Path(os.getenv("MODEL_PATH", "weights"))
        path.mkdir(parents=True, exist_ok=True)

        # Parse CodeFormer weight with validation
        weight_str = os.getenv("CODEFORMER_WEIGHT", "0.7")
        try:
            weight = float(weight_str)
            weight = max(0.0, min(1.0, weight))  # Clamp to [0, 1]
        except ValueError:
            weight = 0.7

        # Parse GPU cache clear frequency with validation
        cache_freq_str = os.getenv("GPU_CACHE_CLEAR_FREQUENCY", "1")
        try:
            cache_freq = int(cache_freq_str)
            cache_freq = max(0, cache_freq)  # Must be >= 0
        except ValueError:
            cache_freq = 1

        return cls(
            name=os.getenv("MODEL_NAME", "RealESRGAN_x4plus"),
            scale=int(os.getenv("MODEL_SCALE", "4")),
            path=path,
            tile_size=int(os.getenv("TILE_SIZE", "0")),
            half_precision=os.getenv("HALF_PRECISION", "true").lower() == "true",
            codeformer_enabled=os.getenv("CODEFORMER_ENABLED", "true").lower() == "true",
            codeformer_weight=weight,
            codeformer_face_upsample=os.getenv("CODEFORMER_FACE_UPSAMPLE", "true").lower() == "true",
            gpu_cache_clear_frequency=cache_freq
        )

    @property
    def model_file(self) -> Path:
        """
        Get full path to model weights file.
        
        Returns:
            Path: Full path to .pth file (e.g., weights/RealESRGAN_x4plus.pth)
        
        Example:
            >>> config.model_file
            PosixPath('weights/RealESRGAN_x4plus.pth')
        """
        return self.path / f"{self.name}.pth"

    @property
    def model_url(self) -> str:
        """
        Get GitHub download URL for model weights.
        
        Returns:
            str: Full URL to download .pth file from GitHub releases
        
        Requirements:
            - Requirement 2.4: Auto-download from GitHub
        
        Example:
            >>> config.model_url
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        """
        base_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0"
        return f"{base_url}/{self.name}.pth"

    @property
    def codeformer_model_file(self) -> Path:
        """
        Get full path to CodeFormer model weights file.

        Returns:
            Path: Full path to codeformer.pth file

        Example:
            >>> config.codeformer_model_file
            PosixPath('weights/codeformer.pth')
        """
        return self.path / "codeformer.pth"

    @property
    def codeformer_model_url(self) -> str:
        """
        Get GitHub download URL for CodeFormer model weights.

        Returns:
            str: Full URL to download codeformer.pth from GitHub releases

        Example:
            >>> config.codeformer_model_url
            'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
        """
        return "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"


@dataclass(frozen=True)
class OutputConfig:
    """Output image configuration."""
    default_resolution: str
    supported_resolutions: List[str]
    format: str
    quality: int

    # Resolution mapping for standard upscaling
    RESOLUTION_MAP = {
        "2k": (2560, 1440),
        "4k": (3840, 2160),
        "8k": (7680, 4320)
    }

    def __post_init__(self):
        """Validate configuration values."""
        # Validate OUTPUT_QUALITY is between 1 and 100
        if not 1 <= self.quality <= 100:
            raise ValueError(f"OUTPUT_QUALITY must be between 1 and 100, got {self.quality}")

    @classmethod
    def from_env(cls) -> "OutputConfig":
        res_str = os.getenv("SUPPORTED_RESOLUTIONS", "2k,4k")
        supported = [r.strip().lower() for r in res_str.split(",")]
        
        quality_str = os.getenv("OUTPUT_QUALITY", "95")
        try:
            quality = int(quality_str)
        except ValueError:
            raise ValueError(f"OUTPUT_QUALITY must be an integer, got '{quality_str}'")

        return cls(
            default_resolution=os.getenv("DEFAULT_TARGET_RESOLUTION", "2k"),
            supported_resolutions=supported,
            format=os.getenv("OUTPUT_FORMAT", "png"),
            quality=quality
        )

    def validate_resolution(self, resolution: str) -> None:
        """
        Validate resolution against supported resolutions.
        
        Args:
            resolution: Target resolution string (e.g., "2k", "4k")
        
        Raises:
            ValueError: If resolution is not in supported_resolutions list
        """
        resolution_lower = resolution.lower()
        if resolution_lower not in self.supported_resolutions:
            supported_str = ", ".join(self.supported_resolutions)
            raise ValueError(
                f"Invalid resolution '{resolution}'. Supported: {supported_str}"
            )

    def get_dimensions(self, resolution: str) -> tuple[int, int]:
        """
        Get dimensions for a resolution with strict validation.
        
        Args:
            resolution: Target resolution string (e.g., "2k", "4k")
        
        Returns:
            Tuple of (width, height) in pixels
        
        Raises:
            ValueError: If resolution is not in RESOLUTION_MAP
        """
        resolution_lower = resolution.lower()
        if resolution_lower not in self.RESOLUTION_MAP:
            available = ", ".join(self.RESOLUTION_MAP.keys())
            raise ValueError(
                f"Resolution '{resolution}' not found in RESOLUTION_MAP. "
                f"Available: {available}"
            )
        return self.RESOLUTION_MAP[resolution_lower]


@dataclass(frozen=True)
class SmartDetectionConfig:
    """
    Smart auto-detection configuration for image analysis.
    
    These thresholds control the sensitivity of detection algorithms.
    All values are loaded from environment variables with sensible defaults.
    
    Attributes:
        grayscale_tolerance: Tolerance for RGB channel comparison (default: 1e-5)
        grayscale_confidence_threshold: Min confidence to suggest colorization (default: 0.9)
        white_mask_threshold: RGB threshold for white pixels 0-255 (default: 240)
        white_mask_percentage: Min white percentage to suggest inpainting (default: 0.30)
        low_res_threshold: Min dimension in pixels (default: 512)
        blur_variance_threshold: Max Laplacian variance for blur (default: 100.0)
        analysis_max_size: Max image size for analysis optimization (default: 1024)
    
    Environment Variables:
        - SMART_GRAYSCALE_TOLERANCE: Grayscale detection tolerance
        - SMART_GRAYSCALE_CONFIDENCE_THRESHOLD: Colorization confidence threshold
        - SMART_WHITE_MASK_THRESHOLD: White pixel RGB threshold
        - SMART_WHITE_MASK_PERCENTAGE: White mask percentage threshold
        - SMART_LOW_RES_THRESHOLD: Low resolution dimension threshold
        - SMART_BLUR_VARIANCE_THRESHOLD: Blur variance threshold
        - SMART_ANALYSIS_MAX_SIZE: Max size for analysis optimization
    
    Requirements:
        - Requirement 6.1-6.8: Detection algorithm thresholds
        - Requirement 7.1: Performance optimization (<200ms analysis)
    
    Example:
        >>> config = SmartDetectionConfig.from_env()
        >>> config.grayscale_tolerance
        1e-05
        >>> config.white_mask_percentage
        0.3
    """
    grayscale_tolerance: float
    grayscale_confidence_threshold: float
    white_mask_threshold: int
    white_mask_percentage: float
    low_res_threshold: int
    blur_variance_threshold: float
    analysis_max_size: int

    @classmethod
    def from_env(cls) -> "SmartDetectionConfig":
        """Load smart detection configuration from environment variables."""
        # Parse grayscale tolerance
        tolerance_str = os.getenv("SMART_GRAYSCALE_TOLERANCE", "1e-5")
        try:
            tolerance = float(tolerance_str)
            tolerance = max(0.0, tolerance)  # Must be >= 0
        except ValueError:
            tolerance = 1e-5

        # Parse grayscale confidence threshold
        conf_str = os.getenv("SMART_GRAYSCALE_CONFIDENCE_THRESHOLD", "0.9")
        try:
            conf_threshold = float(conf_str)
            conf_threshold = max(0.0, min(1.0, conf_threshold))  # Clamp to [0, 1]
        except ValueError:
            conf_threshold = 0.9

        # Parse white mask threshold
        white_threshold_str = os.getenv("SMART_WHITE_MASK_THRESHOLD", "240")
        try:
            white_threshold = int(white_threshold_str)
            white_threshold = max(0, min(255, white_threshold))  # Clamp to [0, 255]
        except ValueError:
            white_threshold = 240

        # Parse white mask percentage
        white_pct_str = os.getenv("SMART_WHITE_MASK_PERCENTAGE", "0.30")
        try:
            white_pct = float(white_pct_str)
            white_pct = max(0.0, min(1.0, white_pct))  # Clamp to [0, 1]
        except ValueError:
            white_pct = 0.30

        # Parse low resolution threshold
        low_res_str = os.getenv("SMART_LOW_RES_THRESHOLD", "512")
        try:
            low_res = int(low_res_str)
            low_res = max(1, low_res)  # Must be >= 1
        except ValueError:
            low_res = 512

        # Parse blur variance threshold
        blur_str = os.getenv("SMART_BLUR_VARIANCE_THRESHOLD", "100.0")
        try:
            blur_threshold = float(blur_str)
            blur_threshold = max(0.0, blur_threshold)  # Must be >= 0
        except ValueError:
            blur_threshold = 100.0

        # Parse analysis max size
        max_size_str = os.getenv("SMART_ANALYSIS_MAX_SIZE", "1024")
        try:
            max_size = int(max_size_str)
            max_size = max(256, max_size)  # Must be >= 256
        except ValueError:
            max_size = 1024

        return cls(
            grayscale_tolerance=tolerance,
            grayscale_confidence_threshold=conf_threshold,
            white_mask_threshold=white_threshold,
            white_mask_percentage=white_pct,
            low_res_threshold=low_res,
            blur_variance_threshold=blur_threshold,
            analysis_max_size=max_size
        )


class Settings:
    """Application settings singleton."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize all configuration and set up PIL security settings."""
        self.api = APIConfig.from_env()
        self.cors = CORSConfig.from_env()
        self.upload = UploadConfig.from_env()
        self.model = ModelConfig.from_env()
        self.output = OutputConfig.from_env()
        self.smart_detection = SmartDetectionConfig.from_env()
        
        # Configure PIL decompression bomb protection
        # PIL default is 89,478,485 pixels
        # We ensure it's set (not None) for security
        from PIL import Image
        if Image.MAX_IMAGE_PIXELS is None:
            Image.MAX_IMAGE_PIXELS = 89478485


settings = Settings()
