"""
AI Model management service.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from app.core import settings, ModelNotFoundError, ImageProcessingError
from app.utils.logging_utils import get_structured_logger

logging.basicConfig(level=logging.INFO)
logger = get_structured_logger(__name__)


class ModelManager:
    """
    Singleton manager for AI upscaling models.
    Handles model loading, caching, and inference.
    """
    _instance: Optional["ModelManager"] = None
    _model: Optional[object] = None
    _lock = threading.Lock()  # ⚡ FIX: Add thread-safe lock

    def __new__(cls) -> "ModelManager":
        """
        Implement Singleton pattern for ModelManager.
        
        Ensures only one instance of ModelManager exists throughout the application.
        This is critical for efficient memory usage as the AI model is large (~17MB).
        
        Returns:
            ModelManager: The singleton instance
        
        Thread Safety:
            Uses double-check locking pattern with threading.Lock for thread-safe
            singleton creation and model loading.
        """
        # ⚡ FIX: Thread-safe singleton with double-check locking
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        """
        Check if AI model is loaded in memory.
        
        This is a read-only property with no side effects - it does NOT trigger
        model loading. Use load() method explicitly to load the model.
        
        Returns:
            bool: True if model is loaded, False otherwise
        
        Requirements:
            - Requirement 2.2: Lazy loading - model not loaded at startup
            - Requirement 2.3: Model loaded on first upscale request
        """
        return self._model is not None

    def load(self) -> None:
        """
        Load the AI model into memory (lazy loading, idempotent).
        
        This method implements lazy loading - the model is only loaded when first needed,
        not at application startup. Subsequent calls are no-ops (idempotent).
        
        Workflow:
            1. Check if model is already loaded (return early if yes)
            2. Download model weights from GitHub if not present locally
            3. Initialize RRDBNet architecture with Real-ESRGAN parameters
            4. Load weights into RealESRGANer inference engine
            5. Log success with timing metrics
        
        Raises:
            ModelNotFoundError: If model download or loading fails
        
        Requirements:
            - Requirement 2.2: Lazy loading (not loaded at startup)
            - Requirement 2.4: Auto-download from GitHub if missing
            - Requirement 2.5: Save to weights/ directory
            - Requirement 2.6: Use RRDBNet architecture with specific parameters
            - Requirement 2.7: Enable FP16 mode if HALF_PRECISION=true
            - Requirement 2.8: Cache model for subsequent requests (idempotent)
            - Requirement 7.5: Log successful load with timing
            - Requirement 7.6: Log errors with context
        
        Performance:
            - First call: ~3-5 seconds (model loading)
            - Subsequent calls: <1ms (cached, no-op)
        
        Thread Safety:
            Uses class-level lock to prevent concurrent model loading.
        """
        # ⚡ FIX: Thread-safe model loading with double-check locking
        if self.is_loaded:
            return
        
        with self._lock:
            # Double-check after acquiring lock
            if self.is_loaded:
                return

            start_time = time.time()
            model_name = settings.model.name
            model_path = settings.model.model_file
            
            try:
                logger.info(
                    "Loading AI model",
                    model_name=model_name,
                    model_path=str(model_path),
                    half_precision=settings.model.half_precision
                )
                
                self._download_if_needed()
                self._load_model()
                
                # Log successful load with timing (Requirement 7.5)
                duration = time.time() - start_time
                logger.info(
                    f"Model '{model_name}' loaded successfully",
                    model_name=model_name,
                    duration_seconds=round(duration, 3),
                    half_precision=settings.model.half_precision,
                    scale=settings.model.scale
                )
                
            except Exception as e:
                # Log error with context (Requirement 7.6)
                duration = time.time() - start_time
                logger.error(
                    f"Failed to load model '{model_name}'",
                    error=e,
                    model_name=model_name,
                    model_path=str(model_path),
                    duration_seconds=round(duration, 3),
                    exc_info=True
                )
                raise ModelNotFoundError(
                    f"Could not load model '{model_name}' from {model_path}: {e}"
                )

    def _download_if_needed(self) -> None:
        """
        Download model weights from GitHub if not present locally.
        
        Checks if model file exists at MODEL_PATH. If not, downloads from GitHub
        releases using torch.hub.download_url_to_file with progress bar.
        
        File Location:
            - Default: weights/RealESRGAN_x4plus.pth
            - Configurable via MODEL_PATH and MODEL_NAME env vars
        
        Download Source:
            - GitHub: xinntao/Real-ESRGAN releases v0.1.0
            - File size: ~17MB
        
        Requirements:
            - Requirement 2.4: Auto-download if model file not exists
            - Requirement 2.5: Save to weights/ directory
        
        Raises:
            Exception: If download fails (network error, disk space, etc.)
        """
        model_file = settings.model.model_file

        if model_file.exists():
            logger.info(
                "Model file found",
                model_file=str(model_file),
                file_size_mb=round(model_file.stat().st_size / (1024 * 1024), 2)
            )
            return

        logger.info(
            "Downloading model",
            url=settings.model.model_url,
            destination=str(model_file)
        )
        
        start_time = time.time()
        torch.hub.download_url_to_file(
            settings.model.model_url,
            str(model_file),
            progress=True
        )
        
        duration = time.time() - start_time
        logger.info(
            "Model download complete",
            model_file=str(model_file),
            file_size_mb=round(model_file.stat().st_size / (1024 * 1024), 2),
            duration_seconds=round(duration, 3)
        )

    def _load_model(self) -> None:
        """
        Initialize RRDBNet architecture and load model weights.
        
        ⚡ OPTIMIZED: Auto-detect device (CUDA/CPU) and configure optimal settings
        
        Creates the Real-ESRGAN model with specific architecture parameters:
        - RRDBNet: Residual in Residual Dense Block Network
        - 3 input/output channels (RGB)
        - 64 feature channels, 23 residual blocks
        - 32 growth channels, 4x scale factor
        
        Then initializes RealESRGANer inference engine with:
        - Model weights from .pth file
        - Auto-detected device (CUDA if available, else CPU)
        - FP16 half precision (only for CUDA to avoid CPU crash)
        - Adaptive tile processing (512 for GPU, 256 for CPU)
        
        Requirements:
            - Requirement 2.6: Use RRDBNet with specific parameters
            - Requirement 2.7: Enable FP16 if HALF_PRECISION=true AND CUDA available
        
        Raises:
            Exception: If model architecture initialization or weight loading fails
        """
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        # ⚡ Auto-detect device (CUDA/CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_half_precision = settings.model.half_precision and device.type == 'cuda'
        
        # ⚡ Adaptive tile size based on device
        # GPU: 512px tiles (balance between speed and VRAM)
        # CPU: 256px tiles (reduce memory usage)
        if settings.model.tile_size > 0:
            tile_size = settings.model.tile_size
        else:
            tile_size = 512 if device.type == 'cuda' else 256
        
        # ⚡ Adaptive tile padding based on device
        # Reduced from 10 to 5 for GPU to minimize overhead
        tile_pad = 5
        
        logger.info(
            "Model configuration",
            device=device.type,
            half_precision=use_half_precision,
            tile_size=tile_size,
            tile_pad=tile_pad
        )

        # Initialize RRDBNet architecture
        model_arch = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=settings.model.scale
        )

        # ⚡ Initialize RealESRGANer with optimized settings
        self._model = RealESRGANer(
            scale=settings.model.scale,
            model_path=str(settings.model.model_file),
            model=model_arch,
            tile=tile_size,
            tile_pad=tile_pad,
            pre_pad=0,
            half=use_half_precision,
            device=device
        )
        
        # ⚡ TORCH 2.0 OPTIMIZATION: channels_last memory format for better GPU performance
        # Note: torch.compile not supported on Windows yet
        # Source: https://github.com/xinntao/Real-ESRGAN/issues/665
        if torch.__version__[0] >= '2' and device.type == 'cuda':
            try:
                logger.info("Applying Torch 2.0 optimizations (channels_last)")
                self._model.model = self._model.model.to(memory_format=torch.channels_last)
                
                # torch.compile only works on Linux/macOS
                import platform
                if platform.system() != 'Windows':
                    logger.info("Applying torch.compile optimization")
                    self._model.model = torch.compile(self._model.model, mode='reduce-overhead')
            except Exception as e:
                logger.warning(f"Could not apply Torch 2.0 optimizations: {e}")

    def upscale(self, image: Image.Image, outscale: int = 4) -> Image.Image:
        """
        Upscale an image using the AI model (Real-ESRGAN).
        
        Automatically loads the model if not already loaded (lazy loading).
        Converts PIL Image to numpy array, runs AI inference, then converts back.
        
        Args:
            image: Input PIL Image (any mode, will be converted internally)
            outscale: Output scale factor (default: 4x)
        
        Returns:
            PIL Image: Upscaled image (4x larger dimensions)
        
        Raises:
            ImageProcessingError: If inference fails (CUDA OOM, model error, etc.)
        
        Requirements:
            - Requirement 3.1: Use Real-ESRGAN with outscale=4
            - Requirement 3.2: Convert PIL to numpy before inference
            - Requirement 3.3: Convert numpy output back to PIL
            - Requirement 3.4: Raise ImageProcessingError on failure
            - Requirement 7.4: Log processing metrics
            - Requirement 7.6: Log errors with context
        
        Performance:
            - Typical: 5-10 seconds for 1920x1080 image on GPU
            - Memory: ~2GB GPU memory for FP16 mode
        
        Example:
            >>> manager = ModelManager()
            >>> input_img = Image.open("photo.jpg")  # 1920x1080
            >>> output_img = manager.upscale(input_img, outscale=4)
            >>> output_img.size  # (7680, 4320) - 4x larger
        """
        if not self.is_loaded:
            self.load()

        width, height = image.size
        start_time = time.time()
        
        # Log inference start (Requirement 7.4)
        logger.info(
            "Starting AI inference",
            input_width=width,
            input_height=height,
            outscale=outscale,
            model=settings.model.name
        )

        try:
            # Convert PIL to numpy array
            img_array = np.array(image)

            # Run inference
            output, _ = self._model.enhance(img_array, outscale=outscale)

            # Convert back to PIL
            result = Image.fromarray(output)
            
            # Log inference success with metrics (Requirement 7.4)
            duration = time.time() - start_time
            output_width, output_height = result.size
            logger.info(
                "AI inference complete",
                input_width=width,
                input_height=height,
                output_width=output_width,
                output_height=output_height,
                outscale=outscale,
                duration_seconds=round(duration, 3)
            )
            
            return result

        except Exception as e:
            # Log inference error with context (Requirement 7.6)
            duration = time.time() - start_time
            logger.error(
                "Model inference failed",
                error=e,
                input_width=width,
                input_height=height,
                outscale=outscale,
                model=settings.model.name,
                duration_seconds=round(duration, 3),
                exc_info=True
            )
            raise ImageProcessingError(
                f"AI upscaling failed for image {width}x{height}px with outscale={outscale}: {e}"
            )

    def get_info(self) -> dict:
        """
        Get model information without triggering load.
        
        Returns model metadata for health checks and monitoring.
        This is a read-only operation with no side effects.
        
        Returns:
            dict: Model information containing:
                - name (str): Model name (e.g., "RealESRGAN_x4plus")
                - scale (int): Scale factor (e.g., 4)
                - loaded (bool): Whether model is currently loaded
                - device (str): Device type (cuda/cpu) if loaded
                - half_precision (bool): Whether FP16 mode is enabled
                - tile_size (int): Tile size for processing
                - model_file (str): Model filename (not full path)
        
        Example:
            >>> manager = ModelManager()
            >>> info = manager.get_info()
            >>> print(info)
            {
                "name": "RealESRGAN_x4plus",
                "scale": 4,
                "loaded": False,
                "device": "cuda",
                "half_precision": True,
                "tile_size": 512,
                "model_file": "RealESRGAN_x4plus.pth"
            }
        """
        info = {
            "name": settings.model.name,
            "scale": settings.model.scale,
            "loaded": self.is_loaded,
            "half_precision": settings.model.half_precision,
            "tile_size": settings.model.tile_size,
            "model_file": settings.model.model_file.name
        }
        
        # ⚡ Add device info if model is loaded
        if self.is_loaded and hasattr(self._model, 'device'):
            info["device"] = str(self._model.device)
        else:
            info["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        
        return info

    def get_upsampler(self):
        """
        Get the Real-ESRGAN upsampler instance.
        
        Public method to access the upsampler for reuse by other services
        (e.g., CodeFormerManager for background enhancement).
        
        Returns:
            RealESRGANer instance
        
        Raises:
            RuntimeError: If model is not loaded
        
        Requirements: 11.6 (reuse existing model)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model
