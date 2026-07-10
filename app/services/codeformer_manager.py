"""
CodeFormer face restoration service.

Uses minimal CodeFormer architecture files with facelib (CodeFormer's version) for face detection.
Note: We use CodeFormer's facelib (not facexlib from PyPI) for face_upsampler support.
"""

import importlib
import logging
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from app.core import settings, ModelNotFoundError
from app.utils.logging_utils import get_structured_logger

# Suppress deprecation warnings from dependencies (torchvision, facexlib, basicsr)
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', message='.*functional_tensor.*', category=UserWarning)

# FIX: Move sys.path modification to a function to avoid import-time side effects
def _ensure_codeformer_in_path():
    """
    Ensure codeformer_minimal is in sys.path for architecture imports.
    
    This is called lazily when needed, not at import time, to avoid
    global side effects and make testing easier.
    """
    backend_root = str(Path(__file__).parent.parent.parent)
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)

from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.registry import ARCH_REGISTRY
from torchvision.transforms.functional import normalize

logging.basicConfig(level=logging.INFO)
logger = get_structured_logger(__name__)


from enum import Enum


class CodeFormerMode(Enum):
    """CodeFormer processing modes."""
    RESTORATION = "restoration"      # Face Restoration - enhance details
    COLORIZATION = "colorization"    # Face Colorization - add color to B&W
    INPAINTING = "inpainting"        # Face Inpainting - fill masked regions


def _get_codeformer_class():
    """Import and return CodeFormer architecture class.

    BasicSR only knows custom architectures after their module is imported.
    Do not swallow import errors here; otherwise the later registry lookup
    hides the real problem behind a misleading "No object named CodeFormer".
    """
    _ensure_codeformer_in_path()

    module = importlib.import_module("codeformer_minimal.codeformer_arch")
    registered_archs = getattr(ARCH_REGISTRY, "_obj_map", {})

    CodeFormerNet = (
        registered_archs.get("CodeFormer")
        or registered_archs.get("CodeFormer_basicsr")
        or getattr(module, "CodeFormer", None)
    )
    if CodeFormerNet is None:
        registered_names = sorted(str(name) for name in registered_archs.keys())
        raise KeyError(
            "CodeFormer architecture class is unavailable "
            f"(registered architectures: {registered_names})"
        )

    if "CodeFormer" not in registered_archs and "CodeFormer_basicsr" not in registered_archs:
        logger.warning(
            "CodeFormer missing from BasicSR registry; using direct class import",
            registered_architectures=sorted(str(name) for name in registered_archs.keys()),
        )

    return CodeFormerNet


class CodeFormerManager:
    """Singleton manager for CodeFormer with full features."""

    _instance: Optional["CodeFormerManager"] = None
    _lock = threading.Lock()  # OPTIMIZATION: Thread-safe singleton
    _models: dict = {}  # mode -> model
    _detection_model: Optional[torch.nn.Module] = None

    # Model configurations
    MODEL_CONFIGS = {
        CodeFormerMode.RESTORATION: {
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
            "codebook_size": 1024,
            "connect_list": ['32', '64', '128', '256'],
            "w": 0.7,  # Default fidelity weight
            "adain": True
        },
        CodeFormerMode.COLORIZATION: {
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_colorization.pth",
            "codebook_size": 1024,
            "connect_list": ['32', '64', '128'],
            "w": 0,  # Fixed w=0 for colorization
            "adain": True
        },
        CodeFormerMode.INPAINTING: {
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_inpainting.pth",
            "codebook_size": 512,
            "connect_list": ['32', '64', '128'],
            "w": 1,  # Fixed w=1 for inpainting
            "adain": False
        }
    }

    def __new__(cls) -> "CodeFormerManager":
        # OPTIMIZATION: Thread-safe singleton pattern
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def is_enabled(self) -> bool:
        return settings.model.codeformer_enabled

    def load(self, mode: CodeFormerMode = CodeFormerMode.RESTORATION) -> None:
        """Load CodeFormer model for specific mode."""
        # OPTIMIZATION: Thread-safe model loading
        with self._lock:
            if mode in self._models:
                return

            start_time = time.time()
            config = self.MODEL_CONFIGS[mode]

            try:
                logger.info(f"Loading CodeFormer {mode.value} model")

                # Download/load model file
                model_file = self._get_model_file(mode, config["url"])

                # Load model
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                checkpoint = torch.load(model_file, map_location='cpu')['params_ema']

                CodeFormerNet = _get_codeformer_class()
                net = CodeFormerNet(
                    dim_embd=512,
                    codebook_size=config["codebook_size"],
                    n_head=8,
                    n_layers=9,
                    connect_list=config["connect_list"]
                )
                net.load_state_dict(checkpoint)
                net.eval()
                net = net.to(device)

                self._models[mode] = net
                logger.info(f"CodeFormer {mode.value} loaded", device=str(device))

                # Load face detector if not loaded
                if self._detection_model is None:
                    self._load_face_detector()

                duration = time.time() - start_time
                logger.info(f"CodeFormer {mode.value} ready", duration_seconds=round(duration, 3))

            except Exception as e:
                logger.error(f"Failed to load CodeFormer {mode.value}", error=e, exc_info=True)
                raise ModelNotFoundError(f"Could not load CodeFormer {mode.value}: {e}")

    def _get_model_file(self, mode: CodeFormerMode, url: str) -> Path:
        """Get model file path, download if needed."""
        from torch.hub import download_url_to_file

        model_dir = settings.model.path / "CodeFormer"
        model_dir.mkdir(parents=True, exist_ok=True)

        if mode == CodeFormerMode.RESTORATION:
            model_file = model_dir / "codeformer.pth"
        elif mode == CodeFormerMode.COLORIZATION:
            model_file = model_dir / "codeformer_colorization.pth"
        else:  # INPAINTING
            model_file = model_dir / "codeformer_inpainting.pth"

        if model_file.exists():
            return model_file

        logger.info(f"Downloading {mode.value} model...")
        download_url_to_file(url, str(model_file), progress=True)
        return model_file

    def _load_face_detector(self) -> None:
        """Load RetinaFace detector."""
        # IMPORTANT: Import from facelib (CodeFormer's version) not facexlib
        from facelib.detection import init_detection_model

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # CodeFormer's facelib API: init_detection_model(model_name, half=False, device='cuda')
        # Model weights will be auto-downloaded to weights/facelib/ directory
        self._detection_model = init_detection_model(
            'retinaface_resnet50',
            half=False,
            device=device
        )
        logger.info("Face detector loaded")
    
    def _get_background_upsampler(self, bg_upscale: int):
        """
        Get Real-ESRGAN upsampler for background enhancement.
        
        REUSES ModelManager's model to avoid duplicate loading.
        Returns None if bg_upscale=1 (Real-ESRGAN x4plus does not support 1x).
        
        Args:
            bg_upscale: Upscale factor (1, 2, or 4)
        
        Returns:
            RealESRGANer instance or None (if bg_upscale=1)
        
        Requirements: 11.4, 11.6 (handle bg_upscale=1, reuse existing model)
        """
        # bg_upscale=1 means skip enhancement (Real-ESRGAN x4plus doesn't support 1x)
        if bg_upscale == 1:
            return None
        
        from app.services.model_manager import ModelManager
        
        model_manager = ModelManager()
        
        # Ensure model is loaded
        if not model_manager.is_loaded:
            model_manager.load()
        
        # Use public method to get upsampler instead of accessing private _model
        return model_manager.get_upsampler()

    def enhance_faces(
        self,
        image: Image.Image,
        weight: Optional[float] = None,
        face_upsample: Optional[bool] = None,
        mode: CodeFormerMode = CodeFormerMode.RESTORATION,
        background_enhance: bool = True,  # NEW
        bg_upscale: int = 2  # NEW
    ) -> Image.Image:
        """
        Enhance faces using specified mode with optional background enhancement.
        
        NEW PARAMETERS:
        - background_enhance: Whether to enhance background with Real-ESRGAN
        - bg_upscale: Background upscale factor (1, 2, or 4)
        
        WORKFLOW (following official CodeFormer):
        1. Detect and align faces
        2. Enhance each face with CodeFormer
        3. IF background_enhance=True AND bg_upscale != 1:
           - Enhance background with Real-ESRGAN (bg_upscale factor)
           - Paste enhanced faces onto enhanced background
        4. ELSE:
           - Paste enhanced faces onto original background
        5. Clear GPU cache to prevent memory overflow
        
        Args:
            image: PIL Image
            weight: Fidelity weight (0-1), higher = more original detail (restoration only)
            face_upsample: Whether to upsample face regions
            mode: Processing mode (restoration/colorization/inpainting)
            background_enhance: Whether to enhance background with Real-ESRGAN
            bg_upscale: Background upscale factor (1, 2, or 4)

        Returns:
            PIL Image with enhanced faces
        
        Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
        """
        if not self.is_enabled:
            return image

        # Load model for this mode if needed
        if mode not in self._models:
            self.load(mode)

        weight = weight if weight is not None else settings.model.codeformer_weight
        face_upsample = face_upsample if face_upsample is not None else settings.model.codeformer_face_upsample

        start_time = time.time()
        config = self.MODEL_CONFIGS[mode]

        try:
            # IMPORTANT: Import from facelib (CodeFormer's version) not facexlib
            # facelib has face_upsampler parameter support in paste_faces_to_input_image()
            from facelib.utils.face_restoration_helper import FaceRestoreHelper

            # Convert PIL to BGR
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            net = self._models[mode]
            device = next(net.parameters()).device

            # For colorization mode, colorize full image first
            if mode == CodeFormerMode.COLORIZATION:
                # Check if image is grayscale
                is_gray = len(img_bgr.shape) == 2 or (
                    len(img_bgr.shape) == 3 and 
                    np.allclose(img_bgr[:,:,0], img_bgr[:,:,1]) and 
                    np.allclose(img_bgr[:,:,1], img_bgr[:,:,2])
                )
                
                if is_gray:
                    h, w = img_bgr.shape[:2]
                    img_512 = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
                    img_t = img2tensor(img_512 / 255., bgr2rgb=True, float32=True)
                    normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    img_t = img_t.unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = net(img_t, w=0, adain=True)[0]
                        colorized = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                    
                    img_bgr = cv2.resize(colorized, (w, h), interpolation=cv2.INTER_LINEAR).astype('uint8')
                    del img_t, output, colorized
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            upscale_factor = bg_upscale if (background_enhance and bg_upscale != 1) else 1
            
            face_helper = FaceRestoreHelper(
                upscale_factor,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,
                device=device
            )
            face_helper.read_image(img_bgr)
            
            # Inpainting mode: use full 512x512 image, skip face detection
            if mode == CodeFormerMode.INPAINTING:
                if img_bgr.shape[:2] != (512, 512):
                    img_bgr = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
                    face_helper.input_img = img_bgr
                face_helper.cropped_faces = [img_bgr.copy()]
                num_faces = 1
            else:
                face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                face_helper.align_warp_face()
                num_faces = len(face_helper.cropped_faces)
            
            if num_faces == 0:
                if background_enhance and bg_upscale != 1:
                    try:
                        bg_upsampler = self._get_background_upsampler(bg_upscale)
                        if bg_upsampler is not None:
                            bg_img = bg_upsampler.enhance(img_bgr, outscale=bg_upscale)[0]
                            result_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            return Image.fromarray(result_rgb)
                    except Exception as e:
                        logger.warning("Background enhancement failed (no faces)", error=str(e))
                
                result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                return Image.fromarray(result_rgb)

            cache_freq = settings.model.gpu_cache_clear_frequency
            
            for idx, cropped_face in enumerate(face_helper.cropped_faces):
                face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                face_t = face_t.unsqueeze(0).to(device)

                with torch.no_grad():
                    w = config["w"] if config["w"] is not None else weight
                    adain = config["adain"]

                    if mode == CodeFormerMode.INPAINTING:
                        # Inpainting: detect white pixels (mask) in normalized tensor
                        # White pixels (255,255,255) normalize to (1,1,1), sum = 3
                        mask = torch.zeros(512, 512)
                        m_ind = torch.sum(face_t[0], dim=0)  # Sum across RGB channels
                        mask[m_ind == 3] = 1.0  # Mark white pixels as mask
                        mask = mask.view(1, 1, 512, 512).to(device)
                        
                        # Log mask statistics for debugging
                        mask_pixels = int(torch.sum(mask).item())
                        mask_percentage = (mask_pixels / (512 * 512)) * 100
                        logger.info(
                            f"Inpainting mask detected",
                            face_index=idx + 1,
                            mask_pixels=mask_pixels,
                            mask_percentage=f"{mask_percentage:.2f}%"
                        )
                        
                        # Run model with w=1, adain=False for inpainting
                        output_face = net(face_t, w=w, adain=adain)[0]
                        output = (1 - mask) * face_t + mask * output_face
                        del mask, output_face
                    else:
                        output = net(face_t, w=w, adain=adain)[0]

                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

                del output, face_t
                
                face_helper.add_restored_face(restored_face.astype('uint8'))
                
                if torch.cuda.is_available() and cache_freq > 0 and (idx + 1) % cache_freq == 0:
                    torch.cuda.empty_cache()
            
            if torch.cuda.is_available() and cache_freq > 0:
                torch.cuda.empty_cache()

            # Inpainting: return restored face directly
            if mode == CodeFormerMode.INPAINTING:
                if len(face_helper.restored_faces) > 0:
                    restored_img = face_helper.restored_faces[0]
                    
                    if background_enhance and bg_upscale != 1:
                        try:
                            bg_upsampler = self._get_background_upsampler(bg_upscale)
                            if bg_upsampler is not None:
                                restored_img = bg_upsampler.enhance(restored_img, outscale=bg_upscale)[0]
                        except Exception as e:
                            logger.warning("Background upscale failed for inpainting", error=str(e))
                    
                    result_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    logger.info(
                        "Face inpainting complete",
                        num_faces=1,
                        weight=weight,
                        duration_seconds=round(time.time() - start_time, 3)
                    )
                    return Image.fromarray(result_rgb.astype('uint8'))
                else:
                    result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(result_rgb)

            bg_img = None
            face_upsampler = None
            
            if background_enhance and bg_upscale != 1:
                try:
                    bg_upsampler = self._get_background_upsampler(bg_upscale)
                    if bg_upsampler is not None:
                        # Enhance background with Real-ESRGAN
                        bg_img = bg_upsampler.enhance(img_bgr, outscale=bg_upscale)[0]
                        
                        # CRITICAL FIX: Ensure bg_img has EXACT size expected by facexlib
                        # facexlib calculates: w_up = face_helper.input_img.width * upscale_factor
                        #                      h_up = face_helper.input_img.height * upscale_factor
                        # Then resizes upsample_img to (w_up, h_up)
                        # We must ensure bg_img matches this EXACTLY to avoid quality loss
                        h_input, w_input = face_helper.input_img.shape[:2]
                        expected_h = int(h_input * upscale_factor)
                        expected_w = int(w_input * upscale_factor)
                        
                        # Check if bg_img size matches expected size
                        if bg_img.shape[0] != expected_h or bg_img.shape[1] != expected_w:
                            logger.info(
                                "Adjusting bg_img size to match face_helper.input_img dimensions",
                                face_helper_input_size=(h_input, w_input),
                                upscale_factor=upscale_factor,
                                expected_size=(expected_h, expected_w),
                                bg_img_size=bg_img.shape[:2]
                            )
                            bg_img = cv2.resize(bg_img, (expected_w, expected_h), interpolation=cv2.INTER_LANCZOS4)
                        
                        logger.info(
                            "Background enhanced",
                            bg_upscale=bg_upscale,
                            original_size=img_bgr.shape[:2],
                            enhanced_size=bg_img.shape[:2]
                        )
                        
                        # FIX: When background is upscaled, ALWAYS upscale faces to match
                        # This ensures faces have same resolution as background
                        # Use bg_upsampler (same model used for background) for consistency
                        face_upsampler = bg_upsampler
                        logger.info(
                            "Face upsampler enabled to match background resolution",
                            bg_upscale=bg_upscale,
                            face_upsample_setting=face_upsample
                        )
                    else:
                        # bg_upscale=1, skip enhancement
                        logger.info("bg_upscale=1, skipping background enhancement")
                except Exception as e:
                    # Graceful fallback on failure
                    logger.warning(
                        "Background enhancement failed, using original",
                        error=str(e),
                        bg_upscale=bg_upscale
                    )
                    bg_img = None
            else:
                if background_enhance and bg_upscale == 1:
                    # Log when skipping due to bg_upscale=1
                    logger.info(
                        "bg_upscale=1, skipping background enhancement "
                        "(Real-ESRGAN x4plus does not support 1x)"
                    )
                
                # If no background upsampling but face_upsample is enabled, create face_upsampler
                if face_upsample:
                    try:
                        face_upsampler = self._get_background_upsampler(2)  # Use 2x for face upsample
                    except Exception as e:
                        logger.warning("Failed to create face upsampler", error=str(e))
                        face_upsampler = None

            # Paste faces back onto background
            face_helper.get_inverse_affine(None)
            
            # Paste faces using CodeFormer's facelib with face_upsampler support
            # CodeFormer's facelib handles face upsampling internally when face_upsampler is provided
            if face_upsampler is not None:
                result_bgr = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img,
                    face_upsampler=face_upsampler
                )
                logger.info(
                    "Faces pasted with upsampling",
                    face_upsampler_enabled=True,
                    bg_img_provided=bg_img is not None
                )
            else:
                result_bgr = face_helper.paste_faces_to_input_image(upsample_img=bg_img)
                logger.info(
                    "Faces pasted without upsampling",
                    face_upsampler_enabled=False,
                    bg_img_provided=bg_img is not None
                )
            
            # Clear GPU cache to prevent memory overflow
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("GPU cache cleared after pasting faces")

            # Convert back to RGB
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(result_rgb)

            duration = time.time() - start_time
            logger.info(
                f"Face {mode.value} complete",
                num_faces=num_faces,
                weight=w,
                duration_seconds=round(duration, 3)
            )

            return result_image

        except Exception as e:
            logger.error(f"Face {mode.value} failed", error=e, exc_info=True)
            return image



    def get_info(self) -> dict:
        """Get CodeFormer manager state."""
        return {
            "enabled": self.is_enabled,
            "loaded_modes": [m.value for m in self._models.keys()],
            "weight": settings.model.codeformer_weight,
            "face_upsample": settings.model.codeformer_face_upsample,
        }
    
    def get_mode_info(self, mode: CodeFormerMode) -> dict:
        """Get configuration for a specific mode."""
        config = self.MODEL_CONFIGS[mode]
        return {
            **config,
            "loaded": mode in self._models
        }
    
    def get_all_modes_info(self) -> dict:
        """Get configuration for all modes."""
        return {
            mode.value: self.get_mode_info(mode)
            for mode in CodeFormerMode
        }
