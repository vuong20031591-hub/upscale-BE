"""
Background job processor for AI upscale with progress tracking.

This service handles:
- Background processing of upscale jobs
- Progress tracking and updates
- Result encoding and storage

Requirements: 10.5 (async processing)
"""

import time
import asyncio
import base64
import logging
from typing import Dict, Any

from fastapi.concurrency import run_in_threadpool

from app.services import ImageProcessor
from app.models import Resolution
from app.utils.logging_utils import get_structured_logger

logger = get_structured_logger(__name__)


class UpscaleJobProcessor:
    """
    Service for processing upscale jobs in background with progress tracking.
    
    Usage:
        processor = UpscaleJobProcessor(progress_store)
        await processor.process_job(job_id, file_info, "2k", True)
    """
    
    def __init__(self, progress_store: Dict[str, Dict[str, Any]]):
        """
        Initialize job processor.
        
        Args:
            progress_store: Shared dictionary for storing job progress
        """
        self.progress_store = progress_store
    
    async def process_job(
        self,
        job_id: str,
        file_info,
        target_resolution: str,
        enhance_faces: bool = True
    ):
        """
        Process upscale job in background with progress updates.
        
        Args:
            job_id: Unique job identifier
            file_info: File information object
            target_resolution: Target resolution string (e.g., "2k", "4k")
            enhance_faces: Whether to apply face enhancement
        
        Updates progress_store with job status and results.
        """
        start_time = time.time()

        try:
            processor = ImageProcessor()
            resolution = self._get_resolution(target_resolution)

            # Get original image dimensions
            original_image = file_info.to_image()
            original_width, original_height = original_image.size

            # Update: Loading model (5%)
            self.progress_store[job_id].update({
                "status": "loading",
                "progress": 5,
                "message": "Đang tải AI model...",
                "original_width": original_width,
                "original_height": original_height
            })
            await asyncio.sleep(0.1)

            # Update: Processing start (10%)
            self.progress_store[job_id].update({
                "status": "processing",
                "progress": 10,
                "message": "Đang xử lý ảnh với AI..."
            })

            # Start progress updates in background
            progress_task = asyncio.create_task(
                self._update_progress_during_inference(job_id)
            )

            # Process image in thread pool (CPU-bound, ~3.7s)
            result = await run_in_threadpool(
                processor.process,
                file_info,
                resolution,
                True,  # use_ai=True
                enhance_faces
            )
            
            # Cancel progress updates
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
            
            # Update: Encoding (90%)
            self.progress_store[job_id].update({
                "status": "encoding",
                "progress": 90,
                "message": "Đang mã hóa kết quả..."
            })
            
            # Encode to bytes
            buffer = result.to_bytes(quality=processor.config.quality)
            
            # Convert to base64 for JSON response
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Update: Complete (100%)
            self.progress_store[job_id].update({
                "status": "complete",
                "progress": 100,
                "message": "Hoàn thành!",
                "result": {
                    "image_data": image_b64,
                    "resolution": result.resolution,
                    "scale_factor": result.scale_factor,
                    "width": result.final_width,
                    "height": result.final_height,
                    "original_width": original_width,
                    "original_height": original_height,
                    "processing_time_ms": processing_time_ms,
                    "file_size_bytes": len(buffer.getvalue())
                }
            })
            
            logger.info(
                "AI stream upscale completed",
                job_id=job_id,
                resolution=result.resolution,
                scale_factor=result.scale_factor,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            logger.error(
                "AI stream upscale failed",
                job_id=job_id,
                error=str(e),
                exc_info=True
            )
            self.progress_store[job_id].update({
                "status": "error",
                "progress": 0,
                "message": "Xử lý thất bại",
                "error": str(e)
            })
    
    async def _update_progress_during_inference(self, job_id: str):
        """
        Update progress smoothly during AI inference (10% → 85%).
        
        Args:
            job_id: Job identifier to update
        """
        for i in range(10, 85, 5):
            await asyncio.sleep(0.25)  # Update every 250ms
            if job_id in self.progress_store:
                self.progress_store[job_id].update({
                    "progress": i,
                    "message": f"Đang xử lý ảnh... {i}%"
                })
    
    @staticmethod
    def _get_resolution(resolution: str) -> Resolution:
        """
        Convert string to Resolution enum with strict validation.
        
        Args:
            resolution: Resolution string (e.g., "2k", "4k")
        
        Returns:
            Resolution enum
        
        Raises:
            ValueError: If resolution is not a valid enum value
        """
        try:
            return Resolution(resolution.lower())
        except ValueError:
            valid_values = ", ".join([r.value for r in Resolution])
            raise ValueError(f"Invalid resolution '{resolution}'. Valid values: {valid_values}")
