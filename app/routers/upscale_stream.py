"""
Streaming upscale endpoints with progress tracking.
"""

import asyncio
import json
import uuid
from typing import AsyncGenerator
from fastapi import APIRouter, File, Form, UploadFile, status, BackgroundTasks
from sse_starlette.sse import EventSourceResponse

from app.services.upscale_job_processor import UpscaleJobProcessor
from app.utils import read_upload_file
from app.utils.logging_utils import get_structured_logger

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_synced_user
from app.db import get_db
from app.models.orm import User
from app.services.quota import check_and_consume


logger = get_structured_logger(__name__)
router = APIRouter(prefix="/upscale", tags=["Streaming Upscaling"])

# Progress tracking dictionary (in-memory, production should use Redis)
_progress_store: dict[str, dict] = {}


@router.post(
    "/ai/stream",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start AI upscale with progress tracking",
    description="Start AI upscale job and return job ID for progress tracking"
)
async def upscale_ai_stream(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image file (JPG/PNG, max 10MB)"),
    target_resolution: str = Form("2k", description="Target resolution: 2k or 4k"),
    enhance_faces: bool = Form(True, description="Apply CodeFormer face enhancement")
,
    user: User = Depends(get_synced_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Start AI upscale job with progress tracking.
    Returns job_id to track progress via SSE endpoint.

    - **file**: Image file (JPG/PNG)
    - **target_resolution**: Target resolution (2k or 4k)
    - **enhance_faces**: Enable CodeFormer face restoration (default: True)
    """
    job_id = str(uuid.uuid4())

    file_info = await read_upload_file(file)
    # Consume quota AFTER file đọc thành công (tránh trừ quota trên upload lỗi).
    await check_and_consume(db, user)

    logger.info(
        "AI stream upscale request received",
        job_id=job_id,
        filename=file_info.filename,
        file_size_bytes=file_info.size,
        target_resolution=target_resolution,
        enhance_faces=enhance_faces
    )

    # Initialize progress
    _progress_store[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Đang khởi tạo...",
        "result": None,
        "error": None
    }

    # Start background processing
    job_processor = UpscaleJobProcessor(_progress_store)
    background_tasks.add_task(
        job_processor.process_job,
        job_id,
        file_info,
        target_resolution,
        enhance_faces
    )

    return {"job_id": job_id}


@router.get(
    "/progress/{job_id}",
    summary="Track upscale progress via SSE",
    description="Server-Sent Events endpoint for real-time progress updates"
)
async def track_progress(job_id: str):
    """
    SSE endpoint to track upscale progress in real-time.
    Client connects va receives progress updates until completion.
    """
    async def event_generator() -> AsyncGenerator[dict, None]:
        """Generate SSE events for progress updates."""
        try:
            while True:
                if job_id not in _progress_store:
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": "Job not found"})
                    }
                    break
                
                progress_data = _progress_store[job_id]
                
                yield {
                    "event": "progress",
                    "data": json.dumps(progress_data)
                }
                
                # Stop if job is complete or failed
                if progress_data["status"] in ["complete", "error"]:
                    # Cleanup after 30 seconds
                    await asyncio.sleep(30)
                    if job_id in _progress_store:
                        del _progress_store[job_id]
                    break
                
                # Poll every 500ms
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"SSE error for job {job_id}", error=str(e), exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({"error": "Stream error"})
            }
    
    return EventSourceResponse(event_generator())
