"""
Health check endpoints with structured logging and Prometheus metrics.
"""

import json
import logging
import time
from functools import wraps

from fastapi import APIRouter, status, Request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from app.core import settings
from app.services import ModelManager

router = APIRouter(prefix="/health", tags=["Health"])
logger = logging.getLogger(__name__)

# Prometheus metrics (Task 12)
HEALTH_REQUEST_COUNT = Counter(
    'health_request_count',
    'Total number of health endpoint requests',
    ['endpoint']
)

HEALTH_RESPONSE_TIME = Histogram(
    'health_response_time_seconds',
    'Response time of health endpoints in seconds',
    ['endpoint']
)

MODEL_LOAD_STATUS = Gauge(
    'model_load_status',
    'Current model load status (1=loaded, 0=not loaded)'
)


def log_request_json(endpoint: str):
    """Decorator for structured JSON logging of requests."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            request = kwargs.get('request')

            try:
                result = await func(*args, **kwargs)
                response_time = time.perf_counter() - start_time

                # Task 11: Structured JSON logging
                from datetime import datetime, timezone
                log_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "info",
                    "endpoint": endpoint,
                    "response_time_ms": round(response_time * 1000, 3),
                    "model_status": getattr(result, 'get', lambda x: None)('model_loaded') if isinstance(result, dict) else None,
                    "status_code": 200
                }
                logger.info(f"Health request: {json.dumps(log_data)}")

                # Task 12: Update Prometheus metrics
                HEALTH_REQUEST_COUNT.labels(endpoint=endpoint).inc()
                HEALTH_RESPONSE_TIME.labels(endpoint=endpoint).observe(response_time)

                if endpoint == "/health/ready" and isinstance(result, dict):
                    model_loaded = result.get('model_loaded', False)
                    MODEL_LOAD_STATUS.set(1 if model_loaded else 0)

                return result
            except Exception as e:
                response_time = time.perf_counter() - start_time
                from datetime import datetime, timezone
                log_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "error",
                    "endpoint": endpoint,
                    "response_time_ms": round(response_time * 1000, 3),
                    "error": str(e),
                    "status_code": 500
                }
                logger.error(f"Health request failed: {json.dumps(log_data)}")

                # Task 12: Still record metrics for failed requests
                HEALTH_REQUEST_COUNT.labels(endpoint=endpoint).inc()
                HEALTH_RESPONSE_TIME.labels(endpoint=endpoint).observe(response_time)
                raise

        return wrapper
    return decorator


@router.get("", status_code=status.HTTP_200_OK)
@log_request_json("/health")
async def health_check(request: Request = None):
    """Basic health check."""
    return {
        "status": "healthy",
        "service": "image-upscale-api",
        "version": "1.0.0"
    }


@router.get("/ready", status_code=status.HTTP_200_OK)
@log_request_json("/health/ready")
async def readiness_check(request: Request = None):
    """
    Readiness probe - checks if model is loaded.

    Returns ready=false with default model_info if ModelManager fails.
    """
    try:
        from app.services.codeformer_manager import CodeFormerManager

        model_manager = ModelManager()
        codeformer_manager = CodeFormerManager()
        return {
            "ready": model_manager.is_loaded,
            "model_loaded": model_manager.is_loaded,
            "model_info": model_manager.get_info(),
            "codeformer_info": codeformer_manager.get_info()
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}", exc_info=True)
        # Return default model_info from settings per Requirement 5.2
        return {
            "ready": False,
            "model_loaded": False,
            "model_info": {
                "name": settings.model.name,
                "scale": settings.model.scale,
                "loaded": False,
                "half_precision": settings.model.half_precision,
                "model_file": settings.model.model_file.name
            },
            "codeformer_info": {
                "enabled": settings.model.codeformer_enabled,
                "loaded": False,
                "weight": settings.model.codeformer_weight,
                "face_upsample": settings.model.codeformer_face_upsample
            }
        }


@router.get("/config", status_code=status.HTTP_200_OK)
@log_request_json("/health/config")
async def get_config(request: Request = None):
    """Get current configuration (safe values only)."""
    try:
        return {
            # Note: api.host and api.port intentionally not exposed
            # per security requirements (internal deployment details)
            "upload": {
                "max_size_mb": settings.upload.max_file_size / (1024 * 1024),
                "allowed_types": list(settings.upload.allowed_content_types)
            },
            "output": {
                "supported_resolutions": settings.output.supported_resolutions,
                "default_resolution": settings.output.default_resolution,
                "format": settings.output.format,
                "quality": settings.output.quality
            }
        }
    except Exception as e:
        logger.error(f"Config endpoint failed: {e}", exc_info=True)
        raise


# Task 12: Prometheus metrics endpoint
@router.get("/metrics", status_code=status.HTTP_200_OK)
async def metrics():
    """
    Prometheus metrics endpoint.

    Exports metrics in Prometheus format for monitoring:
    - health_request_count: Total requests per endpoint
    - health_response_time_seconds: Response time histogram
    - model_load_status: Current model load status (1=loaded, 0=not loaded)
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
