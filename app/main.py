"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from app.core import AppException, settings
from app.middleware import RateLimiter
from app.routers import (
    health_router,
    upscale_basic_router,
    upscale_face_router,
    upscale_smart_router,
    upscale_stream_router,
)
from app.routers.jobs import router as jobs_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload AI model at startup."""
    logger.info("Preloading AI model at startup...")
    try:
        from app.services.model_manager import ModelManager

        ModelManager().load()
        logger.info("AI model preloaded successfully")
    except Exception as e:
        logger.warning(f"Failed to preload AI model: {e}. Will load on first request.")
    yield
    logger.info("Application shutdown complete")


app = FastAPI(
    title="Image Upscale API",
    description="AI-powered image upscaling service using Real-ESRGAN",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS: allow localhost dev + any Lovable preview/published subdomain.
# Override via settings.cors.origin_regex if provided.
_default_cors_regex = (
    r"^(https?://localhost(:\d+)?|https?://127\.0\.0\.1(:\d+)?"
    r"|https://([a-z0-9-]+\.)*lovable\.(app|dev))$"
)
_cors_regex = getattr(settings.cors, "origin_regex", None) or _default_cors_regex

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=_cors_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Faces-Detected",
        "X-Processing-Time",
        "X-Mode-Used",
        "X-Weight-Used",
        "X-Background-Enhanced",
        "X-BG-Upscale",
        "X-Warning",
        "Content-Disposition",
        "X-Analysis-Grayscale",
        "X-Analysis-Grayscale-Confidence",
        "X-Analysis-White-Mask",
        "X-Analysis-White-Mask-Confidence",
        "X-Analysis-Low-Resolution",
        "X-Analysis-Low-Resolution-Confidence",
        "X-Analysis-Blur",
        "X-Analysis-Blur-Confidence",
        "X-Suggested-Mode",
        "X-Analysis-Time",
        "X-Original-Width",
        "X-Original-Height",
        "X-Final-Width",
        "X-Final-Height",
        "X-Request-ID",
    ],
)

app.add_middleware(
    RateLimiter,
    requests_per_minute=settings.api.rate_limit_per_minute,
    trusted_proxies=settings.api.trusted_proxies,
)


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


app.include_router(health_router)
app.include_router(upscale_basic_router)
app.include_router(upscale_face_router)
app.include_router(upscale_smart_router)
app.include_router(upscale_stream_router)
app.include_router(jobs_router)


@app.get("/")
async def root():
    return {"service": "Image Upscale API", "version": "1.0.0", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
    )
