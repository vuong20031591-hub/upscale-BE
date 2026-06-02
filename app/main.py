"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request

from app.core import settings, AppException
from app.routers import (
    health_router,
    upscale_basic_router,
    upscale_face_router,
    upscale_smart_router,
    upscale_stream_router
)
from app.middleware import RateLimiter


def create_application() -> FastAPI:
    """Application factory pattern."""
    app = FastAPI(
        title="Image Upscale API",
        description="AI-powered image upscaling service using Real-ESRGAN",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.origins,
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
            # Smart upscale analysis headers
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
            # Resolution metadata headers
            "X-Original-Width",
            "X-Original-Height",
            "X-Final-Width",
            "X-Final-Height"
        ],
    )
    
    # ⚡ OPTIMIZATION: Rate limiting middleware (configurable via RATE_LIMIT_PER_MINUTE and TRUSTED_PROXIES env vars)
    app.add_middleware(
        RateLimiter, 
        requests_per_minute=settings.api.rate_limit_per_minute,
        trusted_proxies=settings.api.trusted_proxies
    )

    # Exception handlers
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

    # Register routers
    app.include_router(health_router)
    app.include_router(upscale_basic_router)
    app.include_router(upscale_face_router)
    app.include_router(upscale_smart_router)
    app.include_router(upscale_stream_router)

    @app.get("/")
    async def root():
        return {
            "service": "Image Upscale API",
            "version": "1.0.0",
            "docs": "/docs"
        }

    # ⚡ OPTIMIZATION: Preload AI model at startup for faster first request
    @app.on_event("startup")
    async def startup_event():
        """Preload AI model to avoid lazy loading delay on first request."""
        from app.services.model_manager import ModelManager
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info("Preloading AI model at startup...")
        
        try:
            manager = ModelManager()
            manager.load()
            logger.info("AI model preloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to preload AI model: {e}. Will load on first request.")

    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug
    )
