from app.routers.health import router as health_router
from app.routers.upscale_basic import router as upscale_basic_router
from app.routers.upscale_face import router as upscale_face_router
from app.routers.upscale_smart import router as upscale_smart_router
from app.routers.upscale_stream import router as upscale_stream_router

__all__ = [
    'health_router',
    'upscale_basic_router',
    'upscale_face_router',
    'upscale_smart_router',
    'upscale_stream_router'
]
