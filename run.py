#!/usr/bin/env python3
"""
Script to run the application.
"""

import uvicorn
from app.core import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug
    )
