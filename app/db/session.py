"""
Async SQLAlchemy engine + session factory (Sprint 3).

Reads DATABASE_URL. Example (RDS/Postgres):
  postgresql+asyncpg://user:pass@host:5432/upscale
"""
from __future__ import annotations

import os
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://upscale:upscale@localhost:5432/upscale",
)

# echo can be toggled via SQL_ECHO=true for local debugging
_echo = os.getenv("SQL_ECHO", "false").lower() == "true"

engine = create_async_engine(
    DATABASE_URL,
    echo=_echo,
    pool_pre_ping=True,
    pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding an AsyncSession."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
