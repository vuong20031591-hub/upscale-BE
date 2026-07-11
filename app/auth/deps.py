"""
FastAPI dependencies for authenticated routes.

- `get_current_user`   : verify JWT only (không chạm DB) — dùng cho endpoint read-only nhẹ.
- `get_synced_user`    : verify JWT + upsert row vào RDS (`users`) và cập nhật
                         `last_login_at`. Dùng cho mọi endpoint cần user record
                         (quota, jobs, storage...).
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.orm import User

from .cognito import CognitoAuthError, verify_token

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    sub: str
    email: str | None
    tier: str  # "free" | "pro"
    raw_claims: dict


def _extract_tier(claims: dict) -> str:
    tier = claims.get("custom:tier") or claims.get("tier") or "free"
    tier = str(tier).lower()
    if tier not in {"free", "pro"}:
        tier = "free"
    return tier


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> CurrentUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        claims = verify_token(credentials.credentials)
    except CognitoAuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return CurrentUser(
        sub=claims["sub"],
        email=claims.get("email") or "",
        tier=_extract_tier(claims),
        raw_claims=claims,
    )


async def get_synced_user(
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Verify JWT + upsert user vào RDS. Trả về ORM `User` row."""
    # Import cục bộ để tránh circular (quota.py import auth).
    from app.services.quota import ensure_user

    return await ensure_user(
        db,
        sub=current.sub,
        email=current.email or "",
        tier_claim=current.tier,
    )

