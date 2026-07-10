"""
FastAPI dependencies for authenticated routes.

Usage:
    from fastapi import Depends
    from app.auth import get_current_user, require_tier, CurrentUser

    @router.post("/upscale")
    async def upscale(user: CurrentUser = Depends(get_current_user)):
        ...

    @router.post("/upscale/4k")
    async def upscale_4k(user: CurrentUser = Depends(require_tier("pro"))):
        ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .cognito import CognitoAuthError, verify_token

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    sub: str
    email: str | None
    tier: str  # "free" | "pro"
    raw_claims: dict


def _extract_tier(claims: dict) -> str:
    # Cognito custom attributes are prefixed with "custom:"
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
        email=claims.get("email"),
        tier=_extract_tier(claims),
        raw_claims=claims,
    )


def require_tier(min_tier: str) -> Callable:
    """Return a dependency that enforces `min_tier` (free < pro)."""
    order = {"free": 0, "pro": 1}
    if min_tier not in order:
        raise ValueError(f"Unknown tier: {min_tier}")

    async def _dep(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if order[user.tier] < order[min_tier]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {min_tier} tier",
            )
        return user

    return _dep
