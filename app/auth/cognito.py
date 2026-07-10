"""
AWS Cognito JWT verification.

Verifies access/id tokens issued by a Cognito User Pool using the pool's JWKS.
JWKS is fetched once and cached in-process (Cognito rotates keys rarely).

Env vars (see .env.example):
    COGNITO_REGION           e.g. ap-southeast-1
    COGNITO_USER_POOL_ID     e.g. ap-southeast-1_yMHGGcfAO
    COGNITO_CLIENT_ID        app client id (used as `aud`/`client_id` check)
    COGNITO_TOKEN_USE        "access" (default) or "id"
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import httpx
from jose import jwk, jwt
from jose.utils import base64url_decode

logger = logging.getLogger(__name__)

_JWKS_CACHE: Dict[str, Any] = {"keys": None, "fetched_at": 0.0}
_JWKS_TTL_SECONDS = 3600  # 1h


class CognitoAuthError(Exception):
    """Raised when a token fails verification."""


def _region() -> str:
    v = os.getenv("COGNITO_REGION")
    if not v:
        raise CognitoAuthError("COGNITO_REGION is not configured")
    return v


def _user_pool_id() -> str:
    v = os.getenv("COGNITO_USER_POOL_ID")
    if not v:
        raise CognitoAuthError("COGNITO_USER_POOL_ID is not configured")
    return v


def _client_id() -> str:
    v = os.getenv("COGNITO_CLIENT_ID")
    if not v:
        raise CognitoAuthError("COGNITO_CLIENT_ID is not configured")
    return v


def _token_use() -> str:
    return os.getenv("COGNITO_TOKEN_USE", "access").lower()


def issuer_url() -> str:
    return f"https://cognito-idp.{_region()}.amazonaws.com/{_user_pool_id()}"


def jwks_url() -> str:
    return f"{issuer_url()}/.well-known/jwks.json"


def _fetch_jwks(force: bool = False) -> list[dict]:
    now = time.time()
    if (
        not force
        and _JWKS_CACHE["keys"] is not None
        and now - _JWKS_CACHE["fetched_at"] < _JWKS_TTL_SECONDS
    ):
        return _JWKS_CACHE["keys"]

    url = jwks_url()
    logger.info("Fetching Cognito JWKS from %s", url)
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()

    _JWKS_CACHE["keys"] = data.get("keys", [])
    _JWKS_CACHE["fetched_at"] = now
    return _JWKS_CACHE["keys"]


def _find_key(kid: str) -> Optional[dict]:
    for key in _fetch_jwks():
        if key.get("kid") == kid:
            return key
    # Miss → refresh once in case Cognito rotated keys
    for key in _fetch_jwks(force=True):
        if key.get("kid") == kid:
            return key
    return None


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify a Cognito JWT and return its claims.

    Raises CognitoAuthError on any failure.
    """
    if not token:
        raise CognitoAuthError("Missing token")

    try:
        headers = jwt.get_unverified_headers(token)
    except Exception as exc:
        raise CognitoAuthError(f"Malformed token header: {exc}") from exc

    kid = headers.get("kid")
    if not kid:
        raise CognitoAuthError("Token header missing kid")

    key_dict = _find_key(kid)
    if key_dict is None:
        raise CognitoAuthError(f"Signing key not found for kid={kid}")

    public_key = jwk.construct(key_dict)

    try:
        message, encoded_sig = token.rsplit(".", 1)
    except ValueError as exc:
        raise CognitoAuthError("Malformed token") from exc

    decoded_sig = base64url_decode(encoded_sig.encode("utf-8"))
    if not public_key.verify(message.encode("utf-8"), decoded_sig):
        raise CognitoAuthError("Signature verification failed")

    try:
        claims = jwt.get_unverified_claims(token)
    except Exception as exc:
        raise CognitoAuthError(f"Malformed token claims: {exc}") from exc

    # exp
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)) or exp < time.time():
        raise CognitoAuthError("Token expired")

    # iss
    if claims.get("iss") != issuer_url():
        raise CognitoAuthError("Invalid issuer")

    # token_use
    want_use = _token_use()
    got_use = claims.get("token_use")
    if got_use != want_use:
        raise CognitoAuthError(
            f"Invalid token_use: got={got_use} want={want_use}"
        )

    # audience / client_id
    client_id = _client_id()
    if want_use == "id":
        if claims.get("aud") != client_id:
            raise CognitoAuthError("Invalid audience")
    else:  # access token
        if claims.get("client_id") != client_id:
            raise CognitoAuthError("Invalid client_id")

    return claims
