# Sprint 2 — AWS Cognito Auth

## Env

```
COGNITO_REGION=ap-southeast-1
COGNITO_USER_POOL_ID=ap-southeast-1_yMHGGcfAO
COGNITO_CLIENT_ID=76irjqk5s74o350mondlk86vj6
COGNITO_TOKEN_USE=access   # or "id"
```

## Protect an endpoint

```python
from fastapi import APIRouter, Depends
from app.auth import get_current_user, require_tier, CurrentUser

router = APIRouter()

@router.post("/upscale")
async def upscale(user: CurrentUser = Depends(get_current_user)):
    return {"sub": user.sub, "tier": user.tier}

@router.post("/upscale/4k")
async def upscale_4k(user: CurrentUser = Depends(require_tier("pro"))):
    return {"ok": True}
```

## How verification works

1. Extract `kid` from JWT header.
2. Fetch JWKS from `https://cognito-idp.{region}.amazonaws.com/{pool_id}/.well-known/jwks.json` (cached 1h).
3. Verify RS256 signature.
4. Check `exp`, `iss`, `token_use`, `client_id` / `aud`.
5. Read `custom:tier` claim (default `free`).

## Custom attribute

The User Pool must have a custom string attribute `tier` (mutable). Assign a
user to `pro` via AWS Console → Users → Edit → `custom:tier = pro`, or via
`AdminUpdateUserAttributes` API.

## Testing locally

```bash
# Get a token from the Hosted UI or via cognito-idp CLI
export TOKEN="eyJraWQi..."
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/whoami
```
