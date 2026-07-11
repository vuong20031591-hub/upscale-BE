"""
Quota service — atomic check & consume 1 job per period.

- free: N jobs/day (QUOTA_FREE_PER_DAY, default 5)
- pro:  M jobs/month (QUOTA_PRO_PER_MONTH, default 500)
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import UsageQuota, User, UserTier

FREE_PER_DAY = int(os.getenv("QUOTA_FREE_PER_DAY", "5"))
PRO_PER_MONTH = int(os.getenv("QUOTA_PRO_PER_MONTH", "500"))


def _period_key(tier: UserTier, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y-%m") if tier == UserTier.pro else now.strftime("%Y-%m-%d")


def _limit_for(tier: UserTier) -> int:
    return PRO_PER_MONTH if tier == UserTier.pro else FREE_PER_DAY


async def ensure_user(
    db: AsyncSession, sub: str, email: str, tier_claim: str
) -> User:
    """
    Idempotent upsert từ JWT claim. INSERT lần đầu, các lần sau UPDATE
    email/tier/last_login_at. Trả về User row hiện tại.
    """
    tier = UserTier.pro if (tier_claim or "").lower() == "pro" else UserTier.free
    stmt = (
        pg_insert(User)
        .values(id=sub, email=email, tier=tier, last_login_at=func.now())
        .on_conflict_do_update(
            index_elements=[User.id],
            set_={
                "email": email,
                "tier": tier,
                "last_login_at": func.now(),
            },
        )
        .returning(User)
    )
    row = (await db.execute(stmt)).scalar_one()
    # Commit gộp cùng transaction của check_and_consume (tránh double-commit/request).
    return row


async def check_and_consume(db: AsyncSession, user: User) -> UsageQuota:
    limit = _limit_for(user.tier)
    period = _period_key(user.tier)

    stmt = (
        pg_insert(UsageQuota)
        .values(user_id=user.id, period_key=period, jobs_used=1)
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[UsageQuota.user_id, UsageQuota.period_key],
        set_={"jobs_used": UsageQuota.jobs_used + 1},
        where=(UsageQuota.jobs_used < limit),
    ).returning(UsageQuota)

    row = (await db.execute(stmt)).scalar_one_or_none()
    await db.commit()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Quota exceeded for tier '{user.tier.value}': {limit}/{period}",
        )
    return row
