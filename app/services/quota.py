"""
Quota service — atomic check & consume 1 job per period.

- free: N jobs/day (QUOTA_FREE_PER_DAY, default 5)
- pro:  M jobs/month (QUOTA_PRO_PER_MONTH, default 500)

Single INSERT ... ON CONFLICT DO UPDATE ... WHERE guards against race conditions.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import HTTPException, status
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


async def ensure_user(db: AsyncSession, sub: str, email: str, tier_claim: str) -> User:
    """Idempotent upsert từ JWT claim. Trả về User row."""
    tier = UserTier.pro if (tier_claim or "").lower() == "pro" else UserTier.free
    stmt = (
        pg_insert(User)
        .values(id=sub, email=email, tier=tier)
        .on_conflict_do_update(
            index_elements=[User.id],
            set_={"email": email, "tier": tier},
        )
        .returning(User)
    )
    row = (await db.execute(stmt)).scalar_one()
    await db.commit()
    return row


async def check_and_consume(db: AsyncSession, user: User) -> UsageQuota:
    """
    Atomic: INSERT (jobs_used=1) ON CONFLICT DO UPDATE SET jobs_used = jobs_used + 1
    WHERE jobs_used < limit. Nếu rowcount=0 → 429.
    """
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
