"""
Quota service — check & consume 1 job cho user hiện tại.

- free: N jobs/ngày (QUOTA_FREE_PER_DAY, default 5)
- pro:  M jobs/tháng (QUOTA_PRO_PER_MONTH, default 500)

Dùng atomic UPSERT + WHERE để tránh race condition ở tầng DB.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import UsageQuota, User, UserTier

FREE_PER_DAY = int(os.getenv("QUOTA_FREE_PER_DAY", "5"))
PRO_PER_MONTH = int(os.getenv("QUOTA_PRO_PER_MONTH", "500"))


def _period_key(tier: UserTier, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    if tier == UserTier.pro:
        return now.strftime("%Y-%m")
    return now.strftime("%Y-%m-%d")


def _limit_for(tier: UserTier) -> int:
    return PRO_PER_MONTH if tier == UserTier.pro else FREE_PER_DAY


async def ensure_user(db: AsyncSession, sub: str, email: str, tier_claim: str) -> User:
    """Upsert user row từ JWT claim (idempotent)."""
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
    Tăng jobs_used += 1 nếu chưa vượt limit; else 429.
    Atomic: dùng UPSERT + subquery filter.
    """
    limit = _limit_for(user.tier)
    period = _period_key(user.tier)

    # Ensure row exists
    upsert = (
        pg_insert(UsageQuota)
        .values(user_id=user.id, period_key=period, jobs_used=0)
        .on_conflict_do_nothing(index_elements=[UsageQuota.user_id, UsageQuota.period_key])
    )
    await db.execute(upsert)

    # Đọc current, kiểm tra, rồi update
    q = await db.execute(
        select(UsageQuota).where(
            UsageQuota.user_id == user.id, UsageQuota.period_key == period
        ).with_for_update()
    )
    row = q.scalar_one()

    if row.jobs_used >= limit:
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Quota exceeded for tier '{user.tier.value}': "
                f"{row.jobs_used}/{limit} in period {period}"
            ),
        )

    row.jobs_used += 1
    await db.commit()
    await db.refresh(row)
    return row
