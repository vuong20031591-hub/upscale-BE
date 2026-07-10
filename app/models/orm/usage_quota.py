"""UsageQuota — số job đã dùng trong period hiện tại."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.orm.base import Base


class UsageQuota(Base):
    """
    Một row / user / period.
    - free tier: period = ngày UTC (YYYY-MM-DD)
    - pro tier:  period = tháng UTC (YYYY-MM)
    """

    __tablename__ = "usage_quotas"

    user_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    period_key: Mapped[str] = mapped_column(String(16), primary_key=True)
    jobs_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
