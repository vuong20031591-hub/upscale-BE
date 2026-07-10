"""initial: users, jobs, usage_quotas

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-10
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    user_tier = sa.Enum("free", "pro", name="user_tier")
    job_status = sa.Enum("queued", "processing", "succeeded", "failed", name="job_status")
    job_mode = sa.Enum(
        "basic",
        "smart",
        "face_restoration",
        "face_colorization",
        "face_inpainting",
        name="job_mode",
    )
    user_tier.create(op.get_bind(), checkfirst=True)
    job_status.create(op.get_bind(), checkfirst=True)
    job_mode.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "users",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("tier", user_tier, nullable=False, server_default="free"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(64), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", job_status, nullable=False, server_default="queued"),
        sa.Column("mode", job_mode, nullable=False),
        sa.Column("input_key", sa.String(512), nullable=True),
        sa.Column("output_key", sa.String(512), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_jobs_user_id", "jobs", ["user_id"])
    op.create_index("ix_jobs_status", "jobs", ["status"])

    op.create_table(
        "usage_quotas",
        sa.Column("user_id", sa.String(64), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("period_key", sa.String(16), primary_key=True),
        sa.Column("jobs_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("usage_quotas")
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_index("ix_jobs_user_id", table_name="jobs")
    op.drop_table("jobs")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
    sa.Enum(name="job_mode").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="job_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="user_tier").drop(op.get_bind(), checkfirst=True)
