"""Jobs router — list & get status."""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_current_user
from app.db import get_db
from app.models.orm import Job, JobStatus

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("")
async def list_my_jobs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status_filter: Optional[JobStatus] = Query(None, alias="status"),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Job).where(Job.user_id == user.sub).order_by(Job.created_at.desc())
    if status_filter is not None:
        stmt = stmt.where(Job.status == status_filter)
    stmt = stmt.limit(limit).offset(offset)
    rows = (await db.execute(stmt)).scalars().all()
    return {
        "items": [
            {
                "id": str(r.id),
                "status": r.status.value,
                "mode": r.mode.value,
                "output_key": r.output_key,
                "error": r.error,
                "created_at": r.created_at.isoformat(),
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            }
            for r in rows
        ],
        "limit": limit,
        "offset": offset,
    }


@router.get("/{job_id}")
async def get_job(
    job_id: uuid.UUID,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    row = (
        await db.execute(select(Job).where(Job.id == job_id, Job.user_id == user.sub))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return {
        "id": str(row.id),
        "status": row.status.value,
        "mode": row.mode.value,
        "input_key": row.input_key,
        "output_key": row.output_key,
        "error": row.error,
        "created_at": row.created_at.isoformat(),
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
    }
