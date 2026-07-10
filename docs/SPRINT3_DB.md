# Sprint 3 — Database (RDS Postgres) + Quota

## Thêm mới

- `app/db/session.py` — async SQLAlchemy engine + `get_db` dependency.
- `app/models/orm/` — `User`, `Job`, `UsageQuota` + enums (`UserTier`, `JobStatus`, `JobMode`).
- `app/services/quota.py` — `ensure_user()` (upsert từ JWT), `check_and_consume()` (atomic per-period counter).
- `app/routers/jobs.py` — `GET /jobs`, `GET /jobs/{id}` (chỉ trả jobs của chính user).
- `alembic/` + `alembic.ini` + migration `0001_initial`.

## Biến môi trường mới

```
DATABASE_URL=postgresql+asyncpg://upscale:upscale@postgres:5432/upscale
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
SQL_ECHO=false

# Quota
QUOTA_FREE_PER_DAY=5
QUOTA_PRO_PER_MONTH=500
```

## Chạy migration

```bash
# local (compose đã khởi động postgres)
export DATABASE_URL=postgresql+asyncpg://upscale:upscale@localhost:5432/upscale
alembic upgrade head
```

Trên RDS: cùng lệnh, đổi `DATABASE_URL` sang connection string RDS (dùng IAM auth hoặc password), chạy trong bastion/EC2 cùng VPC.

## Sử dụng quota trong endpoint

Ví dụ áp cho `/upscale/basic`:

```python
from fastapi import Depends
from app.auth.deps import get_current_user
from app.db import get_db
from app.services.quota import ensure_user, check_and_consume

@router.post("/upscale")
async def upscale(..., claims=Depends(get_current_user), db=Depends(get_db)):
    user = await ensure_user(db, claims["sub"], claims["email"], claims.get("tier", "free"))
    await check_and_consume(db, user)
    # ... xử lý upscale
```

Sprint 4 sẽ thay xử lý sync bằng flow S3 presign + SQS + worker; khi đó `check_and_consume` gọi ở endpoint `POST /jobs` thay vì trong `/upscale/*` legacy.

## Local dev với Postgres

`docker-compose.yml` đã có service `postgres:16`. Chạy:

```bash
docker compose up -d postgres
alembic upgrade head
docker compose up api
```

## Notes

- `usage_quotas` là 1 row / user / period. `period_key`:
  - free tier: `YYYY-MM-DD` (reset mỗi ngày UTC)
  - pro tier:  `YYYY-MM` (reset mỗi tháng UTC)
  Không cần cron reset — key thay đổi tự nhiên theo thời gian.
- `check_and_consume` dùng `SELECT ... FOR UPDATE` để tránh race giữa 2 request cùng lúc.
- Job xử lý inline (`/upscale/*` cũ) vẫn hoạt động; Sprint 4 sẽ chuyển sang async qua `POST /jobs`.
