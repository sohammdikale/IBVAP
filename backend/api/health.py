"""Health/status endpoint — confirms the API and DB are reachable."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.models.database import get_db

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health_check(db: Session = Depends(get_db)) -> dict:
    settings = get_settings()

    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    return {
        "app_name": settings.app_name,
        "status": "online" if db_ok else "degraded",
        "database": "connected" if db_ok else "unreachable",
        "environment": settings.app_env,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
