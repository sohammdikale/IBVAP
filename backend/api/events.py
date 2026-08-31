"""Events API (Phase 5) — a basic listing/filtering view; the full Events management page (search, date range, click-through detail) is Phase 9."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.models.database import get_db
from backend.models.event import Event
from backend.models.schemas import EventOut

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("", response_model=list[EventOut])
def list_events(
    camera_id: str | None = Query(None),
    event_type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    query = db.query(Event)
    if camera_id:
        query = query.filter(Event.camera_id == camera_id)
    if event_type:
        query = query.filter(Event.event_type == event_type)
    return query.order_by(Event.timestamp.desc()).limit(limit).all()


@router.get("/{event_id}", response_model=EventOut)
def get_event(event_id: int, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
    return event


@router.get("/{event_id}/snapshot")
def get_event_snapshot(event_id: int, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
    if not event.snapshot_path or not Path(event.snapshot_path).exists():
        raise HTTPException(status_code=404, detail="No snapshot available for this event")
    return FileResponse(event.snapshot_path, media_type="image/jpeg")
