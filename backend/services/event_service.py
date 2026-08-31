"""
Event service.

Unlike camera_service, this is called from VideoStreamManager's background
capture thread, not from a FastAPI request — so it can't use the get_db
request-scoped dependency. It opens and closes its own short-lived session
per call instead.
"""

import uuid
from datetime import datetime, timezone

from backend.config import get_settings
from backend.models.database import SessionLocal
from backend.models.event import Event, EventType, Severity
from backend.utils.logger import get_logger
from backend.utils.video_utils import encode_jpeg

logger = get_logger(__name__)
settings = get_settings()


def save_evidence_snapshot(camera_id: str, event_type: EventType, frame) -> str | None:
    """Encode and save a frame as evidence; returns the saved path, or None on failure."""
    jpeg = encode_jpeg(frame)
    if jpeg is None:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"{camera_id}_{event_type.value}_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
    path = settings.evidence_dir / filename

    try:
        path.write_bytes(jpeg)
        return str(path)
    except OSError:
        logger.exception("Failed to save evidence snapshot for camera %s", camera_id)
        return None


def create_event(
    camera_id: str,
    event_type: EventType,
    severity: Severity,
    object_type: str | None = None,
    track_id: int | None = None,
    confidence: float | None = None,
    description: str | None = None,
    frame=None,
) -> None:
    """
    Persist one event, with an optional evidence snapshot.

    Opens its own DB session since it's called from a background thread,
    not a FastAPI request.
    """
    snapshot_path = save_evidence_snapshot(camera_id, event_type, frame) if frame is not None else None

    db = SessionLocal()
    try:
        event = Event(
            camera_id=camera_id,
            event_type=event_type,
            object_type=object_type,
            track_id=track_id,
            confidence=confidence,
            severity=severity,
            snapshot_path=snapshot_path,
            description=description,
        )
        db.add(event)
        db.commit()
        logger.info("Event logged: %s camera=%s severity=%s track=%s", event_type.value, camera_id, severity.value, track_id)
    except Exception:
        logger.exception("Failed to write event for camera %s", camera_id)
        db.rollback()
    finally:
        db.close()
