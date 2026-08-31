"""
Camera service.

Owns two things:
1. DB CRUD for the Camera table.
2. An in-memory registry mapping camera_id -> running VideoStreamManager,
   since the actual capture threads are process-local and don't belong in
   the database.
"""

from pathlib import Path

from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.core.anpr_engine import build_anpr_engine
from backend.core.detector import build_detector
from backend.core.intrusion_engine import parse_zone
from backend.core.tracker import build_tracker
from backend.core.video_processor import VideoStreamManager
from backend.models.camera import Camera, CameraStatus, CameraType
from backend.models.schemas import CameraCreate, CameraUpdate
from backend.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# camera_id -> VideoStreamManager, for cameras currently started in this process.
_active_managers: dict[str, VideoStreamManager] = {}

# The YOLO and EasyOCR models are expensive to load — build each once and
# share across every camera's stream manager rather than loading per-camera.
_detector = None
_anpr_engine = None


def _get_shared_detector():
    global _detector
    if _detector is None:
        _detector = build_detector()
    return _detector


def _get_shared_anpr_engine():
    global _anpr_engine
    if _anpr_engine is None:
        _anpr_engine = build_anpr_engine()
    return _anpr_engine


class CameraNotFoundError(Exception):
    pass


class CameraAlreadyExistsError(Exception):
    pass


def create_camera(db: Session, payload: CameraCreate) -> Camera:
    existing = db.query(Camera).filter(Camera.camera_id == payload.camera_id).first()
    if existing is not None:
        raise CameraAlreadyExistsError(f"Camera '{payload.camera_id}' already exists")

    camera = Camera(
        camera_id=payload.camera_id,
        name=payload.name,
        source_type=payload.source_type,
        source_uri=payload.source_uri,
        location=payload.location,
        status=CameraStatus.OFFLINE,
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)
    logger.info("Camera created: %s (%s)", camera.camera_id, camera.source_type.value)
    return camera


def create_camera_from_upload(db: Session, camera_id: str, name: str, location: str | None, saved_path: Path) -> Camera:
    payload = CameraCreate(
        camera_id=camera_id,
        name=name,
        source_type=CameraType.UPLOADED_VIDEO,
        source_uri=str(saved_path),
        location=location,
    )
    return create_camera(db, payload)


def list_cameras(db: Session) -> list[Camera]:
    return db.query(Camera).order_by(Camera.created_at.desc()).all()


def get_camera(db: Session, camera_id: str) -> Camera:
    camera = db.query(Camera).filter(Camera.camera_id == camera_id).first()
    if camera is None:
        raise CameraNotFoundError(f"Camera '{camera_id}' not found")
    return camera


def update_camera(db: Session, camera_id: str, payload: CameraUpdate) -> Camera:
    camera = get_camera(db, camera_id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(camera, field, value)
    db.commit()
    db.refresh(camera)
    return camera


def delete_camera(db: Session, camera_id: str) -> None:
    camera = get_camera(db, camera_id)
    stop_camera_stream(camera_id)  # ensure no orphaned thread keeps running
    db.delete(camera)
    db.commit()
    logger.info("Camera deleted: %s", camera_id)


def start_camera_stream(db: Session, camera_id: str) -> VideoStreamManager:
    """Start (or return the already-running) capture thread for a camera."""
    camera = get_camera(db, camera_id)

    existing = _active_managers.get(camera_id)
    if existing is not None and existing.status == CameraStatus.ONLINE:
        return existing

    manager = VideoStreamManager(
        camera_id=camera.camera_id,
        source_uri=camera.source_uri,
        source_type=camera.source_type,
        process_fps=settings.process_fps,
        detector=_get_shared_detector(),
        tracker=build_tracker(),  # fresh instance per camera — ByteTrack keeps internal per-stream state
        restricted_zone=parse_zone(camera.restricted_zone),
        anpr_engine=_get_shared_anpr_engine(),
    )
    manager.start()
    _active_managers[camera_id] = manager

    camera.status = manager.status
    db.commit()

    return manager


def stop_camera_stream(camera_id: str) -> None:
    manager = _active_managers.pop(camera_id, None)
    if manager is not None:
        manager.stop()


def get_active_manager(camera_id: str) -> VideoStreamManager | None:
    return _active_managers.get(camera_id)


def sync_camera_status(db: Session, camera_id: str) -> Camera:
    """Refresh the DB status field from the live manager's current state, if running."""
    camera = get_camera(db, camera_id)
    manager = _active_managers.get(camera_id)
    if manager is not None:
        camera.status = manager.status
        db.commit()
        db.refresh(camera)
    return camera
