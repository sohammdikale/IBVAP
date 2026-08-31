"""Camera management API (Phase 4)."""

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.core.video_processor import mjpeg_stream
from backend.models.camera import CameraStatus
from backend.models.database import get_db
from backend.models.schemas import (
    CameraCreate,
    CameraOut,
    CameraStatusOut,
    CameraUpdate,
    DetectionSummaryOut,
    PlateReadOut,
    TrackOut,
)
from backend.services import camera_service
from backend.services.camera_service import CameraAlreadyExistsError, CameraNotFoundError
from backend.utils.logger import get_logger
from backend.utils.video_utils import is_allowed_video_file

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.post("", response_model=CameraOut, status_code=201)
def create_camera(payload: CameraCreate, db: Session = Depends(get_db)):
    """Register an RTSP or webcam camera. For uploaded video files, use POST /upload instead."""
    try:
        return camera_service.create_camera(db, payload)
    except CameraAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/upload", response_model=CameraOut, status_code=201)
async def upload_camera_video(
    name: str = Form(...),
    location: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a video file and register it as a demo/uploaded-video camera."""
    if not file.filename or not is_allowed_video_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .mp4, .avi, .mov, or .mkv")

    camera_id = f"CAM-{uuid.uuid4().hex[:8].upper()}"
    dest_path = settings.videos_dir / f"{camera_id}_{file.filename}"

    with dest_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    try:
        return camera_service.create_camera_from_upload(db, camera_id, name, location, dest_path)
    except CameraAlreadyExistsError as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("", response_model=list[CameraOut])
def list_cameras(db: Session = Depends(get_db)):
    return camera_service.list_cameras(db)


@router.get("/{camera_id}", response_model=CameraOut)
def get_camera(camera_id: str, db: Session = Depends(get_db)):
    try:
        return camera_service.sync_camera_status(db, camera_id)
    except CameraNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{camera_id}", response_model=CameraOut)
def update_camera(camera_id: str, payload: CameraUpdate, db: Session = Depends(get_db)):
    try:
        return camera_service.update_camera(db, camera_id, payload)
    except CameraNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{camera_id}", status_code=204)
def delete_camera(camera_id: str, db: Session = Depends(get_db)):
    try:
        camera_service.delete_camera(db, camera_id)
    except CameraNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{camera_id}/start", response_model=CameraStatusOut)
def start_camera(camera_id: str, db: Session = Depends(get_db)):
    try:
        manager = camera_service.start_camera_stream(db, camera_id)
    except CameraNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return CameraStatusOut(camera_id=camera_id, status=manager.status, error_message=manager.error_message)


@router.post("/{camera_id}/stop", response_model=CameraStatusOut)
def stop_camera(camera_id: str, db: Session = Depends(get_db)):
    try:
        camera_service.get_camera(db, camera_id)  # 404 if unknown
    except CameraNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    camera_service.stop_camera_stream(camera_id)
    camera = camera_service.get_camera(db, camera_id)
    camera.status = CameraStatus.OFFLINE
    db.commit()

    return CameraStatusOut(camera_id=camera_id, status=CameraStatus.OFFLINE)


@router.get("/{camera_id}/stream")
def stream_camera(camera_id: str, db: Session = Depends(get_db)):
    try:
        camera_service.get_camera(db, camera_id)  # 404 if unknown
    except CameraNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    manager = camera_service.get_active_manager(camera_id)
    if manager is None or manager.status != CameraStatus.ONLINE:
        raise HTTPException(status_code=409, detail="Camera is not running. Start it first via /start.")

    return StreamingResponse(
        mjpeg_stream(manager, target_fps=settings.process_fps),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/{camera_id}/detections", response_model=DetectionSummaryOut)
def get_detections(camera_id: str, db: Session = Depends(get_db)):
    try:
        camera_service.get_camera(db, camera_id)  # 404 if unknown
    except CameraNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    manager = camera_service.get_active_manager(camera_id)
    if manager is None or manager.status != CameraStatus.ONLINE:
        raise HTTPException(status_code=409, detail="Camera is not running. Start it first via /start.")

    counts = manager.get_latest_counts()
    latest = manager.get_latest_detection()
    demo_mode = latest.demo_mode if latest is not None else False

    return DetectionSummaryOut(camera_id=camera_id, person=counts["person"], vehicle=counts["vehicle"], demo_mode=demo_mode)


@router.get("/{camera_id}/tracks", response_model=list[TrackOut])
def get_tracks(camera_id: str, db: Session = Depends(get_db)):
    """Currently active tracked objects for a camera — persistent IDs, dwell time, direction, speed estimate."""
    try:
        camera_service.get_camera(db, camera_id)  # 404 if unknown
    except CameraNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    manager = camera_service.get_active_manager(camera_id)
    if manager is None or manager.status != CameraStatus.ONLINE:
        raise HTTPException(status_code=409, detail="Camera is not running. Start it first via /start.")

    return [
        TrackOut(
            track_id=record.track_id,
            class_name=record.class_name,
            first_seen=record.first_seen,
            last_seen=record.last_seen,
            dwell_seconds=round(record.dwell_seconds(), 2),
            direction=record.direction(),
            speed_px_per_second=record.speed_px_per_second(),
        )
        for record in manager.get_active_tracks()
    ]


@router.get("/{camera_id}/plates", response_model=list[PlateReadOut])
def get_plates(camera_id: str, db: Session = Depends(get_db)):
    """Currently known plate reads for vehicle tracks on this camera — one OCR attempt per track (Phase 6)."""
    try:
        camera_service.get_camera(db, camera_id)  # 404 if unknown
    except CameraNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    manager = camera_service.get_active_manager(camera_id)
    if manager is None or manager.status != CameraStatus.ONLINE:
        raise HTTPException(status_code=409, detail="Camera is not running. Start it first via /start.")

    plates = []
    for track_id, result in manager.get_active_plates().items():
        if result is None:
            plates.append(PlateReadOut(track_id=track_id, plate_text=None, confidence=None, valid_format=None))
        else:
            plates.append(
                PlateReadOut(
                    track_id=track_id,
                    plate_text=result.normalized_text,
                    confidence=result.confidence,
                    valid_format=result.valid_format,
                )
            )
    return plates
