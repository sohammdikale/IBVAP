"""Pydantic schemas for the cameras and events APIs."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from backend.models.camera import CameraStatus, CameraType


class CameraCreate(BaseModel):
    """Body for creating an RTSP or webcam camera. Uploaded videos use the /upload endpoint instead."""

    camera_id: str
    name: str
    source_type: CameraType
    source_uri: str  # RTSP URL, or webcam index as a string e.g. "0"
    location: str | None = None


class CameraUpdate(BaseModel):
    name: str | None = None
    source_uri: str | None = None
    location: str | None = None
    restricted_zone: str | None = None


class CameraOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    camera_id: str
    name: str
    source_type: CameraType
    source_uri: str
    location: str | None
    status: CameraStatus
    restricted_zone: str | None
    created_at: datetime


class CameraStatusOut(BaseModel):
    camera_id: str
    status: CameraStatus
    error_message: str | None = None


class DetectionSummaryOut(BaseModel):
    camera_id: str
    person: int
    vehicle: int
    demo_mode: bool


class TrackOut(BaseModel):
    track_id: int
    class_name: str
    first_seen: datetime
    last_seen: datetime
    dwell_seconds: float
    direction: str | None
    speed_px_per_second: float | None


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: str
    event_type: str
    object_type: str | None
    track_id: int | None
    confidence: float | None
    timestamp: datetime
    severity: str
    snapshot_path: str | None
    description: str | None
    status: str


class PlateReadOut(BaseModel):
    track_id: int
    plate_text: str | None
    confidence: float | None
    valid_format: bool | None
