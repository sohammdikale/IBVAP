"""Camera ORM model."""

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.database import Base


class CameraType(str, enum.Enum):
    RTSP = "rtsp"
    UPLOADED_VIDEO = "uploaded_video"
    WEBCAM = "webcam"


class CameraStatus(str, enum.Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DISABLED = "disabled"
    ERROR = "error"


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    source_type: Mapped[CameraType] = mapped_column(Enum(CameraType), default=CameraType.WEBCAM)
    source_uri: Mapped[str] = mapped_column(String(500))  # RTSP URL, file path, or webcam index
    location: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[CameraStatus] = mapped_column(Enum(CameraStatus), default=CameraStatus.OFFLINE)

    # Restricted zone as a JSON-encoded list of [x, y] polygon points (added Phase 5)
    restricted_zone: Mapped[str | None] = mapped_column(String(2000), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"<Camera {self.camera_id} ({self.status})>"
