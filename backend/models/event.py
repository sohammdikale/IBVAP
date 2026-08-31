"""Event ORM model — every detection/analytics event is logged here."""

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.database import Base


class EventType(str, enum.Enum):
    PERSON_DETECTED = "PERSON_DETECTED"
    VEHICLE_DETECTED = "VEHICLE_DETECTED"
    FACE_DETECTED = "FACE_DETECTED"
    KNOWN_FACE = "KNOWN_FACE"
    UNKNOWN_FACE = "UNKNOWN_FACE"
    PLATE_DETECTED = "PLATE_DETECTED"
    INTRUSION = "INTRUSION"
    LOITERING = "LOITERING"
    NIGHT_MOVEMENT = "NIGHT_MOVEMENT"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"


class Severity(str, enum.Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(50), ForeignKey("cameras.camera_id"), index=True)
    event_type: Mapped[EventType] = mapped_column(Enum(EventType), index=True)
    object_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    severity: Mapped[Severity] = mapped_column(Enum(Severity), default=Severity.INFO)
    snapshot_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="new")  # new | acknowledged | resolved

    def __repr__(self) -> str:
        return f"<Event {self.event_type} cam={self.camera_id} sev={self.severity}>"
