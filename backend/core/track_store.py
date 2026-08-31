"""
Per-camera track history.

Turns a stream of tracked Detections into a small in-memory record per
track_id — how long it's been on screen, which way it's moving, and a rough
speed estimate (Section 8: "Position / Direction / Speed estimation if
possible / Time first detected / Time last detected"). Zone entered/exited
is added in Phase 5 once restricted zones exist.

Speed is in pixels/second, not a real-world unit — there's no camera
calibration in this prototype, so we don't claim an accuracy we don't have.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from backend.core.interfaces import Detection

# 8-way compass direction from a movement vector, image coordinates
# (y grows downward, so dy is negated before the angle is taken).
_DIRECTIONS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


@dataclass
class TrackRecord:
    track_id: int
    class_name: str
    first_seen: datetime
    last_seen: datetime
    positions: list[tuple[float, float, datetime]] = field(default_factory=list)  # (cx, cy, timestamp)

    def dwell_seconds(self) -> float:
        return (self.last_seen - self.first_seen).total_seconds()

    def direction(self) -> str | None:
        if len(self.positions) < 2:
            return None
        x1, y1, _ = self.positions[0]
        x2, y2, _ = self.positions[-1]
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) < 5 and abs(dy) < 5:
            return "stationary"
        angle = math.degrees(math.atan2(-dy, dx))
        index = round(angle / 45) % 8
        return _DIRECTIONS[index]

    def speed_px_per_second(self) -> float | None:
        if len(self.positions) < 2:
            return None
        x1, y1, t1 = self.positions[0]
        x2, y2, t2 = self.positions[-1]
        dt = (t2 - t1).total_seconds()
        if dt <= 0:
            return None
        distance = math.hypot(x2 - x1, y2 - y1)
        return round(distance / dt, 1)


class TrackHistory:
    """
    Tracks active trajectories for a single camera.

    Not independently thread-safe — the owning VideoStreamManager already
    serializes access to it under its own lock, same as latest_jpeg.
    """

    STALE_AFTER_SECONDS = 5.0
    MAX_POSITIONS = 30  # cap memory per track; only recent trajectory matters for direction/speed

    def __init__(self) -> None:
        self._records: dict[int, TrackRecord] = {}

    def update(self, detections: list[Detection]) -> None:
        now = datetime.now(timezone.utc)

        for det in detections:
            if det.track_id is None:
                continue  # untracked (e.g. NullTracker fallback) — nothing to record

            cx = (det.bbox[0] + det.bbox[2]) / 2
            cy = (det.bbox[1] + det.bbox[3]) / 2

            record = self._records.get(det.track_id)
            if record is None:
                record = TrackRecord(track_id=det.track_id, class_name=det.class_name, first_seen=now, last_seen=now)
                self._records[det.track_id] = record

            record.last_seen = now
            record.positions.append((cx, cy, now))
            if len(record.positions) > self.MAX_POSITIONS:
                record.positions.pop(0)

        self._prune_stale(now)

    def _prune_stale(self, now: datetime) -> None:
        stale_ids = [
            track_id
            for track_id, record in self._records.items()
            if (now - record.last_seen).total_seconds() > self.STALE_AFTER_SECONDS
        ]
        for track_id in stale_ids:
            del self._records[track_id]

    def active_tracks(self) -> list[TrackRecord]:
        return list(self._records.values())
