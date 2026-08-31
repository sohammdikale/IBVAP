"""
Multi-object tracking engine (Phase 4).

ByteTracker wraps `supervision`'s ByteTrack implementation and is the real
engine — every person/vehicle gets a persistent track_id that survives
across frames (Section 8). NullTracker is an explicit pass-through fallback
if `supervision` fails to load, same philosophy as DemoDetector: never
silently pretend untracked detections are tracked.

One tracker instance is needed PER CAMERA — ByteTrack keeps internal frame
history, so sharing one instance across cameras would corrupt tracking for
all of them. Callers must build a fresh instance per camera (see
camera_service.start_camera_stream).
"""

import numpy as np

from backend.core.detector import RELEVANT_CLASSES
from backend.core.interfaces import Detection, Tracker
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# supervision's Detections needs integer class_ids; build a stable mapping
# from our class name strings so IDs are consistent run to run.
_CLASS_NAMES = sorted(RELEVANT_CLASSES)
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(_CLASS_NAMES)}
ID_TO_CLASS_NAME = {i: name for name, i in CLASS_NAME_TO_ID.items()}


class ByteTracker(Tracker):
    """Real tracker backed by supervision's ByteTrack."""

    def __init__(self) -> None:
        import supervision as sv  # imported lazily so Phases 1-3 don't require it

        self._sv = sv
        self._tracker = sv.ByteTrack()

    def update(self, detections: list[Detection]) -> list[Detection]:
        sv = self._sv

        if not detections:
            sv_detections = sv.Detections.empty()
        else:
            xyxy = np.array([d.bbox for d in detections], dtype=float)
            confidence = np.array([d.confidence for d in detections], dtype=float)
            class_id = np.array([CLASS_NAME_TO_ID.get(d.class_name, 0) for d in detections], dtype=int)
            sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

        tracked = self._tracker.update_with_detections(sv_detections)

        results: list[Detection] = []
        for i in range(len(tracked)):
            x1, y1, x2, y2 = (int(v) for v in tracked.xyxy[i])
            class_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
            class_name = ID_TO_CLASS_NAME.get(class_id, "object")
            confidence = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None
            results.append(
                Detection(
                    class_name=class_name,
                    confidence=round(confidence, 4),
                    bbox=(x1, y1, x2, y2),
                    track_id=track_id,
                )
            )
        return results


class NullTracker(Tracker):
    """Explicit fallback if ByteTrack can't be initialized — passes detections through with no track_id."""

    def update(self, detections: list[Detection]) -> list[Detection]:
        return detections


def build_tracker() -> Tracker:
    """Factory: try ByteTracker, fall back to NullTracker (with a loud warning) on failure."""
    try:
        return ByteTracker()
    except Exception:
        logger.exception("Could not initialize ByteTrack — falling back to NullTracker (no persistent track IDs).")
        return NullTracker()
