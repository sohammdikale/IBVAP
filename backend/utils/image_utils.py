"""Drawing helpers for detection overlays on video frames."""

import cv2
import numpy as np

from backend.core.interfaces import Detection

# BGR colors per class group — person distinct from vehicles, for a scannable feed.
_PERSON_COLOR = (60, 200, 60)  # green
_VEHICLE_COLOR = (60, 140, 255)  # orange
_DEMO_BANNER_COLOR = (0, 0, 255)  # red

_VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}


def draw_zone(frame: np.ndarray, zone: list[tuple[float, float]] | None) -> np.ndarray:
    """Draw the restricted-zone polygon outline (Section 9's virtual fence) onto a copy of the frame."""
    if not zone:
        return frame
    annotated = frame.copy()
    points = np.array(zone, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated, [points], isClosed=True, color=(0, 0, 255), thickness=2)
    return annotated


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    demo_mode: bool = False,
    plate_lookup: dict[int, object] | None = None,
) -> np.ndarray:
    """
    Draw bounding boxes + labels onto a copy of the frame and return it.

    plate_lookup, if given, maps track_id -> PlateReadResult (or None) from
    the ANPR engine (Phase 6) — recognized plate text is drawn under the
    vehicle's box when available.
    """
    annotated = frame.copy()

    for det in detections:
        color = _PERSON_COLOR if det.class_name == "person" else _VEHICLE_COLOR
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name} {det.confidence:.0%}"
        if det.track_id is not None:
            label = f"#{det.track_id} {label}"

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        plate_result = plate_lookup.get(det.track_id) if plate_lookup and det.track_id is not None else None
        if plate_result is not None:
            plate_label = f"Plate: {plate_result.normalized_text} ({plate_result.confidence:.0%})"
            (pw, ph), _ = cv2.getTextSize(plate_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y2), (x1 + pw + 4, y2 + ph + 6), (0, 0, 0), -1)
            cv2.putText(annotated, plate_label, (x1 + 2, y2 + ph + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    if demo_mode:
        cv2.putText(
            annotated,
            "DEMO MODE — NO REAL AI DETECTION",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _DEMO_BANNER_COLOR,
            2,
        )

    return annotated


def count_by_class(detections: list[Detection]) -> dict[str, int]:
    """e.g. {'person': 3, 'vehicle': 1} — vehicle classes are summed together."""
    counts = {"person": 0, "vehicle": 0}
    for det in detections:
        if det.class_name == "person":
            counts["person"] += 1
        elif det.class_name in _VEHICLE_CLASSES:
            counts["vehicle"] += 1
    return counts
