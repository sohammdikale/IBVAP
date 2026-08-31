"""
Object detection engine (Phase 3).

YOLODetector wraps Ultralytics YOLO and is the real engine. DemoDetector is
an explicit, clearly-labeled fallback used only if the YOLO model can't be
loaded (e.g. no weights, no internet on first run, unsupported hardware) —
per the project rule, we never let a mock silently pass as a real
prediction, so DetectionResult.demo_mode is always set and surfaced by the
API/dashboard.
"""

from pathlib import Path

import numpy as np

from backend.config import get_settings
from backend.core.interfaces import Detection, DetectionResult, Detector
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Classes IBVAP cares about (Section 7): person + vehicle types.
# COCO class names — kept as a set for O(1) filtering of YOLO's full 80-class output.
RELEVANT_CLASSES = {"person", "car", "motorcycle", "bus", "truck", "bicycle"}


class YOLODetector(Detector):
    """Real detector backed by an Ultralytics YOLO model."""

    def __init__(self, model_path: str, confidence_threshold: float) -> None:
        from ultralytics import YOLO  # imported lazily so Phase 1/2 don't require torch installed

        self.confidence_threshold = confidence_threshold
        self._model = YOLO(model_path)
        logger.info("YOLO model loaded from %s", model_path)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        results = self._model.predict(source=frame, conf=self.confidence_threshold, verbose=False)

        detections: list[Detection] = []
        for r in results:
            for box in r.boxes:
                class_name = self._model.names[int(box.cls[0])]
                if class_name not in RELEVANT_CLASSES:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=round(float(box.conf[0]), 4),
                        bbox=(x1, y1, x2, y2),
                    )
                )

        return DetectionResult(detections=detections, demo_mode=False)


class DemoDetector(Detector):
    """
    Explicit fallback when no real model is available.

    Returns no detections rather than fabricating plausible-looking boxes —
    "no AI available" must never be dressed up as "AI found nothing here".
    demo_mode=True is always set so callers (API, dashboard) can display it
    honestly instead of hiding the fact that this isn't real inference.
    """

    def detect(self, frame: np.ndarray) -> DetectionResult:
        return DetectionResult(detections=[], demo_mode=True)


def build_detector() -> Detector:
    """
    Factory: try to load the real YOLO detector; fall back to DemoDetector
    (with a loud warning) if that fails for any reason.
    """
    settings = get_settings()
    try:
        return YOLODetector(settings.yolo_model, settings.confidence_threshold)
    except Exception:
        logger.exception(
            "Could not load YOLO model '%s' — falling back to DemoDetector "
            "(no real detections will be produced until this is fixed).",
            settings.yolo_model,
        )
        return DemoDetector()
