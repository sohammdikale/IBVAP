"""
Abstract engine interfaces.

Every AI stage (detection now; tracking/face/ANPR in later phases) is built
against one of these small interfaces. A real engine and a demo/fallback
engine share the same call signature, so:

- the video pipeline never needs to know which one it's talking to
- switching models later is a config change, not a rewrite
- demo mode is explicit (DetectionResult.demo_mode) rather than the app
  silently pretending fake output is a real prediction
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Detection:
    """A single detected object in one frame."""

    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates
    track_id: int | None = None  # populated starting Phase 4


@dataclass
class DetectionResult:
    """Everything a detector produces for one frame."""

    detections: list[Detection]
    demo_mode: bool  # True if this came from a non-AI fallback, never silently hidden


class Detector(ABC):
    """Interface every object-detection engine implements."""

    @abstractmethod
    def detect(self, frame) -> DetectionResult:
        """Run detection on a single BGR frame (as returned by OpenCV) and return results."""
        raise NotImplementedError


class Tracker(ABC):
    """
    Interface every multi-object tracking engine implements (Phase 4).

    A tracker consumes one frame's detections (no track_id yet) and returns
    the same detections with persistent track_id values assigned — matching
    each box to the same real-world object it matched in previous frames.
    """

    @abstractmethod
    def update(self, detections: list[Detection]) -> list[Detection]:
        raise NotImplementedError
