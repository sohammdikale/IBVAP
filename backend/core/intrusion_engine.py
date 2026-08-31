"""
Virtual fence / intrusion detection engine (Phase 5, Section 9).

A restricted zone is an operator-defined polygon (list of [x, y] pixel
points) stored per-camera. Each tracked object's ground-contact point
(bottom-center of its bounding box — the closest approximation to "where
its feet/wheels are" from a single 2D box) is tested against the polygon
each frame.
"""

import json

import cv2
import numpy as np

from backend.core.interfaces import Detection
from backend.utils.logger import get_logger

logger = get_logger(__name__)

ZonePolygon = list[tuple[float, float]]


def parse_zone(raw: str | None) -> ZonePolygon | None:
    """
    Parse a stored zone string (JSON array of [x, y] pairs) into a polygon.

    Returns None for missing/invalid input rather than raising — an
    unparsable zone should disable intrusion checking for that camera, not
    crash the capture thread.
    """
    if not raw:
        return None
    try:
        points = json.loads(raw)
        polygon = [(float(p[0]), float(p[1])) for p in points]
        if len(polygon) < 3:
            logger.warning("Restricted zone has fewer than 3 points; ignoring")
            return None
        return polygon
    except (ValueError, TypeError, IndexError):
        logger.warning("Could not parse restricted zone %r; ignoring", raw)
        return None


def ground_point(detection: Detection) -> tuple[float, float]:
    """Bottom-center of the bounding box — the object's approximate ground contact point."""
    x1, y1, x2, y2 = detection.bbox
    return ((x1 + x2) / 2, y2)


def is_inside_zone(point: tuple[float, float], zone: ZonePolygon) -> bool:
    contour = np.array(zone, dtype=np.float32).reshape((-1, 1, 2))
    result = cv2.pointPolygonTest(contour, point, False)
    return result >= 0  # inside or exactly on the boundary
