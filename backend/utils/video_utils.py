"""Small helpers for frame encoding and upload validation."""

from pathlib import Path

import cv2
import numpy as np

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def is_allowed_video_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes | None:
    """Encode a BGR frame (as returned by OpenCV) to JPEG bytes."""
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return buffer.tobytes()


def frame_brightness(frame: np.ndarray) -> float:
    """Mean grayscale brightness of a frame, 0-255. Used by night detection (Phase 5)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))
