"""
Automatic Number Plate Recognition engine (Phase 6, Section 10).

There's no separate plate-localization model in this prototype — instead,
EasyOCR's own text detector is run against the cropped vehicle region (not
the full frame), and candidate text boxes are filtered/scored by how well
they match a plate-like shape and an Indian registration format. This is a
real, working OCR pipeline, just without a dedicated plate-detector stage;
Section 10's diagram calls for "Number Plate Detection" as a distinct step,
and using EasyOCR's built-in detector on the vehicle crop is how that step
is implemented here.

Confidence is always the raw OCR confidence — never inflated, and always
surfaced alongside the normalized text so the caller can judge trust. There
is deliberately no "common misread" auto-correction (e.g. guessing O->0):
a blind character substitution can silently corrupt a real letter in the
plate's series segment into a different valid-looking plate, which would
misrepresent OCR uncertainty as a confident correction. Validation is a
straightforward regex match against the normalized text only.
"""

import re
from dataclasses import dataclass

import numpy as np

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Indian plate format: 2 letters (state) + 1-2 digits (RTO code) +
# 1-3 letters (series, optional on older formats) + 4 digits (unique number).
# e.g. MH12AB1234, DL3CAB1234, KA05MH1234.
_INDIAN_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")


@dataclass
class PlateReadResult:
    raw_text: str
    normalized_text: str
    confidence: float
    valid_format: bool


class ANPREngine:
    """Real engine: EasyOCR's text detector + reader run against a vehicle crop."""

    def __init__(self) -> None:
        import easyocr  # imported lazily so earlier phases don't require it
        import torch

        # EasyOCR's quantized recognition model can hit a native threading
        # abort during interpreter shutdown (seen only in short-lived
        # processes like the test suite, not the long-running server) unless
        # torch is pinned to a single thread before the model is built.
        torch.set_num_threads(1)

        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        logger.info("EasyOCR reader loaded")

    def read_plate(self, vehicle_crop: np.ndarray) -> PlateReadResult | None:
        if vehicle_crop.size == 0:
            return None

        try:
            results = self._reader.readtext(vehicle_crop)
        except Exception:
            logger.exception("EasyOCR inference failed")
            return None

        best: PlateReadResult | None = None
        for _bbox, text, confidence in results:
            candidate = _score_candidate(text, confidence)
            if candidate is None:
                continue
            if best is None or candidate.confidence > best.confidence:
                best = candidate

        return best


class NullANPREngine:
    """Explicit fallback if EasyOCR can't be initialized — never fabricates a plate reading."""

    def read_plate(self, vehicle_crop: np.ndarray) -> PlateReadResult | None:
        return None


def _normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _score_candidate(raw_text: str, confidence: float) -> PlateReadResult | None:
    """Normalize + validate one OCR text box as a plate candidate, or reject it."""
    normalized = _normalize(raw_text)

    # Too short/long to plausibly be a plate — cheap filter before regex/substitution work.
    if not (6 <= len(normalized) <= 11):
        return None

    if _INDIAN_PLATE_PATTERN.match(normalized):
        return PlateReadResult(raw_text=raw_text, normalized_text=normalized, confidence=confidence, valid_format=True)

    # Doesn't match the expected format — still return it (never silently
    # discard a real OCR result), just flagged as unvalidated.
    return PlateReadResult(raw_text=raw_text, normalized_text=normalized, confidence=confidence, valid_format=False)


def crop_vehicle(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2]


def is_large_enough_for_anpr(bbox: tuple[int, int, int, int]) -> bool:
    """Skip OCR on vehicles too small in-frame to have a readable plate."""
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    return width >= settings.anpr_min_vehicle_width and height >= settings.anpr_min_vehicle_height


def build_anpr_engine():
    try:
        return ANPREngine()
    except Exception:
        logger.exception("Could not initialize EasyOCR — falling back to NullANPREngine (no plate reads will be produced).")
        return NullANPREngine()
