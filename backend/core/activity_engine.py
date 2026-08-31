"""
Suspicious activity engine (Phase 5, Section 12).

Deliberately rule-based and limited to one measurable signal for this
phase — dwell time exceeding a configurable threshold ("loitering") — per
the project's explicit instruction not to pretend a full behavior-
recognition model exists. Additional signals (repeated zone approaches,
multiple people entering a zone, vehicle stopped in a restricted area) can
be added here later without changing the interface other engines use.
"""

from backend.core.track_store import TrackRecord


def is_loitering(record: TrackRecord, threshold_seconds: int) -> bool:
    return record.dwell_seconds() >= threshold_seconds
