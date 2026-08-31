"""
Night-time movement detection engine (Phase 5, Section 13).

Combines two independent, cheap checks per Section 13's own pseudocode:
"if movement_detected and brightness < threshold: create NIGHT_MOVEMENT
alert" — a configured night-hours window, and actual frame brightness (so a
well-lit night scene near a floodlight doesn't spuriously trigger).
"""

from datetime import datetime, time

from backend.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_hhmm(value: str) -> time:
    hour, minute = (int(part) for part in value.split(":"))
    return time(hour=hour, minute=minute)


def is_night_time(now: datetime, night_start: str, night_end: str) -> bool:
    """
    True if `now` falls within the configured night window.

    Handles overnight windows correctly (e.g. 19:00-06:00 wraps past
    midnight); a same-day window (e.g. 22:00-23:00) works the normal way.
    """
    try:
        start = _parse_hhmm(night_start)
        end = _parse_hhmm(night_end)
    except (ValueError, AttributeError):
        logger.warning("Invalid night hours %r-%r; night detection disabled", night_start, night_end)
        return False

    current = now.time()

    if start <= end:
        return start <= current <= end
    # Overnight window (e.g. 19:00 -> 06:00): night if after start OR before end.
    return current >= start or current <= end
