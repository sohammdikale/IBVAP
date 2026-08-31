"""
Structured application logging.

Every module should get its logger via `get_logger(__name__)` rather than
calling `logging.getLogger` directly, so formatting/level stay consistent
across the whole app (console now, file/JSON handlers can be added later
without touching call sites).
"""

import logging
import sys

from backend.config import get_settings

_CONFIGURED = False


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured module-level logger."""
    _configure_root_logger()
    return logging.getLogger(name)
