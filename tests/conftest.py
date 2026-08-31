"""
Shared test fixtures.

Loading the YOLO and EasyOCR models is a one-time, multi-second cost each.
Without warming them up here, whichever test file happens to run first eats
that cost inside its own short per-test wait loop and can time out — not a
real bug, just test ordering. Loading them once, session-wide, before any
test runs removes that flakiness.
"""

import pytest

from backend.services import camera_service


@pytest.fixture(scope="session", autouse=True)
def warm_up_shared_detector():
    camera_service._get_shared_detector()


@pytest.fixture(scope="session", autouse=True)
def warm_up_shared_anpr_engine():
    camera_service._get_shared_anpr_engine()


@pytest.fixture(scope="session", autouse=True)
def _stop_all_camera_streams_at_session_end():
    """
    Ensure no capture thread is left running when the test process exits.

    Capture threads are daemons calling into OpenCV/PyTorch native code; an
    abrupt process exit while one is mid-frame can trigger a native abort
    during interpreter shutdown. Stopping every stream a test started avoids
    that regardless of which tests ran or in what order.
    """
    yield
    for camera_id in list(camera_service._active_managers.keys()):
        camera_service.stop_camera_stream(camera_id)
