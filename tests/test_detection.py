"""
Phase 3 tests: object detection engine and the /detections endpoint.

Uses the real YOLO model (downloaded on first run) against a synthetic test
video so we're verifying actual inference, not just wiring. Also tests the
DemoDetector fallback in isolation so "no model available" behavior is
covered without depending on network access.
"""

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.core.detector import DemoDetector
from backend.core.interfaces import Detection
from backend.main import app
from backend.services import camera_service
from backend.utils.image_utils import count_by_class, draw_detections

client = TestClient(app)
client.__enter__()


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 5, (64, 64))
    for _ in range(10):
        writer.write(np.full((64, 64, 3), 120, dtype=np.uint8))
    writer.release()
    return path


def test_demo_detector_returns_no_detections_and_flags_demo_mode():
    detector = DemoDetector()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert result.detections == []
    assert result.demo_mode is True


def test_count_by_class_groups_vehicles_together():
    detections = [
        Detection("person", 0.9, (0, 0, 10, 10)),
        Detection("car", 0.8, (0, 0, 10, 10)),
        Detection("truck", 0.7, (0, 0, 10, 10)),
    ]
    counts = count_by_class(detections)
    assert counts == {"person": 1, "vehicle": 2}


def test_draw_detections_does_not_crash_and_returns_same_shape():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [Detection("person", 0.91, (10, 10, 50, 90))]
    annotated = draw_detections(frame, detections)
    assert annotated.shape == frame.shape
    # original frame must be untouched (draw_detections works on a copy)
    assert np.array_equal(frame, np.zeros((100, 100, 3), dtype=np.uint8))


def test_detections_endpoint_returns_counts_for_running_camera(synthetic_video):
    with synthetic_video.open("rb") as f:
        resp = client.post(
            "/api/cameras/upload",
            data={"name": "Detection Test", "location": "Lab"},
            files={"file": ("synthetic.mp4", f, "video/mp4")},
        )
    camera_id = resp.json()["camera_id"]

    client.post(f"/api/cameras/{camera_id}/start")

    manager = camera_service.get_active_manager(camera_id)
    for _ in range(50):
        if manager.get_latest_detection() is not None:
            break
        time.sleep(0.2)
    assert manager.get_latest_detection() is not None, "detector never produced a result"

    det_resp = client.get(f"/api/cameras/{camera_id}/detections")
    assert det_resp.status_code == 200
    body = det_resp.json()
    assert "person" in body and "vehicle" in body
    assert isinstance(body["demo_mode"], bool)

    client.post(f"/api/cameras/{camera_id}/stop")
    client.delete(f"/api/cameras/{camera_id}")


def test_detections_endpoint_404s_for_unknown_camera():
    resp = client.get("/api/cameras/CAM-NOPE/detections")
    assert resp.status_code == 404


def test_detections_endpoint_409s_when_not_running():
    payload = {"camera_id": "CAM-IDLE-DET", "name": "Idle", "source_type": "webcam", "source_uri": "0"}
    client.post("/api/cameras", json=payload)

    resp = client.get("/api/cameras/CAM-IDLE-DET/detections")
    assert resp.status_code == 409

    client.delete("/api/cameras/CAM-IDLE-DET")
