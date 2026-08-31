"""
Phase 2 tests: camera CRUD, uploaded-video capture/streaming, error handling.

A tiny synthetic .mp4 is generated on the fly with OpenCV so the test suite
doesn't depend on any external media file, and works headlessly.
"""

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import camera_service

# TestClient must run as a context manager for FastAPI's lifespan (which
# calls init_db() to create tables) to execute. __enter__ here mirrors what
# `with TestClient(app) as client:` does, for a client shared across every
# test function in this module.
client = TestClient(app)
client.__enter__()


@pytest.fixture
def synthetic_video(tmp_path):
    """A 2-second, 5fps solid-color test video."""
    path = tmp_path / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 5, (64, 64))
    for _ in range(10):
        frame = np.full((64, 64, 3), 120, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_create_and_list_rtsp_camera():
    payload = {
        "camera_id": "CAM-TEST-01",
        "name": "Test Gate",
        "source_type": "rtsp",
        "source_uri": "rtsp://example.invalid/stream",
        "location": "North",
    }
    resp = client.post("/api/cameras", json=payload)
    assert resp.status_code == 201
    assert resp.json()["status"] == "offline"

    listed = client.get("/api/cameras").json()
    assert any(c["camera_id"] == "CAM-TEST-01" for c in listed)

    client.delete("/api/cameras/CAM-TEST-01")


def test_duplicate_camera_id_rejected():
    payload = {
        "camera_id": "CAM-DUP",
        "name": "Dup",
        "source_type": "webcam",
        "source_uri": "0",
    }
    first = client.post("/api/cameras", json=payload)
    assert first.status_code == 201

    second = client.post("/api/cameras", json=payload)
    assert second.status_code == 409

    client.delete("/api/cameras/CAM-DUP")


def test_starting_invalid_source_reports_error_not_crash():
    payload = {
        "camera_id": "CAM-BAD",
        "name": "Bad Source",
        "source_type": "rtsp",
        "source_uri": "rtsp://255.255.255.255/nonexistent",
    }
    client.post("/api/cameras", json=payload)

    resp = client.post("/api/cameras/CAM-BAD/start")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert body["error_message"] is not None

    client.post("/api/cameras/CAM-BAD/stop")
    client.delete("/api/cameras/CAM-BAD")


def test_upload_start_stream_and_stop(synthetic_video):
    with synthetic_video.open("rb") as f:
        resp = client.post(
            "/api/cameras/upload",
            data={"name": "Uploaded Demo", "location": "Lab"},
            files={"file": ("synthetic.mp4", f, "video/mp4")},
        )
    assert resp.status_code == 201
    camera = resp.json()
    camera_id = camera["camera_id"]
    assert camera["source_type"] == "uploaded_video"

    start_resp = client.post(f"/api/cameras/{camera_id}/start")
    assert start_resp.status_code == 200
    assert start_resp.json()["status"] == "online"

    # Give the capture thread a moment to produce a frame. With detection +
    # tracking now running per frame, first-frame latency is higher than
    # Phase 2 alone (YOLO inference on CPU), so allow more time.
    manager = camera_service.get_active_manager(camera_id)
    for _ in range(100):
        if manager.get_latest_jpeg() is not None:
            break
        time.sleep(0.2)
    assert manager.get_latest_jpeg() is not None, "capture thread never produced a frame"

    stop_resp = client.post(f"/api/cameras/{camera_id}/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] == "offline"

    client.delete(f"/api/cameras/{camera_id}")


def test_stream_endpoint_rejects_when_not_running():
    payload = {
        "camera_id": "CAM-NOTRUNNING",
        "name": "Idle",
        "source_type": "webcam",
        "source_uri": "0",
    }
    client.post("/api/cameras", json=payload)

    resp = client.get("/api/cameras/CAM-NOTRUNNING/stream")
    assert resp.status_code == 409

    client.delete("/api/cameras/CAM-NOTRUNNING")


def test_upload_rejects_unsupported_file_type(tmp_path):
    bad_file = tmp_path / "not_a_video.txt"
    bad_file.write_text("hello")
    with bad_file.open("rb") as f:
        resp = client.post(
            "/api/cameras/upload",
            data={"name": "Bad Upload"},
            files={"file": ("not_a_video.txt", f, "text/plain")},
        )
    assert resp.status_code == 400
