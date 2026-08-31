"""
Phase 5 tests: virtual fence / intrusion, loitering, night detection, and
the events API — including a real end-to-end check that a moving object
inside a defined zone actually produces an INTRUSION Event row with a saved
evidence snapshot.
"""

import json
import time
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.core.activity_engine import is_loitering
from backend.core.interfaces import Detection
from backend.core.intrusion_engine import ground_point, is_inside_zone, parse_zone
from backend.core.night_detection import is_night_time
from backend.core.track_store import TrackRecord
from backend.main import app
from backend.services import camera_service

client = TestClient(app)
client.__enter__()


# --- Intrusion engine ---

def test_parse_zone_valid_json():
    zone = parse_zone("[[0,0],[100,0],[100,100],[0,100]]")
    assert zone == [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def test_parse_zone_rejects_too_few_points():
    assert parse_zone("[[0,0],[100,0]]") is None


def test_parse_zone_rejects_invalid_json():
    assert parse_zone("not json") is None


def test_parse_zone_none_for_empty():
    assert parse_zone(None) is None
    assert parse_zone("") is None


def test_ground_point_is_bottom_center_of_bbox():
    det = Detection("person", 0.9, (10, 20, 30, 60))
    assert ground_point(det) == (20.0, 60.0)


def test_is_inside_zone_true_for_point_inside():
    zone = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert is_inside_zone((50, 50), zone) is True


def test_is_inside_zone_false_for_point_outside():
    zone = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert is_inside_zone((200, 200), zone) is False


# --- Night detection ---

def test_is_night_time_overnight_window():
    # 19:00 -> 06:00 window
    late_night = datetime(2026, 1, 1, 23, 0, tzinfo=timezone.utc)
    early_morning = datetime(2026, 1, 1, 5, 0, tzinfo=timezone.utc)
    midday = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)
    assert is_night_time(late_night, "19:00", "06:00") is True
    assert is_night_time(early_morning, "19:00", "06:00") is True
    assert is_night_time(midday, "19:00", "06:00") is False


def test_is_night_time_same_day_window():
    inside = datetime(2026, 1, 1, 22, 30, tzinfo=timezone.utc)
    outside = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert is_night_time(inside, "22:00", "23:00") is True
    assert is_night_time(outside, "22:00", "23:00") is False


def test_is_night_time_invalid_config_returns_false():
    now = datetime(2026, 1, 1, 23, 0, tzinfo=timezone.utc)
    assert is_night_time(now, "not-a-time", "06:00") is False


# --- Loitering / activity engine ---

def test_is_loitering_true_when_over_threshold():
    now = datetime.now(timezone.utc)
    record = TrackRecord(track_id=1, class_name="person", first_seen=now - timedelta(seconds=40), last_seen=now)
    assert is_loitering(record, threshold_seconds=30) is True


def test_is_loitering_false_when_under_threshold():
    now = datetime.now(timezone.utc)
    record = TrackRecord(track_id=1, class_name="person", first_seen=now - timedelta(seconds=5), last_seen=now)
    assert is_loitering(record, threshold_seconds=30) is False


# --- Full pipeline: intrusion event actually gets logged ---

@pytest.fixture
def moving_object_video(tmp_path):
    """A video where a bright block moves left-to-right — enough motion for detection isn't guaranteed
    without a real object, so this fixture is used only for shape/plumbing tests, not detection content."""
    path = tmp_path / "moving.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 5, (200, 200))
    for i in range(15):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_events_endpoint_empty_list_shape(moving_object_video):
    with moving_object_video.open("rb") as f:
        resp = client.post(
            "/api/cameras/upload",
            data={"name": "Event Plumbing Test", "location": "Lab"},
            files={"file": ("moving.mp4", f, "video/mp4")},
        )
    camera_id = resp.json()["camera_id"]

    events_resp = client.get("/api/events", params={"camera_id": camera_id})
    assert events_resp.status_code == 200
    assert events_resp.json() == []  # nothing logged yet, camera never started

    client.delete(f"/api/cameras/{camera_id}")


def test_zone_can_be_set_and_retrieved_via_camera_update():
    payload = {"camera_id": "CAM-ZONE-TEST", "name": "Zone Test", "source_type": "webcam", "source_uri": "0"}
    client.post("/api/cameras", json=payload)

    zone_json = json.dumps([[10, 10], [200, 10], [200, 200], [10, 200]])
    r = client.put("/api/cameras/CAM-ZONE-TEST", json={"restricted_zone": zone_json})
    assert r.status_code == 200
    assert r.json()["restricted_zone"] == zone_json

    fetched = client.get("/api/cameras/CAM-ZONE-TEST")
    assert fetched.json()["restricted_zone"] == zone_json

    client.delete("/api/cameras/CAM-ZONE-TEST")


def test_intrusion_event_logged_end_to_end(tmp_path):
    """
    Builds a video with a real photo (people + bus) positioned so it falls
    inside a defined restricted zone, starts the camera, and confirms an
    INTRUSION event actually lands in the database with a saved snapshot —
    not just that the code path runs.
    """
    import urllib.request

    img_path = tmp_path / "source.jpg"
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
        img_path,
    )
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    video_path = tmp_path / "intrusion.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
    for _ in range(15):
        writer.write(img)
    writer.release()

    with video_path.open("rb") as f:
        resp = client.post(
            "/api/cameras/upload",
            data={"name": "Intrusion Test", "location": "Lab"},
            files={"file": ("intrusion.mp4", f, "video/mp4")},
        )
    camera_id = resp.json()["camera_id"]

    # Zone covering the whole frame guarantees every detected object is "inside".
    zone_json = json.dumps([[0, 0], [w, 0], [w, h], [0, h]])
    client.put(f"/api/cameras/{camera_id}", json={"restricted_zone": zone_json})

    client.post(f"/api/cameras/{camera_id}/start")

    manager = camera_service.get_active_manager(camera_id)
    events_found = []
    for _ in range(50):
        events_found = client.get("/api/events", params={"camera_id": camera_id, "event_type": "INTRUSION"}).json()
        if events_found:
            break
        time.sleep(0.3)

    assert events_found, "no INTRUSION event was logged for an object inside a full-frame zone"
    event = events_found[0]
    assert event["severity"] == "HIGH"
    assert event["camera_id"] == camera_id
    assert event["snapshot_path"] is not None

    # Snapshot endpoint actually serves a real file.
    snap_resp = client.get(f"/api/events/{event['id']}/snapshot")
    assert snap_resp.status_code == 200
    assert snap_resp.headers["content-type"] == "image/jpeg"
    assert len(snap_resp.content) > 0

    client.post(f"/api/cameras/{camera_id}/stop")
    client.delete(f"/api/cameras/{camera_id}")


def test_events_endpoint_404_for_unknown_event():
    resp = client.get("/api/events/999999")
    assert resp.status_code == 404


def test_snapshot_endpoint_404_for_unknown_event():
    resp = client.get("/api/events/999999/snapshot")
    assert resp.status_code == 404
