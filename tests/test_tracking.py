"""
Phase 4 tests: tracking engine and track history.

Uses the real ByteTrack implementation (via `supervision`) — verifies track
IDs actually persist across frames for a moving object, not just that the
code runs. Direction/speed math is checked against a hand-computed case so
regressions in the geometry aren't silently possible.
"""

import time
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.core.interfaces import Detection
from backend.core.track_store import TrackHistory, TrackRecord
from backend.core.tracker import ByteTracker, NullTracker, build_tracker
from backend.main import app
from backend.services import camera_service

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


def test_bytetrack_assigns_stable_id_to_a_moving_object():
    tracker = ByteTracker()

    frame1 = [Detection("person", 0.9, (10, 10, 50, 50))]
    tracked1 = tracker.update(frame1)
    assert len(tracked1) == 1
    assert tracked1[0].track_id is not None
    first_id = tracked1[0].track_id

    # Same object, moved slightly — ByteTrack should keep the same ID.
    frame2 = [Detection("person", 0.88, (15, 15, 55, 55))]
    tracked2 = tracker.update(frame2)
    assert len(tracked2) == 1
    assert tracked2[0].track_id == first_id


def test_bytetrack_assigns_different_ids_to_distinct_objects():
    tracker = ByteTracker()
    frame = [
        Detection("person", 0.9, (10, 10, 50, 50)),
        Detection("car", 0.85, (200, 200, 260, 260)),
    ]
    tracked = tracker.update(frame)
    assert len(tracked) == 2
    ids = {d.track_id for d in tracked}
    assert len(ids) == 2  # two distinct objects, two distinct IDs


def test_null_tracker_passes_through_without_assigning_ids():
    tracker = NullTracker()
    detections = [Detection("person", 0.9, (0, 0, 10, 10))]
    result = tracker.update(detections)
    assert result == detections
    assert result[0].track_id is None


def test_build_tracker_returns_working_tracker():
    tracker = build_tracker()
    result = tracker.update([Detection("person", 0.9, (0, 0, 10, 10))])
    assert len(result) == 1


def test_track_history_computes_dwell_direction_and_speed():
    history = TrackHistory()
    now = datetime.now(timezone.utc)

    # Manually seed a record with a controlled trajectory to check the math,
    # rather than relying on real elapsed time in the test process.
    record = TrackRecord(track_id=1, class_name="person", first_seen=now, last_seen=now)
    record.positions = [(0.0, 0.0, now), (100.0, 0.0, now + timedelta(seconds=2))]
    record.last_seen = now + timedelta(seconds=2)

    assert record.dwell_seconds() == pytest.approx(2.0)
    assert record.direction() == "E"  # moved purely in +x
    assert record.speed_px_per_second() == pytest.approx(50.0)  # 100px / 2s


def test_track_history_prunes_stale_tracks():
    history = TrackHistory()
    history.STALE_AFTER_SECONDS = 0.01  # force near-immediate staleness for the test

    history.update([Detection("person", 0.9, (0, 0, 10, 10), track_id=1)])
    assert len(history.active_tracks()) == 1

    time.sleep(0.05)
    history.update([])  # no detections this cycle -> triggers pruning check
    assert len(history.active_tracks()) == 0


def test_tracks_endpoint_returns_entries_for_running_camera(synthetic_video):
    with synthetic_video.open("rb") as f:
        resp = client.post(
            "/api/cameras/upload",
            data={"name": "Tracking Test", "location": "Lab"},
            files={"file": ("synthetic.mp4", f, "video/mp4")},
        )
    camera_id = resp.json()["camera_id"]

    client.post(f"/api/cameras/{camera_id}/start")

    manager = camera_service.get_active_manager(camera_id)
    for _ in range(50):
        if manager.get_latest_detection() is not None:
            break
        time.sleep(0.2)

    tracks_resp = client.get(f"/api/cameras/{camera_id}/tracks")
    assert tracks_resp.status_code == 200
    assert isinstance(tracks_resp.json(), list)  # empty is fine (blank synthetic frame -> no detections)

    client.post(f"/api/cameras/{camera_id}/stop")
    client.delete(f"/api/cameras/{camera_id}")


def test_tracks_endpoint_404s_for_unknown_camera():
    resp = client.get("/api/cameras/CAM-NOPE/tracks")
    assert resp.status_code == 404


def test_tracks_endpoint_409s_when_not_running():
    payload = {"camera_id": "CAM-IDLE-TRACK", "name": "Idle", "source_type": "webcam", "source_uri": "0"}
    client.post("/api/cameras", json=payload)

    resp = client.get("/api/cameras/CAM-IDLE-TRACK/tracks")
    assert resp.status_code == 409

    client.delete("/api/cameras/CAM-IDLE-TRACK")
