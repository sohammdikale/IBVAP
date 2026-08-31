"""
Video processing engine.

One VideoStreamManager per camera. Each runs its own daemon thread that
continuously reads frames from the source (RTSP / webcam / uploaded video
file) and keeps the latest frame available as JPEG bytes. The FastAPI
process never blocks on frame I/O — routes just read whatever the
background thread last produced.

If a detector is attached (Phase 3), each frame is run through it and
detections are drawn onto the streamed frame. If a tracker is also attached
(Phase 4), detections are assigned persistent track_ids before drawing, and
a per-camera TrackHistory accumulates dwell time / direction / speed
estimates for whatever's currently being tracked.

Phase 5 adds three rule engines running on top of the tracked detections:
virtual-fence intrusion, loitering (dwell time), and night-time movement.
Each writes an Event row (with an evidence snapshot) via event_service the
first time it fires for a given track, not on every frame — see the
_alerted_* sets below.
"""

import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

import cv2
import numpy as np

from backend.config import get_settings
from backend.core.activity_engine import is_loitering
from backend.core.anpr_engine import PlateReadResult, crop_vehicle, is_large_enough_for_anpr
from backend.core.interfaces import DetectionResult, Detector, Tracker
from backend.core.intrusion_engine import ZonePolygon, ground_point, is_inside_zone
from backend.core.night_detection import is_night_time
from backend.core.track_store import TrackHistory, TrackRecord
from backend.models.camera import CameraStatus, CameraType
from backend.models.event import EventType, Severity
from backend.services import event_service
from backend.utils.image_utils import count_by_class, draw_detections, draw_zone
from backend.utils.logger import get_logger
from backend.utils.video_utils import encode_jpeg, frame_brightness

logger = get_logger(__name__)
settings = get_settings()

# Objects these rule engines care about — matches the detector's RELEVANT_CLASSES.
_TRACKED_CLASSES = {"person", "car", "motorcycle", "bus", "truck", "bicycle"}
_VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

# Minimum gap between two NIGHT_MOVEMENT events for the same camera, so a
# dark scene with continuous motion doesn't flood the event log.
_NIGHT_ALERT_COOLDOWN_SECONDS = 60

# Hook for future phases: called with (camera_id, frame) on every captured
# frame, before it's discarded. Left available for phases beyond detection
# (e.g. ANPR/face cropping) that want the raw frame independent of the
# detector below.
FrameProcessor = Callable[[str, np.ndarray], None]


class VideoStreamManager:
    """Owns the capture thread and latest-frame buffer for a single camera."""

    def __init__(
        self,
        camera_id: str,
        source_uri: str,
        source_type: CameraType,
        process_fps: int = 10,
        on_frame: FrameProcessor | None = None,
        detector: Detector | None = None,
        tracker: Tracker | None = None,
        restricted_zone: ZonePolygon | None = None,
        anpr_engine=None,
    ) -> None:
        self.camera_id = camera_id
        self.source_uri = source_uri
        self.source_type = source_type
        self.process_fps = max(process_fps, 1)
        self._on_frame = on_frame
        self._detector = detector
        self._tracker = tracker
        self._track_history = TrackHistory()
        self._restricted_zone = restricted_zone
        self._anpr_engine = anpr_engine

        # Dedup state for Phase 5 rule engines — fire once per track, not every frame.
        self._intrusion_alerted_ids: set[int] = set()
        self._loitering_alerted_ids: set[int] = set()
        self._last_night_alert_at: datetime | None = None

        # Phase 6: one ANPR attempt per vehicle track_id — cached so we don't
        # re-run OCR every frame for the same vehicle. None means "not
        # attempted yet"; a PlateReadResult or the sentinel _NO_PLATE_FOUND
        # means the attempt has already happened.
        self._plate_cache: dict[int, PlateReadResult | None] = {}
        self._plate_logged_ids: set[int] = set()

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_detection: DetectionResult | None = None

        self.status: CameraStatus = CameraStatus.OFFLINE
        self.error_message: str | None = None

    def _resolve_source(self) -> int | str:
        """Webcam sources are numeric indices; RTSP/file sources are used as-is."""
        if self.source_type == CameraType.WEBCAM:
            try:
                return int(self.source_uri)
            except ValueError:
                return 0
        return self.source_uri

    def start(self) -> bool:
        """Open the source and start the capture thread. Returns success/failure."""
        if self._running:
            return True

        source = self._resolve_source()
        self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            self.status = CameraStatus.ERROR
            self.error_message = f"Unable to open video source: {self.source_uri!r}"
            logger.warning("Camera %s failed to open: %s", self.camera_id, self.error_message)
            return False

        self._running = True
        self.status = CameraStatus.ONLINE
        self.error_message = None
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info("Camera %s started (%s)", self.camera_id, self.source_type.value)
        return True

    def _read_loop(self) -> None:
        frame_interval = 1.0 / self.process_fps
        assert self._cap is not None

        while self._running:
            ok, frame = self._cap.read()

            if not ok:
                if self.source_type == CameraType.UPLOADED_VIDEO:
                    # Loop demo/uploaded videos instead of ending the stream.
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.status = CameraStatus.ERROR
                self.error_message = "Lost connection to video source"
                logger.warning("Camera %s lost connection", self.camera_id)
                break

            jpeg = None
            if self._detector is not None:
                try:
                    result = self._detector.detect(frame)
                    tracked_detections = result.detections
                    if self._tracker is not None:
                        tracked_detections = self._tracker.update(tracked_detections)
                        with self._lock:
                            self._track_history.update(tracked_detections)
                    result = DetectionResult(detections=tracked_detections, demo_mode=result.demo_mode)

                    annotated = draw_zone(frame, self._restricted_zone)
                    with self._lock:
                        plate_lookup = dict(self._plate_cache)
                    annotated = draw_detections(annotated, result.detections, demo_mode=result.demo_mode, plate_lookup=plate_lookup)

                    # Evidence snapshots use the annotated frame (boxes + zone
                    # outline visible) so a reviewer can see exactly what
                    # triggered the alert without cross-referencing the stream.
                    self._run_rule_engines(annotated, tracked_detections)

                    jpeg = encode_jpeg(annotated)
                    with self._lock:
                        self._latest_detection = result
                except Exception:
                    logger.exception("Detection failed for camera %s; streaming raw frame", self.camera_id)
                    jpeg = encode_jpeg(frame)
            else:
                jpeg = encode_jpeg(frame)

            if jpeg is not None:
                with self._lock:
                    self._latest_jpeg = jpeg

            if self._on_frame is not None:
                try:
                    self._on_frame(self.camera_id, frame)
                except Exception:
                    logger.exception("Frame processor failed for camera %s", self.camera_id)

            time.sleep(frame_interval)

        self._running = False

    def _run_rule_engines(self, frame: np.ndarray, tracked_detections: list) -> None:
        """Intrusion, loitering, night-movement, and ANPR checks — Phase 5/6, Sections 9/10/12/13."""
        self._check_intrusion(frame, tracked_detections)
        self._check_loitering(frame)
        self._check_night_movement(frame, tracked_detections)
        self._check_anpr(frame, tracked_detections)

    def _check_intrusion(self, frame: np.ndarray, tracked_detections: list) -> None:
        if not self._restricted_zone:
            return

        currently_inside: set[int] = set()

        for det in tracked_detections:
            if det.track_id is None or det.class_name not in _TRACKED_CLASSES:
                continue

            inside = is_inside_zone(ground_point(det), self._restricted_zone)
            if not inside:
                continue

            currently_inside.add(det.track_id)
            if det.track_id in self._intrusion_alerted_ids:
                continue  # already alerted for this track's current visit

            self._intrusion_alerted_ids.add(det.track_id)
            event_service.create_event(
                camera_id=self.camera_id,
                event_type=EventType.INTRUSION,
                severity=Severity.HIGH,
                object_type=det.class_name,
                track_id=det.track_id,
                confidence=det.confidence,
                description=f"{det.class_name} entered restricted zone",
                frame=frame,
            )

        # Allow re-triggering if a track leaves and later re-enters the zone.
        self._intrusion_alerted_ids &= currently_inside

    def _check_loitering(self, frame: np.ndarray) -> None:
        for record in self._track_history.active_tracks():
            if record.track_id in self._loitering_alerted_ids:
                continue
            if not is_loitering(record, settings.loitering_seconds):
                continue

            self._loitering_alerted_ids.add(record.track_id)
            event_service.create_event(
                camera_id=self.camera_id,
                event_type=EventType.LOITERING,
                severity=Severity.MEDIUM,
                object_type=record.class_name,
                track_id=record.track_id,
                description=f"{record.class_name} present for {record.dwell_seconds():.0f}s (threshold {settings.loitering_seconds}s)",
                frame=frame,
            )

    def _check_night_movement(self, frame: np.ndarray, tracked_detections: list) -> None:
        if not tracked_detections:
            return

        now = datetime.now(timezone.utc)
        if not is_night_time(now, settings.night_start, settings.night_end):
            return
        if frame_brightness(frame) >= settings.night_brightness_threshold:
            return
        if self._last_night_alert_at is not None:
            elapsed = (now - self._last_night_alert_at).total_seconds()
            if elapsed < _NIGHT_ALERT_COOLDOWN_SECONDS:
                return

        self._last_night_alert_at = now
        event_service.create_event(
            camera_id=self.camera_id,
            event_type=EventType.NIGHT_MOVEMENT,
            severity=Severity.MEDIUM,
            description=f"Movement detected during night hours ({len(tracked_detections)} object(s))",
            frame=frame,
        )

    def _check_anpr(self, frame: np.ndarray, tracked_detections: list) -> None:
        """One OCR attempt per vehicle track_id — Phase 6, Section 10."""
        if self._anpr_engine is None:
            return

        for det in tracked_detections:
            if det.track_id is None or det.class_name not in _VEHICLE_CLASSES:
                continue
            if det.track_id in self._plate_cache:
                continue  # already attempted for this vehicle
            if not is_large_enough_for_anpr(det.bbox):
                continue  # too small in-frame to plausibly read yet; retry next frame it's seen

            crop = crop_vehicle(frame, det.bbox)
            result = self._anpr_engine.read_plate(crop)
            with self._lock:
                self._plate_cache[det.track_id] = result

            if result is None or result.confidence < settings.anpr_min_confidence:
                continue
            if det.track_id in self._plate_logged_ids:
                continue

            self._plate_logged_ids.add(det.track_id)
            event_service.create_event(
                camera_id=self.camera_id,
                event_type=EventType.PLATE_DETECTED,
                severity=Severity.LOW,
                object_type=det.class_name,
                track_id=det.track_id,
                confidence=result.confidence,
                description=f"Plate {result.normalized_text} ({'valid format' if result.valid_format else 'unvalidated format'})",
                frame=frame,
            )

    def get_latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def get_latest_detection(self) -> DetectionResult | None:
        with self._lock:
            return self._latest_detection

    def get_latest_counts(self) -> dict[str, int]:
        result = self.get_latest_detection()
        if result is None:
            return {"person": 0, "vehicle": 0}
        return count_by_class(result.detections)

    def get_active_tracks(self) -> list[TrackRecord]:
        with self._lock:
            return self._track_history.active_tracks()

    def get_active_plates(self) -> dict[int, PlateReadResult | None]:
        with self._lock:
            return dict(self._plate_cache)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
        self.status = CameraStatus.OFFLINE
        logger.info("Camera %s stopped", self.camera_id)


def mjpeg_stream(manager: VideoStreamManager, target_fps: int = 10):
    """Generator yielding an MJPEG multipart stream from a running manager's latest frames."""
    boundary = b"--frame"
    interval = 1.0 / max(target_fps, 1)

    while manager.status == CameraStatus.ONLINE:
        frame = manager.get_latest_jpeg()
        if frame is None:
            time.sleep(0.05)
            continue
        yield (
            boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(interval)
