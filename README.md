# IBVAP — Intelligent Border Video Analytics Platform

> **Status: Phase 6 of 10 — ANPR.**
> This README grows with each phase. Vehicles now get automatic number
> plate recognition — one OCR attempt per vehicle track, validated against
> Indian plate format, logged as a `PLATE_DETECTED` event with an evidence
> snapshot. Face detection/recognition is not implemented yet — that's
> Phase 7.

## 1. Overview

IBVAP turns existing IP CCTV cameras into an intelligent surveillance
platform — object/vehicle detection, tracking, ANPR, face recognition,
intrusion and suspicious-activity detection, night-movement alerts, and a
live analytics dashboard — without needing dedicated smart-camera hardware.

## 2. Problem Statement

Border Out Posts and checkpoints run CCTV that only provides live view and
recording. Advanced analysis requires continuous human monitoring.

## 3. Solution

A modular, AI-driven pipeline (detection → tracking → analysis → rules →
alerts → dashboard) added on top of standard RTSP/IP camera streams.

## 4. Features (target — built incrementally across 10 phases)

- Human & vehicle detection and tracking
- Face detection/recognition (demo identities only)
- ANPR (Indian plate formats)
- Virtual fence / intrusion detection
- Loitering & suspicious activity rules
- Night-time movement detection
- Real-time alerts (WebSocket) with severity levels
- Full event logging + evidence snapshots
- Live dashboard, camera management, analytics

## 5. Architecture

See the data-flow diagram and phase roadmap discussed with the project
owner — added here in full once Phase 9 (dashboard) is complete.

## 6. Tech Stack

- **Backend:** FastAPI, Uvicorn, WebSockets
- **AI/CV (from Phase 3):** Ultralytics YOLO, OpenCV, ByteTrack, EasyOCR
- **DB:** SQLAlchemy + SQLite (Postgres-ready)
- **Frontend:** Streamlit shell now, React-ready backend for later
- **Storage:** local filesystem under `data/`, path configurable

## 7. Installation (Phase 1)

Requires Python 3.11.

```bash
git clone <repository>
cd ibvap

python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## 8. Environment Setup

```bash
copy .env.example .env      # Windows
cp .env.example .env        # macOS/Linux
```

Defaults work out of the box for local development — no values are required
to change for Phase 1.

## 9. Model Setup

Phase 3 uses [Ultralytics YOLO](https://docs.ultralytics.com/) (`yolo11n.pt`
— the nano model, CPU-runnable) for detection, and Phase 4 uses ByteTrack
(via the `supervision` package) for tracking. YOLO weights download
automatically the first time the backend starts a camera stream, saved to
`models/yolo/yolo11n.pt`. This requires internet access once; after that
it's cached locally. ByteTrack needs no separate weights.

If YOLO fails to load for any reason (no internet, unsupported hardware,
corrupted download), IBVAP does **not** silently fake detections — it falls
back to an explicit `DemoDetector` that returns zero detections and marks
every response `"demo_mode": true`. Likewise, if ByteTrack fails to
initialize, tracking falls back to `NullTracker`, which passes detections
through with no track_id rather than inventing fake IDs. The dashboard shows
a visible warning banner when detection demo mode is active, and a red
"DEMO MODE" label is burned into the video overlay itself. Check the
backend terminal log for the reason if you see either fallback.

To use a larger/more accurate detection model, change `YOLO_MODEL` in
`.env` to e.g. `yolo11s.pt` or `yolo11m.pt` (larger = slower on CPU, more
accurate).

Phase 6 uses [EasyOCR](https://github.com/JaidedAI/EasyOCR) for ANPR —
there's no dedicated plate-detector model in this prototype; EasyOCR's own
text detector runs against each detected vehicle's cropped region instead.
Its detection + recognition models (~65MB total) download automatically on
first use and are cached locally. If EasyOCR fails to load, ANPR falls back
to `NullANPREngine`, which never fabricates a plate reading — vehicles
simply show no plate text until it's fixed. There is deliberately no
"OCR misread correction" (e.g. guessing O↔0) — an earlier version of this
had one, but testing showed it could silently corrupt a real letter in the
plate's series segment into a different valid-looking plate, so it was
removed. Confidence and format-validity are always shown as-is.

## 10. Running the Application

**Backend:**

```bash
uvicorn backend.main:app --reload
```

Visit http://localhost:8000/api/health and http://localhost:8000/docs.

**Frontend** (separate terminal, same venv active):

```bash
streamlit run frontend/app.py
```

From the dashboard you can now:
- Add a camera via RTSP URL, webcam index, or by uploading a demo video file
- Start/stop a camera's capture thread
- View its live feed inline, **with YOLO bounding boxes + persistent track IDs + the restricted-zone outline drawn on people and vehicles**
- See live person/vehicle counts per camera and totals across all active cameras
- Expand **Active Tracks** on the live view to see each tracked object's ID, dwell time, direction, and estimated speed
- Define a **restricted zone (virtual fence)** per camera as a JSON polygon, and clear it
- Browse **Recent Events** (intrusion / loitering / night movement / plate detected) with their evidence snapshots, filterable by type
- Expand **Recognized Plates** on the live view to see OCR'd plate text, confidence, and format-validity per vehicle track

**Zone changes take effect the next time that camera is stopped and started** — the running capture thread doesn't hot-reload a new polygon mid-stream.

## 11. Demo Mode

Working now: use the **Upload Video** tab in "Add Camera" to register any
`.mp4/.avi/.mov/.mkv` file as a camera — no physical camera or RTSP source
required. Uploaded videos loop automatically when they reach the end, so a
short clip can run indefinitely for a demo.

## 12. CCTV / RTSP Setup

Add a camera via `POST /api/cameras` (or the RTSP tab in the dashboard) with
`source_type: "rtsp"` and `source_uri: "rtsp://user:pass@ip:554/stream"`.
Credentials live only in the DB/env, never in frontend code, per the
project's security rules.

## 13. API Documentation

Auto-generated by FastAPI at `/docs` (Swagger) and `/redoc`. Current
endpoints:

```
GET    /api/health

POST   /api/cameras
POST   /api/cameras/upload         (multipart: name, location, file)
GET    /api/cameras
GET    /api/cameras/{camera_id}
PUT    /api/cameras/{camera_id}
DELETE /api/cameras/{camera_id}
POST   /api/cameras/{camera_id}/start
POST   /api/cameras/{camera_id}/stop
GET    /api/cameras/{camera_id}/stream       (MJPEG, boxes + track IDs + zone outline drawn server-side, only while running)
GET    /api/cameras/{camera_id}/detections   (live person/vehicle counts, only while running)
GET    /api/cameras/{camera_id}/tracks       (active tracked objects: ID, dwell time, direction, speed estimate)
GET    /api/cameras/{camera_id}/plates       (recognized plates per active vehicle track)

GET    /api/events                (list, filterable by ?camera_id= and ?event_type=)
GET    /api/events/{event_id}
GET    /api/events/{event_id}/snapshot   (evidence JPEG)
```

Restricted zone is set/cleared via `PUT /api/cameras/{camera_id}` with a
`restricted_zone` field — a JSON string of `[[x,y], ...]` polygon points, or
`null` to clear it.

## 14. Screenshots

Added once the dashboard (Phase 9) exists.

## 15. Project Structure

```
ibvap/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── api/{health,cameras,events}.py
│   ├── core/{interfaces,detector,tracker,track_store,intrusion_engine,activity_engine,night_detection,anpr_engine,video_processor}.py
│   ├── models/{database,camera,event,alert,schemas}.py
│   ├── services/{camera_service,event_service}.py
│   └── utils/{logger,video_utils,image_utils}.py
├── frontend/app.py
├── models/yolo/yolo11n.pt       (auto-downloaded)
├── models/face/                 (placeholder)
├── models/anpr/                 (EasyOCR models cache elsewhere; this stays a placeholder)
├── data/{videos,faces,events,evidence}/
├── tests/{test_health,test_cameras,test_detection,test_tracking,test_intrusion,test_anpr}.py
├── requirements.txt
├── .env.example
└── README.md
```

Full structure (including `core/`, `services/`, and remaining `api/`
modules) fills in phase by phase — see the codebase for what currently
exists.

## 16. Testing

```bash
pip install pytest httpx
pytest
```

Phase 1 covers: app boot, health endpoint status/shape.
Phase 2 adds: camera CRUD, duplicate-ID rejection, uploaded-video capture
(a synthetic test video is generated on the fly), MJPEG stream availability,
and graceful failure on an unreachable RTSP source.
Phase 3 adds: real YOLO inference against a synthetic test video, the
DemoDetector fallback in isolation, bounding-box drawing, and the
`/detections` endpoint (including its 404/409 cases).
Phase 4 adds: ByteTrack ID persistence across frames (same object keeps its
ID, distinct objects get distinct IDs), the NullTracker fallback, dwell/
direction/speed math against hand-computed cases, stale-track pruning, and
the `/tracks` endpoint.
Phase 5 adds: zone parsing/point-in-polygon math, overnight vs. same-day
night-window logic, loitering threshold logic, and — most importantly — a
full end-to-end test that starts a camera with a real photo (people + a
bus) inside a defined zone and confirms an actual `INTRUSION` Event row
lands in the database with a real, retrievable evidence snapshot.
Phase 6 adds: real (non-mocked) EasyOCR inference against rendered plate
text, Indian-format validation/normalization, the `NullANPREngine`
fallback, and a full pipeline test that composites a plate onto a real
vehicle photo and confirms a `PLATE_DETECTED` event is logged with the
correct OCR'd text.

**Known cosmetic issue:** the full suite may print `Aborted` after
`pytest`'s own summary line. This is a native-library thread-cleanup quirk
during the *test process's* interpreter shutdown (traced to the combination
of the intrusion pipeline test with camera capture threads) — it happens
*after* every test has already been reported, and the summary line ("53
passed", zero failures) is authoritative. It doesn't occur in the actual
running application, since `uvicorn` is a long-lived server process, not a
short-lived script like `pytest`.

## 17. Docker

Introduced in Phase 10.

## 18. Future Improvements

Tracked per-phase; consolidated here once Phase 10 completes.

## 19. Limitations

- No face recognition yet (Phase 7).
- No dedicated plate-localization model — ANPR runs EasyOCR's text detector
  against the cropped vehicle region rather than a separate plate-bbox
  stage. Works well when the plate is clearly visible in the vehicle crop;
  does not claim OCR accuracy it hasn't been measured on, and there is
  deliberately no "misread correction" guessing (see Model Setup above).
- ANPR attempts once per vehicle track (cached) — a vehicle whose plate
  wasn't readable in its first large-enough frame won't be retried later in
  that same track's lifetime.
- Suspicious-activity detection is limited to loitering (dwell time over a
  threshold) for this phase — the other signals listed in the spec
  (repeated approaches, multiple people in a zone, vehicle stopped in a
  zone) are not implemented yet.
- Zone edits apply on the camera's next start, not live mid-stream.
- No alert acknowledgement workflow yet (Phase 8) — events are logged and
  visible, but there's no "acknowledge"/"resolve" action in the UI yet.
- YOLO confidence threshold and model size are configurable in `.env` but
  ship at defaults tuned for a CPU laptop, not maximum accuracy.
- Track IDs can occasionally "flip" (a new ID assigned to the same object)
  after a several-second occlusion or gap in detection — this is standard
  ByteTrack behavior on CPU-speed inference, not a bug in this integration.
- Speed/direction are pixel-space estimates (no camera calibration in this
  prototype) — useful for relative comparison, not real-world units.
- Detection/tracking results are never dressed up as more certain than they
  are — demo/fallback mode is always explicitly flagged, never hidden.

## 20. Responsible Use & Privacy Notes

- Face recognition is designed for **authorized deployments only**, is
  configurable to disable entirely, and uses **demo/synthetic identities**
  in this repository — never real biometric data.
- Data/evidence retention periods are configurable, not indefinite by
  default.
- No RTSP credentials are ever exposed to frontend code.

## 21. License

Add your institution's/project's license here.
