"""
IBVAP dashboard — Phase 5.

Adds a restricted-zone editor (per camera, JSON polygon points) and a
recent-events panel (intrusion / loitering / night-movement, with evidence
snapshots). Zone changes take effect the next time that camera is started —
see the note in the zone editor. Full alert acknowledgement workflow and the
richer Events search/filter page are later phases.

Run with:
    streamlit run frontend/app.py
"""

import json

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title="IBVAP", layout="wide")

st.title("IBVAP")
st.caption("Intelligent Border Video Analytics Platform")

# --- System status ---
try:
    resp = requests.get(f"{API_BASE}/health", timeout=3)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status", "unknown")
    dot = "🟢" if status == "online" else "🟡" if status == "degraded" else "🔴"
    st.subheader(f"{dot} SYSTEM STATUS: {status.upper()}")
    backend_up = True
except requests.exceptions.RequestException as exc:
    st.error(f"🔴 Backend unreachable: {exc}")
    st.info("Start the backend first: `uvicorn backend.main:app --reload`")
    backend_up = False

st.divider()

if backend_up:
    # --- Add camera ---
    with st.expander("➕ Add Camera", expanded=False):
        tab_rtsp, tab_webcam, tab_upload = st.tabs(["RTSP", "Webcam", "Upload Video (Demo Mode)"])

        with tab_rtsp:
            with st.form("add_rtsp"):
                cid = st.text_input("Camera ID", value="CAM-01", key="rtsp_id")
                cname = st.text_input("Name", value="BOP North Gate", key="rtsp_name")
                curi = st.text_input("RTSP URL", placeholder="rtsp://user:pass@ip:554/stream")
                cloc = st.text_input("Location", key="rtsp_loc")
                if st.form_submit_button("Add RTSP Camera"):
                    r = requests.post(
                        f"{API_BASE}/cameras",
                        json={"camera_id": cid, "name": cname, "source_type": "rtsp", "source_uri": curi, "location": cloc},
                    )
                    if r.ok:
                        st.success(f"Added {cid}")
                    else:
                        try:
                            detail = r.json().get("detail", r.text)
                        except ValueError:
                            detail = r.text
                        st.error(detail)

        with tab_webcam:
            with st.form("add_webcam"):
                cid = st.text_input("Camera ID", value="CAM-WEBCAM", key="wc_id")
                cname = st.text_input("Name", value="Dev Webcam", key="wc_name")
                idx = st.number_input("Webcam Index", min_value=0, value=0, step=1)
                cloc = st.text_input("Location", key="wc_loc")
                if st.form_submit_button("Add Webcam"):
                    r = requests.post(
                        f"{API_BASE}/cameras",
                        json={"camera_id": cid, "name": cname, "source_type": "webcam", "source_uri": str(idx), "location": cloc},
                    )
                    if r.ok:
                        st.success(f"Added {cid}")
                    else:
                        try:
                            detail = r.json().get("detail", r.text)
                        except ValueError:
                            detail = r.text
                        st.error(detail)

        with tab_upload:
            with st.form("add_upload"):
                cname = st.text_input("Name", value="Demo Feed", key="up_name")
                cloc = st.text_input("Location", key="up_loc")
                vfile = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
                if st.form_submit_button("Upload & Add") and vfile is not None:
                    files = {"file": (vfile.name, vfile.getvalue())}
                    r = requests.post(f"{API_BASE}/cameras/upload", data={"name": cname, "location": cloc}, files=files)
                    if r.ok:
                        st.success("Uploaded and added")
                    else:
                        try:
                            detail = r.json().get("detail", r.text)
                        except ValueError:
                            detail = r.text
                        st.error(detail)

    st.divider()

    # --- Camera list + controls ---
    st.subheader("Cameras")
    cams = requests.get(f"{API_BASE}/cameras").json()

    if not cams:
        st.info("No cameras yet — add one above (RTSP, webcam, or upload a demo video).")

    # First pass: gather live totals from any running cameras before rendering stats.
    total_person = 0
    total_vehicle = 0
    demo_mode_active = False
    detections_by_camera = {}
    for cam in cams:
        if cam["status"] == "online":
            det = requests.get(f"{API_BASE}/cameras/{cam['camera_id']}/detections").json()
            detections_by_camera[cam["camera_id"]] = det
            total_person += det["person"]
            total_vehicle += det["vehicle"]
            demo_mode_active = demo_mode_active or det["demo_mode"]

    stat_cols = st.columns(4)
    stat_cols[0].metric("ACTIVE CAMERAS", sum(1 for c in cams if c["status"] == "online"))
    stat_cols[1].metric("TOTAL CAMERAS", len(cams))
    stat_cols[2].metric("PERSONS (live)", total_person)
    stat_cols[3].metric("VEHICLES (live)", total_vehicle)

    selected_stream_id = None

    for cam in cams:
        with st.container(border=True):
            dot = "🟢" if cam["status"] == "online" else "🔴" if cam["status"] == "error" else "⚪"
            c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
            c1.markdown(f"**{cam['camera_id']}** — {cam['name']}  \n{dot} {cam['status'].upper()}")
            c2.caption(f"{cam['source_type']} · {cam.get('location') or '—'}")

            if c3.button("Start / View", key=f"start_{cam['camera_id']}"):
                _ = requests.post(f"{API_BASE}/cameras/{cam['camera_id']}/start")
                st.rerun()

            if c4.button("Stop", key=f"stop_{cam['camera_id']}"):
                _ = requests.post(f"{API_BASE}/cameras/{cam['camera_id']}/stop")
                st.rerun()

            if cam["status"] == "online":
                selected_stream_id = cam["camera_id"]
                det = detections_by_camera.get(cam["camera_id"])
                if det:
                    st.caption(f"👤 Persons: {det['person']}   🚗 Vehicles: {det['vehicle']}")

    if demo_mode_active:
        st.warning(
            "⚠️ DEMO MODE: the YOLO model could not be loaded, so detections above are "
            "empty placeholders, not real AI output. Check the backend logs for why the "
            "model failed to load."
        )

    if selected_stream_id:
        st.divider()
        st.subheader(f"Live: {selected_stream_id}")
        st.caption("Bounding boxes and track IDs are drawn server-side by the detection/tracking engine.")
        st.markdown(
            f'<img src="{API_BASE}/cameras/{selected_stream_id}/stream" style="max-width:100%;border-radius:8px;">',
            unsafe_allow_html=True,
        )

        tracks = requests.get(f"{API_BASE}/cameras/{selected_stream_id}/tracks").json()
        with st.expander(f"Active Tracks ({len(tracks)})", expanded=False):
            if not tracks:
                st.caption("No active tracks right now.")
            else:
                rows = [
                    {
                        "ID": t["track_id"],
                        "Class": t["class_name"],
                        "Dwell (s)": t["dwell_seconds"],
                        "Direction": t["direction"] or "—",
                        "Speed (px/s)": t["speed_px_per_second"] if t["speed_px_per_second"] is not None else "—",
                    }
                    for t in tracks
                ]
                st.table(rows)

        # --- Recognized plates (Phase 6) ---
        plates = requests.get(f"{API_BASE}/cameras/{selected_stream_id}/plates").json()
        with st.expander(f"Recognized Plates ({len([p for p in plates if p['plate_text']])})", expanded=False):
            readable = [p for p in plates if p["plate_text"]]
            if not readable:
                st.caption("No plates read yet — plates are attempted once per vehicle track when it's large enough in-frame.")
            else:
                rows = [
                    {
                        "Track ID": p["track_id"],
                        "Plate": p["plate_text"],
                        "Confidence": f"{p['confidence']:.0%}" if p["confidence"] is not None else "—",
                        "Format": "✅ Valid" if p["valid_format"] else "⚠️ Unvalidated",
                    }
                    for p in readable
                ]
                st.table(rows)

        # --- Restricted zone editor ---
        selected_cam = next((c for c in cams if c["camera_id"] == selected_stream_id), None)
        with st.expander("🚫 Restricted Zone (Virtual Fence)", expanded=False):
            st.caption(
                "Polygon points as JSON, e.g. `[[50,50],[400,50],[400,300],[50,300]]` — pixel "
                "coordinates on the raw video frame. Changes apply the next time this camera is "
                "**stopped and started again**."
            )
            current_zone = selected_cam.get("restricted_zone") if selected_cam else None
            zone_input = st.text_area("Zone polygon (JSON)", value=current_zone or "", key="zone_input", height=80)
            zc1, zc2 = st.columns(2)
            if zc1.button("Save Zone", key="save_zone"):
                try:
                    parsed = json.loads(zone_input) if zone_input.strip() else None
                    if parsed is not None and (not isinstance(parsed, list) or len(parsed) < 3):
                        st.error("Zone must be a JSON array of at least 3 [x, y] points.")
                    else:
                        r = requests.put(
                            f"{API_BASE}/cameras/{selected_stream_id}",
                            json={"restricted_zone": json.dumps(parsed) if parsed else None},
                        )
                        if r.ok:
                            st.success("Zone saved — restart this camera for it to take effect.")
                        else:
                            st.error(r.text)
                except json.JSONDecodeError:
                    st.error("Invalid JSON.")
            if zc2.button("Clear Zone", key="clear_zone"):
                r = requests.put(f"{API_BASE}/cameras/{selected_stream_id}", json={"restricted_zone": None})
                if r.ok:
                    st.success("Zone cleared — restart this camera for it to take effect.")
                else:
                    st.error(r.text)

    st.divider()

    # --- Recent events ---
    st.subheader("Recent Events")
    event_type_filter = st.selectbox(
        "Filter by type",
        ["All", "INTRUSION", "LOITERING", "NIGHT_MOVEMENT"],
        key="event_type_filter",
    )
    params = {"limit": 20}
    if event_type_filter != "All":
        params["event_type"] = event_type_filter
    events = requests.get(f"{API_BASE}/events", params=params).json()

    if not events:
        st.caption("No events logged yet.")
    else:
        for ev in events:
            with st.container(border=True):
                sev_icon = {"CRITICAL": "🟥", "HIGH": "🟧", "MEDIUM": "🟨", "LOW": "🟦", "INFO": "⬜"}.get(ev["severity"], "⬜")
                ec1, ec2 = st.columns([4, 1])
                ec1.markdown(
                    f"{sev_icon} **{ev['event_type']}** — {ev['camera_id']}  \n"
                    f"{ev['description'] or ''}  \n"
                    f"🕒 {ev['timestamp']}"
                )
                if ev.get("snapshot_path"):
                    ec2.image(f"{API_BASE}/events/{ev['id']}/snapshot", width=160)
