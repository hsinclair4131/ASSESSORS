# live_drone_dashboard_local.py
# SCIO → HAMID
# In-process drone simulation (no sockets) with moving map route, trail,
# travel/hover delays, and image capture at each bridge.

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ==============================
# Config (edit paths if needed)
# ==============================
DEFAULT_VDOT = r"C:\Users\hamid\SENIOR CAPSTONE PROJECT_1\BRIDGE CORROSION VDOT"
LIVE_ROOT    = r"C:\Users\hamid\SENIOR CAPSTONE PROJECT_1\BRIDGE CORROSION LIVE"
FRAME_W, FRAME_H = 1024, 768
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Optional auto-refresh so the UI pulls new telemetry every second
try:
    from streamlit_autorefresh import st_autorefresh
    HAVE_AUTOREFRESH = True
except Exception:
    HAVE_AUTOREFRESH = False


# ==============================
# Shared state (persists through reruns)
# ==============================
@dataclass
class Telemetry:
    t: float = 0.0
    mission: str = ""
    lat: float = 37.53000
    lon: float = -77.43000
    alt: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    battery: float = 100.0
    status: str = "IDLE"
    bridge: str = ""

class Shared:
    def __init__(self):
        self.tel = Telemetry()
        self.last_image_bgr = None
        self.last_image_name = ""
        self.log = []
        self.save_root = Path(LIVE_ROOT)
        self.lock = threading.Lock()
        self.simulator_running = False
        self.stop_requested = False
        # Route/trail for map animation
        self.route = []          # list[[lon,lat], ...]
        self.route_ix = 0
        self.trail = []          # flown track for PathLayer

def _append_log(shared: "Shared", msg: str):
    with shared.lock:
        ts = time.strftime("%H:%M:%S")
        shared.log.append(f"[{ts}] {msg}")
        shared.log = shared.log[-400:]

# Persist one Shared object across reruns
if "shared_obj" not in st.session_state:
    st.session_state["shared_obj"] = Shared()
shared: Shared = st.session_state["shared_obj"]
shared.save_root.mkdir(parents=True, exist_ok=True)


# ==============================
# Helpers: dataset discovery & image I/O
# ==============================
def _pick_triplet_any(fld: Path):
    """Pick 3 images. Prefer filenames with LOW/MED/SEV (or MEDIUM/SEVERE); else first three."""
    imgs = [p for p in sorted(fld.iterdir()) if p.is_file() and p.suffix.lower() in EXTS]
    if len(imgs) < 3:
        return None
    def pick(tag):
        tag = tag.lower()
        for p in imgs:
            if tag in p.stem.lower():
                return p
        return None
    low = pick("low")
    med = pick("med") or pick("medium")
    sev = pick("sev") or pick("severe")
    if not (low and med and sev):
        low, med, sev = imgs[:3]
    return low, med, sev

def _collect_triplets(root: Path):
    trips = []
    for fld in sorted([p for p in root.glob("IMG *") if p.is_dir()], key=lambda x: x.name):
        t = _pick_triplet_any(fld)
        if t:
            trips.append((fld.name, *t))
    return trips

def _publish_telem(shared: Shared, dt, mission, lat, lon, alt, yaw, speed, battery, bridge=None, status=None):
    with shared.lock:
        tel = shared.tel
        tel.t = round(dt, 1)
        tel.mission = mission
        tel.lat = round(float(lat), 7)
        tel.lon = round(float(lon), 7)
        tel.alt = round(float(alt), 1)
        tel.yaw = round(float(yaw), 1)
        tel.speed = round(float(speed), 2)
        tel.battery = round(float(battery), 1)
        if bridge is not None:
            tel.bridge = bridge
        if status is not None:
            tel.status = status

def _publish_frame(shared: Shared, bridge: str, path: Path):
    out_dir = shared.save_root / bridge
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        _append_log(shared, f"Read ERROR: {path.name}")
        return
    bgr = cv2.resize(img, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    out_path = out_dir / path.name
    if not cv2.imwrite(str(out_path), bgr):
        _append_log(shared, f"Write ERROR: {out_path.name}")
        return
    with shared.lock:
        shared.last_image_bgr = bgr
        shared.last_image_name = f"{bridge}/{path.name}"
        # trail gets a point each time we shoot
        shared.trail.append([shared.tel.lon, shared.tel.lat])
        shared.trail = shared.trail[-2000:]
    _append_log(shared, f"Saved frame: {out_path}")


# ==============================
# Route building & movement
# ==============================
def _build_waypoints(center_lat: float, center_lon: float, n: int, radius_km: float = 0.7):
    """n evenly spaced waypoints around a circle centered at (lat,lon)."""
    if n < 1:
        return []
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * np.cos(np.radians(center_lat)))
    wps = []
    for k in range(n):
        th = 2 * np.pi * k / n
        lat = center_lat + dlat * np.sin(th)
        lon = center_lon + dlon * np.cos(th)
        wps.append((lat, lon))
    return wps

def _interp(a, b, t):
    return a + (b - a) * t

def _bearing_deg(lat1, lon1, lat2, lon2):
    """Approx bearing for yaw (degrees)."""
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
         math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(lon2 - lon1)))
    brng = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return brng


# ==============================
# Simulator (runs in a thread; NO sockets)
# ==============================
def start_simulator(shared: Shared, vdot_root: str, tick_hz: float,
                    travel_secs: float, hover_secs: float, cruise_alt: float):
    if shared.simulator_running:
        _append_log(shared, "Simulator already running.")
        return

    def move_linear(lat_from, lon_from, lat_to, lon_to, duration_s, mission_id, speed_kts=12.0, battery_rate=0.10):
        """Move with linear interpolation for given duration, publishing telemetry at tick rate."""
        tick = max(0.1, 1.0 / max(0.2, tick_hz))
        steps = max(1, int(duration_s / tick))
        lat_prev, lon_prev = lat_from, lon_from
        for i in range(1, steps + 1):
            if shared.stop_requested:
                break
            t = i / steps
            lat = _interp(lat_from, lat_to, t)
            lon = _interp(lon_from, lon_to, t)
            yaw = _bearing_deg(lat_prev, lon_prev, lat, lon)
            lat_prev, lon_prev = lat, lon
            # simple speed/battery models
            speed = 0.5144 * speed_kts   # m/s approx (for display)
            with shared.lock:
                b = max(5.0, shared.tel.battery - battery_rate)
            _publish_telem(shared, time.time() - t0, mission_id, lat, lon, cruise_alt, yaw, speed, b, status="TRANSIT")
            time.sleep(tick)

    def hover_and_shoot(bridge, files, mission_id):
        """Hover for hover_secs; capture 3 frames spaced across hover window."""
        tick = max(0.1, 1.0 / max(0.2, tick_hz))
        steps = max(1, int(hover_secs / tick))
        capture_steps = {max(1, int(0.2 * steps)) : files[0],
                         max(1, int(0.5 * steps)) : files[1],
                         max(1, int(0.8 * steps)) : files[2]}
        for i in range(1, steps + 1):
            if shared.stop_requested:
                break
            with shared.lock:
                lat, lon = shared.tel.lat, shared.tel.lon
                b = max(5.0, shared.tel.battery - 0.06)
            _publish_telem(shared, time.time() - t0, mission_id, lat, lon, cruise_alt, shared.tel.yaw, 0.2, b,
                           bridge=bridge, status="ON-STATION")
            if i in capture_steps:
                _publish_frame(shared, bridge, capture_steps[i])
            time.sleep(tick)

    def run():
        shared.simulator_running = True
        shared.stop_requested = False
        try:
            root = Path(vdot_root)
            trips = _collect_triplets(root)
            if not trips:
                _append_log(shared, f"Path ERROR: {root} has no usable images.")
                return

            # Mission & home pos
            mission_id = f"sim_{int(time.time())}"
            with shared.lock:
                lat_home, lon_home = shared.tel.lat, shared.tel.lon
                shared.trail = []
            _append_log(shared, f"Mission {mission_id} — arming")
            _publish_telem(shared, 0, mission_id, lat_home, lon_home, 0, 0, 0, 100, status="ARMING")

            # Takeoff to cruise_alt
            _append_log(shared, "Takeoff")
            global t0
            t0 = time.time()
            climb_steps = max(1, int( (cruise_alt/3.0) ))  # ~0.3 s/step at 1 Hz
            for _ in range(climb_steps):
                if shared.stop_requested: break
                with shared.lock:
                    b = max(5.0, shared.tel.battery - 0.10)
                _publish_telem(shared, time.time() - t0, mission_id, lat_home, lon_home,
                               min(cruise_alt, shared.tel.alt + 3.0), shared.tel.yaw, 0.0, b, status="CLIMB")
                time.sleep(max(0.1, 1.0 / max(0.2, tick_hz)))

            # Build waypoints: one per bridge
            wps = _build_waypoints(lat_home, lon_home, n=len(trips), radius_km=0.7)
            with shared.lock:
                shared.route = [[lon, lat] for (lat, lon) in wps]  # for PathLayer
                shared.route_ix = 0

            lat_cur, lon_cur = lat_home, lon_home

            # Visit each waypoint (bridge)
            for (bridge, low, med, sev), (lat_wp, lon_wp) in zip(trips, wps):
                if shared.stop_requested: break
                _append_log(shared, f"Transit → {bridge}")
                move_linear(lat_cur, lon_cur, lat_wp, lon_wp, travel_secs, mission_id)
                lat_cur, lon_cur = lat_wp, lon_wp

                _append_log(shared, f"On-station {bridge}")
                hover_and_shoot(bridge, (low, med, sev), mission_id)

            # Return to launch
            _append_log(shared, "Return-to-Launch")
            move_linear(lat_cur, lon_cur, lat_home, lon_home, travel_secs, mission_id)

            # Land
            _append_log(shared, "Landing")
            descend_steps = max(1, int((cruise_alt/3.0)))
            for _ in range(descend_steps):
                if shared.stop_requested: break
                with shared.lock:
                    b = max(5.0, shared.tel.battery - 0.10)
                _publish_telem(shared, time.time() - t0, mission_id, lat_home, lon_home,
                               max(0.0, shared.tel.alt - 3.0), shared.tel.yaw, 0.0, b, status="LANDING")
                time.sleep(max(0.1, 1.0 / max(0.2, tick_hz)))

            _append_log(shared, "Landed")
            _publish_telem(shared, time.time() - t0, mission_id, lat_home, lon_home, 0.0, 0.0, 0.0, shared.tel.battery, status="LANDED")

        except Exception as e:
            _append_log(shared, f"Simulator ERROR: {e}")
        finally:
            shared.simulator_running = False

    threading.Thread(target=run, daemon=True).start()

def stop_simulator(shared: Shared):
    shared.stop_requested = True
    _append_log(shared, "Stop requested (will end after current step).")


# ==============================
# UI
# ==============================
st.set_page_config(page_title="Bridge Drone — Live Sim (No Network)", layout="wide")

st.sidebar.title("Drone Sim Controls")
vdot_root = st.sidebar.text_input("VDOT image root", DEFAULT_VDOT)

# Sim timing controls
tick_hz   = st.sidebar.slider("Sim tick rate (updates/sec)", 0.2, 3.0, 1.0, 0.2)
travel_s  = st.sidebar.slider("Travel time between bridges (s)", 2, 20, 8, 1)
hover_s   = st.sidebar.slider("Hover / process at bridge (s)", 2, 20, 6, 1)
cruise_m  = st.sidebar.slider("Cruise altitude (m)", 10, 60, 25, 1)

# Auto-refresh
auto = st.sidebar.checkbox("Auto-refresh", value=True)
if auto and HAVE_AUTOREFRESH:
    st_autorefresh(interval=1000, key="tick")  # 1 second

# Buttons
b1, b2, b3 = st.sidebar.columns(3)
if b1.button("Launch", use_container_width=True):
    _append_log(shared, "Launch pressed")
    start_simulator(shared, vdot_root, tick_hz, travel_s, hover_s, cruise_m)
if b2.button("Stop", use_container_width=True):
    stop_simulator(shared)
if b3.button("Test path", use_container_width=True):
    trips = _collect_triplets(Path(vdot_root))
    if trips:
        st.sidebar.success(f"Found {len(trips)} triplet(s).")
        _append_log(shared, f"Path OK: {vdot_root} — {len(trips)} triplet(s)")
    else:
        st.sidebar.error("No usable images found (need ≥3 images per 'IMG *' folder).")
        _append_log(shared, f"Path ERROR: {vdot_root} has no images/triplets")

# Metrics
st.title("Bridge Inspection — Live Drone Simulation (No Network)")
with shared.lock:
    tel = shared.tel
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Status", tel.status)
m2.metric("Alt (m)", f"{tel.alt:.1f}")
m3.metric("Speed (m/s)", f"{tel.speed:.2f}")
m4.metric("Battery (%)", f"{tel.battery:.0f}")
m5.metric("Lat", f"{tel.lat:.5f}")
m6.metric("Lon", f"{tel.lon:.5f}")

left, right = st.columns([2, 1])

# Map with route + trail + moving dot
with left:
    st.subheader("Map / Position")
    try:
        import pydeck as pdk
        with shared.lock:
            route = getattr(shared, "route", [])
            trail = list(shared.trail)
            cur_lon, cur_lat = shared.tel.lon, shared.tel.lat

        view = pdk.ViewState(latitude=cur_lat, longitude=cur_lon, zoom=14, pitch=30)
        layers = []

        if route:
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    data=[{"path": route}],
                    get_color=[140, 140, 140],
                    width_scale=1,
                    width_min_pixels=2,
                )
            )
        if trail:
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    data=[{"path": trail}],
                    get_color=[255, 0, 0],
                    width_scale=2,
                    width_min_pixels=3,
                )
            )
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=[{"lat": cur_lat, "lon": cur_lon}],
                get_position='[lon, lat]',
                get_radius=25,
                get_fill_color=[255, 0, 0],
            )
        )

        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style=None))
    except Exception as e:
        st.info(f"Map unavailable: {e}")

    st.subheader("Latest Camera Frame")
    with shared.lock:
        frame = None if shared.last_image_bgr is None else shared.last_image_bgr.copy()
        fname = shared.last_image_name
    if frame is not None:
        st.caption(fname or "—")
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        st.info("Awaiting first frame… (Click Test path, then Launch)")

# Mission + logs
with right:
    st.subheader("Mission / Bridge")
    with shared.lock:
        tel2 = shared.tel
    st.write(f"Mission: {tel2.mission or '—'}")
    st.write(f"Current bridge: {tel2.bridge or '—'}")
    st.subheader("Event Log")
    with shared.lock:
        for line in reversed(shared.log[-150:]):
            st.write(line)

st.caption("SCIO • In-process sim → no sockets • Images saved to: " + LIVE_ROOT)
