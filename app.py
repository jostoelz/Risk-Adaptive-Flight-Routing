"""
==============================================================================
app.py  --  PHASE 2: STREAMLIT FRONTEND  (Ground Control Dashboard)
Project: Autonomous Risk-Based Drone Patrolling using Informative Path Planning
Target Event: ETH AI Sprint (June 18, 2026)
==============================================================================

Interactive Streamlit dashboard wrapping the Phase 1 mathematical core
(model.py). Renders, on a single pydeck map:

    * the pasture polygon overlay,
    * the dynamic 2D Gaussian risk heatmap as semi-transparent coloured blocks,
    * the livestock as moving dots (synthetic herd motion in SIMULATION MODE),
    * the live-updating 3D drone trajectory string + planned next waypoints,
    * any active virtual-wolf threats.

The sidebar exposes every hardware parameter as a live slider; changing any of
them instantly re-derives the per-cell flight envelope (max safe altitude &
velocity). The main panel hosts the prominent 'Trigger Virtual Wolf Spawn'
button which injects a threat near natural cover and forces the risk matrix in
model.py to recalibrate on the next cycle.

Run with:   streamlit run app.py
==============================================================================
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import runtime_state as rs
from model import RiskModel, DEFAULT_SWISS_PASTURE_POLYGON

# ------------------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Drone Patrol — Risk-Based Path Planning",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded",
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_HW_PATH = os.path.join(_HERE, "agent_context", "hardware_constraints.txt")
_M_PER_DEG_LAT = math.radians(1.0) * 6_371_000.0  # metres per degree latitude


# ==============================================================================
# Helpers
# ==============================================================================

def _m_per_deg_lon(lat_deg: float) -> float:
    return _M_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def risk_to_rgba(r: float) -> List[int]:
    """Map a risk value in [0,1] to a green→yellow→red semi-transparent colour."""
    r = float(np.clip(r, 0.0, 1.0))
    red = int(255 * min(1.0, r * 2.0))
    green = int(255 * min(1.0, (1.0 - r) * 2.0))
    blue = 40
    alpha = int(35 + 175 * r)  # higher risk = more opaque
    return [red, green, blue, alpha]


def build_heatmap_records(grid, risk: np.ndarray) -> List[dict]:
    """One filled square polygon per in-polygon cell, coloured by risk."""
    lon_edges, lat_edges = grid.cell_edges()
    ny, nx = grid.shape
    records: List[dict] = []
    for ry in range(ny):
        for rx in range(nx):
            if not grid.mask[ry, rx]:
                continue
            r = float(risk[ry, rx])
            lon0, lon1 = lon_edges[rx], lon_edges[rx + 1]
            lat0, lat1 = lat_edges[ry], lat_edges[ry + 1]
            records.append({
                "polygon": [
                    [lon0, lat0], [lon1, lat0],
                    [lon1, lat1], [lon0, lat1], [lon0, lat0],
                ],
                "color": risk_to_rgba(r),
                "risk": round(r, 3),
            })
    return records


def init_herd(n: int, bbox, rng) -> np.ndarray:
    """Initialise the herd as two clusters inside the pasture bounding box."""
    lon_min, lat_min, lon_max, lat_max = bbox
    span_lon, span_lat = lon_max - lon_min, lat_max - lat_min
    centres = [
        (lon_min + 0.35 * span_lon, lat_min + 0.40 * span_lat),
        (lon_min + 0.68 * span_lon, lat_min + 0.62 * span_lat),
    ]
    pts = []
    for i in range(n):
        cx, cy = centres[i % len(centres)]
        pts.append([
            cx + rng.normal(0, 0.10 * span_lon * 0.25),
            cy + rng.normal(0, 0.10 * span_lat * 0.25),
        ])
    return np.array(pts, dtype=float)


def advance_herd(pos: np.ndarray, heading: np.ndarray, bbox, lat0, dt_s, rng,
                 graze_speed=0.6, cohesion=0.15) -> Tuple[np.ndarray, np.ndarray]:
    """Correlated random walk with light herd cohesion (boids-lite grazing)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    mlon = _m_per_deg_lon(lat0)
    n = pos.shape[0]

    # Perturb heading; bias slightly toward the herd centroid for clustering.
    heading = heading + rng.normal(0.0, 0.5, size=n)
    centroid = pos.mean(axis=0)
    to_centre = centroid - pos
    centre_ang = np.arctan2(to_centre[:, 1] * _M_PER_DEG_LAT,
                            to_centre[:, 0] * mlon)
    heading = (1 - cohesion) * heading + cohesion * centre_ang

    step_m = graze_speed * dt_s
    dx_m = step_m * np.cos(heading)
    dy_m = step_m * np.sin(heading)
    pos = pos.copy()
    pos[:, 0] += dx_m / mlon
    pos[:, 1] += dy_m / _M_PER_DEG_LAT

    # Reflect off the pasture boundary (keep a small margin).
    pad_lon = 0.04 * (lon_max - lon_min)
    pad_lat = 0.04 * (lat_max - lat_min)
    for k, (lo, hi, pad) in enumerate([
        (lon_min, lon_max, pad_lon), (lat_min, lat_max, pad_lat)
    ]):
        below = pos[:, k] < lo + pad
        above = pos[:, k] > hi - pad
        pos[below, k] = lo + pad
        pos[above, k] = hi - pad
        heading[below | above] += math.pi  # turn around
    return pos, heading


def advance_drone(drone: Tuple[float, float], waypoints, lat0, dt_s
                  ) -> Tuple[float, float]:
    """Move the drone toward its first planned waypoint at that cell's speed."""
    if not waypoints:
        return drone
    lon, lat = drone
    target = waypoints[0]
    mlon = _m_per_deg_lon(lat0)
    dx_m = (target["lon"] - lon) * mlon
    dy_m = (target["lat"] - lat) * _M_PER_DEG_LAT
    dist = math.hypot(dx_m, dy_m)
    if dist < 1e-6:
        return drone
    speed = max(float(target.get("speed_ms", 5.0)), 0.5)
    step = min(speed * dt_s, dist)
    ux, uy = dx_m / dist, dy_m / dist
    new_lon = lon + (step * ux) / mlon
    new_lat = lat + (step * uy) / _M_PER_DEG_LAT
    return (new_lon, new_lat)


# ==============================================================================
# REAL DRONE MODE — FastAPI Ground Control integration
# ==============================================================================

@st.cache_resource(show_spinner=False)
def ensure_embedded_api(host: str, port: int) -> dict:
    """Start the FastAPI Ground Control server once per Streamlit process.

    Cached so it runs a single time regardless of reruns/sessions. Lets a single
    `streamlit run app.py` expose both the dashboard and the live API endpoints.
    """
    import api
    return api.serve_in_background(host=host, port=port)


def http_get_json(url: str, timeout: float = 1.5) -> Optional[dict]:
    """Best-effort GET returning parsed JSON, or None on any failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def http_post_json(url: str, payload: dict, timeout: float = 3.0) -> Optional[dict]:
    """Best-effort POST of a JSON body returning parsed JSON, or None on failure."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


# ==============================================================================
# Session-state bootstrap
# ==============================================================================

def get_model(cell_size_m: float) -> RiskModel:
    """Build (and cache) the RiskModel; rebuild only when the grid changes."""
    sig = ("default_swiss", round(cell_size_m, 2))
    if st.session_state.get("_model_sig") != sig:
        with st.spinner("Fetching OpenStreetMap features & fitting GPR baseline…"):
            st.session_state.model = RiskModel(
                polygon=DEFAULT_SWISS_PASTURE_POLYGON,
                cell_size_m=cell_size_m,
                hardware_path=_HW_PATH,
            )
        st.session_state._model_sig = sig
        _reset_simulation()  # geometry changed -> restart entities
    return st.session_state.model


def _bbox_from_model(model: RiskModel):
    g = model.grid
    return (float(g.lons.min()), float(g.lats.min()),
            float(g.lons.max()), float(g.lats.max()))


def _reset_simulation() -> None:
    """Reset herd, drone trail, threats and tick counter."""
    model = st.session_state.get("model")
    st.session_state.rng = np.random.default_rng(7)
    st.session_state.tick = 0
    st.session_state.threats = []
    st.session_state.drone_trail = []
    if model is not None:
        bbox = _bbox_from_model(model)
        n = int(st.session_state.get("n_animals", 24))
        st.session_state.herd = init_herd(n, bbox, st.session_state.rng)
        st.session_state.heading = st.session_state.rng.uniform(
            -math.pi, math.pi, size=n)
        st.session_state.drone = (
            0.5 * (bbox[0] + bbox[2]),
            bbox[1] + 0.10 * (bbox[3] - bbox[1]),
        )


# ==============================================================================
# SIDEBAR — controls
# ==============================================================================

st.sidebar.title("🛩️ Control Station")

mode = st.sidebar.radio(
    "Runtime mode",
    ["SIMULATION MODE", "REAL DRONE MODE"],
    index=0,
    help="Simulation synthesises herd motion for the jury demo. Real Drone mode "
         "consumes live telemetry from the FastAPI Ground Control endpoints.",
)
is_sim = mode == "SIMULATION MODE"

st.sidebar.markdown("### 🎥 Camera & Edge AI")
fov_h = st.sidebar.slider("Camera FoV — horizontal (°)", 30.0, 120.0, 75.0, 1.0)
fov_v = st.sidebar.slider("Camera FoV — vertical (°)", 20.0, 100.0, 50.0, 1.0)
fps = st.sidebar.slider("Edge inference rate (FPS)", 0.25, 10.0, 1.12, 0.01,
                        help="Raspberry-Pi object-detection throughput limit.")
min_wolf_px = st.sidebar.slider("Min wolf size on sensor (px)", 10, 120, 40, 1,
                                help="Drives the maximum detection altitude.")

st.sidebar.markdown("### 🚁 Flight Envelope")
min_alt = st.sidebar.slider("Safe minimum altitude (m)", 3.0, 40.0, 8.0, 0.5)
max_vel = st.sidebar.slider("Max drone velocity (m/s)", 3.0, 30.0, 15.0, 0.5)
weather_buf = st.sidebar.slider(
    "Weather / visibility buffer (%)", 0.0, 60.0, 0.0, 1.0,
    help="Extra image overlap demanded by poor visibility — flies lower & slower.")
overlap = st.sidebar.slider("Base image overlap (%)", 0.0, 80.0, 20.0, 1.0)

st.sidebar.markdown("### 🐑 Risk Tuning")
livestock_gain = st.sidebar.slider("Livestock risk gain", 0.0, 6.0, 2.5, 0.1,
                                   help="How strongly herd density amplifies risk.")
n_waypoints = st.sidebar.slider("Waypoints per cycle", 3, 5, 4, 1,
                                help="Receding-horizon output length.")

st.sidebar.markdown("### ⏯️ Simulation")
n_animals = st.sidebar.slider("Number of livestock", 4, 60, 24, 1)
sim_speed = st.sidebar.slider("Sim seconds per tick", 0.5, 8.0, 2.0, 0.5)
auto_play = st.sidebar.toggle(
    "▶ Auto-play / live refresh", value=False,
    help="Simulation: advances the synthetic herd. Real Drone: polls the live "
         "telemetry feed on each interval.")
refresh_s = st.sidebar.slider("Refresh interval (s)", 0.2, 3.0, 0.7, 0.1)
cell_size_m = st.sidebar.slider("Grid cell size (m)", 10.0, 50.0, 20.0, 5.0)

st.session_state.n_animals = n_animals

c1, c2 = st.sidebar.columns(2)
step_clicked = c1.button("⏭ Step", width="stretch", disabled=not is_sim)
reset_clicked = c2.button("🔄 Reset", width="stretch")

# --- Real Drone Ground Control configuration ---
st.sidebar.markdown("### 📡 Ground Control (Real Drone)")
api_host = st.sidebar.text_input("Embedded API host", value="0.0.0.0",
                                 disabled=is_sim)
api_port = st.sidebar.number_input("Embedded API port", min_value=1024,
                                   max_value=65535, value=8000, step=1,
                                   disabled=is_sim)
api_base = st.sidebar.text_input(
    "API base URL (the Pi POSTs here)", value=f"http://localhost:{int(api_port)}",
    disabled=is_sim,
    help="Standalone deployment: point this at the host running `uvicorn api:app`.")


# ==============================================================================
# MODEL + STATE
# ==============================================================================

model = get_model(cell_size_m)
bbox = _bbox_from_model(model)
lat0 = 0.5 * (bbox[1] + bbox[3])

# If the herd size changed, re-seed the simulation entities.
if "herd" not in st.session_state or st.session_state.herd.shape[0] != n_animals:
    _reset_simulation()
if reset_clicked:
    _reset_simulation()
    st.session_state._live_seq = None
    if not is_sim:
        rs.reset()  # clear the live telemetry feed too

# Push live hardware-slider values into the model BEFORE recomputation, so the
# flight envelope updates instantly.
hw = model.hw
hw.camera_fov_h_deg = fov_h
hw.camera_fov_v_deg = fov_v
hw.inference_rate_fps = fps
hw.min_detection_size_wolf_px = float(min_wolf_px)
hw.min_safety_altitude_m = min_alt
hw.drone_max_velocity_ms = max_vel
hw.default_image_overlap_pct = overlap
hw.weather_visibility_buffer_pct = weather_buf
model.livestock_gain = livestock_gain


# ==============================================================================
# MAIN PANEL — header + actions
# ==============================================================================

st.title("🐺 Autonomous Risk-Based Drone Patrolling")
st.caption("Informative Path Planning over a live Gaussian risk field — "
           "ETH AI Sprint demo. Static planning is forbidden: the engine emits "
           "only the next 3–5 receding-horizon waypoints each cycle.")

act1, act2, act3 = st.columns([2, 1, 1])
with act1:
    wolf_clicked = st.button(
        "🚨  TRIGGER VIRTUAL WOLF SPAWN  🐺",
        type="primary",
        width="stretch",
        disabled=not is_sim,
        help="Injects a threat coordinate near the forest edge and forces the "
             "risk matrix to recalibrate dynamically (Simulation mode).",
    )
with act2:
    clear_threats = st.button("🧹 Clear threats", width="stretch",
                              disabled=not is_sim)
with act3:
    advance_once = st.button(
        "🔁 Advance frame" if is_sim else "🔄 Refresh feed", width="stretch")

# --- process action buttons (mutate state before recomputation) ---
if clear_threats:
    st.session_state.threats = []
if wolf_clicked:
    loc = model.sample_threat_location(index=len(st.session_state.threats))
    st.session_state.threats.append(loc)
    st.toast(f"🐺 Wolf spawned at {loc[1]:.5f}, {loc[0]:.5f} — recalibrating risk!")


# ==============================================================================
# DATA FEED  — synthetic (Simulation) vs live endpoints (Real Drone)
# ==============================================================================

if is_sim:
    # ---- SIMULATION MODE: synthetically generated herd + autonomous drone ----
    do_advance = auto_play or step_clicked or advance_once
    if do_advance:
        st.session_state.tick += 1
        st.session_state.herd, st.session_state.heading = advance_herd(
            st.session_state.herd, st.session_state.heading, bbox, lat0,
            sim_speed, st.session_state.rng,
        )

    herd = st.session_state.herd
    threats = list(st.session_state.threats)
    drone = st.session_state.drone

    threats_arr = np.array(threats, dtype=float) if threats else None
    result = model.update(
        livestock_lonlat=herd,
        drone_lonlat=drone,
        threats_lonlat=threats_arr,
        n_waypoints=n_waypoints,
    )

    # Fly the drone along the freshly planned path, then log the trajectory.
    if do_advance:
        st.session_state.drone = advance_drone(
            st.session_state.drone, result.waypoints, lat0, sim_speed)
        st.session_state.drone_trail.append(list(st.session_state.drone))
        st.session_state.drone_trail = st.session_state.drone_trail[-150:]
    drone = st.session_state.drone

else:
    # ---- REAL DRONE MODE: feed comes from the FastAPI endpoints ----
    # Start the embedded Ground Control server once (one-click deployment).
    api_status = ensure_embedded_api(api_host, int(api_port))

    # Pull the live shared state (written by POST /update_herd). Prefer an HTTP
    # GET /state against the configured base URL; fall back to the local store.
    live = http_get_json(f"{api_base.rstrip('/')}/state") or rs.load_state()
    tele = live.get("telemetry")

    if tele:
        ls = tele.get("livestock") or []
        herd = (np.array([[lon, lat] for lat, lon in ls], dtype=float)
                if ls else np.empty((0, 2)))
        th = tele.get("threats") or []
        threats = [[lon, lat] for lat, lon in th]  # (lon, lat) for the map
        drone = (float(tele["drone_lon"]), float(tele["drone_lat"]))

        # Append to the flown trajectory only when a NEW packet arrived.
        last_seq = st.session_state.get("_live_seq")
        if live.get("seq") != last_seq:
            st.session_state._live_seq = live.get("seq")
            st.session_state.tick = int(live.get("seq") or 0)
            st.session_state.drone_trail.append([drone[0], drone[1]])
            st.session_state.drone_trail = st.session_state.drone_trail[-150:]
    else:
        # No telemetry yet: keep the map centred with an empty herd.
        herd = np.empty((0, 2))
        threats = []
        drone = st.session_state.drone

    threats_arr = np.array(threats, dtype=float) if threats else None
    result = model.update(
        livestock_lonlat=(herd if len(herd) else None),
        drone_lonlat=drone,
        threats_lonlat=threats_arr,
        n_waypoints=n_waypoints,
    )

trail = st.session_state.drone_trail


# ==============================================================================
# STATUS METRICS
# ==============================================================================

valid_risk = result.risk[result.grid.mask]
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Threat level", f"{valid_risk.max():.2f}",
          delta="WOLF ACTIVE" if threats else None,
          delta_color="inverse" if threats else "off")
m2.metric("Max detection alt.", f"{result.h_max_m:.1f} m")
m3.metric("Min safe alt.", f"{result.h_min_m:.1f} m")
m4.metric("Active threats", f"{len(threats)}")
m5.metric("Cycle / tick", f"{st.session_state.tick}")

if not is_sim:
    served = api_status.get("started") if isinstance(api_status, dict) else False
    healthy = http_get_json(f"{api_base.rstrip('/')}/health") is not None
    badge = "🟢 reachable" if healthy else "🟠 starting / unreachable"
    st.info(
        f"**REAL DRONE MODE** — synthetic generation is **OFF**. Ground Control "
        f"API embedded on `{api_host}:{int(api_port)}` ({badge}). The Raspberry "
        f"Pi pushes `[drone GPS, livestock]` to **POST {api_base}/update_herd** "
        f"and the flight controller pulls **GET {api_base}/next_waypoints**."
    )

    with st.expander("📡 Endpoint console (live)", expanded=(tele is None)):
        st.markdown(
            f"**POST** `{api_base}/update_herd` — push telemetry  \n"
            f"**GET** `{api_base}/next_waypoints` — 3–5 × `[Lat, Lon, Alt, Speed]`"
        )
        cco1, cco2 = st.columns(2)
        with cco1:
            if st.button("📨 Push test telemetry (simulate the Pi)",
                         width="stretch"):
                # Build a plausible packet from the pasture geometry.
                test_ls = [[
                    bbox[1] + 0.45 * (bbox[3] - bbox[1]) + 0.0001 * (i % 3),
                    bbox[0] + 0.40 * (bbox[2] - bbox[0]) + 0.0001 * (i // 3),
                ] for i in range(8)]
                wolf = model.sample_threat_location(index=0)  # (lon, lat)
                packet = {
                    "drone_lat": bbox[1] + 0.10 * (bbox[3] - bbox[1]),
                    "drone_lon": 0.5 * (bbox[0] + bbox[2]),
                    "drone_alt": 25.0,
                    "livestock": test_ls,
                    "threats": [[wolf[1], wolf[0]]],  # wire order [lat, lon]
                    "n_waypoints": n_waypoints,
                }
                resp = http_post_json(f"{api_base.rstrip('/')}/update_herd", packet)
                if resp:
                    st.success(f"Telemetry ingested (seq {resp['seq']}, "
                               f"{resp['count']} waypoints returned).")
                else:
                    st.error("POST failed — is the API reachable at the base URL?")
                st.rerun()
        with cco2:
            if st.button("🧪 GET /next_waypoints", width="stretch"):
                st.session_state._endpoint_probe = http_get_json(
                    f"{api_base.rstrip('/')}/next_waypoints")

        probe = st.session_state.get("_endpoint_probe")
        if probe is not None:
            st.caption("Latest GET /next_waypoints response (served to the drone):")
            st.json(probe)

    if tele is None:
        st.warning("Waiting for the first telemetry packet. Use **Push test "
                   "telemetry** above, or POST to the endpoint, e.g.:")
        st.code(
            f"curl -X POST {api_base}/update_herd -H 'Content-Type: application/json' "
            f"-d '{{\"drone_lat\": {0.5*(bbox[1]+bbox[3]):.5f}, "
            f"\"drone_lon\": {0.5*(bbox[0]+bbox[2]):.5f}, "
            f"\"livestock\": [[{0.5*(bbox[1]+bbox[3]):.5f}, "
            f"{0.5*(bbox[0]+bbox[2]):.5f}]], \"n_waypoints\": 4}}'",
            language="bash",
        )


# ==============================================================================
# PYDECK MAP
# ==============================================================================

layers = []

# 1) Risk heatmap (semi-transparent coloured blocks)
heat_records = build_heatmap_records(result.grid, result.risk)
layers.append(pdk.Layer(
    "PolygonLayer",
    data=heat_records,
    get_polygon="polygon",
    get_fill_color="color",
    stroked=False,
    filled=True,
    pickable=True,
    auto_highlight=True,
))

# 2) Pasture polygon overlay (outline)
poly_coords = [[float(lon), float(lat)] for lon, lat in result.grid.polygon]
layers.append(pdk.Layer(
    "PolygonLayer",
    data=[{"polygon": poly_coords}],
    get_polygon="polygon",
    get_fill_color=[255, 255, 255, 10],
    get_line_color=[255, 255, 255, 220],
    line_width_min_pixels=2,
    stroked=True,
    filled=True,
))

# 3) Livestock dots
if herd is not None and len(herd):
    herd_df = pd.DataFrame(herd, columns=["lon", "lat"])
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=herd_df,
        get_position=["lon", "lat"],
        get_fill_color=[255, 255, 255, 230],
        get_line_color=[20, 20, 20, 255],
        line_width_min_pixels=1,
        stroked=True,
        get_radius=4,
        radius_min_pixels=3,
        radius_max_pixels=8,
        pickable=False,
    ))

# 4) Planned waypoint path (drone -> next waypoints)
if result.waypoints:
    wp_path = [list(drone)] + [[w["lon"], w["lat"]] for w in result.waypoints]
    layers.append(pdk.Layer(
        "PathLayer",
        data=[{"path": wp_path}],
        get_path="path",
        get_color=[0, 200, 255, 200],
        get_width=2.5,
        width_min_pixels=2,
    ))
    wp_df = pd.DataFrame(result.waypoints)
    wp_df["order"] = [str(i + 1) for i in range(len(wp_df))]
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=wp_df,
        get_position=["lon", "lat"],
        get_fill_color=[0, 200, 255, 180],
        get_radius=5,
        radius_min_pixels=4,
        radius_max_pixels=10,
        pickable=True,
    ))
    layers.append(pdk.Layer(
        "TextLayer",
        data=wp_df,
        get_position=["lon", "lat"],
        get_text="order",
        get_size=14,
        get_color=[10, 10, 10, 255],
        get_alignment_baseline="'center'",
    ))

# 5) Drone trajectory string (history)
if len(trail) >= 2:
    layers.append(pdk.Layer(
        "PathLayer",
        data=[{"path": [list(p) for p in trail]}],
        get_path="path",
        get_color=[255, 215, 0, 220],
        get_width=2.0,
        width_min_pixels=2,
    ))

# 6) Drone current position
layers.append(pdk.Layer(
    "ScatterplotLayer",
    data=pd.DataFrame([{"lon": drone[0], "lat": drone[1]}]),
    get_position=["lon", "lat"],
    get_fill_color=[0, 120, 255, 255],
    get_line_color=[255, 255, 255, 255],
    line_width_min_pixels=2,
    stroked=True,
    get_radius=7,
    radius_min_pixels=6,
    radius_max_pixels=12,
))

# 7) Active threats (wolves)
if threats:
    threat_df = pd.DataFrame(threats, columns=["lon", "lat"])
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=threat_df,
        get_position=["lon", "lat"],
        get_fill_color=[220, 0, 0, 230],
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=2,
        stroked=True,
        get_radius=10,
        radius_min_pixels=8,
        radius_max_pixels=18,
    ))

view_state = pdk.ViewState(
    longitude=0.5 * (bbox[0] + bbox[2]),
    latitude=0.5 * (bbox[1] + bbox[3]),
    zoom=15.5,
    pitch=35,
    bearing=0,
)

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    map_style="road",
    tooltip={"text": "risk: {risk}\nalt: {altitude_m} m\nv: {speed_ms} m/s"},
)
st.pydeck_chart(deck, use_container_width=True)


# ==============================================================================
# WAYPOINT TABLE + LEGEND
# ==============================================================================

left, right = st.columns([3, 2])
with left:
    st.subheader("Next 3–5 waypoints (receding horizon)")
    if result.waypoints:
        wp_table = pd.DataFrame(result.waypoints)
        wp_table.index = np.arange(1, len(wp_table) + 1)
        wp_table = wp_table.rename(columns={
            "lat": "Lat", "lon": "Lon", "altitude_m": "Altitude (m)",
            "speed_ms": "Speed (m/s)", "risk": "Risk",
        })
        st.dataframe(
            wp_table.style.format({
                "Lat": "{:.5f}", "Lon": "{:.5f}", "Altitude (m)": "{:.1f}",
                "Speed (m/s)": "{:.1f}", "Risk": "{:.3f}",
            }),
            width="stretch",
        )
    else:
        st.write("No valid waypoints (empty pasture?).")

with right:
    st.subheader("Legend")
    st.markdown(
        """
        - 🟥 **Red blocks** — high wolf-intrusion risk
        - 🟩 **Green blocks** — low risk (semi-transparent grid)
        - ⚪ **White dots** — livestock (IoT GPS)
        - 🔵 **Blue dot** — drone position
        - 🟡 **Gold line** — flown trajectory string
        - 🔵 **Cyan line** — planned next waypoints
        - 🔴 **Red dot** — active wolf threat
        """
    )
    st.caption(
        f"Static features source: **{result.feature_source}** · "
        f"grid {result.grid.shape[0]}×{result.grid.shape[1]} · "
        f"overlap eff. {hw.overlap_fraction*100:.0f}%"
    )

with st.expander("ℹ️ How the pipeline reacts to your controls"):
    st.markdown(
        """
        * **Hardware sliders** re-derive the per-cell *flight envelope* instantly:
          lower `min wolf px` / higher altitude ⇒ faster flight; a larger
          *weather/visibility buffer* forces more image overlap ⇒ the drone flies
          **lower and slower** to avoid blind spots.
        * **Livestock risk gain** controls how strongly herd density (KDE)
          exponentially amplifies the GPR baseline risk.
        * **Trigger Virtual Wolf Spawn** adds a sharply-peaked threat near natural
          cover; the risk matrix renormalises so the planner is pulled toward the
          threat on the very next cycle.
        """
    )


# ==============================================================================
# AUTO-PLAY / LIVE-REFRESH LOOP
# ==============================================================================
# Simulation: advances the synthetic herd each interval.
# Real Drone: polls the live telemetry feed each interval.

if auto_play:
    time.sleep(float(refresh_s))
    st.rerun()
