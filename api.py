"""
==============================================================================
api.py  --  PHASE 3 & 4: REAL DRONE MODE (FastAPI Ground Control Station)
Project: Autonomous Risk-Based Drone Patrolling (ETH AI Sprint 2026)
==============================================================================

FastAPI service that turns the Phase 1 mathematical core (model.py) into a live
Ground Control Station for a physical drone (Raspberry Pi client).

Mandatory endpoints (architecture spec):

    POST /update_herd     Accept live drone GPS + detected livestock (and any
                          detected threats) from the Raspberry Pi. Stores the
                          packet and immediately returns the freshly planned
                          next 3-5 receding-horizon waypoints.

    GET  /next_waypoints  Return ONLY the next 3-5 RHC-optimised 3D waypoints
                          [Lat, Lon, Alt, Speed] for the flight controller,
                          recomputed from the latest telemetry (true rolling
                          horizon -- never a static precomputed tour).

Auxiliary endpoints:

    GET  /                Service banner.
    GET  /health          Liveness probe.
    GET  /state           Latest telemetry + waypoints (consumed by the
                          Streamlit dashboard in REAL DRONE MODE).

Run standalone:           uvicorn api:app --host 0.0.0.0 --port 8000
Run embedded (one-click): app.py starts serve_in_background() in a daemon
                          thread, so a single `streamlit run app.py` exposes
                          both the dashboard and this API on the same host.

All coordinates on the wire use GPS convention [lat, lon]; internally the model
uses (lon, lat) and the conversion happens here at the boundary.
==============================================================================
"""

from __future__ import annotations

import os
import threading
import time
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import runtime_state as rs
from model import RiskModel, DEFAULT_SWISS_PASTURE_POLYGON

_HERE = os.path.dirname(os.path.abspath(__file__))
_HW_PATH = os.path.join(_HERE, "agent_context", "hardware_constraints.txt")
_CELL_SIZE_M = float(os.environ.get("DRONE_CELL_SIZE_M", "20.0"))

app = FastAPI(
    title="Drone Patrol — Ground Control Station",
    version="1.0",
    description="Receding-horizon informative path planning for autonomous "
                "risk-based livestock-protection drone patrols.",
)

# The drone controller is a separate origin, so allow cross-origin calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Lazily-built shared model (GPR fit is expensive -> build once per process).
# ------------------------------------------------------------------------------
_model: Optional[RiskModel] = None
_model_lock = threading.Lock()


def get_model() -> RiskModel:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = RiskModel(
                    polygon=DEFAULT_SWISS_PASTURE_POLYGON,
                    cell_size_m=_CELL_SIZE_M,
                    hardware_path=_HW_PATH,
                )
    return _model


# ==============================================================================
# Pydantic request / response schemas
# ==============================================================================

class HerdUpdate(BaseModel):
    """Telemetry packet pushed by the Raspberry Pi on the drone."""
    drone_lat: float = Field(..., description="Current drone latitude (WGS84).")
    drone_lon: float = Field(..., description="Current drone longitude (WGS84).")
    drone_alt: Optional[float] = Field(None, description="Current AGL altitude (m).")
    livestock: List[List[float]] = Field(
        default_factory=list,
        description="Detected livestock as [[lat, lon], ...].",
    )
    threats: List[List[float]] = Field(
        default_factory=list,
        description="Detected threats (wolves) as [[lat, lon], ...].",
    )
    n_waypoints: int = Field(4, ge=3, le=5, description="RHC horizon length (3-5).")


class Waypoint(BaseModel):
    lat: float
    lon: float
    altitude_m: float
    speed_ms: float
    risk: float


class WaypointsResponse(BaseModel):
    waypoints: List[Waypoint]
    count: int
    seq: int
    computed_at: float
    threat_active: bool
    message: str = "ok"


# ==============================================================================
# Core computation
# ==============================================================================

def _compute_waypoints_from_telemetry(tele: dict) -> List[dict]:
    """Run one receding-horizon cycle on a stored telemetry packet.

    Converts wire [lat, lon] -> model (lon, lat), runs model.update, and returns
    the waypoint dicts (which already carry lat/lon/altitude_m/speed_ms/risk).
    """
    model = get_model()

    livestock = tele.get("livestock") or []
    ll = (np.array([[lon, lat] for lat, lon in livestock], dtype=float)
          if livestock else None)

    threats = tele.get("threats") or []
    tt = (np.array([[lon, lat] for lat, lon in threats], dtype=float)
          if threats else None)

    drone_lonlat = (float(tele["drone_lon"]), float(tele["drone_lat"]))
    n = int(tele.get("n_waypoints", 4))

    result = model.update(
        livestock_lonlat=ll,
        drone_lonlat=drone_lonlat,
        threats_lonlat=tt,
        n_waypoints=n,
    )
    return result.waypoints


# ==============================================================================
# Endpoints
# ==============================================================================

@app.get("/")
def root():
    return {
        "service": "Drone Patrol — Ground Control Station",
        "mode": "REAL DRONE",
        "endpoints": ["POST /update_herd", "GET /next_waypoints", "GET /state",
                      "GET /health"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}


@app.post("/update_herd", response_model=WaypointsResponse)
def update_herd(packet: HerdUpdate):
    """Ingest live telemetry from the drone and return the next waypoints.

    This both persists the packet (so GET /next_waypoints and the dashboard see
    it) and responds immediately with the freshly planned horizon, so the Pi can
    act on a single round-trip.
    """
    state = rs.update_telemetry(
        drone_lat=packet.drone_lat,
        drone_lon=packet.drone_lon,
        livestock=packet.livestock,
        threats=packet.threats,
        drone_alt=packet.drone_alt,
        n_waypoints=packet.n_waypoints,
    )
    waypoints = _compute_waypoints_from_telemetry(state["telemetry"])
    rs.store_waypoints(waypoints)

    return WaypointsResponse(
        waypoints=[Waypoint(**w) for w in waypoints],
        count=len(waypoints),
        seq=int(state["seq"]),
        computed_at=time.time(),
        threat_active=bool(packet.threats),
        message="telemetry ingested; horizon recomputed",
    )


@app.get("/next_waypoints", response_model=WaypointsResponse)
def next_waypoints(n: Optional[int] = None):
    """Return the next 3-5 RHC waypoints for the flight controller.

    Recomputes from the latest stored telemetry on every call (rolling horizon).
    Optional `?n=` overrides the horizon length (clamped to 3-5).
    """
    state = rs.load_state()
    tele = state.get("telemetry")
    if not tele:
        return WaypointsResponse(
            waypoints=[], count=0, seq=int(state.get("seq", 0)),
            computed_at=time.time(), threat_active=False,
            message="no telemetry received yet — POST /update_herd first",
        )

    if n is not None:
        tele = {**tele, "n_waypoints": int(np.clip(n, 3, 5))}

    waypoints = _compute_waypoints_from_telemetry(tele)
    rs.store_waypoints(waypoints)

    return WaypointsResponse(
        waypoints=[Waypoint(**w) for w in waypoints],
        count=len(waypoints),
        seq=int(state.get("seq", 0)),
        computed_at=time.time(),
        threat_active=bool(tele.get("threats")),
        message="ok",
    )


@app.get("/state")
def get_state():
    """Full shared state (telemetry + last waypoints) for the dashboard."""
    return rs.load_state()


# ==============================================================================
# Embedded background server (used by app.py for one-click deployment)
# ==============================================================================

_server_thread: Optional[threading.Thread] = None
_server_started = False


def serve_in_background(host: str = "0.0.0.0", port: int = 8000) -> dict:
    """Start uvicorn in a daemon thread (idempotent). Returns a status dict.

    Lets `streamlit run app.py` expose this API on the same host without a
    second process. On Streamlit Community Cloud only the Streamlit port is
    publicly routed, so for a *publicly reachable* API in real field deployment
    run this module standalone (`uvicorn api:app`) on a host that exposes the
    port; the dashboard can then point REAL DRONE MODE at that URL.
    """
    global _server_thread, _server_started
    if _server_started and _server_thread is not None and _server_thread.is_alive():
        return {"started": True, "host": host, "port": port, "already": True}

    import uvicorn

    def _run():
        try:
            config = uvicorn.Config(app, host=host, port=port, log_level="warning")
            uvicorn.Server(config).run()
        except Exception as exc:  # pragma: no cover - e.g. port already in use
            print(f"[api] embedded server could not start on {host}:{port}: {exc}")

    _server_thread = threading.Thread(target=_run, name="ground-control-api",
                                      daemon=True)
    _server_thread.start()
    _server_started = True
    return {"started": True, "host": host, "port": port, "already": False}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        reload=False,
    )
