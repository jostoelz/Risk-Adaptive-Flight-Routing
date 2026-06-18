"""
==============================================================================
runtime_state.py  --  shared live-telemetry store
Project: Autonomous Risk-Based Drone Patrolling (ETH AI Sprint 2026)
==============================================================================

A tiny, dependency-free, process-safe JSON store that lets the FastAPI Ground
Control endpoints (api.py) and the Streamlit dashboard (app.py) share the live
drone/herd telemetry -- whether they run in the SAME process (FastAPI started in
a background thread inside Streamlit, the one-click Cloud deployment) or in TWO
separate processes (a standalone `uvicorn api:app` next to `streamlit run`).

Writes are atomic (temp file + os.replace), so a concurrent reader always sees
either the previous or the next complete state, never a half-written file.

State schema:
    {
      "telemetry": {                 # last packet from the drone, or None
          "drone_lat": float, "drone_lon": float, "drone_alt": float|None,
          "livestock": [[lat, lon], ...],
          "threats":   [[lat, lon], ...],
          "n_waypoints": int,
          "received_at": float       # unix seconds
      },
      "waypoints": [ {lat, lon, altitude_m, speed_ms, risk}, ... ],
      "seq": int,                    # increments on every telemetry update
      "updated_at": float
    }
==============================================================================
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Dict, List, Optional

# State file location. Override with DRONE_STATE_PATH; defaults to the system
# temp dir so it is always writeable (including on Streamlit Community Cloud).
STATE_PATH = os.environ.get(
    "DRONE_STATE_PATH",
    os.path.join(tempfile.gettempdir(), "drone_patrol_state.json"),
)

_LOCK = threading.Lock()  # guards same-process writers


def _default_state() -> Dict:
    return {"telemetry": None, "waypoints": [], "seq": 0, "updated_at": None}


def load_state() -> Dict:
    """Read the current shared state (returns a fresh default if none exists)."""
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return _default_state()


def save_state(state: Dict) -> None:
    """Atomically persist the state dict."""
    tmp = f"{STATE_PATH}.{os.getpid()}.tmp"
    with _LOCK:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(tmp, STATE_PATH)  # atomic on Windows & POSIX


def update_telemetry(
    drone_lat: float,
    drone_lon: float,
    livestock: List[List[float]],
    threats: Optional[List[List[float]]] = None,
    drone_alt: Optional[float] = None,
    n_waypoints: int = 4,
) -> Dict:
    """Store a new telemetry packet, bump the sequence counter, return new state."""
    state = load_state()
    state["telemetry"] = {
        "drone_lat": float(drone_lat),
        "drone_lon": float(drone_lon),
        "drone_alt": (float(drone_alt) if drone_alt is not None else None),
        "livestock": [[float(a), float(b)] for a, b in (livestock or [])],
        "threats": [[float(a), float(b)] for a, b in (threats or [])],
        "n_waypoints": int(n_waypoints),
        "received_at": time.time(),
    }
    state["seq"] = int(state.get("seq", 0)) + 1
    state["updated_at"] = time.time()
    save_state(state)
    return state


def store_waypoints(waypoints: List[Dict]) -> None:
    """Cache the most recently computed waypoints alongside the telemetry."""
    state = load_state()
    state["waypoints"] = waypoints
    state["updated_at"] = time.time()
    save_state(state)


def reset() -> None:
    """Clear the store (used by the dashboard 'reset' control)."""
    save_state(_default_state())
