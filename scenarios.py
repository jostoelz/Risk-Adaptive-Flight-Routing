"""
scenarios.py — synthetic livestock-behaviour engine for the demo.

Defines the four jury-demo scenarios selectable from the Streamlit sidebar. Each
scenario provides:
  * init_scenario()    -> initial herd positions, headings and a `meta` dict.
  * advance_scenario() -> one tick of the scenario-specific movement model.

Pure numpy/math (no Streamlit, no model import) so it stays a clean, testable
simulation engine. All coordinates are (lon, lat); movement is computed in metres
via a local equirectangular scale and converted back.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

_EARTH_RADIUS_M = 6_371_000.0
_M_PER_DEG_LAT = math.radians(1.0) * _EARTH_RADIUS_M

# Sidebar dropdown options (order = display order; first is the default).
SCENARIOS = [
    "Compact Herd",
    "Wide Split (2 Groups)",
    "Scattered / Fragmented",
]

SCENARIO_HELP = {
    "Compact Herd": "Tight single cluster grazing slowly — smooth centred Bodyguard orbit.",
    "Wide Split (2 Groups)": "Herd separates into two groups ~60 m apart — SVD split + hysteresis lock.",
    "Scattered / Fragmented": "Animals scattered across the field — KDE expands; drone sweeps the area.",
}

# Target separation for the split scenario (metres).
SPLIT_TARGET_SEP_M = 60.0


def _m_per_deg_lon(lat_deg: float) -> float:
    return _M_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def _clamp_reflect(pos: np.ndarray, heading: np.ndarray, bbox) -> None:
    """Keep animals inside the pasture (with a margin); reflect heading at edges."""
    lon_min, lat_min, lon_max, lat_max = bbox
    pad_lon = 0.04 * (lon_max - lon_min)
    pad_lat = 0.04 * (lat_max - lat_min)
    for k, (lo, hi, pad) in enumerate([(lon_min, lon_max, pad_lon),
                                       (lat_min, lat_max, pad_lat)]):
        below = pos[:, k] < lo + pad
        above = pos[:, k] > hi - pad
        pos[below, k] = lo + pad
        pos[above, k] = hi - pad
        heading[below | above] += math.pi


def init_scenario(
    name: str,
    n: int,
    bbox: Tuple[float, float, float, float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Initialise herd positions/headings and scenario metadata.

    Returns (pos (n,2) lon/lat, heading (n,), meta). `meta` may carry:
      * "groups" -> (n,) int labels (split scenario)
    """
    n = int(max(n, 4))
    lon_min, lat_min, lon_max, lat_max = bbox
    dlon, dlat = lon_max - lon_min, lat_max - lat_min
    lat0 = 0.5 * (lat_min + lat_max)
    mlon = _m_per_deg_lon(lat0)
    meta: Dict = {"name": name, "groups": None}
    heading = rng.uniform(-math.pi, math.pi, size=n)

    if name == "Wide Split (2 Groups)":
        cx, cy = lon_min + 0.5 * dlon, lat_min + 0.5 * dlat
        groups = np.arange(n) % 2
        half0 = 10.0  # initial half-separation in metres along the lon axis
        pos = np.zeros((n, 2), dtype=float)
        sgn = np.where(groups == 0, -1.0, 1.0)
        pos[:, 0] = cx + sgn * half0 / mlon + rng.normal(0, 6.0 / mlon, n)
        pos[:, 1] = cy + rng.normal(0, 6.0 / _M_PER_DEG_LAT, n)
        meta["groups"] = groups

    elif name == "Scattered / Fragmented":
        pos = np.column_stack([
            rng.uniform(lon_min + 0.1 * dlon, lon_max - 0.1 * dlon, n),
            rng.uniform(lat_min + 0.1 * dlat, lat_max - 0.1 * dlat, n),
        ]).astype(float)

    else:  # "Compact Herd" (default)
        cx, cy = lon_min + 0.45 * dlon, lat_min + 0.45 * dlat
        pos = np.column_stack([
            rng.normal(cx, 7.0 / mlon, n),
            rng.normal(cy, 7.0 / _M_PER_DEG_LAT, n),
        ]).astype(float)

    return pos, heading, meta


def advance_scenario(
    name: str,
    pos: np.ndarray,
    heading: np.ndarray,
    meta: Optional[Dict],
    bbox: Tuple[float, float, float, float],
    lat0: float,
    dt_s: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Advance the herd one tick according to the active scenario."""
    pos = np.asarray(pos, dtype=float).copy()
    heading = np.asarray(heading, dtype=float).copy()
    meta = dict(meta) if meta else {"name": name}
    n = pos.shape[0]
    mlon = _m_per_deg_lon(lat0)
    mlat = _M_PER_DEG_LAT

    if name == "Wide Split (2 Groups)" and meta.get("groups") is not None:
        groups = np.asarray(meta["groups"])
        if groups.shape[0] == n and (groups == 0).any() and (groups == 1).any():
            cA = pos[groups == 0].mean(axis=0)
            cB = pos[groups == 1].mean(axis=0)
            sep = math.hypot((cA[0] - cB[0]) * mlon, (cA[1] - cB[1]) * mlat)
            v_push = 0.9 if sep < SPLIT_TARGET_SEP_M else 0.0  # m/s apart until 60 m
            graze = 0.22
            for gi, direction in ((0, -1.0), (1, 1.0)):
                m = groups == gi
                cnt = int(m.sum())
                dx = direction * v_push * dt_s + rng.normal(0, graze * dt_s, cnt)
                dy = rng.normal(0, graze * dt_s, cnt)
                pos[m, 0] += dx / mlon
                pos[m, 1] += dy / mlat
            _clamp_reflect(pos, heading, bbox)
            return pos, heading, meta

    if name == "Scattered / Fragmented":
        # Independent wanderers (no cohesion), high heading noise.
        heading += rng.normal(0.0, 0.8, n)
        v = 0.6
        pos[:, 0] += v * dt_s * np.cos(heading) / mlon
        pos[:, 1] += v * dt_s * np.sin(heading) / mlat
        _clamp_reflect(pos, heading, bbox)
        return pos, heading, meta

    # "Compact Herd" (default): slow correlated walk with strong cohesion.
    heading += rng.normal(0.0, 0.4, n)
    centroid = pos.mean(axis=0)
    to_c = centroid - pos
    c_ang = np.arctan2(to_c[:, 1] * mlat, to_c[:, 0] * mlon)
    coh = 0.25
    heading = (1 - coh) * heading + coh * c_ang
    v = 0.3
    pos[:, 0] += v * dt_s * np.cos(heading) / mlon
    pos[:, 1] += v * dt_s * np.sin(heading) / mlat
    _clamp_reflect(pos, heading, bbox)
    return pos, heading, meta
