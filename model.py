"""
==============================================================================
model.py  --  PHASE 1: MATHEMATICAL CORE
Project: Autonomous Risk-Based Drone Patrolling using Informative Path Planning
Target Event: ETH AI Sprint (June 18, 2026)
==============================================================================

This module is the pure mathematical / computational engine. It contains NO UI
and NO networking code (Streamlit & FastAPI belong to later phases). Everything
here is deterministic, importable and unit-testable.

Pipeline implemented (see agent_context/architecture.txt):

    1. Geo + Grid       -> discretise the pasture polygon into a 2D cell grid.
    2. Static features  -> fetch forest edges / creeks / hedges from OpenStreetMap
                           (osmnx, fallback overpy, fallback synthetic) using a
                           hardcoded Swiss-pasture bounding box as a safe default.
    3. Baseline risk    -> GaussianProcessRegressor (scikit-learn) interpolates a
                           smooth continuous risk heatmap from the feature anchors.
    4. Livestock layer  -> scipy.stats.gaussian_kde turns the live GPS cloud into a
                           herd-density field that *exponentially* scales the risk.
    5. Normalisation    -> the final cell-risk matrix is strictly clamped to [0, 1].
    6. Hardware mapping -> per-cell max safe altitude & velocity derived natively
                           from hardware_constraints.txt (FoV, GSD, FPS, overlap,
                           accel limits, battery).
    7. Path planning    -> ergodicity / MDP / TSP heuristics written natively in
                           NumPy + SciPy (no external planning packages), producing
                           a receding-horizon list of the next 3-5 3D waypoints.

Dependencies: numpy, scipy, scikit-learn (hard); osmnx OR overpy (optional, the
module degrades gracefully to synthetic features when neither is available).
==============================================================================
"""

from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

# ------------------------------------------------------------------------------
# Optional geospatial backends. The model MUST boot even if these are missing,
# so every import is guarded and we record which backend is available.
# ------------------------------------------------------------------------------
try:  # primary backend
    import osmnx as ox  # type: ignore

    _OSMNX_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    ox = None
    _OSMNX_AVAILABLE = False

try:  # secondary backend
    import overpy  # type: ignore

    _OVERPY_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    overpy = None
    _OVERPY_AVAILABLE = False


# ==============================================================================
# 0. DEFAULTS & CONSTANTS
# ==============================================================================

# Mean Earth radius used for the local equirectangular projection (metres).
_EARTH_RADIUS_M = 6_371_000.0

# Approximate body length of an adult European wolf along the camera sensor.
# Used together with MIN_DETECTION_SIZE_WOLF_PX to derive the max flight altitude.
WOLF_CHARACTERISTIC_SIZE_M = 1.10

# ------------------------------------------------------------------------------
# HARDCODED SWISS PASTURE FALLBACK (Rule 1).
# A small realistic alpine grazing parcel in known wolf territory
# (Graubünden, near Davos/Klosters). Roughly a 280 m x 260 m pasture.
# Order: (lon, lat) pairs, closed-ring not required (we close it ourselves).
# ------------------------------------------------------------------------------
DEFAULT_SWISS_PASTURE_POLYGON: List[Tuple[float, float]] = [
    (9.84800, 46.79900),
    (9.85160, 46.79900),
    (9.85160, 46.80130),
    (9.84800, 46.80130),
]

# Risk values used as GPR training targets.
_RISK_FEATURE_VALUE = 1.0      # a cell sitting on a static risk feature
_RISK_BACKGROUND_VALUE = 0.05  # open ground far from any cover


# ==============================================================================
# 1. HARDWARE CONSTRAINTS  (Rule 5)
# ==============================================================================

@dataclass
class HardwareConstraints:
    """Typed container for hardware_constraints.txt plus derived physics."""

    # --- Camera & edge AI ---
    camera_fov_h_deg: float = 75.0
    camera_fov_v_deg: float = 50.0
    camera_res_w_px: int = 1920
    camera_res_h_px: int = 1080
    inference_rate_fps: float = 1.12
    min_detection_size_wolf_px: float = 40.0

    # --- Drone physics ---
    drone_max_velocity_ms: float = 15.0
    drone_max_battery_min: float = 25.0
    drone_accel_limit_ms2: float = 2.5
    drone_decel_limit_ms2: float = 2.5

    # --- Safety & environment ---
    min_safety_altitude_m: float = 8.0
    default_image_overlap_pct: float = 20.0

    # --- Tunable assumptions (not in the file) ---
    wolf_size_m: float = WOLF_CHARACTERISTIC_SIZE_M
    # Extra image overlap demanded by poor weather / visibility. Added on top of
    # the base overlap, it forces the drone lower & slower (UI sidebar slider).
    weather_visibility_buffer_pct: float = 0.0

    @classmethod
    def from_file(cls, path: str) -> "HardwareConstraints":
        """Parse the KEY = VALUE config file, ignoring '#' comments & sections."""
        defaults = cls()
        if not os.path.isfile(path):
            warnings.warn(f"hardware_constraints file not found at {path}; using defaults.")
            return defaults

        # Map file keys -> dataclass attribute names.
        key_map = {
            "CAMERA_FOV_HORIZONTAL_DEG": "camera_fov_h_deg",
            "CAMERA_FOV_VERTICAL_DEG": "camera_fov_v_deg",
            "CAMERA_RESOLUTION_WIDTH_PX": "camera_res_w_px",
            "CAMERA_RESOLUTION_HEIGHT_PX": "camera_res_h_px",
            "RASPBERRY_PI_INFERENCE_RATE_FPS": "inference_rate_fps",
            "MIN_DETECTION_SIZE_WOLF_PX": "min_detection_size_wolf_px",
            "DRONE_MAX_VELOCITY_MS": "drone_max_velocity_ms",
            "DRONE_MAX_BATTERY_LIFE_MIN": "drone_max_battery_min",
            "DRONE_ACCELERATION_LIMIT_MS2": "drone_accel_limit_ms2",
            "DRONE_DECELERATION_LIMIT_MS2": "drone_decel_limit_ms2",
            "MIN_SAFETY_ALTITUDE_ABOVE_OBSTACLES_M": "min_safety_altitude_m",
            "DEFAULT_IMAGE_OVERLAP_PERCENTAGE": "default_image_overlap_pct",
        }
        int_attrs = {"camera_res_w_px", "camera_res_h_px"}

        parsed: Dict[str, float] = {}
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.split("#", 1)[0].strip()  # drop inline comments
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip()
                if key in key_map and val:
                    try:
                        num = float(val)
                    except ValueError:
                        continue
                    attr = key_map[key]
                    parsed[attr] = int(round(num)) if attr in int_attrs else num

        return cls(**{**defaults.__dict__, **parsed})

    # ---------------------------------------------------------------- physics

    @property
    def overlap_fraction(self) -> float:
        """Effective image overlap (base + weather buffer) as a fraction in [0, 0.95]."""
        eff = (self.default_image_overlap_pct + self.weather_visibility_buffer_pct) / 100.0
        return float(np.clip(eff, 0.0, 0.95))

    def max_detection_altitude_m(self) -> float:
        """
        Highest altitude at which a wolf still spans MIN_DETECTION_SIZE_WOLF_PX.

        Ground Sample Distance (vertical):
            footprint_v(h) = 2 * h * tan(FoV_v / 2)        [m on the ground]
            metres_per_pixel(h) = footprint_v(h) / res_h
        Detection requires:
            wolf_size_m / metres_per_pixel(h) >= min_px
        Solving for h:
            h_max = wolf_size_m * res_h / (min_px * 2 * tan(FoV_v / 2))
        """
        half_fov_v = math.radians(self.camera_fov_v_deg / 2.0)
        denom = self.min_detection_size_wolf_px * 2.0 * math.tan(half_fov_v)
        if denom <= 0:
            return self.min_safety_altitude_m
        h_max = self.wolf_size_m * self.camera_res_h_px / denom
        # Never below the obstacle-clearance floor.
        return max(h_max, self.min_safety_altitude_m)

    def ground_footprint_m(self, altitude_m: float) -> Tuple[float, float]:
        """Along-/cross-track ground footprint (width_m, height_m) at altitude."""
        w = 2.0 * altitude_m * math.tan(math.radians(self.camera_fov_h_deg / 2.0))
        h = 2.0 * altitude_m * math.tan(math.radians(self.camera_fov_v_deg / 2.0))
        return w, h

    def max_velocity_for_altitude(self, altitude_m: float) -> float:
        """
        Maximum velocity that still guarantees gap-free coverage given the
        1.12 FPS processing limit and the required image overlap.

            distance_between_frames = v / fps
            need: distance_between_frames <= footprint_along * (1 - overlap)
            => v_max = fps * footprint_along * (1 - overlap)

        We take the *smaller* (vertical) footprint as the along-track dimension
        to stay conservative, then clamp to the drone's hardware velocity ceiling.
        """
        _, footprint_along = self.ground_footprint_m(altitude_m)
        v_cover = self.inference_rate_fps * footprint_along * (1.0 - self.overlap_fraction)
        return float(min(v_cover, self.drone_max_velocity_ms))

    def cornering_velocity(self, turn_radius_m: float) -> float:
        """Max speed through a turn of given radius under the accel limit:
        a = v^2 / r  =>  v = sqrt(a * r)."""
        turn_radius_m = max(turn_radius_m, 1e-6)
        v = math.sqrt(self.drone_accel_limit_ms2 * turn_radius_m)
        return float(min(v, self.drone_max_velocity_ms))

    def max_range_m(self, velocity_ms: float) -> float:
        """Maximum traversable distance on one battery charge at a given speed."""
        return float(velocity_ms * self.drone_max_battery_min * 60.0)


# ==============================================================================
# 2. GEO HELPERS & GRID  (local equirectangular projection)
# ==============================================================================

@dataclass
class Grid:
    """A regular lat/lon cell grid plus its local-metre projection."""

    lats: np.ndarray            # (ny,) cell-centre latitudes
    lons: np.ndarray            # (nx,) cell-centre longitudes
    lat0: float                 # projection origin (centroid lat)
    lon0: float                 # projection origin (centroid lon)
    cell_size_m: float
    polygon: np.ndarray         # (k, 2) closed ring of (lon, lat)
    mask: np.ndarray            # (ny, nx) bool: True == inside polygon

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.lats.size, self.lons.size)

    @property
    def mesh_lonlat(self) -> Tuple[np.ndarray, np.ndarray]:
        """(LON, LAT) meshgrids, shape (ny, nx)."""
        LON, LAT = np.meshgrid(self.lons, self.lats)
        return LON, LAT

    def cell_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lon_edges, lat_edges) bounding each cell, for square rendering.

        Edges have length nx+1 / ny+1 and sit half a step around the centres.
        """
        def _edges(centres: np.ndarray) -> np.ndarray:
            if centres.size == 1:
                step = 1e-4
                return np.array([centres[0] - step / 2, centres[0] + step / 2])
            step = centres[1] - centres[0]
            return np.concatenate([centres - step / 2.0, [centres[-1] + step / 2.0]])

        return _edges(self.lons), _edges(self.lats)

    def lonlat_to_xy(self, lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Equirectangular projection to local metres around (lat0, lon0)."""
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        x = np.radians(lon - self.lon0) * _EARTH_RADIUS_M * math.cos(math.radians(self.lat0))
        y = np.radians(lat - self.lat0) * _EARTH_RADIUS_M
        return x, y

    def mesh_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        LON, LAT = self.mesh_lonlat
        return self.lonlat_to_xy(LON, LAT)


def _ensure_closed_ring(polygon: Sequence[Tuple[float, float]]) -> np.ndarray:
    ring = np.asarray(polygon, dtype=float)
    if ring.ndim != 2 or ring.shape[1] != 2:
        raise ValueError("polygon must be a sequence of (lon, lat) pairs")
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])
    return ring


def _point_in_polygon(px: np.ndarray, py: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Vectorised ray-casting point-in-polygon test.

    px, py : arrays of the same shape (the query points, in any consistent CRS).
    ring   : (k, 2) closed polygon in the SAME CRS as the points.
    Returns a boolean array shaped like px.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    inside = np.zeros(px.shape, dtype=bool)
    xs, ys = ring[:, 0], ring[:, 1]
    n = len(ring) - 1  # last vertex == first
    j = n - 1
    for i in range(n):
        xi, yi = xs[i], ys[i]
        xj, yj = xs[j], ys[j]
        # Does the horizontal ray from the point cross edge (j -> i)?
        cond = ((yi > py) != (yj > py))
        denom = np.where(np.abs(yj - yi) < 1e-15, 1e-15, yj - yi)
        x_cross = (xj - xi) * (py - yi) / denom + xi
        inside ^= cond & (px < x_cross)
        j = i
    return inside


def build_grid(
    polygon: Optional[Sequence[Tuple[float, float]]] = None,
    cell_size_m: float = 25.0,
) -> Grid:
    """Discretise the pasture polygon into a regular grid (Rule fallback safe)."""
    if polygon is None or len(polygon) < 3:
        warnings.warn("No valid polygon supplied; using hardcoded Swiss pasture fallback.")
        polygon = DEFAULT_SWISS_PASTURE_POLYGON

    ring = _ensure_closed_ring(polygon)
    lon_min, lat_min = ring[:, 0].min(), ring[:, 1].min()
    lon_max, lat_max = ring[:, 0].max(), ring[:, 1].max()

    lat0 = 0.5 * (lat_min + lat_max)
    lon0 = 0.5 * (lon_min + lon_max)

    # metres per degree at this latitude
    m_per_deg_lat = math.radians(1.0) * _EARTH_RADIUS_M
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(lat0))

    span_y_m = (lat_max - lat_min) * m_per_deg_lat
    span_x_m = (lon_max - lon_min) * m_per_deg_lon

    ny = max(int(math.ceil(span_y_m / cell_size_m)), 2)
    nx = max(int(math.ceil(span_x_m / cell_size_m)), 2)
    # Keep the grid affordable for the in-browser receding-horizon loop.
    ny, nx = min(ny, 200), min(nx, 200)

    lats = np.linspace(lat_min, lat_max, ny)
    lons = np.linspace(lon_min, lon_max, nx)

    grid = Grid(
        lats=lats, lons=lons, lat0=lat0, lon0=lon0,
        cell_size_m=cell_size_m, polygon=ring,
        mask=np.zeros((ny, nx), dtype=bool),
    )
    LON, LAT = grid.mesh_lonlat
    grid.mask = _point_in_polygon(LON, LAT, ring)
    if not grid.mask.any():
        # Degenerate polygon -> treat the whole bounding box as valid.
        grid.mask[:] = True
    return grid


# ==============================================================================
# 3. STATIC RISK FEATURES FROM OPENSTREETMAP  (Rule 1)
# ==============================================================================

# OSM tag groups that correlate with wolf approach corridors / natural cover.
_OSM_FEATURE_TAGS: Dict[str, Dict[str, object]] = {
    "forest": {"natural": ["wood"], "landuse": ["forest"]},
    "creek":  {"waterway": ["stream", "river", "ditch"], "natural": ["water"]},
    "hedge":  {"barrier": ["hedge"], "natural": ["tree_row", "scrub"]},
}


@dataclass
class StaticFeatures:
    """Sampled (lon, lat) points lying on static risk features."""

    points_lonlat: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    source: str = "none"
    # Forest-edge points are tracked separately so the dashboard can colour the
    # deep-red "within-20-m-of-forest" danger band precisely.
    forest_points_lonlat: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2)))

    def __len__(self) -> int:
        return int(self.points_lonlat.shape[0])


def _sample_geometry_points(geom, max_pts: int = 40) -> List[Tuple[float, float]]:
    """Extract representative (lon, lat) points from a shapely geometry."""
    pts: List[Tuple[float, float]] = []
    try:
        gtype = geom.geom_type
        if gtype == "Point":
            pts.append((geom.x, geom.y))
        elif gtype in ("LineString", "LinearRing"):
            coords = list(geom.coords)
            pts.extend((c[0], c[1]) for c in coords)
        elif gtype in ("Polygon",):
            coords = list(geom.exterior.coords)  # forest *edges* -> boundary
            pts.extend((c[0], c[1]) for c in coords)
        elif gtype.startswith("Multi") or gtype == "GeometryCollection":
            for sub in geom.geoms:
                pts.extend(_sample_geometry_points(sub, max_pts))
    except Exception:
        return []
    if len(pts) > max_pts:  # thin out dense ways
        idx = np.linspace(0, len(pts) - 1, max_pts).astype(int)
        pts = [pts[i] for i in idx]
    return pts


def fetch_static_features(grid: Grid) -> StaticFeatures:
    """
    Fetch forest edges / creeks / hedges inside the polygon from OpenStreetMap.

    Tries osmnx first, then overpy, then falls back to a deterministic synthetic
    feature set so that the mathematical core NEVER crashes on boot (Rule 1).
    """
    ring = grid.polygon
    # shapely polygon expects (lon, lat) ordering -> (x, y).
    if _OSMNX_AVAILABLE:
        feats = _fetch_features_osmnx(ring)
        if feats is not None and len(feats):
            return feats
    if _OVERPY_AVAILABLE:
        feats = _fetch_features_overpy(ring)
        if feats is not None and len(feats):
            return feats
    return _synthetic_features(grid)


def _fetch_features_osmnx(ring: np.ndarray) -> Optional[StaticFeatures]:
    try:  # shapely is an osmnx dependency
        from shapely.geometry import Polygon as ShapelyPolygon
    except Exception:  # pragma: no cover
        return None

    poly = ShapelyPolygon([(lon, lat) for lon, lat in ring])
    collected: List[Tuple[float, float]] = []
    forest: List[Tuple[float, float]] = []
    for name, tags in _OSM_FEATURE_TAGS.items():
        try:
            # osmnx >=1.0 exposes features_from_polygon; older uses geometries_*.
            fetch = getattr(ox, "features_from_polygon", None) or \
                getattr(ox.geometries, "geometries_from_polygon", None)
            if fetch is None:
                return None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gdf = fetch(poly, tags)
            if gdf is None or len(gdf) == 0:
                continue
            pts_this: List[Tuple[float, float]] = []
            for geom in gdf.geometry:
                pts_this.extend(_sample_geometry_points(geom))
            collected.extend(pts_this)
            if name == "forest":
                forest.extend(pts_this)
        except Exception:
            continue
    if not collected:
        return None
    return StaticFeatures(
        np.asarray(collected, dtype=float), source="osmnx",
        forest_points_lonlat=(np.asarray(forest, dtype=float)
                              if forest else np.empty((0, 2))),
    )


def _fetch_features_overpy(ring: np.ndarray) -> Optional[StaticFeatures]:
    try:
        api = overpy.Overpass()
    except Exception:  # pragma: no cover
        return None
    lon_min, lat_min = ring[:, 0].min(), ring[:, 1].min()
    lon_max, lat_max = ring[:, 0].max(), ring[:, 1].max()
    bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    query = f"""
    (
      way["natural"="wood"]({bbox});
      way["landuse"="forest"]({bbox});
      way["waterway"]({bbox});
      way["barrier"="hedge"]({bbox});
      way["natural"="tree_row"]({bbox});
    );
    out geom;
    """
    try:
        result = api.query(query)
    except Exception:
        return None
    collected: List[Tuple[float, float]] = []
    forest: List[Tuple[float, float]] = []
    for way in getattr(result, "ways", []):
        pts_this: List[Tuple[float, float]] = []
        for node in way.get_nodes(resolve_missing=False) or []:
            try:
                pts_this.append((float(node.lon), float(node.lat)))
            except Exception:
                continue
        collected.extend(pts_this)
        tags = getattr(way, "tags", {}) or {}
        if tags.get("natural") == "wood" or tags.get("landuse") == "forest":
            forest.extend(pts_this)
    if not collected:
        return None
    return StaticFeatures(
        np.asarray(collected, dtype=float), source="overpy",
        forest_points_lonlat=(np.asarray(forest, dtype=float)
                              if forest else np.empty((0, 2))),
    )


def _synthetic_features(grid: Grid) -> StaticFeatures:
    """Deterministic stand-in features so the demo always has risk structure.

    Places a 'forest edge' line along the northern boundary and a diagonal
    'creek' through the parcel -- both common high-risk wolf corridors.
    """
    lon_min, lon_max = grid.lons.min(), grid.lons.max()
    lat_min, lat_max = grid.lats.min(), grid.lats.max()

    t = np.linspace(0.0, 1.0, 24)
    # Forest edge near the top (north) of the parcel.
    forest_lon = lon_min + t * (lon_max - lon_min)
    forest_lat = np.full_like(t, lat_max - 0.06 * (lat_max - lat_min))
    # Creek cutting diagonally across the parcel.
    creek_lon = lon_min + 0.15 * (lon_max - lon_min) + t * 0.7 * (lon_max - lon_min)
    creek_lat = lat_min + 0.85 * (lat_max - lat_min) - t * 0.7 * (lat_max - lat_min)

    pts = np.column_stack([
        np.concatenate([forest_lon, creek_lon]),
        np.concatenate([forest_lat, creek_lat]),
    ])
    forest_pts = np.column_stack([forest_lon, forest_lat])
    return StaticFeatures(pts, source="synthetic", forest_points_lonlat=forest_pts)


# ==============================================================================
# 4. BASELINE RISK VIA GAUSSIAN PROCESS REGRESSION  (Rule 2)
# ==============================================================================

def compute_baseline_risk(grid: Grid, features: StaticFeatures) -> np.ndarray:
    """
    Interpolate a smooth continuous baseline risk field with a
    GaussianProcessRegressor trained on feature anchors (risk=1) and a sparse
    set of background anchors (risk~0). Output shaped (ny, nx), values in [0, 1].
    """
    ny, nx = grid.shape

    # Work in local metres so the RBF length-scale is physically meaningful.
    GX, GY = grid.mesh_xy()
    query = np.column_stack([GX.ravel(), GY.ravel()])

    if len(features) == 0:
        # No features -> flat low baseline.
        return np.full((ny, nx), _RISK_BACKGROUND_VALUE, dtype=float)

    fx, fy = grid.lonlat_to_xy(features.points_lonlat[:, 0], features.points_lonlat[:, 1])
    feat_xy = np.column_stack([fx, fy])

    # Background anchors: a coarse lattice over the bounding box, kept only where
    # they are far from any feature so we do not fight the high-risk anchors.
    bg_n = 6
    bx = np.linspace(GX.min(), GX.max(), bg_n)
    by = np.linspace(GY.min(), GY.max(), bg_n)
    BX, BY = np.meshgrid(bx, by)
    bg_xy = np.column_stack([BX.ravel(), BY.ravel()])
    tree = cKDTree(feat_xy)
    dist_bg, _ = tree.query(bg_xy, k=1)
    far = dist_bg > max(grid.cell_size_m * 2.0, 30.0)
    bg_xy = bg_xy[far]

    # Thin features so GPR stays cheap (O(n^3) training).
    if feat_xy.shape[0] > 120:
        idx = np.linspace(0, feat_xy.shape[0] - 1, 120).astype(int)
        feat_xy = feat_xy[idx]

    X_train = np.vstack([feat_xy, bg_xy]) if len(bg_xy) else feat_xy
    y_train = np.concatenate([
        np.full(feat_xy.shape[0], _RISK_FEATURE_VALUE),
        np.full(len(bg_xy), _RISK_BACKGROUND_VALUE),
    ]) if len(bg_xy) else np.full(feat_xy.shape[0], _RISK_FEATURE_VALUE)

    # RBF length-scale ~ how far the "scent corridor" risk diffuses from cover.
    length_scale = max(grid.cell_size_m * 3.0, 40.0)
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * RBF(length_scale=length_scale, length_scale_bounds=(10.0, 500.0))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, alpha=1e-6, n_restarts_optimizer=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpr.fit(X_train, y_train)
        mean = gpr.predict(query)

    baseline = mean.reshape(ny, nx)
    baseline = np.clip(baseline, 0.0, 1.0)
    return baseline


def compute_forest_proximity(
    grid: Grid,
    forest_points_lonlat: np.ndarray,
    red_radius_m: float = 20.0,
    fade_m: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-cell proximity to the nearest OpenStreetMap forest edge.

    Returns (proximity, distance_m), both shaped (ny, nx):
      * distance_m  -- straight-line metres from each cell centre to the closest
                       forest-edge sample point (np.inf if no forest present).
      * proximity   -- in [0, 1]: exactly 1.0 within `red_radius_m` (the deep-red
                       danger band), then a SMOOTH exponential decay toward 0 over
                       `fade_m` metres as cells move toward the open centre.

    This is the geometric basis for the map's red→yellow→green colouring and is
    folded into the planning baseline so the drone is drawn to forest edges.
    """
    ny, nx = grid.shape
    if forest_points_lonlat is None or len(forest_points_lonlat) == 0:
        return (np.zeros((ny, nx), dtype=float),
                np.full((ny, nx), np.inf, dtype=float))

    fx, fy = grid.lonlat_to_xy(
        forest_points_lonlat[:, 0], forest_points_lonlat[:, 1])
    tree = cKDTree(np.column_stack([fx, fy]))

    GX, GY = grid.mesh_xy()
    query = np.column_stack([GX.ravel(), GY.ravel()])
    dist, _ = tree.query(query, k=1)
    dist = dist.reshape(ny, nx)

    over = np.maximum(0.0, dist - float(red_radius_m))
    proximity = np.exp(-over / max(float(fade_m), 1e-6))
    proximity[dist <= red_radius_m] = 1.0
    return np.clip(proximity, 0.0, 1.0), dist


# ==============================================================================
# 5. LIVESTOCK DENSITY VIA KDE  (Rule 2)
# ==============================================================================

def compute_livestock_density(
    grid: Grid,
    livestock_lonlat: Optional[np.ndarray],
) -> np.ndarray:
    """
    Kernel-density estimate of the herd over the grid (scipy.stats.gaussian_kde).
    Returns a field in [0, 1] (0 == no animals nearby). Shaped (ny, nx).
    """
    ny, nx = grid.shape
    if livestock_lonlat is None or len(livestock_lonlat) == 0:
        return np.zeros((ny, nx), dtype=float)

    livestock_lonlat = np.asarray(livestock_lonlat, dtype=float).reshape(-1, 2)
    lx, ly = grid.lonlat_to_xy(livestock_lonlat[:, 0], livestock_lonlat[:, 1])
    sample = np.vstack([lx, ly])  # shape (2, N)

    if sample.shape[1] < 3 or np.allclose(sample.std(axis=1), 0):
        # Too few / coincident points for a covariance -> use a fixed Gaussian bump.
        return _fixed_bump_density(grid, lx, ly)

    try:
        kde = gaussian_kde(sample)  # Scott's rule bandwidth
    except Exception:
        return _fixed_bump_density(grid, lx, ly)

    GX, GY = grid.mesh_xy()
    query = np.vstack([GX.ravel(), GY.ravel()])
    dens = kde(query).reshape(ny, nx)

    dmax = dens.max()
    if dmax > 0:
        dens = dens / dmax  # normalise to [0, 1]
    return dens


def _fixed_bump_density(grid: Grid, lx: np.ndarray, ly: np.ndarray) -> np.ndarray:
    """Fallback density: sum of fixed-width Gaussians around each animal."""
    GX, GY = grid.mesh_xy()
    sigma = max(grid.cell_size_m * 1.5, 20.0)
    dens = np.zeros(GX.shape, dtype=float)
    for x0, y0 in zip(np.atleast_1d(lx), np.atleast_1d(ly)):
        dens += np.exp(-((GX - x0) ** 2 + (GY - y0) ** 2) / (2.0 * sigma ** 2))
    dmax = dens.max()
    if dmax > 0:
        dens /= dmax
    return dens


# ==============================================================================
# 5b. ACTIVE THREAT FIELD  (Virtual Wolf Spawn)
# ==============================================================================

def compute_threat_field(
    grid: Grid,
    threats_lonlat: Optional[np.ndarray],
    sigma_cells: float = 1.6,
) -> np.ndarray:
    """
    Localised, sharply-peaked Gaussian field around each active threat (wolf).

    A detected/spawned wolf is a hard, near-certain danger, so its kernel is much
    tighter than the herd KDE. Returns a field in [0, 1], shaped (ny, nx). When
    fed into combine_risk() it dominates the matrix locally and forces the whole
    risk map -- and therefore the path planner -- to recalibrate.
    """
    ny, nx = grid.shape
    if threats_lonlat is None or len(threats_lonlat) == 0:
        return np.zeros((ny, nx), dtype=float)

    threats_lonlat = np.asarray(threats_lonlat, dtype=float).reshape(-1, 2)
    tx, ty = grid.lonlat_to_xy(threats_lonlat[:, 0], threats_lonlat[:, 1])

    GX, GY = grid.mesh_xy()
    sigma = max(sigma_cells * grid.cell_size_m, 1e-6)
    field = np.zeros(GX.shape, dtype=float)
    for x0, y0 in zip(np.atleast_1d(tx), np.atleast_1d(ty)):
        field += np.exp(-((GX - x0) ** 2 + (GY - y0) ** 2) / (2.0 * sigma ** 2))
    fmax = field.max()
    if fmax > 0:
        field /= fmax
    return field


# ==============================================================================
# 6. RISK COMBINATION + STRICT NORMALISATION  (Rule 3)
# ==============================================================================

def combine_risk(
    baseline: np.ndarray,
    density: np.ndarray,
    mask: np.ndarray,
    livestock_gain: float = 2.5,
    threat_field: Optional[np.ndarray] = None,
    threat_weight: float = 25.0,
) -> np.ndarray:
    """
    Combine the GPR baseline with the KDE livestock density. Dense clusters
    EXPONENTIALLY scale the underlying cell risk (architecture spec), then the
    matrix is STRICTLY clamped/normalised to [0, 1] to prevent overflow (Rule 3).

        risk_raw = baseline * exp(gain * density) + threat_weight * threat
        risk     = (risk_raw - min) / (max - min)         # -> [0, 1]

    An active threat (virtual wolf) is added with a large weight so it dominates
    the matrix locally and dynamically recalibrates the whole map after spawn.
    """
    baseline = np.clip(np.nan_to_num(baseline, nan=0.0), 0.0, 1.0)
    density = np.clip(np.nan_to_num(density, nan=0.0), 0.0, 1.0)
    gain = float(np.clip(livestock_gain, 0.0, 10.0))  # bound the exponent

    risk_raw = baseline * np.exp(gain * density)  # exp arg in [0, gain] -> finite

    if threat_field is not None:
        tf = np.clip(np.nan_to_num(threat_field, nan=0.0), 0.0, 1.0)
        risk_raw = risk_raw + float(max(threat_weight, 0.0)) * tf

    # Restrict normalisation statistics to valid (in-polygon) cells.
    valid = risk_raw[mask] if mask.any() else risk_raw.ravel()
    rmin, rmax = float(np.min(valid)), float(np.max(valid))
    if rmax - rmin < 1e-12:
        risk = np.zeros_like(risk_raw)
    else:
        risk = (risk_raw - rmin) / (rmax - rmin)

    risk = np.clip(risk, 0.0, 1.0)
    risk[~mask] = 0.0  # cells outside the pasture carry no patrol value
    return risk


# ==============================================================================
# 7. PER-CELL HARDWARE ENVELOPE  (Rule 5)
# ==============================================================================

def compute_flight_envelope(
    risk: np.ndarray,
    hw: HardwareConstraints,
) -> Dict[str, np.ndarray]:
    """
    Map every cell's risk to a dynamically safe altitude & velocity.

    Logic: high-risk cells are inspected from LOWER altitude (better GSD, the
    wolf spans more pixels) but must therefore fly SLOWER (smaller footprint ->
    tighter overlap constraint at 1.12 FPS). Low-risk cells fly high & fast.

        altitude(r) = h_max - r * (h_max - h_min)      # r in [0,1]
        velocity(r) = hardware-derived coverage cap at that altitude
    """
    h_min = float(hw.min_safety_altitude_m)
    h_max = float(hw.max_detection_altitude_m())
    if h_max < h_min:
        h_max = h_min

    r = np.clip(risk, 0.0, 1.0)
    altitude = h_max - r * (h_max - h_min)

    # Vectorise the per-altitude velocity computation.
    tan_v = math.tan(math.radians(hw.camera_fov_v_deg / 2.0))
    footprint_along = 2.0 * altitude * tan_v
    v_cover = hw.inference_rate_fps * footprint_along * (1.0 - hw.overlap_fraction)
    velocity = np.minimum(v_cover, hw.drone_max_velocity_ms)

    return {
        "altitude_m": altitude,
        "velocity_ms": velocity,
        "h_min_m": h_min,
        "h_max_m": h_max,
    }


# ==============================================================================
# 8. NATIVE PATH-PLANNING HEURISTICS  (Rule 4: NumPy/SciPy only)
# ==============================================================================

def ergodic_spectral_coefficients(
    risk: np.ndarray,
    mask: np.ndarray,
    n_modes: int = 6,
) -> np.ndarray:
    """
    Fourier (cosine) spectral decomposition of the target risk distribution,
    used as the ergodic reference. The ergodic objective (Mathew & Mezić) drives
    the time-averaged visitation statistics of the trajectory toward this
    distribution -- i.e. time-in-cell proportional to risk.

    Returns the (n_modes x n_modes) matrix of normalised mode coefficients
    phi_k of the spatial risk PDF over the unit square.
    """
    ny, nx = risk.shape
    pdf = np.where(mask, risk, 0.0)
    total = pdf.sum()
    if total <= 0:
        return np.zeros((n_modes, n_modes))
    pdf = pdf / total

    ys = (np.arange(ny) + 0.5) / ny
    xs = (np.arange(nx) + 0.5) / nx

    coeffs = np.zeros((n_modes, n_modes))
    for ky in range(n_modes):
        cos_y = np.cos(math.pi * ky * ys)              # (ny,)
        for kx in range(n_modes):
            cos_x = np.cos(math.pi * kx * xs)          # (nx,)
            basis = np.outer(cos_y, cos_x)             # (ny, nx)
            coeffs[ky, kx] = np.sum(pdf * basis)
    # Normalise so the dominant (0,0) mode is 1.
    if abs(coeffs[0, 0]) > 1e-12:
        coeffs = coeffs / coeffs[0, 0]
    return coeffs


def mdp_value_iteration(
    risk: np.ndarray,
    mask: np.ndarray,
    gamma: float = 0.9,
    n_iter: int = 60,
) -> np.ndarray:
    """
    Native Markov Decision Process value iteration over the 4-connected grid.

    Reward = cell risk. The converged value function V(cell) ranks cells by the
    long-horizon discounted risk reachable from them -> used for next-best-cell
    selection in the receding-horizon loop. Returns V shaped (ny, nx).
    """
    ny, nx = risk.shape
    reward = np.where(mask, risk, -1.0)  # leaving the pasture is penalised
    V = np.where(mask, risk, 0.0).astype(float)

    for _ in range(n_iter):
        # Neighbour value lookups via padded shifts (Dirichlet-like boundary).
        up = np.vstack([V[:1, :], V[:-1, :]])
        down = np.vstack([V[1:, :], V[-1:, :]])
        left = np.hstack([V[:, :1], V[:, :-1]])
        right = np.hstack([V[:, 1:], V[:, -1:]])
        best_neighbour = np.maximum.reduce([up, down, left, right])
        V_new = reward + gamma * best_neighbour
        V_new = np.where(mask, V_new, 0.0)
        if np.max(np.abs(V_new - V)) < 1e-4:
            V = V_new
            break
        V = V_new
    return V


def _greedy_then_2opt(coords: np.ndarray, max_iter: int = 200) -> List[int]:
    """Nearest-neighbour TSP construction refined by 2-opt (open tour).

    coords : (m, 2) array of waypoint positions (local metres).
    Returns an ordering (list of indices into coords).
    """
    m = coords.shape[0]
    if m <= 2:
        return list(range(m))

    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

    # --- nearest-neighbour construction ---
    unvisited = set(range(m))
    start = 0
    tour = [start]
    unvisited.remove(start)
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda j: D[last, j])
        tour.append(nxt)
        unvisited.remove(nxt)

    # --- 2-opt refinement (open path: do not wrap the last edge) ---
    def tour_len(t: List[int]) -> float:
        return float(sum(D[t[i], t[i + 1]] for i in range(len(t) - 1)))

    improved = True
    it = 0
    best_len = tour_len(tour)
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(1, m - 1):
            for k in range(i + 1, m):
                new_tour = tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]
                new_len = tour_len(new_tour)
                if new_len + 1e-9 < best_len:
                    tour, best_len = new_tour, new_len
                    improved = True
    return tour


def plan_receding_horizon(
    risk: np.ndarray,
    envelope: Dict[str, np.ndarray],
    grid: Grid,
    drone_lonlat: Tuple[float, float],
    n_waypoints: int = 4,
    n_candidates: int = 12,
) -> List[Dict[str, float]]:
    """
    Receding-horizon planner: returns ONLY the next `n_waypoints` (3-5) 3D
    waypoints, never a full static tour (architecture: static planning forbidden).

    Method:
      1. MDP value iteration ranks cells by long-horizon reachable risk.
      2. Take the top `n_candidates` in-polygon cells (the current information
         frontier) near the drone, weighted by value AND proximity.
      3. Order them into a fluid flight string with the native TSP heuristic
         (MTSP degenerates to a single salesperson for one drone).
      4. Attach per-cell altitude & velocity from the hardware envelope.

    Output: list of dicts {lat, lon, altitude_m, speed_ms, risk}.
    """
    n_waypoints = int(np.clip(n_waypoints, 3, 5))
    ny, nx = risk.shape

    V = mdp_value_iteration(risk, grid.mask)

    # Drone position in local metres.
    dx, dy = grid.lonlat_to_xy(np.array([drone_lonlat[0]]), np.array([drone_lonlat[1]]))
    drone_xy = np.array([dx[0], dy[0]])

    GX, GY = grid.mesh_xy()
    valid_idx = np.argwhere(grid.mask)
    if valid_idx.size == 0:
        return []

    # Score = MDP value discounted by travel distance from the drone.
    cell_xy = np.column_stack([GX[grid.mask], GY[grid.mask]])
    dist = np.linalg.norm(cell_xy - drone_xy[None, :], axis=1)
    dist_scale = max(grid.cell_size_m * 4.0, 50.0)
    score = V[grid.mask] * np.exp(-dist / dist_scale)

    n_take = min(n_candidates, score.size)
    top = np.argsort(score)[::-1][:n_take]
    cand_rows = valid_idx[top]            # (n_take, 2) -> (row=y, col=x)
    cand_xy = cell_xy[top]

    # Order candidates into a smooth string, starting from the drone.
    coords = np.vstack([drone_xy, cand_xy])
    order = _greedy_then_2opt(coords)
    # Drop the drone's own start node, keep the planned visiting order.
    ordered_cand = [i - 1 for i in order if i != 0]

    waypoints: List[Dict[str, float]] = []
    for ci in ordered_cand[:n_waypoints]:
        ry, rx = cand_rows[ci]
        waypoints.append({
            "lat": float(grid.lats[ry]),
            "lon": float(grid.lons[rx]),
            "altitude_m": float(envelope["altitude_m"][ry, rx]),
            "speed_ms": float(envelope["velocity_ms"][ry, rx]),
            "risk": float(risk[ry, rx]),
        })
    return waypoints


# ==============================================================================
# 9. ORCHESTRATOR
# ==============================================================================

@dataclass
class RiskModelResult:
    grid: Grid
    baseline_risk: np.ndarray
    livestock_density: np.ndarray
    risk: np.ndarray
    altitude_m: np.ndarray
    velocity_ms: np.ndarray
    waypoints: List[Dict[str, float]]
    feature_source: str
    h_min_m: float
    h_max_m: float
    forest_proximity: np.ndarray          # [0,1]; 1.0 within 20 m of forest edge
    forest_distance_m: np.ndarray         # metres to nearest forest edge (inf if none)
    forest_points_lonlat: np.ndarray      # the OSM forest-edge sample points


class RiskModel:
    """End-to-end mathematical core. Build once, then call `update()` per cycle."""

    def __init__(
        self,
        polygon: Optional[Sequence[Tuple[float, float]]] = None,
        cell_size_m: float = 25.0,
        hardware_path: Optional[str] = None,
        livestock_gain: float = 2.5,
    ):
        self.hw = (
            HardwareConstraints.from_file(hardware_path)
            if hardware_path else HardwareConstraints()
        )
        self.livestock_gain = livestock_gain
        self.grid = build_grid(polygon, cell_size_m=cell_size_m)
        # Static layers are fetched once (they do not change between cycles).
        self.features = fetch_static_features(self.grid)
        self.baseline_gpr = compute_baseline_risk(self.grid, self.features)
        # Geometric forest-edge proximity: 1.0 within 20 m (the deep-red band),
        # smoothly fading toward the open centre.
        self.forest_proximity, self.forest_distance_m = compute_forest_proximity(
            self.grid, self.features.forest_points_lonlat)
        # The forest band anchors the high end of the static baseline so the
        # planner is genuinely drawn to forest edges and the map reads true.
        self.baseline = np.clip(
            np.maximum(self.baseline_gpr, self.forest_proximity), 0.0, 1.0)

    def update(
        self,
        livestock_lonlat: Optional[np.ndarray],
        drone_lonlat: Tuple[float, float],
        threats_lonlat: Optional[np.ndarray] = None,
        n_waypoints: int = 4,
    ) -> RiskModelResult:
        """One receding-horizon cycle: recompute live risk and next waypoints.

        `threats_lonlat` carries any active virtual-wolf spawns; passing them in
        forces the risk matrix (and hence the planner) to recalibrate.
        """
        density = compute_livestock_density(self.grid, livestock_lonlat)
        threat_field = compute_threat_field(self.grid, threats_lonlat)
        risk = combine_risk(
            self.baseline, density, self.grid.mask, self.livestock_gain,
            threat_field=threat_field,
        )
        env = compute_flight_envelope(risk, self.hw)
        waypoints = plan_receding_horizon(
            risk, env, self.grid, drone_lonlat, n_waypoints=n_waypoints,
        )
        return RiskModelResult(
            grid=self.grid,
            baseline_risk=self.baseline,
            livestock_density=density,
            risk=risk,
            altitude_m=env["altitude_m"],
            velocity_ms=env["velocity_ms"],
            waypoints=waypoints,
            feature_source=self.features.source,
            h_min_m=env["h_min_m"],
            h_max_m=env["h_max_m"],
            forest_proximity=self.forest_proximity,
            forest_distance_m=self.forest_distance_m,
            forest_points_lonlat=self.features.forest_points_lonlat,
        )

    def sample_threat_location(self, index: int = 0) -> Tuple[float, float]:
        """Pick a plausible wolf-spawn coordinate near natural cover.

        Wolves approach from forest edges / hedges / creeks, exactly where the
        GPR baseline risk peaks. We select among the highest-baseline in-polygon
        cells (deterministic in `index`) and jitter inside the cell.
        """
        base = np.where(self.grid.mask, self.baseline, -np.inf)
        ny, nx = self.grid.shape
        order = np.argsort(base.ravel())[::-1]
        n_valid = int(self.grid.mask.sum())
        pool = max(5, int(0.15 * n_valid))
        top = order[:min(pool, n_valid)]
        if top.size == 0:
            return (float(self.grid.lons.mean()), float(self.grid.lats.mean()))
        flat = int(top[index % top.size])
        ry, rx = divmod(flat, nx)

        # Deterministic sub-cell jitter so repeated spawns are not identical.
        dlon = (self.grid.lons[1] - self.grid.lons[0]) if nx > 1 else 0.0
        dlat = (self.grid.lats[1] - self.grid.lats[0]) if ny > 1 else 0.0
        jx = (((index * 37) % 7) / 7.0 - 0.5) * 0.6
        jy = (((index * 53) % 5) / 5.0 - 0.5) * 0.6
        lon = float(self.grid.lons[rx] + jx * dlon)
        lat = float(self.grid.lats[ry] + jy * dlat)
        return (lon, lat)


# ==============================================================================
# 10. SELF-TEST / DEMO
# ==============================================================================

def _demo() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    hw_path = os.path.join(here, "agent_context", "hardware_constraints.txt")

    print("=" * 70)
    print("PHASE 1 MATHEMATICAL CORE -- SELF TEST")
    print("=" * 70)
    print(f"osmnx available : {_OSMNX_AVAILABLE}")
    print(f"overpy available: {_OVERPY_AVAILABLE}")

    model = RiskModel(
        polygon=None,                 # -> hardcoded Swiss pasture fallback
        cell_size_m=20.0,
        hardware_path=hw_path,
    )

    hw = model.hw
    print("\n[HARDWARE ENVELOPE]")
    print(f"  Max detection altitude : {hw.max_detection_altitude_m():6.2f} m")
    print(f"  Min safety altitude    : {hw.min_safety_altitude_m:6.2f} m")
    print(f"  V @ max altitude       : {hw.max_velocity_for_altitude(hw.max_detection_altitude_m()):6.2f} m/s")
    print(f"  V @ min altitude       : {hw.max_velocity_for_altitude(hw.min_safety_altitude_m):6.2f} m/s")
    print(f"  Range @ V_max          : {hw.max_range_m(hw.drone_max_velocity_ms):6.0f} m")

    print(f"\n[GRID] shape={model.grid.shape}, "
          f"valid cells={int(model.grid.mask.sum())}, "
          f"feature source={model.features.source}, "
          f"feature points={len(model.features)}")

    # Synthetic herd: two clusters inside the parcel (Simulation-mode style).
    lon0, lon1 = model.grid.lons.min(), model.grid.lons.max()
    lat0, lat1 = model.grid.lats.min(), model.grid.lats.max()
    rng = np.random.default_rng(42)
    cluster_a = np.column_stack([
        rng.normal(lon0 + 0.35 * (lon1 - lon0), 0.00015, 18),
        rng.normal(lat0 + 0.40 * (lat1 - lat0), 0.00010, 18),
    ])
    cluster_b = np.column_stack([
        rng.normal(lon0 + 0.70 * (lon1 - lon0), 0.00012, 12),
        rng.normal(lat0 + 0.65 * (lat1 - lat0), 0.00010, 12),
    ])
    livestock = np.vstack([cluster_a, cluster_b])
    drone = (lon0 + 0.5 * (lon1 - lon0), lat0 + 0.1 * (lat1 - lat0))

    result = model.update(livestock, drone, n_waypoints=5)

    print("\n[RISK MATRIX]")
    valid = result.risk[result.grid.mask]
    print(f"  min={valid.min():.3f}  max={valid.max():.3f}  mean={valid.mean():.3f}")
    assert valid.min() >= 0.0 and valid.max() <= 1.0, "risk not normalised to [0,1]!"
    print("  -> strictly normalised in [0, 1]  OK")

    print("\n[NEXT WAYPOINTS] (receding horizon)")
    for i, wp in enumerate(result.waypoints, 1):
        print(f"  {i}. lat={wp['lat']:.5f} lon={wp['lon']:.5f} "
              f"alt={wp['altitude_m']:5.1f} m  v={wp['speed_ms']:4.1f} m/s  "
              f"risk={wp['risk']:.3f}")

    print("\n[ERGODIC SPECTRAL COEFFICIENTS] (4x4 head)")
    coeffs = ergodic_spectral_coefficients(result.risk, result.grid.mask, n_modes=4)
    with np.printoptions(precision=3, suppress=True):
        print(coeffs)

    print("\nSelf-test completed successfully.")


if __name__ == "__main__":
    _demo()
