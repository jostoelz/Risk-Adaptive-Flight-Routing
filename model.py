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
from scipy.interpolate import make_interp_spline
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

# Spacing (metres) of the virtual nodes interpolated ALONG OSM feature edges.
# Densifying the polylines turns sparse OSM vertices into a continuous, unbroken
# high-risk buffer along the whole forest-pasture boundary.
FEATURE_DENSIFY_SPACING_M = 7.0
# Hard cap on densified feature points so a huge OSM way cannot explode the KDTree.
_MAX_FEATURE_POINTS = 4000

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


def _densify_polyline_lonlat(
    coords: Sequence[Tuple[float, float]],
    spacing_m: float,
    ref_lat: float,
) -> List[Tuple[float, float]]:
    """Insert virtual nodes every ~spacing_m metres along a (lon, lat) polyline.

    Uses a local equirectangular scale at ``ref_lat`` to measure segment lengths
    in metres, then linearly subdivides each segment. This converts sparse OSM
    vertices into a continuous run of points so the downstream distance field has
    no gaps along the edge.
    """
    pts = [(float(c[0]), float(c[1])) for c in coords]
    if len(pts) < 2:
        return pts
    m_per_deg_lat = math.radians(1.0) * _EARTH_RADIUS_M
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(ref_lat))
    spacing_m = max(float(spacing_m), 1e-6)

    out: List[Tuple[float, float]] = []
    for (lo0, la0), (lo1, la1) in zip(pts[:-1], pts[1:]):
        dx = (lo1 - lo0) * m_per_deg_lon
        dy = (la1 - la0) * m_per_deg_lat
        seg_m = math.hypot(dx, dy)
        n = max(1, int(math.ceil(seg_m / spacing_m)))
        for k in range(n):  # include start, exclude end (added by next segment)
            t = k / n
            out.append((lo0 + t * (lo1 - lo0), la0 + t * (la1 - la0)))
    out.append(pts[-1])
    return out


def _cap_points(arr: np.ndarray, max_n: int = _MAX_FEATURE_POINTS) -> np.ndarray:
    """Stride-subsample a point array if it exceeds max_n (keeps it bounded)."""
    if len(arr) > max_n:
        stride = int(math.ceil(len(arr) / max_n))
        return arr[::stride]
    return arr


def _sample_geometry_points(
    geom,
    ref_lat: float,
    spacing_m: float = FEATURE_DENSIFY_SPACING_M,
) -> List[Tuple[float, float]]:
    """Extract DENSIFIED (lon, lat) points from a shapely geometry.

    Lines and polygon boundaries are interpolated every ~spacing_m metres so the
    forest-pasture edge becomes a continuous high-risk buffer rather than a few
    scattered OSM nodes.
    """
    pts: List[Tuple[float, float]] = []
    try:
        gtype = geom.geom_type
        if gtype == "Point":
            pts.append((geom.x, geom.y))
        elif gtype in ("LineString", "LinearRing"):
            pts.extend(_densify_polyline_lonlat(list(geom.coords), spacing_m, ref_lat))
        elif gtype == "Polygon":
            # forest *edges* -> the exterior boundary, densified
            pts.extend(_densify_polyline_lonlat(
                list(geom.exterior.coords), spacing_m, ref_lat))
        elif gtype.startswith("Multi") or gtype == "GeometryCollection":
            for sub in geom.geoms:
                pts.extend(_sample_geometry_points(sub, ref_lat, spacing_m))
    except Exception:
        return []
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
    ref_lat = float(np.mean(ring[:, 1]))
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
                pts_this.extend(_sample_geometry_points(geom, ref_lat))
            collected.extend(pts_this)
            if name == "forest":
                forest.extend(pts_this)
        except Exception:
            continue
    if not collected:
        return None
    return StaticFeatures(
        _cap_points(np.asarray(collected, dtype=float)), source="osmnx",
        forest_points_lonlat=(_cap_points(np.asarray(forest, dtype=float))
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
    ref_lat = float(np.mean(ring[:, 1]))
    collected: List[Tuple[float, float]] = []
    forest: List[Tuple[float, float]] = []
    for way in getattr(result, "ways", []):
        node_pts: List[Tuple[float, float]] = []
        for node in way.get_nodes(resolve_missing=False) or []:
            try:
                node_pts.append((float(node.lon), float(node.lat)))
            except Exception:
                continue
        # Densify along the way so the edge buffer is continuous.
        pts_this = _densify_polyline_lonlat(node_pts, FEATURE_DENSIFY_SPACING_M, ref_lat)
        collected.extend(pts_this)
        tags = getattr(way, "tags", {}) or {}
        if tags.get("natural") == "wood" or tags.get("landuse") == "forest":
            forest.extend(pts_this)
    if not collected:
        return None
    return StaticFeatures(
        _cap_points(np.asarray(collected, dtype=float)), source="overpy",
        forest_points_lonlat=(_cap_points(np.asarray(forest, dtype=float))
                              if forest else np.empty((0, 2))),
    )


def _synthetic_features(grid: Grid) -> StaticFeatures:
    """Deterministic stand-in features so the demo always has risk structure.

    Places a 'forest edge' line along the northern boundary and a diagonal
    'creek' through the parcel -- both common high-risk wolf corridors.
    """
    lon_min, lon_max = grid.lons.min(), grid.lons.max()
    lat_min, lat_max = grid.lats.min(), grid.lats.max()

    ref_lat = float(np.mean(grid.lats))
    # Forest edge near the top (north) of the parcel -- two endpoints, densified.
    forest_lat_val = lat_max - 0.06 * (lat_max - lat_min)
    forest_line = [(lon_min, forest_lat_val), (lon_max, forest_lat_val)]
    # Creek cutting diagonally across the parcel.
    creek_line = [
        (lon_min + 0.15 * (lon_max - lon_min), lat_min + 0.85 * (lat_max - lat_min)),
        (lon_min + 0.85 * (lon_max - lon_min), lat_min + 0.15 * (lat_max - lat_min)),
    ]
    forest_dens = _densify_polyline_lonlat(
        forest_line, FEATURE_DENSIFY_SPACING_M, ref_lat)
    creek_dens = _densify_polyline_lonlat(
        creek_line, FEATURE_DENSIFY_SPACING_M, ref_lat)

    pts = np.asarray(forest_dens + creek_dens, dtype=float)
    forest_pts = np.asarray(forest_dens, dtype=float)
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
# 5. HERD PROXIMITY FIELD VIA KDE  (Rule 2)  — the dynamic half of V2.0 risk
# ==============================================================================

# Bandwidth multiplier applied to Silverman's rule in the SCATTERED state, to
# melt scattered individual animals into one continuous area-coverage cloud.
SCATTERED_BW_FACTOR = 2.5


def compute_herd_proximity(
    grid: Grid,
    livestock_lonlat: Optional[np.ndarray],
    guard_radius_m: float = 70.0,
    scattered: bool = False,
) -> np.ndarray:
    """
    Smooth herd-proximity field H in [0, 1] (scipy.stats.gaussian_kde).

    Two regimes:
      * Normal (compact / locked cluster): the KDE bandwidth is widened to roughly
        `guard_radius_m`, giving a tight protective zone around the herd centroid.
      * SCATTERED (scattered=True): the bandwidth is Silverman's rule × 2.5
        (`SCATTERED_BW_FACTOR`), melting the individual animal dots into one large
        continuous risk cloud spanning the whole active pasture — so the planner
        sweeps the area rather than locking onto a single isolated animal.

    A fixed Gaussian-bump fallback covers tiny/degenerate herds. Recomputed from
    live coordinates each cycle. Shaped (ny, nx).
    """
    ny, nx = grid.shape
    if livestock_lonlat is None or len(livestock_lonlat) == 0:
        return np.zeros((ny, nx), dtype=float)

    ll = np.asarray(livestock_lonlat, dtype=float).reshape(-1, 2)
    lx, ly = grid.lonlat_to_xy(ll[:, 0], ll[:, 1])
    sample = np.vstack([lx, ly])  # shape (2, N)

    field = None
    if sample.shape[1] >= 3 and not np.allclose(sample.std(axis=1), 0.0):
        try:
            if scattered:
                # Silverman's rule, then significantly widened.
                kde = gaussian_kde(sample, bw_method="silverman")
                kde.set_bandwidth(bw_method=kde.factor * SCATTERED_BW_FACTOR)
            else:
                # Widen the kernel so its std ≈ guard radius (factor ≈ guard/spread).
                spread = max(float(np.hypot(sample[0].std(), sample[1].std())), 1.0)
                bw = float(np.clip(guard_radius_m / spread, 0.15, 25.0))
                kde = gaussian_kde(sample, bw_method=bw)
            GX, GY = grid.mesh_xy()
            field = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(ny, nx)
        except Exception:
            field = None

    if field is None:
        sigma = guard_radius_m * (SCATTERED_BW_FACTOR if scattered else 1.0)
        field = _fixed_bump_density(grid, lx, ly, sigma_m=sigma)

    fmax = field.max()
    if fmax > 0:
        field = field / fmax  # normalise to [0, 1]
    return field


def _fixed_bump_density(grid: Grid, lx: np.ndarray, ly: np.ndarray,
                        sigma_m: Optional[float] = None) -> np.ndarray:
    """Fallback herd field: sum of fixed-width Gaussians around each animal."""
    GX, GY = grid.mesh_xy()
    sigma = float(sigma_m) if sigma_m else max(grid.cell_size_m * 1.5, 20.0)
    dens = np.zeros(GX.shape, dtype=float)
    for x0, y0 in zip(np.atleast_1d(lx), np.atleast_1d(ly)):
        dens += np.exp(-((GX - x0) ** 2 + (GY - y0) ** 2) / (2.0 * sigma ** 2))
    dmax = dens.max()
    if dmax > 0:
        dens /= dmax
    return dens


# Herd dispersion above which the state is declared SCATTERED (metres, RMS
# radius from the centroid) when there is no clean two-cluster split.
SCATTER_SPREAD_M = 60.0

# Herd state constants.
STATE_COMPACT = "compact"
STATE_SPLIT = "split"
STATE_SCATTERED = "scattered"


def select_focus_cluster(
    livestock_lonlat: Optional[np.ndarray],
    drone_lonlat: Optional[Tuple[float, float]],
    lock_centroid: Optional[Tuple[float, float]] = None,
    hysteresis_m: float = 50.0,
    min_separation_m: float = 45.0,
    gap_factor: float = 2.2,
    scatter_spread_m: float = SCATTER_SPREAD_M,
    force_scattered: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]], str,
           Optional[np.ndarray], Optional[Tuple[float, float]]]:
    """
    Classify the herd and pick what the drone should track. Three states:

      * SCATTERED -- checked FIRST as a hard override: if `force_scattered` is set
                     (the scenario is explicitly Scattered) OR the omnidirectional
                     RMS spread exceeds `scatter_spread_m`, the SVD split check and
                     all cluster-locking are COMPLETELY BYPASSED. The whole herd is
                     returned (no sub-sampling, no lock) so a wide KDE covers the
                     field and the drone area-sweeps. This stops the mode-collision
                     flip-flop where a stochastic gap spuriously triggered a split.
      * SPLIT     -- only reachable when NOT scattered: two clean SVD clusters
                     (largest principal-axis gap clearly bimodal AND groups far
                     apart). Locks onto ONE group with hysteresis.
      * COMPACT   -- a single tight cluster -> orbit its centroid.

    Returns (subset_lonlat, chosen_centroid, state, guarded_mask, other_centroid):
      * guarded_mask  -- bool over the INPUT order: True for the guarded subgroup
                         (all True unless SPLIT). Frontend dims the rest.
      * other_centroid-- (lon, lat) of the deprioritised cluster (SPLIT only).
    chosen_centroid is None outside the SPLIT state (the caller clears its lock).
    """
    if livestock_lonlat is None:
        return None, None, STATE_COMPACT, None, None
    pts = np.asarray(livestock_lonlat, dtype=float).reshape(-1, 2)
    n = pts.shape[0]
    if n == 0:
        return pts, None, STATE_COMPACT, np.zeros(0, dtype=bool), None

    # Local metric scale (lon/lat -> metres) at the herd's latitude.
    ref_lat = float(np.mean(pts[:, 1]))
    mlat = math.radians(1.0) * _EARTH_RADIUS_M
    mlon = mlat * math.cos(math.radians(ref_lat))

    def _m(a, b):  # distance in metres between two (lon, lat) points
        return math.hypot((a[0] - b[0]) * mlon, (a[1] - b[1]) * mlat)

    # Omnidirectional RMS spread (metres) about the centroid.
    c = pts.mean(axis=0)
    X = np.column_stack([(pts[:, 0] - c[0]) * mlon, (pts[:, 1] - c[1]) * mlat])
    spread_m = float(np.sqrt(np.mean(np.sum(X ** 2, axis=1)))) if n else 0.0

    # --- HARD SCATTERED OVERRIDE (checked before any SVD / lock) ---------------
    # An explicit Scattered scenario, or genuinely wide dispersion, forces the
    # area-coverage state. We skip the SVD entirely and return ALL coordinates
    # with NO lock, so minor stochastic motion can never flip us into Split-lock.
    if force_scattered or spread_m > scatter_spread_m:
        return pts, None, STATE_SCATTERED, np.ones(n, dtype=bool), None

    # Below here the herd is NOT scattered -> any fallback collapses to COMPACT.
    def _compact():
        return pts, None, STATE_COMPACT, np.ones(n, dtype=bool), None

    if n < 4 or drone_lonlat is None:
        return _compact()

    # Try a clean two-cluster split along the principal axis (largest gap).
    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
    except Exception:
        return _compact()
    proj = X @ vt[0]
    order = np.argsort(proj)
    gaps = np.diff(proj[order])
    if gaps.size == 0:
        return _compact()
    gi = int(np.argmax(gaps))
    max_gap = float(gaps[gi])
    med_gap = float(np.median(gaps))
    med_gap = med_gap if med_gap > 1e-9 else 1e-9

    idx_a, idx_b = order[:gi + 1], order[gi + 1:]
    if idx_a.size < 2 or idx_b.size < 2:
        return _compact()

    ca = tuple(pts[idx_a].mean(axis=0))
    cb = tuple(pts[idx_b].mean(axis=0))
    sep_m = _m(ca, cb)
    if not (max_gap > gap_factor * med_gap and sep_m > min_separation_m):
        return _compact()

    da, db = _m(ca, drone_lonlat), _m(cb, drone_lonlat)

    if lock_centroid is not None:
        # Continuity: the group whose centroid is nearest the locked one.
        if _m(ca, lock_centroid) <= _m(cb, lock_centroid):
            cont_idx, cont_c, cont_d = idx_a, ca, da
            alt_idx, alt_c, alt_d = idx_b, cb, db
        else:
            cont_idx, cont_c, cont_d = idx_b, cb, db
            alt_idx, alt_c, alt_d = idx_a, ca, da
        if alt_d + hysteresis_m < cont_d:      # other group decisively closer
            chosen_idx, chosen_c, other_c = alt_idx, alt_c, cont_c
        else:
            chosen_idx, chosen_c, other_c = cont_idx, cont_c, alt_c
    else:
        if da <= db:
            chosen_idx, chosen_c, other_c = idx_a, ca, cb
        else:
            chosen_idx, chosen_c, other_c = idx_b, cb, ca

    guarded_mask = np.zeros(n, dtype=bool)
    guarded_mask[chosen_idx] = True
    return (pts[chosen_idx],
            (float(chosen_c[0]), float(chosen_c[1])),
            STATE_SPLIT,
            guarded_mask,
            (float(other_c[0]), float(other_c[1])))


# ==============================================================================
# 6. RISK COMBINATION + STRICT NORMALISATION  (Rule 3)  — V2.0 dynamic coupling
# ==============================================================================

def combine_risk(
    forest_proximity: np.ndarray,
    herd_proximity: np.ndarray,
    baseline_gpr: np.ndarray,
    mask: np.ndarray,
    livestock_gain: float = 2.5,
    forest_boost: float = 0.30,
    w_static: float = 0.08,
) -> np.ndarray:
    """
    PHILOSOPHY A — "Bodyguard Mode": the risk objective is centred on the herd.

        herd_core = herd_proximity * (1 + forest_boost * forest_proximity)
        risk_raw  = livestock_gain * herd_core + w_static * baseline_gpr
        risk      = (risk_raw - min) / (max - min)          # strictly -> [0, 1]

    Design (Bodyguard):
      * The HERD field H is the dominant driver, so the peak risk sits ON the herd
        (a tight cluster -> a single hotspot; a scattered herd -> a broad cloud,
        because H is built from a deliberately wide KDE in the SCATTERED state).
      * The forest field acts only as a MINOR MULTIPLIER (1 + forest_boost * F),
        gently emphasising the flank facing the tree line; it can never out-score
        the herd itself.
      * The faint w_static term preserves creek/hedge structure.
    Because H is recomputed from live coordinates each cycle, the hotspot/cloud
    moves with the herd and the RHC planner re-routes to follow it.

    The matrix is strictly min-max normalised to [0, 1] to prevent overflow.
    """
    F = np.clip(np.nan_to_num(forest_proximity, nan=0.0), 0.0, 1.0)
    H = np.clip(np.nan_to_num(herd_proximity, nan=0.0), 0.0, 1.0)
    B = np.clip(np.nan_to_num(baseline_gpr, nan=0.0), 0.0, 1.0)
    g = float(np.clip(livestock_gain, 0.0, 10.0))
    fb = float(max(forest_boost, 0.0))

    herd_core = H * (1.0 + fb * F)          # herd-centred; forest only amplifies
    risk_raw = g * herd_core + float(max(w_static, 0.0)) * B

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


def _mdp_policy_rollout(
    V: np.ndarray,
    mask: np.ndarray,
    start_rc: Tuple[int, int],
    n_steps: int,
) -> List[Tuple[int, int]]:
    """Greedy steepest-ascent rollout of the MDP-optimal policy over value V.

    From `start_rc`, repeatedly step to the highest-value 8-connected, in-polygon,
    not-yet-visited neighbour. This climbs the value surface toward the herd peak
    and then traces a ring around it (the visited set forces it outward into an
    orbit). Returns up to `n_steps` (row, col) cells AHEAD of the drone — the
    start cell itself is never included, so the drone always has somewhere to go.
    """
    ny, nx = V.shape
    sr, sc = start_rc
    sr = int(np.clip(sr, 0, ny - 1))
    sc = int(np.clip(sc, 0, nx - 1))
    visited = {(sr, sc)}
    cur = (sr, sc)
    path: List[Tuple[int, int]] = []
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for _ in range(int(n_steps)):
        best = None
        best_v = -np.inf
        for dr, dc in neigh:
            r, c = cur[0] + dr, cur[1] + dc
            if 0 <= r < ny and 0 <= c < nx and mask[r, c] and (r, c) not in visited:
                if V[r, c] > best_v:
                    best_v = V[r, c]
                    best = (r, c)
        if best is None:
            break
        path.append(best)
        visited.add(best)
        cur = best
    return path


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
      1. MDP value iteration ranks cells by long-horizon reachable risk; the
         value peak sits on the herd hotspot.
      2. Take the top `n_candidates` in-polygon cells by value, with a *gentle*
         pasture-scale distance preference so the drone is drawn toward the
         hotspot (and its near flank) rather than trapped locally. The drone's
         OWN cell is explicitly excluded, so a waypoint is always somewhere to
         travel TO — never "stay put".
      3. Order them into a fluid flight string with the native TSP heuristic
         (MTSP degenerates to a single salesperson for one drone).
      4. Attach per-cell altitude & velocity from the hardware envelope.

    Output: list of dicts {lat, lon, altitude_m, speed_ms, risk}.
    """
    n_waypoints = int(np.clip(n_waypoints, 3, 5))
    ny, nx = risk.shape

    V = mdp_value_iteration(risk, grid.mask)
    if not grid.mask.any():
        return []

    # The drone's current grid cell (nearest centre).
    d_rx = int(np.argmin(np.abs(grid.lons - drone_lonlat[0])))
    d_ry = int(np.argmin(np.abs(grid.lats - drone_lonlat[1])))

    # Roll out the MDP-optimal policy (greedy steepest-ascent on V) from the
    # drone cell. The value field peaks on the herd, so the rollout climbs
    # monotonically toward the herd and then circles it: a STABLE goal-directed
    # track that never oscillates in place. This is the native MTSP/MDP engine
    # collapsed to a single-agent rolling horizon.
    cells = _mdp_policy_rollout(V, grid.mask, (d_ry, d_rx), n_waypoints)
    if not cells:
        return []

    waypoints: List[Dict[str, float]] = []
    for ry, rx in cells:
        waypoints.append({
            "lat": float(grid.lats[ry]),
            "lon": float(grid.lons[rx]),
            "altitude_m": float(envelope["altitude_m"][ry, rx]),
            "speed_ms": float(envelope["velocity_ms"][ry, rx]),
            "risk": float(risk[ry, rx]),
        })
    return waypoints


def plan_orbit(
    centroid_lonlat: Tuple[float, float],
    drone_lonlat: Tuple[float, float],
    grid: Grid,
    risk: np.ndarray,
    envelope: Dict[str, np.ndarray],
    n_waypoints: int = 4,
    radius_m: float = 22.0,
    arc_span_deg: float = 150.0,
    direction: int = 1,
) -> List[Dict[str, float]]:
    """
    Orbit-Guard planner (Philosophy A Pro): a TRUE geometric bodyguard ring.

    Rather than letting the MDP planner hunt the noisy individual-animal KDE bumps
    at the cluster centre — which causes the waypoint chattering / cross-cutting —
    we place the next `n_waypoints` on a smooth circle of `radius_m` around the
    herd cluster's dynamic centroid, located AHEAD of the drone's current angular
    position (fixed orbit direction) so the path is stable tick-to-tick and the
    drone progresses smoothly around the ring. As the centroid drifts with the
    herd, the ring translates, giving continuous tactical tracking. The returned
    waypoints are fed through the same cubic B-spline smoother for rendering.

    Output: list of {lat, lon, altitude_m, speed_ms, risk} on the orbit ring.
    """
    n = int(np.clip(n_waypoints, 3, 5))
    clon, clat = float(centroid_lonlat[0]), float(centroid_lonlat[1])

    mlat = math.radians(1.0) * _EARTH_RADIUS_M
    mlon = mlat * math.cos(math.radians(clat))

    # Drone's current bearing around the centroid (radians).
    dxm = (drone_lonlat[0] - clon) * mlon
    dym = (drone_lonlat[1] - clat) * mlat
    theta0 = math.atan2(dym, dxm) if math.hypot(dxm, dym) > 1e-6 else 0.0

    dstep = math.radians(float(arc_span_deg)) / n
    direction = 1 if direction >= 0 else -1
    ny, nx = grid.shape

    waypoints: List[Dict[str, float]] = []
    for k in range(1, n + 1):
        th = theta0 + direction * k * dstep
        wlon = clon + radius_m * math.cos(th) / mlon
        wlat = clat + radius_m * math.sin(th) / mlat
        rx = int(np.clip(np.argmin(np.abs(grid.lons - wlon)), 0, nx - 1))
        ry = int(np.clip(np.argmin(np.abs(grid.lats - wlat)), 0, ny - 1))
        waypoints.append({
            "lat": float(wlat),
            "lon": float(wlon),
            "altitude_m": float(envelope["altitude_m"][ry, rx]),
            "speed_ms": float(envelope["velocity_ms"][ry, rx]),
            "risk": float(risk[ry, rx]),
        })
    return waypoints


# Area-coverage sweep tuning (SCATTERED state).
SWEEP_PHASE_STEP = 0.13      # radians the lead target advances each scattered cycle
SWEEP_LEAD = 0.38            # phase lead between successive sweep waypoints
PRECESSION_RATE = 0.05       # Method A: Δφ per tick -> the figure-8 slowly rotates


def plan_area_sweep(
    centroid_lonlat: Tuple[float, float],
    half_x_m: float,
    half_y_m: float,
    grid: Grid,
    risk: np.ndarray,
    envelope: Dict[str, np.ndarray],
    n_waypoints: int,
    phase: float,
    precession: float = 0.0,
    lead: float = SWEEP_LEAD,
) -> List[Dict[str, float]]:
    """
    Area-Coverage planner (SCATTERED state): a moving, ROTATING figure-8 target
    (Method A — phase-shifting / rotating Lissajous).

    We drive a dynamic target along a 1:2 Lissajous (figure-8):
        bx = half_x·sin t ,  by = half_y·sin 2t        (t = phase + k·lead)
    `phase` advances every cycle so the target leads the drone along the curve.
    The whole curve is then rotated in space by the precession angle Δφ:
        dx = bx·cos Δφ − by·sin Δφ
        dy = bx·sin Δφ + by·cos Δφ
    As Δφ = tick·PRECESSION_RATE grows, the figure-8's spatial orientation slowly
    precesses, so the horizontal sweep lines shift vertically over time and the
    previous blind/dead zones are continuously wiped — achieving full coverage of
    the whole pasture meshgrid. Waypoints are clamped to the grid (stay in view).

    Output: list of {lat, lon, altitude_m, speed_ms, risk}.
    """
    n = int(np.clip(n_waypoints, 3, 5))
    cx, cy = float(centroid_lonlat[0]), float(centroid_lonlat[1])
    mlat = math.radians(1.0) * _EARTH_RADIUS_M
    mlon = mlat * math.cos(math.radians(cy))
    ax = max(float(half_x_m), 20.0)
    ay = max(float(half_y_m), 15.0)
    cos_p, sin_p = math.cos(precession), math.sin(precession)  # rotation by Δφ

    ny, nx = grid.shape
    lon_lo, lon_hi = float(grid.lons.min()), float(grid.lons.max())
    lat_lo, lat_hi = float(grid.lats.min()), float(grid.lats.max())

    waypoints: List[Dict[str, float]] = []
    for k in range(1, n + 1):
        t = phase + k * lead
        bx = ax * math.sin(t)             # base figure-8 (wide horizontal)
        by = ay * math.sin(2.0 * t)       # 1:2 Lissajous -> 8 on its side
        dx = bx * cos_p - by * sin_p      # rotate the curve by the precession Δφ
        dy = bx * sin_p + by * cos_p
        wlon = min(max(cx + dx / mlon, lon_lo), lon_hi)
        wlat = min(max(cy + dy / mlat, lat_lo), lat_hi)
        rx = int(np.clip(np.argmin(np.abs(grid.lons - wlon)), 0, nx - 1))
        ry = int(np.clip(np.argmin(np.abs(grid.lats - wlat)), 0, ny - 1))
        waypoints.append({
            "lat": float(wlat),
            "lon": float(wlon),
            "altitude_m": float(envelope["altitude_m"][ry, rx]),
            "speed_ms": float(envelope["velocity_ms"][ry, rx]),
            "risk": float(risk[ry, rx]),
        })
    return waypoints


# ==============================================================================
# 8b. AERODYNAMIC PATH SMOOTHING  (Cubic spline over the RHC/TSP waypoints)
# ==============================================================================

def smooth_path_lonlat(
    points: Sequence[Sequence[float]],
    samples_per_segment: int = 14,
) -> np.ndarray:
    """
    Smooth a discrete waypoint path into an aerodynamically feasible curve.

    The grid-based RHC/TSP planner emits waypoints on cell centres, which strung
    together produce rigid ~90-degree corners no real drone can fly. We fit a
    cubic B-spline (scipy.interpolate.make_interp_spline, degree min(3, n-1))
    parameterised by cumulative chord length and resample it densely.

    The spline is INTERPOLATING -- it passes exactly through every planned
    waypoint -- so each cell's safe altitude/velocity target is still honoured;
    only the connecting trajectory between them is rounded into a smooth curve.

    Parameters
    ----------
    points : sequence of [lon, lat] (e.g. [drone, wp1, wp2, ...]).
    Returns
    -------
    (M, 2) array of [lon, lat] along the smoothed curve (the original points when
    there are fewer than 3 distinct nodes -- nothing to smooth).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 2:
        return pts.copy()

    # Drop consecutive duplicates (a strictly increasing parameter is required).
    keep = np.concatenate([[True], np.any(np.abs(np.diff(pts, axis=0)) > 0, axis=1)])
    pts = pts[keep]
    n = pts.shape[0]
    if n < 3:
        return pts.copy()

    seg = np.sqrt((np.diff(pts, axis=0) ** 2).sum(axis=1))
    t = np.concatenate([[0.0], np.cumsum(seg)])
    if t[-1] <= 0:
        return pts.copy()
    t = t / t[-1]

    k = min(3, n - 1)
    try:
        spline = make_interp_spline(t, pts, k=k)
    except Exception:
        return pts.copy()
    tt = np.linspace(0.0, 1.0, (n - 1) * max(2, samples_per_segment) + 1)
    return np.asarray(spline(tt), dtype=float)


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
    # Adaptive tracking-state context (frontend rendering only; physics unaffected).
    state: str = "compact"                # "compact" | "split" | "scattered"
    is_split: bool = False                # herd split into two guarded/other groups
    is_scattered: bool = False            # wide dispersion -> area-coverage sweep
    herd_guarded_mask: Optional[np.ndarray] = None   # bool over input herd order
    guarded_centroid: Optional[Tuple[float, float]] = None     # (lon, lat)
    other_centroid: Optional[Tuple[float, float]] = None       # (lon, lat) deprioritised


class RiskModel:
    """End-to-end mathematical core. Build once, then call `update()` per cycle."""

    def __init__(
        self,
        polygon: Optional[Sequence[Tuple[float, float]]] = None,
        cell_size_m: float = 25.0,
        hardware_path: Optional[str] = None,
        livestock_gain: float = 2.5,
        orbit_radius_m: float = 22.0,
    ):
        self.hw = (
            HardwareConstraints.from_file(hardware_path)
            if hardware_path else HardwareConstraints()
        )
        self.livestock_gain = livestock_gain
        # Bodyguard orbit ring radius around the herd centroid (Philosophy A Pro).
        self.orbit_radius_m = float(orbit_radius_m)
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
        # Hysteresis lock for split-herd tracking (which sub-group we guard).
        self._locked_cluster_centroid: Optional[Tuple[float, float]] = None
        # Advancing phase for the SCATTERED-state figure-8 area-coverage sweep.
        self._sweep_phase: float = 0.0

    def update(
        self,
        livestock_lonlat: Optional[np.ndarray],
        drone_lonlat: Tuple[float, float],
        n_waypoints: int = 4,
        force_scattered: bool = False,
        tick: int = 0,
    ) -> RiskModelResult:
        """One receding-horizon cycle: recompute live risk and next waypoints.

        Adaptive behaviour switch:
          * COMPACT / SPLIT -> Orbit Mode: a tight KDE around the (locked) cluster
            centroid; the planner flies a geometric bodyguard ring around it.
          * SCATTERED       -> Area-Coverage Mode: a deliberately wide KDE melts the
            animals into one cloud and a moving figure-8 target sweeps the field.

        `force_scattered=True` (the scenario is explicitly Scattered) hard-locks
        the SCATTERED state: SVD split detection and cluster-locking are bypassed,
        eliminating the Area-Coverage <-> Split-lock flip-flop from stochastic
        animal motion.
        """
        (focus_subset, focus_centroid, state,
         guarded_mask, other_centroid) = select_focus_cluster(
            livestock_lonlat, drone_lonlat, self._locked_cluster_centroid,
            force_scattered=force_scattered)
        self._locked_cluster_centroid = focus_centroid
        is_split = state == STATE_SPLIT
        is_scattered = state == STATE_SCATTERED

        herd_proximity = compute_herd_proximity(
            self.grid, focus_subset, scattered=is_scattered)
        # V2.0: couple the static forest-edge field to the live herd proximity.
        risk = combine_risk(
            self.forest_proximity, herd_proximity, self.baseline_gpr,
            self.grid.mask, self.livestock_gain,
        )
        env = compute_flight_envelope(risk, self.hw)

        # Adaptive planner:
        #   SCATTERED        -> Area-Coverage: a moving figure-8 sweep target so the
        #                       drone draws large sweeping curves (never freezes).
        #   COMPACT / SPLIT  -> Orbit Mode: a tight geometric ring on the centroid.
        #   no herd          -> MDP policy rollout (patrol).
        has_herd = focus_subset is not None and len(focus_subset) > 0
        if is_scattered and has_herd:
            pts = np.asarray(focus_subset, dtype=float).reshape(-1, 2)
            cen = pts.mean(axis=0)
            mlat = math.radians(1.0) * _EARTH_RADIUS_M
            mlon = mlat * math.cos(math.radians(float(cen[1])))
            xs = (pts[:, 0] - cen[0]) * mlon
            ys = (pts[:, 1] - cen[1]) * mlat
            half_x = 0.9 * 0.5 * float(xs.max() - xs.min()) if pts.shape[0] > 1 else 0.0
            half_y = 0.9 * 0.5 * float(ys.max() - ys.min()) if pts.shape[0] > 1 else 0.0
            self._sweep_phase += SWEEP_PHASE_STEP
            # Method A: rotate (precess) the figure-8 by Δφ = tick · 0.05 so the
            # sweep orientation slowly turns and wipes out horizontal blind spots.
            precession = float(tick) * PRECESSION_RATE
            waypoints = plan_area_sweep(
                (float(cen[0]), float(cen[1])), half_x, half_y,
                self.grid, risk, env, n_waypoints, self._sweep_phase,
                precession=precession,
            )
        elif has_herd:
            cen = np.asarray(focus_subset, dtype=float).reshape(-1, 2).mean(axis=0)
            waypoints = plan_orbit(
                (float(cen[0]), float(cen[1])), drone_lonlat, self.grid, risk, env,
                n_waypoints=n_waypoints, radius_m=self.orbit_radius_m,
            )
        else:
            waypoints = plan_receding_horizon(
                risk, env, self.grid, drone_lonlat, n_waypoints=n_waypoints,
            )
        return RiskModelResult(
            grid=self.grid,
            baseline_risk=self.baseline,
            livestock_density=herd_proximity,
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
            state=state,
            is_split=is_split,
            is_scattered=is_scattered,
            herd_guarded_mask=guarded_mask,
            guarded_centroid=focus_centroid,
            other_centroid=other_centroid,
        )


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
    print(f"  tracking state = {result.state}")

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
