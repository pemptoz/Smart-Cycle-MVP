"""Cycleway network model shared by map matching, events and the API.

A network is a GeoJSON FeatureCollection of LineString *segments* (~50–100 m) with properties
`segment_id, way_id, name, highway, is_cycleway, length_m`. Files written by `fetch-osm` are
already segmented. Any other LineString file (e.g. the SYNTHETIC `data/synthetic/routes.geojson`)
is segmented on the fly, ids `seg_<route_id>_<k>`.

Coordinates are projected with a local equirectangular projection around the network centroid
(metres; accurate enough at city scale, no pyproj).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EARTH_M_PER_DEG = 111_320.0
DEFAULT_NETWORK = "data/osm/cycleways.geojson"
MAX_SEGMENT_M = 100.0


class Projector:
    """Local equirectangular projection (lat/lon degrees <-> metres)."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = EARTH_M_PER_DEG * math.cos(math.radians(lat0))
        self.ky = EARTH_M_PER_DEG

    def to_xy(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky

    def to_latlon(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return y / self.ky + self.lat0, x / self.kx + self.lon0


def polyline_length_m(coords: list[list[float]]) -> float:
    """Length of a [lon, lat] polyline in metres (local projection at its first vertex)."""
    if len(coords) < 2:
        return 0.0
    a = np.asarray(coords, dtype=float)
    p = Projector(a[0, 1], a[0, 0])
    x, y = p.to_xy(a[:, 1], a[:, 0])
    return float(np.hypot(np.diff(x), np.diff(y)).sum())


def segmentise(coords: list[list[float]], max_len: float = MAX_SEGMENT_M
               ) -> list[list[list[float]]]:
    """Split a [lon, lat] polyline into n = ceil(L / max_len) pieces of equal length.

    With max_len = 100 every piece of a way longer than 100 m is 50–100 m long; shorter ways
    stay in one piece. Cut points are interpolated; original vertices are kept inside pieces.
    """
    a = np.asarray(coords, dtype=float)
    if len(a) < 2:
        return []
    p = Projector(a[0, 1], a[0, 0])
    x, y = p.to_xy(a[:, 1], a[:, 0])
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    total = float(cum[-1])
    if total <= 0:
        return []
    n = max(1, math.ceil(total / max_len - 1e-9))
    cuts = np.linspace(0.0, total, n + 1)
    pieces = []
    for k in range(n):
        s0, s1 = cuts[k], cuts[k + 1]
        inner = [i for i in range(len(a)) if s0 < cum[i] < s1]
        pts = [_interp(a, cum, s0)] + [list(map(float, a[i])) for i in inner] \
            + [_interp(a, cum, s1)]
        pieces.append(pts)
    return pieces


def _interp(a: np.ndarray, cum: np.ndarray, s: float) -> list[float]:
    return [float(np.interp(s, cum, a[:, 0])), float(np.interp(s, cum, a[:, 1]))]


@dataclass
class Network:
    ids: list[str]
    props: list[dict]
    coords: list[np.ndarray]  # per segment (k, 2) [lon, lat]
    proj: Projector
    # flattened edges (one row per polyline edge), projected metres
    ex0: np.ndarray
    ey0: np.ndarray
    ex1: np.ndarray
    ey1: np.ndarray
    eseg: np.ndarray  # segment index of each edge
    ecum: np.ndarray  # distance along the segment at the start of each edge
    adj: list[set[int]]
    index: dict[str, int] = field(default_factory=dict)
    synthetic: bool = False
    source: str = ""

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def is_cycleway(self) -> np.ndarray:
        return np.array([bool(p.get("is_cycleway", True)) for p in self.props])

    def connected(self, i: int, j: int) -> bool:
        return j in self.adj[i]

    def segment_geojson(self, i: int, extra: dict | None = None) -> dict:
        return {"type": "Feature",
                "properties": {"segment_id": self.ids[i], **self.props[i], **(extra or {})},
                "geometry": {"type": "LineString",
                             "coordinates": [[round(lo, 7), round(la, 7)]
                                             for lo, la in self.coords[i]]}}


def build_network(features: list[dict], synthetic: bool = False, source: str = "") -> Network:
    """Build a Network from segment features (each with properties.segment_id)."""
    ids, props, coords = [], [], []
    for f in features:
        pr = dict(f.get("properties") or {})
        sid = str(pr.pop("segment_id"))
        c = np.asarray(f["geometry"]["coordinates"], dtype=float)
        if len(c) < 2:
            continue
        pr.setdefault("name", "Unnamed cycleway")
        pr.setdefault("is_cycleway", True)
        pr.setdefault("length_m", round(polyline_length_m(c.tolist()), 1))
        ids.append(sid)
        props.append(pr)
        coords.append(c)
    if not ids:
        raise ValueError("network has no LineString segment")
    allc = np.vstack(coords)
    proj = Projector(float(allc[:, 1].mean()), float(allc[:, 0].mean()))
    ex0, ey0, ex1, ey1, eseg, ecum = [], [], [], [], [], []
    for k, c in enumerate(coords):
        x, y = proj.to_xy(c[:, 1], c[:, 0])
        d = np.hypot(np.diff(x), np.diff(y))
        ex0.append(x[:-1]); ey0.append(y[:-1]); ex1.append(x[1:]); ey1.append(y[1:])
        eseg.append(np.full(len(d), k)); ecum.append(np.concatenate([[0.0], np.cumsum(d)[:-1]]))
    # connectivity: segments sharing an endpoint (OSM shared nodes have identical coordinates)
    ends: dict[tuple[int, int], list[int]] = {}
    for k, c in enumerate(coords):
        for pt in (c[0], c[-1]):
            key = (round(pt[0] * 1e7), round(pt[1] * 1e7))
            ends.setdefault(key, []).append(k)
    adj: list[set[int]] = [set() for _ in ids]
    for segs in ends.values():
        for i in segs:
            adj[i].update(j for j in segs if j != i)
    return Network(ids=ids, props=props, coords=coords, proj=proj,
                   ex0=np.concatenate(ex0), ey0=np.concatenate(ey0),
                   ex1=np.concatenate(ex1), ey1=np.concatenate(ey1),
                   eseg=np.concatenate(eseg).astype(int), ecum=np.concatenate(ecum), adj=adj,
                   index={s: i for i, s in enumerate(ids)}, synthetic=synthetic, source=source)


def network_from_geojson(gj: dict, source: str = "") -> Network:
    """Accept an already segmented file (properties.segment_id) or plain LineStrings."""
    feats = [f for f in gj.get("features", [])
             if (f.get("geometry") or {}).get("type") == "LineString"]
    synthetic = bool(gj.get("synthetic", False))
    if feats and all("segment_id" in (f.get("properties") or {}) for f in feats):
        return build_network(feats, synthetic=synthetic, source=source)
    segs = []
    for n, f in enumerate(feats):
        pr = f.get("properties") or {}
        way = str(pr.get("route_id") or pr.get("way_id") or pr.get("id") or n)
        for k, pts in enumerate(segmentise(f["geometry"]["coordinates"])):
            segs.append({"type": "Feature",
                         "properties": {"segment_id": f"seg_{way}_{k}", "way_id": way,
                                        "name": pr.get("name") or (f"Synthetic route {way}"
                                                                   if synthetic else
                                                                   "Unnamed cycleway"),
                                        "highway": pr.get("highway", "cycleway"),
                                        "is_cycleway": bool(pr.get("is_cycleway", True)),
                                        "length_m": round(polyline_length_m(pts), 1)},
                         "geometry": {"type": "LineString", "coordinates": pts}})
    return build_network(segs, synthetic=synthetic, source=source)


def default_network_path() -> Path:
    return Path(os.environ.get("SMARTCYCLE_NETWORK") or DEFAULT_NETWORK)


def load_network(path: str | Path | None = None) -> Network | None:
    """Load a network file; None if it does not exist (pipeline then runs unmatched)."""
    p = Path(path) if path else default_network_path()
    if not p.exists():
        return None
    return network_from_geojson(json.loads(p.read_text()), source=str(p))
