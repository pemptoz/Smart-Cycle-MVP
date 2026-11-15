"""Hand-made SYNTHETIC route polylines in central Dublin (not real cycleway geometry)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BBOX = {"lat_min": 53.33, "lat_max": 53.36, "lon_min": -6.28, "lon_max": -6.24}
EARTH_M_PER_DEG = 111_320.0

# (lat, lon) vertices, all inside BBOX. Shapes are loosely inspired by the quays, a canal,
# a north-south axis, a diagonal and a small loop, but they are invented.
ROUTES: dict[str, list[tuple[float, float]]] = {
    "syn_quays": [(53.3462, -6.2780), (53.3465, -6.2700), (53.3470, -6.2600),
                  (53.3475, -6.2500), (53.3478, -6.2420)],
    "syn_canal": [(53.3330, -6.2780), (53.3335, -6.2650), (53.3345, -6.2550),
                  (53.3370, -6.2450), (53.3400, -6.2410)],
    "syn_north_south": [(53.3590, -6.2610), (53.3520, -6.2605), (53.3470, -6.2595),
                        (53.3400, -6.2585), (53.3340, -6.2590)],
    "syn_diagonal": [(53.3580, -6.2760), (53.3520, -6.2700), (53.3450, -6.2640),
                     (53.3380, -6.2560)],
    "syn_loop": [(53.3420, -6.2520), (53.3440, -6.2460), (53.3400, -6.2440),
                 (53.3390, -6.2500), (53.3420, -6.2520)],
}


def _xy(coords: np.ndarray, lat0: float) -> np.ndarray:
    """Local equirectangular projection to metres."""
    k = np.cos(np.radians(lat0))
    return np.column_stack([coords[:, 1] * EARTH_M_PER_DEG * k, coords[:, 0] * EARTH_M_PER_DEG])


class Polyline:
    def __init__(self, coords: list[tuple[float, float]]):
        self.ll = np.asarray(coords, dtype=float)
        self.lat0 = float(self.ll[:, 0].mean())
        xy = _xy(self.ll, self.lat0)
        seg = np.hypot(*np.diff(xy, axis=0).T)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.cum[-1])

    def at(self, s: np.ndarray) -> np.ndarray:
        """(lat, lon) at distances s, riding back and forth along the polyline."""
        u = np.mod(np.asarray(s, dtype=float), 2 * self.length)
        u = np.where(u > self.length, 2 * self.length - u, u)
        lat = np.interp(u, self.cum, self.ll[:, 0])
        lon = np.interp(u, self.cum, self.ll[:, 1])
        return np.column_stack([lat, lon])


def routes_geojson() -> dict:
    feats = [{
        "type": "Feature",
        "properties": {"route_id": rid, "synthetic": True},
        "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in pts]},
    } for rid, pts in ROUTES.items()]
    return {"type": "FeatureCollection", "synthetic": True,
            "note": "Hand-made synthetic polylines, NOT real OSM cycleways.", "features": feats}


def write_routes(path: str | Path) -> None:
    Path(path).write_text(json.dumps(routes_geojson(), indent=1))
