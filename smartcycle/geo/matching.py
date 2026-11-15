"""Map matching of GPS fixes onto network segments (HMM-lite / Viterbi, numpy only).

- Candidates of a fix: segments within r = max(15 m, 2 x gps_accuracy), at most 8 (nearest
  first). If none: the nearest segment within 40 m. If none either: the fix is unmatched and
  breaks the Viterbi chain (raw GPS kept).
- Emission cost: 0.5 (d / sigma)^2 with sigma = max(8 m, accuracy), minus a mild bonus when the
  segment is a cycleway (`is_cycleway=true`).
- Transition cost: 0 same segment, 0.5 connected segments (shared endpoint), 3 otherwise, plus a
  large penalty when the snapped jump is implausible given speed * dt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartcycle.geo.network import Network

MATCH_CFG = {
    "r_min_m": 15.0, "r_acc_factor": 2.0, "fallback_m": 40.0, "max_candidates": 8,
    "sigma_min_m": 8.0, "default_accuracy_m": 8.0, "cycleway_bonus": 0.3,
    "cost_same": 0.0, "cost_connected": 0.5, "cost_other": 3.0,
    "jump_slack_m": 30.0, "jump_factor": 2.0, "cost_jump": 10.0,
}

MATCH_COLUMNS = ["segment_id", "snapped_lat", "snapped_lon", "distance_along_m",
                 "snap_dist_m", "matched"]


def _project(x: np.ndarray, y: np.ndarray, net: Network):
    """Distances from n points to every edge: (d, px, py, along) arrays of shape (n, E)."""
    dx, dy = net.ex1 - net.ex0, net.ey1 - net.ey0
    l2 = np.maximum(dx * dx + dy * dy, 1e-12)
    t = ((x[:, None] - net.ex0) * dx + (y[:, None] - net.ey0) * dy) / l2
    t = np.clip(t, 0.0, 1.0)
    px, py = net.ex0 + t * dx, net.ey0 + t * dy
    d = np.hypot(x[:, None] - px, y[:, None] - py)
    along = net.ecum + t * np.sqrt(l2)
    return d, px, py, along


def _candidates(x: np.ndarray, y: np.ndarray, radius: np.ndarray, net: Network,
                chunk: int = 256) -> list[list[tuple[int, float, float, float, float]]]:
    """Per point: list of (segment index, distance, snapped x, snapped y, along) nearest first."""
    cfg = MATCH_CFG
    out: list[list[tuple[int, float, float, float, float]]] = []
    for s in range(0, len(x), chunk):
        d, px, py, al = _project(x[s:s + chunk], y[s:s + chunk], net)
        for i in range(d.shape[0]):
            r = radius[s + i]
            near = np.flatnonzero(d[i] <= max(r, cfg["fallback_m"]))
            if near.size == 0:
                out.append([])
                continue
            near = near[np.argsort(d[i, near], kind="stable")]
            seen: dict[int, tuple] = {}
            for e in near:
                k = int(net.eseg[e])
                if k not in seen:
                    seen[k] = (k, float(d[i, e]), float(px[i, e]), float(py[i, e]),
                               float(al[i, e]))
            cands = list(seen.values())
            inside = [c for c in cands if c[1] <= r][:cfg["max_candidates"]]
            out.append(inside if inside else cands[:1])  # fallback: nearest within 40 m
    return out


def snap_point(lat: float, lon: float, net: Network | None, max_m: float | None = None
               ) -> dict:
    """Nearest segment of one point (no HMM). Unmatched if nothing within `max_m` (40 m)."""
    max_m = MATCH_CFG["fallback_m"] if max_m is None else max_m
    res = {"segment_id": None, "snapped_lat": lat, "snapped_lon": lon,
           "distance_along_m": np.nan, "snap_dist_m": np.nan, "matched": False}
    if net is None or not np.isfinite(lat) or not np.isfinite(lon):
        return res
    x, y = net.proj.to_xy(np.array([lat]), np.array([lon]))
    d, px, py, al = _project(x, y, net)
    e = int(d[0].argmin())
    if d[0, e] > max_m:
        return res
    sl, so = net.proj.to_latlon(px[0, e], py[0, e])
    return {"segment_id": net.ids[int(net.eseg[e])], "snapped_lat": float(sl),
            "snapped_lon": float(so), "distance_along_m": float(al[0, e]),
            "snap_dist_m": float(d[0, e]), "matched": True}


def match_track(fixes: pd.DataFrame, net: Network | None) -> pd.DataFrame:
    """Viterbi map matching of GPS fixes.

    `fixes` columns: t_s, latitude, longitude, optional gps_accuracy_m and speed_mps.
    Returns a DataFrame aligned with `fixes` (same index) with MATCH_COLUMNS. Unmatched fixes
    keep their raw position as snapped_lat/lon and segment_id None.
    """
    cfg = MATCH_CFG
    n = len(fixes)
    out = pd.DataFrame({"segment_id": pd.Series([None] * n, dtype=object),
                        "snapped_lat": fixes["latitude"].to_numpy(dtype=float),
                        "snapped_lon": fixes["longitude"].to_numpy(dtype=float),
                        "distance_along_m": np.nan, "snap_dist_m": np.nan,
                        "matched": False}, index=fixes.index)
    if net is None or n == 0:
        return out
    lat = fixes["latitude"].to_numpy(dtype=float)
    lon = fixes["longitude"].to_numpy(dtype=float)
    t = fixes["t_s"].to_numpy(dtype=float)
    acc = (fixes["gps_accuracy_m"].to_numpy(dtype=float) if "gps_accuracy_m" in fixes
           else np.full(n, np.nan))
    acc = np.where(np.isfinite(acc), acc, cfg["default_accuracy_m"])
    spd = (fixes["speed_mps"].to_numpy(dtype=float) if "speed_mps" in fixes
           else np.full(n, np.nan))
    x, y = net.proj.to_xy(lat, lon)
    ok = np.isfinite(x) & np.isfinite(y)
    radius = np.maximum(cfg["r_min_m"], cfg["r_acc_factor"] * acc)
    cands: list[list] = [[] for _ in range(n)]
    idx_ok = np.flatnonzero(ok)
    if idx_ok.size:
        for i, c in zip(idx_ok, _candidates(x[idx_ok], y[idx_ok], radius[idx_ok], net)):
            cands[i] = c
    cyc = net.is_cycleway
    sigma = np.maximum(cfg["sigma_min_m"], acc)

    def emission(i: int) -> np.ndarray:
        return np.array([0.5 * (c[1] / sigma[i]) ** 2 - (cfg["cycleway_bonus"] if cyc[c[0]]
                                                         else 0.0) for c in cands[i]])

    chosen: list[int | None] = [None] * n
    i = 0
    while i < n:  # each run of consecutive fixes with candidates is decoded separately
        if not cands[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and cands[j + 1]:
            j += 1
        cost = emission(i)
        back: list[np.ndarray] = []
        for k in range(i + 1, j + 1):
            prev, cur = cands[k - 1], cands[k]
            dt = max(t[k] - t[k - 1], 0.0)
            raw = float(np.hypot(x[k] - x[k - 1], y[k] - y[k - 1]))
            v = spd[k] if np.isfinite(spd[k]) else 0.0
            max_jump = cfg["jump_factor"] * max(v * dt, raw) + cfg["jump_slack_m"]
            trans = np.empty((len(prev), len(cur)))
            for a, ca in enumerate(prev):
                for b, cb in enumerate(cur):
                    if ca[0] == cb[0]:
                        tc = cfg["cost_same"]
                    elif net.connected(ca[0], cb[0]):
                        tc = cfg["cost_connected"]
                    else:
                        tc = cfg["cost_other"]
                    if np.hypot(cb[2] - ca[2], cb[3] - ca[3]) > max_jump:
                        tc += cfg["cost_jump"]
                    trans[a, b] = tc
            tot = cost[:, None] + trans
            back.append(tot.argmin(axis=0))
            cost = tot.min(axis=0) + emission(k)
        s = int(cost.argmin())
        chosen[j] = s
        for k in range(j, i, -1):
            s = int(back[k - i - 1][s])
            chosen[k - 1] = s
        i = j + 1

    seg_ids, slat, slon, along, sd, m = [], [], [], [], [], []
    for k in range(n):
        if chosen[k] is None:
            seg_ids.append(None); slat.append(lat[k]); slon.append(lon[k])
            along.append(np.nan); sd.append(np.nan); m.append(False)
            continue
        c = cands[k][chosen[k]]
        la, lo = net.proj.to_latlon(c[2], c[3])
        seg_ids.append(net.ids[c[0]]); slat.append(float(la)); slon.append(float(lo))
        along.append(c[4]); sd.append(c[1]); m.append(True)
    out["segment_id"] = pd.Series(seg_ids, index=fixes.index, dtype=object)
    out["snapped_lat"], out["snapped_lon"] = slat, slon
    out["distance_along_m"], out["snap_dist_m"], out["matched"] = along, sd, m
    return out


def match_stats(matched: pd.DataFrame) -> dict:
    """% matched and median / p90 snap distance (metres)."""
    n = len(matched)
    d = matched.loc[matched["matched"], "snap_dist_m"].to_numpy(dtype=float)
    return {"n_fixes": int(n), "pct_matched": round(100.0 * len(d) / n, 1) if n else 0.0,
            "snap_median_m": round(float(np.median(d)), 1) if len(d) else None,
            "snap_p90_m": round(float(np.percentile(d, 90)), 1) if len(d) else None}


def snap_to_segment(lat: float, lon: float, net: Network, segment_id: str) -> dict:
    """Project a point onto one given segment (used to place events on the matched segment)."""
    k = net.index[segment_id]
    x, y = net.proj.to_xy(np.array([lat]), np.array([lon]))
    d, px, py, al = _project(x, y, net)
    edges = np.flatnonzero(net.eseg == k)
    e = int(edges[d[0, edges].argmin()])
    sl, so = net.proj.to_latlon(px[0, e], py[0, e])
    return {"segment_id": segment_id, "snapped_lat": float(sl), "snapped_lon": float(so),
            "distance_along_m": float(al[0, e]), "snap_dist_m": float(d[0, e]), "matched": True}
