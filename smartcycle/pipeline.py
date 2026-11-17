"""One ride in -> events, coverage and track out. Shared by the API and `ingest`.

canonical ride DataFrame -> predict_ride (windows + merged events) -> GPS fixes -> map matching
-> events placed and snapped on segments, severity (uncalibrated heuristic) -> per-segment
coverage (distance ridden, rough distance) -> downsampled matched track for replay.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from smartcycle.features.preprocess import PipelineConfig, haversine_m, prepare_gps
from smartcycle.geo.matching import match_stats, match_track, snap_point, snap_to_segment
from smartcycle.geo.network import Network
from smartcycle.geo.severity import peak_value, severity
from smartcycle.models.predict import predict_ride

MAX_EVENT_FIX_GAP_S = 5.0  # an event uses the segment of a matched fix at most this far in time
MAX_COVERAGE_GAP_S = 5.0  # consecutive fixes further apart do not count as covered distance
TRACK_MAX_POINTS = 3000


@dataclass
class RideResult:
    summary: dict
    events: list[dict]
    coverage: list[dict]  # {segment_id, covered_m, rough_m}
    track: dict  # GeoJSON Feature (LineString) with properties t_ms / matched
    windows: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)


def _iso(ms: float) -> str:
    return dt.datetime.fromtimestamp(ms / 1000.0, dt.timezone.utc).isoformat(timespec="seconds")


def _f(x) -> float | None:
    """float or None (JSON/SQLite friendly)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def process_ride(df: pd.DataFrame, bundle: dict, net: Network | None, mount: str = "handlebar",
                 ride_id: str | None = None) -> RideResult:
    """Run the full pipeline on a canonical ride (output of `load_ride` / `canonicalise`)."""
    cfg = PipelineConfig.from_dict(bundle["config"])
    windows, ev = predict_ride(df, bundle, ride_id)
    epoch = str(df.attrs.get("time_kind", "")).startswith("epoch")
    t0_ms = float(df["timestamp_ms"].iloc[0])
    started_at = _iso(t0_ms) if epoch else None

    fixes, gps_info = prepare_gps(df, cfg.max_gps_accuracy_m)
    fixes = fixes.reset_index(drop=True)
    m = match_track(fixes, net) if len(fixes) else pd.DataFrame(columns=["matched"])

    ft = fixes["t_s"].to_numpy(dtype=float) if len(fixes) else np.zeros(0)
    if len(fixes) >= 2:
        step = haversine_m(fixes["latitude"].to_numpy()[:-1], fixes["longitude"].to_numpy()[:-1],
                           fixes["latitude"].to_numpy()[1:], fixes["longitude"].to_numpy()[1:])
        step = np.where(np.diff(ft) <= MAX_COVERAGE_GAP_S * 6, step, 0.0)
        cum = np.concatenate([[0.0], np.cumsum(step)])
    else:
        cum = np.zeros(len(fixes))
    distance_m = float(cum[-1]) if len(cum) else 0.0

    def dist_between(a: float, b: float) -> float:
        if len(ft) < 2:
            return 0.0
        return float(np.interp(b, ft, cum) - np.interp(a, ft, cum))

    matched_idx = np.flatnonzero(m["matched"].to_numpy(dtype=bool)) if len(m) else np.zeros(0, int)
    events: list[dict] = []
    n_unlocated = 0
    for _, e in ev.iterrows():
        lat, lon = _f(e["lat"]), _f(e["lon"])
        if lat is None or lon is None:
            n_unlocated += 1
            continue
        kind = str(e["label"])
        snap = None
        if net is not None and matched_idx.size:
            k = matched_idx[np.abs(ft[matched_idx] - e["t_peak"]).argmin()]
            if abs(ft[k] - e["t_peak"]) <= MAX_EVENT_FIX_GAP_S:
                snap = snap_to_segment(lat, lon, net, m["segment_id"].iloc[k])
        if snap is None:
            snap = snap_point(lat, lon, net)
        seg = snap["segment_id"]
        ed = e.to_dict()
        events.append({
            "type": kind, "kind": str(e["kind"]), "lat": lat, "lon": lon,
            "snapped_lat": snap["snapped_lat"], "snapped_lon": snap["snapped_lon"],
            "matched": bool(snap["matched"]), "segment_id": seg,
            "way_name": net.props[net.index[seg]].get("name") if seg else None,
            "t_start_ms": int(round(e["t_start"] * 1000)), "t_end_ms": int(round(e["t_end"] * 1000)),
            "t_peak_ms": int(round(e["t_peak"] * 1000)),
            "severity": severity(kind, ed), "confidence": _f(e["mean_confidence"]),
            "peak": _f(peak_value(kind, ed)),
            "length_m": round(dist_between(e["t_start"], e["t_end"]), 1) if kind == "Rough"
            else None,
            "observed_at": _iso(t0_ms + e["t_peak"] * 1000) if epoch else None,
        })

    # coverage: distance between consecutive matched fixes, attributed to the first fix's segment
    rough = [(e["t_start_ms"] / 1000, e["t_end_ms"] / 1000) for e in events if e["type"] == "Rough"]
    cov: dict[str, list[float]] = {}
    if len(m) >= 2:
        seg = m["segment_id"].to_numpy(dtype=object)
        mt = m["matched"].to_numpy(dtype=bool)
        sl, so = m["snapped_lat"].to_numpy(), m["snapped_lon"].to_numpy()
        for i in range(len(m) - 1):
            if not (mt[i] and mt[i + 1]) or ft[i + 1] - ft[i] > MAX_COVERAGE_GAP_S:
                continue
            d = float(haversine_m(sl[i], so[i], sl[i + 1], so[i + 1]))
            c = cov.setdefault(seg[i], [0.0, 0.0])
            c[0] += d
            mid = (ft[i] + ft[i + 1]) / 2
            if any(a <= mid <= b for a, b in rough):
                c[1] += d
    coverage = [{"segment_id": s, "covered_m": round(v[0], 1), "rough_m": round(v[1], 1)}
                for s, v in cov.items()]

    # downsampled track (<= 1 point per second, <= TRACK_MAX_POINTS) for replay
    track_coords, track_t, track_m = [], [], []
    if len(fixes):
        keep = np.concatenate([[True], np.floor(ft[1:]) != np.floor(ft[:-1])])
        idx = np.flatnonzero(keep)
        if len(idx) > TRACK_MAX_POINTS:
            idx = idx[np.linspace(0, len(idx) - 1, TRACK_MAX_POINTS).astype(int)]
        for i in idx:
            track_coords.append([round(float(m["snapped_lon"].iloc[i]), 7),
                                 round(float(m["snapped_lat"].iloc[i]), 7)])
            track_t.append(int(round(ft[i] * 1000)))
            track_m.append(bool(m["matched"].iloc[i]))
    track = {"type": "Feature", "properties": {"t_ms": track_t, "matched": track_m},
             "geometry": {"type": "LineString", "coordinates": track_coords}}

    moving = windows[~windows["stationary"].astype(bool)] if len(windows) else windows
    summary = {
        "started_at": started_at, "time_kind": df.attrs.get("time_kind"), "mount": mount,
        "n_samples": int(len(df)),
        "duration_s": round(float(df["t_s"].iloc[-1]), 1) if len(df) else 0.0,
        "distance_m": round(distance_m, 1),
        "n_windows": int(len(windows)), "n_moving_windows": int(len(moving)),
        "windows_per_class": {str(k): int(v) for k, v in
                              (windows["pred"].value_counts().items() if len(windows) else [])},
        "gps": {**{k: v for k, v in gps_info.items()}, **(match_stats(m) if len(m) else {})},
        "network": None if net is None else {"source": net.source, "synthetic": net.synthetic},
        "n_events": len(events), "n_events_without_gps": n_unlocated,
        "model": bundle.get("model_name"), "model_provenance": bundle.get("data_provenance"),
    }
    return RideResult(summary=summary, events=events, coverage=coverage, track=track,
                      windows=windows)
