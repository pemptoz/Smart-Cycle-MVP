"""Window predictions -> events (pure functions, reused by the API in P03).

- Pothole / Irregularity: windows of the same class whose time gap is <= `gap_s` (overlapping
  windows have a negative gap) are merged into one point event located at the peak.
- Rough: merged the same way into stretches (start/end).
- Smooth and stationary windows never produce events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

POINT_CLASSES = ("Pothole", "Irregularity")
STRETCH_CLASSES = ("Rough",)
EVENT_COLUMNS = ["kind", "label", "t_start", "t_end", "n_windows", "mean_confidence",
                 "peak_mag", "t_peak", "lat", "lon", "peak_vert", "peak_hmag", "rms_vert"]


def merge_events(windows: pd.DataFrame, gap_s: float = 1.0,
                 classes: tuple[str, ...] = POINT_CLASSES + STRETCH_CLASSES) -> pd.DataFrame:
    """Merge window predictions into events.

    `windows` needs columns t_start, t_end, pred; optional: confidence, peak_mag, t_peak,
    lat, lon, stationary, peak_vert, peak_hmag, rms_vert. Returns one row per event with EVENT_COLUMNS
    (kind = "point" | "stretch"; lat/lon/t_peak of the window with the highest peak_mag;
    peak_vert / peak_hmag = max over the windows, rms_vert = mean over the windows).
    """
    if windows.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    w = windows.copy()
    for c, default in (("confidence", np.nan), ("peak_mag", np.nan), ("lat", np.nan),
                       ("lon", np.nan), ("stationary", False), ("peak_vert", np.nan),
                       ("peak_hmag", np.nan), ("rms_vert", np.nan)):
        if c not in w:
            w[c] = default
    if "t_peak" not in w:
        w["t_peak"] = (w["t_start"] + w["t_end"]) / 2
    w = w[~w["stationary"].astype(bool)].sort_values("t_start", kind="stable")
    rows = []
    for cls in classes:
        sub = w[w["pred"] == cls]
        if sub.empty:
            continue
        group: list[int] = []
        last_end = -np.inf
        for i, (t0, t1) in enumerate(zip(sub["t_start"].to_numpy(), sub["t_end"].to_numpy())):
            if group and t0 - last_end > gap_s:
                rows.append(_event(sub.iloc[group], cls))
                group = []
            group.append(i)
            last_end = max(last_end, t1) if len(group) > 1 else t1
        rows.append(_event(sub.iloc[group], cls))
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS).sort_values("t_start",
                                                                  ignore_index=True)


def _event(g: pd.DataFrame, cls: str) -> dict:
    pm = g["peak_mag"].to_numpy(dtype="float64")
    k = int(np.nanargmax(pm)) if np.isfinite(pm).any() else 0
    peak = g.iloc[k]
    return {"kind": "point" if cls in POINT_CLASSES else "stretch", "label": cls,
            "t_start": float(g["t_start"].min()), "t_end": float(g["t_end"].max()),
            "n_windows": int(len(g)), "mean_confidence": float(g["confidence"].mean()),
            "peak_mag": float(peak["peak_mag"]), "t_peak": float(peak["t_peak"]),
            "lat": float(peak["lat"]), "lon": float(peak["lon"]),
            "peak_vert": _nanmax(g["peak_vert"]), "peak_hmag": _nanmax(g["peak_hmag"]),
            "rms_vert": _nanmean(g["rms_vert"])}


def _nanmax(s: pd.Series) -> float:
    a = s.to_numpy(dtype="float64")
    return float(np.nanmax(a)) if np.isfinite(a).any() else float("nan")


def _nanmean(s: pd.Series) -> float:
    a = s.to_numpy(dtype="float64")
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def event_recall(true_events: list[tuple[float, float, str]], windows: pd.DataFrame,
                 cls: str, tol_s: float = 1.0) -> tuple[int, int]:
    """(# labelled `cls` events detected, # labelled `cls` events). A labelled event counts as
    detected if a predicted window of that class overlaps [start - tol, end + tol]."""
    pred = windows[windows["pred"] == cls]
    t0, t1 = pred["t_start"].to_numpy(), pred["t_end"].to_numpy()
    n = hit = 0
    for a, b, lbl in true_events:
        if lbl != cls:
            continue
        n += 1
        if ((t0 < b + tol_s) & (t1 > a - tol_s)).any():
            hit += 1
    return hit, n


def false_alarms(true_events: list[tuple[float, float, str]], pred_events: pd.DataFrame,
                 cls: str, tol_s: float = 1.0) -> int:
    """Predicted `cls` events with no labelled `cls` event within +-tol_s."""
    truth = [(a, b) for a, b, lbl in true_events if lbl == cls]
    n = 0
    for _, e in pred_events[pred_events["label"] == cls].iterrows():
        if not any(e["t_start"] < b + tol_s and e["t_end"] > a - tol_s for a, b in truth):
            n += 1
    return n
