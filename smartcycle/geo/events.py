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
        rows += [_event(sub.iloc[g], cls) for g in _groups(sub, gap_s)]
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS).sort_values("t_start",
                                                                  ignore_index=True)


def _groups(sub: pd.DataFrame, gap_s: float) -> list[list[int]]:
    """Positional index groups of consecutive windows (sorted by t_start) whose gap <= gap_s."""
    out: list[list[int]] = []
    group: list[int] = []
    last_end = -np.inf
    for i, (t0, t1) in enumerate(zip(sub["t_start"].to_numpy(), sub["t_end"].to_numpy())):
        if group and t0 - last_end > gap_s:
            out.append(group)
            group = []
        group.append(i)
        last_end = max(last_end, t1) if len(group) > 1 else t1
    if group:
        out.append(group)
    return out


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


# ---------------------------------------------------------------- event-level evaluation
EVENT_UNIT_S = 2.0  # a predicted event longer than this counts as several predictions


def event_units(windows: pd.DataFrame, cls: str, gap_s: float = 1.0,
                unit_s: float = EVENT_UNIT_S) -> list[tuple[float, float, float]]:
    """Predicted `cls` events as scoring units (t_start, t_end, t_peak). Windows are merged as in
    `merge_events`, then each merged event is cut into chunks of at most `unit_s` seconds of
    window starts, each with its own peak (window of max peak_mag, else window centre). A long
    run of predictions therefore counts as many predictions, not one: predicting a class on a
    whole ride can no longer cost a single false alarm."""
    w = windows
    if "stationary" in w:
        w = w[~w["stationary"].astype(bool)]
    sub = w[w["pred"] == cls].sort_values("t_start", kind="stable")
    if sub.empty:
        return []
    t0, t1 = sub["t_start"].to_numpy(float), sub["t_end"].to_numpy(float)
    tp = sub["t_peak"].to_numpy(float) if "t_peak" in sub else (t0 + t1) / 2
    pm = sub["peak_mag"].to_numpy(float) if "peak_mag" in sub else np.full(len(sub), np.nan)
    units = []
    for g in _groups(sub, gap_s):
        chunk: list[int] = []
        for i in g:
            if chunk and t0[i] - t0[chunk[0]] >= unit_s:
                units.append(_unit(chunk, t0, t1, tp, pm))
                chunk = []
            chunk.append(i)
        units.append(_unit(chunk, t0, t1, tp, pm))
    return units


def _unit(idx: list[int], t0, t1, tp, pm) -> tuple[float, float, float]:
    vals = pm[idx]
    k = idx[int(np.nanargmax(vals))] if np.isfinite(vals).any() else idx[len(idx) // 2]
    return float(t0[idx].min()), float(t1[idx].max()), float(tp[k])


def match_events(true_events: list[tuple[float, float, str]],
                 units: list[tuple[float, float, float]], cls: str,
                 tol_s: float = 1.0) -> tuple[int, int, int]:
    """One-to-one matching of labelled `cls` events with predicted units: a pair is possible when
    the unit's PEAK lies within [start - tol, end + tol] of the labelled interval; pairs are taken
    greedily by increasing distance (peak to interval). Returns (matched, n_labelled, n_units):
    recall = matched / n_labelled, precision = matched / n_units, false alarms = n_units -
    matched."""
    truth = [(a, b) for a, b, lbl in true_events if lbl == cls]
    pairs = []
    for i, (a, b) in enumerate(truth):
        for j, (_, _, p) in enumerate(units):
            if a - tol_s <= p <= b + tol_s:
                pairs.append((max(a - p, 0.0, p - b), i, j))
    used_t, used_u = set(), set()
    for _, i, j in sorted(pairs):
        if i not in used_t and j not in used_u:
            used_t.add(i)
            used_u.add(j)
    return len(used_t), len(truth), len(units)


def event_recall(true_events: list[tuple[float, float, str]], windows: pd.DataFrame,
                 cls: str, tol_s: float = 1.0) -> tuple[int, int]:
    """LEGACY (lenient, used before the P90 review; kept only to print the "before" numbers).
    (# labelled `cls` events detected, # labelled `cls` events). A labelled event counts as
    detected if ANY predicted window of that class overlaps [start - tol, end + tol]."""
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
    """LEGACY (lenient, see `event_recall`): merged predicted `cls` events with no labelled
    `cls` event within +-tol_s; one long merged event counts once whatever its length."""
    truth = [(a, b) for a, b, lbl in true_events if lbl == cls]
    n = 0
    for _, e in pred_events[pred_events["label"] == cls].iterrows():
        if not any(e["t_start"] < b + tol_s and e["t_end"] > a - tol_s for a, b in truth):
            n += 1
    return n
