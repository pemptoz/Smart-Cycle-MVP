"""Sliding windows over a prepared ride, and window labels from label intervals.

Labelling rule (P02):
- if a Pothole or Irregularity interval overlaps the window by >= `event_overlap_s` (0.3 s)
  -> that class (priority Pothole > Irregularity);
- else the majority label by duration among Rough / Smooth;
- windows whose labelled coverage is < `min_label_coverage` of the window are dropped
  (labelled data only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartcycle.features.preprocess import PipelineConfig, Prepared

EVENT_CLASSES = ("Pothole", "Irregularity")  # in priority order


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def window_label(t0: float, t1: float, runs: list[tuple[float, float, str]],
                 event_overlap_s: float = 0.3, min_coverage: float = 0.5) -> str | None:
    """Label of the window [t0, t1) from label intervals (see module docstring)."""
    ov: dict[str, float] = {}
    for a, b, lbl in runs:
        if b <= t0 or a >= t1:
            continue
        ov[lbl] = ov.get(lbl, 0.0) + overlap(t0, t1, a, b)
    for cls in EVENT_CLASSES:
        if ov.get(cls, 0.0) >= event_overlap_s - 1e-9:
            return cls
    if sum(ov.values()) < min_coverage * (t1 - t0) - 1e-9:
        return None
    r, s = ov.get("Rough", 0.0), ov.get("Smooth", 0.0)
    if r == 0.0 and s == 0.0:
        return None  # only a short event overlap: ambiguous, dropped
    return "Rough" if r > s else "Smooth"


def make_windows(prep: Prepared, cfg: PipelineConfig, labelled: bool
                 ) -> tuple[pd.DataFrame, np.ndarray]:
    """Cut the prepared ride into windows.

    Returns (meta, idx): meta has one row per window (ride_id, t_start, t_end, t_peak,
    lat, lon (at t_peak), speed, stationary, peak_mag, label) and idx is the (n, W) array of
    grid indices of each window's samples. Windows crossing a sampling gap are skipped.
    In labelled mode, windows without a label are dropped.
    """
    fs = prep.fs
    w = int(round(cfg.window_s * fs))
    step = max(1, int(round(cfg.step_s * fs)))
    n = len(prep.t)
    cols = ["ride_id", "t_start", "t_end", "t_peak", "lat", "lon", "speed", "stationary",
            "peak_mag", "i0", "label"]
    if n < w:
        return pd.DataFrame(columns=cols), np.zeros((0, w), dtype=int)
    starts = np.arange(0, n - w + 1, step)
    idx = starts[:, None] + np.arange(w)[None, :]
    ok = prep.valid[idx].all(axis=1)
    idx = idx[ok]
    starts = starts[ok]
    dyn = np.hypot(prep.signals["a_vert"], prep.signals["a_hmag"])
    seg = dyn[idx]
    k_peak = idx[np.arange(len(idx)), seg.argmax(axis=1)]
    sp = prep.speed[idx]
    cnt = np.isfinite(sp).sum(axis=1)
    speed = np.where(cnt > 0, np.where(np.isfinite(sp), sp, 0.0).sum(axis=1)
                     / np.maximum(cnt, 1), np.nan)
    stationary = np.isfinite(speed) & (speed < cfg.stationary_mps)
    t_start = prep.t[starts]
    meta = pd.DataFrame({
        "ride_id": prep.ride_id, "t_start": t_start, "t_end": t_start + w / fs,
        "t_peak": prep.t[k_peak], "lat": prep.lat[k_peak], "lon": prep.lon[k_peak],
        "speed": speed, "stationary": stationary, "peak_mag": seg.max(axis=1),
        "i0": starts,  # grid index of the first sample (window samples = i0 .. i0 + W - 1)
    })
    if labelled:
        meta["label"] = [window_label(a, b, prep.label_runs, cfg.event_overlap_s,
                                      cfg.min_label_coverage)
                         for a, b in zip(meta["t_start"], meta["t_end"])]
        keep = meta["label"].notna().to_numpy()
        meta, idx = meta[keep].reset_index(drop=True), idx[keep]
    else:
        meta["label"] = None
    return meta, idx
