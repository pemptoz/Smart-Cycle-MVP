"""`inspect` report: sampling, GPS, gravity/orientation check, labels, warnings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from smartcycle.io.labels import LABELS, load_labels, merge_labels
from smartcycle.io.loader import load_ride

GAP_S = 0.5
STATIONARY_MPS = 1.0


def find_labels(path: Path) -> Path | None:
    for cand in (path.parent / "labels.csv", path.parent.parent / "labels.csv"):
        if cand.exists():
            return cand
    return None


def stationary_windows(df: pd.DataFrame, window_s: float = 1.0) -> tuple[int, int]:
    """(# stationary 1 s windows, # windows with a speed estimate). Speed comes from the GPS
    speed column, else from distance between consecutive fixes (A2: mean speed < 1 m/s)."""
    g = df[df["latitude"].notna() & df["longitude"].notna()]
    if len(g) < 2:
        return 0, 0
    spd = g["speed_mps"].to_numpy()
    if np.isnan(spd).all():
        lat, lon = np.radians(g["latitude"].to_numpy()), np.radians(g["longitude"].to_numpy())
        dx = np.diff(lon) * np.cos(lat[:-1]) * 6_371_000
        dy = np.diff(lat) * 6_371_000
        dt = np.diff(g["t_s"].to_numpy())
        spd = np.concatenate([[np.nan], np.hypot(dx, dy) / np.where(dt > 0, dt, np.nan)])
    win = np.arange(0, df["t_s"].iloc[-1], window_s)
    v = np.interp(win + window_s / 2, g["t_s"].to_numpy()[~np.isnan(spd)], spd[~np.isnan(spd)])
    return int((v < STATIONARY_MPS).sum()), int(len(win))


def inspect_ride(path: str | Path, labels_path: str | Path | None = None,
                 ride_id: str | None = None) -> tuple[str, list[str]]:
    """Return (report text, warnings)."""
    path = Path(path)
    df = load_ride(path)
    rid = ride_id or path.stem
    lbl_file = Path(labels_path) if labels_path else find_labels(path)
    if lbl_file is not None:
        df = merge_labels(df, load_labels(lbl_file), rid)
    out, warn = [], []
    p = out.append
    t = df["t_s"].to_numpy()
    dt = np.diff(t)
    med = float(np.median(dt)) if dt.size else float("nan")
    rate = 1.0 / med if med > 0 else float("nan")
    dur = float(t[-1]) if t.size else 0.0
    gaps = dt[dt > GAP_S]

    p(f"File: {path}")
    p(f"Columns mapped: {df.attrs['column_mapping']}")
    p(f"Time column: {df.attrs['time_kind']}")
    p(f"Samples: {len(df)}   Duration: {dur:.1f} s ({dur / 60:.1f} min)")
    p(f"Native sampling rate: {rate:.1f} Hz (median dt {med * 1000:.2f} ms, "
      f"jitter std {np.std(dt) * 1000:.2f} ms, p95 dt {np.percentile(dt, 95) * 1000:.2f} ms)")
    p(f"Gaps > {GAP_S} s: {gaps.size}" + (f" (max {gaps.max():.2f} s)" if gaps.size else ""))

    fix = df[df["latitude"].notna() & df["longitude"].notna()]
    if fix.empty:
        p("GPS: no fix")
        warn.append("No GPS fix in this file")
    else:
        span = fix["t_s"].iloc[-1] - fix["t_s"].iloc[0]
        fix_hz = (len(fix) - 1) / span if span > 0 else float("nan")
        p(f"GPS fixes: {len(fix)} ({fix_hz:.2f} Hz, {len(fix) / len(df) * 100:.1f}% of rows)")
        p(f"GPS lat {fix['latitude'].min():.5f}..{fix['latitude'].max():.5f}, "
          f"lon {fix['longitude'].min():.5f}..{fix['longitude'].max():.5f}")
        acc = fix["gps_accuracy_m"].dropna()
        p("GPS accuracy (m): " + (f"min {acc.min():.1f} / median {acc.median():.1f} / "
                                  f"max {acc.max():.1f}" if len(acc) else "n/a (no column)"))
        sp = fix["speed_mps"].dropna()
        p("Speed (m/s): " + (f"mean {sp.mean():.2f} / median {sp.median():.2f} / "
                             f"max {sp.max():.2f}" if len(sp) else "n/a (no column)"))
        if fix_hz < 0.5:
            warn.append(f"Low GPS rate ({fix_hz:.2f} Hz)")
    n_stat, n_win = stationary_windows(df)
    p(f"Stationary 1 s windows (speed < {STATIONARY_MPS} m/s): {n_stat} / {n_win}")

    a = df[["accel_x", "accel_y", "accel_z"]].to_numpy()
    norm = np.linalg.norm(a, axis=1)
    p(f"|a| (m/s^2): min {norm.min():.2f} / mean {norm.mean():.2f} / max {norm.max():.2f}")
    rest = a[t <= 5.0]
    mean = rest.mean(axis=0)
    rest_norm = float(np.linalg.norm(mean))
    dom = "xyz"[int(np.argmax(np.abs(mean)))]
    p(f"First 5 s: mean a = ({mean[0]:.2f}, {mean[1]:.2f}, {mean[2]:.2f}), "
      f"|mean a| = {rest_norm:.2f}, std |a| = {norm[t <= 5.0].std():.2f}, "
      f"gravity mostly on {dom} ({'+' if mean['xyz'.index(dom)] > 0 else '-'})")

    if df["label"].notna().any():
        counts = df["label"].value_counts()
        p("Labels (samples, ~seconds):" + (f"  [from {lbl_file}]" if lbl_file else ""))
        for lab in LABELS:
            n = int(counts.get(lab, 0))
            p(f"  {lab:<13}{n:>8}  ~{n / rate:7.1f} s")
        p(f"  {'unlabelled':<13}{int(df['label'].isna().sum()):>8}")
    else:
        p("Labels: none")

    if rate < 20:
        warn.append(f"Low sampling rate ({rate:.1f} Hz < 20 Hz): potholes may be missed")
    if not 9.3 <= rest_norm <= 10.3:
        warn.append(f"|a| over the first 5 s is {rest_norm:.2f} (expected ~9.81): "
                    "gravity missing (linear acceleration?) or phone moving at start")
    if gaps.size and gaps.max() > 1.0:
        warn.append(f"Long gap(s) in the accelerometer stream (max {gaps.max():.2f} s)")
    p("Warnings: " + ("none" if not warn else ""))
    out.extend(f"  - {w}" for w in warn)
    return "\n".join(out), warn
