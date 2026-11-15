"""Preprocessing of one canonical ride: uniform resampling, GPS on the grid, orientation-robust
decomposition of the acceleration, stationary mask.

The native sampling rate is never assumed: it is estimated from the median time step.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

log = logging.getLogger(__name__)

EARTH_R_M = 6_371_000.0
GAP_S = 0.5  # time steps larger than this are gaps: no window may cross them


@dataclass
class PipelineConfig:
    fs: float = 50.0  # target sampling rate (Hz) after resampling
    window_s: float = 1.0
    step_s: float = 0.5
    stationary_mps: float = 1.0  # windows slower than this are dropped (not a class)
    max_gps_accuracy_m: float | None = 30.0  # fixes worse than this are dropped (None = keep)
    gravity_hz: float = 0.5  # zero-phase low-pass cut-off used to estimate gravity
    feature_set: str = "invariant"  # "invariant" | "invariant+fixed"
    include_speed: bool = True
    event_overlap_s: float = 0.3  # min overlap of a Pothole/Irregularity interval with a window
    min_label_coverage: float = 0.5  # fraction of a window that must be labelled (else dropped)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Prepared:
    """One ride on a uniform time grid."""
    ride_id: str
    fs: float
    native_fs: float
    t: np.ndarray  # seconds from the first sample, uniform at 1/fs
    valid: np.ndarray  # False inside sampling gaps (> GAP_S)
    signals: dict[str, np.ndarray]  # a_vert, a_hmag, a_dev, ax, ay, az
    lat: np.ndarray  # interpolated between kept GPS fixes (NaN outside the fix range)
    lon: np.ndarray
    speed: np.ndarray  # m/s on the grid, NaN when unknown
    has_speed: bool
    label_runs: list[tuple[float, float, str]] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def estimate_fs(t_s: np.ndarray) -> float:
    """Native sampling rate. Gaps (steps > 5x the median step) are excluded, then the MEAN of
    the remaining steps is used: the median alone is biased when timestamps are rounded to
    whole milliseconds (e.g. 80 Hz -> steps of 12/13 ms -> median 12 ms -> 83 Hz)."""
    dt = np.diff(np.asarray(t_s, dtype="float64"))
    dt = dt[dt > 0]
    if dt.size == 0:
        raise ValueError("Cannot estimate the sampling rate: fewer than 2 distinct timestamps")
    ok = dt[dt < 5 * np.median(dt)]
    return float(1.0 / ok.mean())


def lowpass(x: np.ndarray, cutoff_hz: float, fs: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0. Falls back to a centred rolling median
    (3 s) when the signal is too short for filtfilt."""
    sos = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    if len(x) > max(padlen, int(2 * fs / cutoff_hz)):  # at least ~2 periods of the cut-off
        return sosfiltfilt(sos, x, axis=0)
    size = max(3, int(3 * fs) | 1)
    return median_filter(x, size=(size,) + (1,) * (x.ndim - 1), mode="nearest")


def resample_accel(t_s: np.ndarray, acc: np.ndarray, fs: float
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Resample (n, 3) acceleration to a uniform grid at `fs`.

    When the native rate is higher than `fs`, an anti-alias zero-phase low-pass (0.4·fs) is
    applied before linear interpolation. Returns (grid t, acc on grid, valid mask, native fs).
    Grid samples inside a gap (native step > GAP_S) are marked invalid.
    """
    t_s = np.asarray(t_s, dtype="float64")
    native = estimate_fs(t_s)
    x = np.asarray(acc, dtype="float64")
    if native > fs * 1.05:
        x = lowpass(x, 0.4 * fs, native, order=4)
    grid = np.arange(0.0, t_s[-1] - t_s[0] + 1e-9, 1.0 / fs) + t_s[0]
    out = np.column_stack([np.interp(grid, t_s, x[:, k]) for k in range(x.shape[1])])
    valid = np.ones(len(grid), dtype=bool)
    steps = np.diff(t_s)
    for i in np.flatnonzero(steps > GAP_S):
        valid[(grid > t_s[i]) & (grid < t_s[i + 1])] = False
    return grid - t_s[0], out, valid, native


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def prepare_gps(df: pd.DataFrame, max_accuracy_m: float | None = 30.0
                ) -> tuple[pd.DataFrame, dict]:
    """Real GPS fixes only (non-NaN lat/lon), optionally filtered on accuracy.

    Speed comes from `speed_mps` when that column has values, otherwise from haversine
    distance between consecutive fixes, smoothed with a 5-fix rolling median.
    Returns (fixes with columns t_s, latitude, longitude, speed_mps; info dict).
    """
    g = df.loc[df["latitude"].notna() & df["longitude"].notna(),
               ["t_s", "latitude", "longitude", "gps_accuracy_m", "speed_mps"]].copy()
    info = {"n_fixes": int(len(g)), "n_fixes_dropped_accuracy": 0, "speed_source": "none"}
    if max_accuracy_m is not None and len(g):
        bad = g["gps_accuracy_m"] > max_accuracy_m
        info["n_fixes_dropped_accuracy"] = int(bad.sum())
        if bad.any():
            log.info("dropped %d GPS fixes with accuracy > %.0f m", int(bad.sum()), max_accuracy_m)
        g = g[~bad]
    g = g.reset_index(drop=True)
    if len(g) == 0:
        return g, info
    if g["speed_mps"].notna().sum() >= 2:
        info["speed_source"] = "speed_mps column"
    elif len(g) >= 2:
        # Centred difference over +-k fixes (GPS noise is divided by the longer baseline),
        # then a 5-fix rolling median.
        k = min(2, len(g) - 1)
        lat, lon, t = (g[c].to_numpy() for c in ("latitude", "longitude", "t_s"))
        i0 = np.clip(np.arange(len(g)) - k, 0, len(g) - 1)
        i1 = np.clip(np.arange(len(g)) + k, 0, len(g) - 1)
        dt = t[i1] - t[i0]
        d = haversine_m(lat[i0], lon[i0], lat[i1], lon[i1])
        v = np.where(dt > 0, d / np.where(dt > 0, dt, 1.0), np.nan)
        g["speed_mps"] = pd.Series(v).rolling(5, center=True, min_periods=1).median().to_numpy()
        info["speed_source"] = "haversine (smoothed)"
    return g, info


def decompose(acc: np.ndarray, fs: float, gravity_hz: float = 0.5) -> dict[str, np.ndarray]:
    """Orientation-robust channels from (n, 3) acceleration including gravity.

    g = zero-phase low-pass of each axis (~0.5 Hz); g_hat = g / |g|.
    a_vert = acc·g_hat − mean|g|; a_h = acc − (acc·g_hat) g_hat, a_hmag = |a_h|;
    a_dev = |acc| − mean|acc|; ax, ay, az = acc − g (fixed-axes, high-passed view).
    a_vert, a_hmag and a_dev do not change under a constant rotation of the phone.
    """
    g = lowpass(acc, gravity_hz, fs)
    gn = np.linalg.norm(g, axis=1)
    g_hat = g / np.where(gn > 1e-6, gn, 1.0)[:, None]
    proj = np.einsum("ij,ij->i", acc, g_hat)
    a_h = acc - proj[:, None] * g_hat
    mag = np.linalg.norm(acc, axis=1)
    hp = acc - g
    return {"a_vert": proj - gn.mean(), "a_hmag": np.linalg.norm(a_h, axis=1),
            "a_dev": mag - mag.mean(), "ax": hp[:, 0], "ay": hp[:, 1], "az": hp[:, 2]}


def label_runs(t_s: np.ndarray, labels: pd.Series) -> list[tuple[float, float, str]]:
    """Per-sample labels -> contiguous runs (start, end, label); end = time of the next sample
    with a different label (NA samples break runs and are not reported)."""
    lab = labels.astype("string").fillna("").to_numpy()
    t = np.asarray(t_s, dtype="float64")
    if len(t) == 0:
        return []
    change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(t)]])
    dt_last = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    runs = []
    for a, b in zip(starts, ends):
        if lab[a] == "":
            continue
        t_end = t[b] if b < len(t) else t[-1] + dt_last
        runs.append((float(t[a]), float(t_end), str(lab[a])))
    return runs


def preprocess_ride(df: pd.DataFrame, cfg: PipelineConfig, ride_id: str | None = None
                    ) -> Prepared:
    """Full preprocessing of one canonical ride DataFrame (output of `load_ride`)."""
    rid = str(ride_id or df.attrs.get("ride_id") or "ride")
    t_s = df["t_s"].to_numpy(dtype="float64")
    acc = df[["accel_x", "accel_y", "accel_z"]].to_numpy(dtype="float64")
    t, acc_u, valid, native = resample_accel(t_s, acc, cfg.fs)
    signals = decompose(acc_u, cfg.fs, cfg.gravity_hz)

    fixes, info = prepare_gps(df, cfg.max_gps_accuracy_m)
    nan = np.full(len(t), np.nan)
    lat, lon, speed = nan.copy(), nan.copy(), nan.copy()
    if len(fixes) >= 1:
        ft = fixes["t_s"].to_numpy()
        inside = (t >= ft[0]) & (t <= ft[-1]) if len(fixes) > 1 else np.zeros(len(t), bool)
        for arr, col in ((lat, "latitude"), (lon, "longitude")):
            arr[:] = np.interp(t, ft, fixes[col].to_numpy())
        sp = fixes["speed_mps"].to_numpy(dtype="float64")
        ok = np.isfinite(sp)
        if ok.sum() >= 2:
            speed = np.interp(t, ft[ok], sp[ok])
            speed[~inside] = np.nan  # no extrapolation of speed beyond the fixes
    has_speed = bool(np.isfinite(speed).any())
    if not has_speed:
        log.warning("ride %s: no GPS speed -> stationary windows are NOT dropped", rid)
    info.update({"native_fs": native, "n_samples": int(len(df)), "has_speed": has_speed,
                 "duration_s": float(t[-1]) if len(t) else 0.0})
    runs = label_runs(t_s, df["label"]) if "label" in df else []
    return Prepared(ride_id=rid, fs=cfg.fs, native_fs=native, t=t, valid=valid,
                    signals=signals, lat=lat, lon=lon, speed=speed, has_speed=has_speed,
                    label_runs=runs, info=info)
