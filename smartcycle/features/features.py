"""Hand-crafted window features.

Per channel: mean, std, rms, min, max, peak-to-peak, skewness, kurtosis, mean |jerk|,
zero-crossing rate, band energies (0–3, 3–8, 8–15, 15–25 Hz, clipped to Nyquist) and dominant
frequency. Channels: the `invariant` set {a_vert, a_hmag, a_dev} (orientation-robust) and the
`fixed` set {ax, ay, az} (phone axes, meaningful for a handlebar mount only).
Extra: mean GPS speed (+ missing indicator), rms(a_vert)/speed, peak ratio a_hmag/a_vert,
relative time of the peak inside the window.

The RMS-per-speed and peak-amplitude features are inspired by the roughness-index and
impact-index idea of the reference papers (Lee et al. 2025, smartphone-based bicycle pavement
assessment); nothing beyond that idea is claimed — these are not calibrated indices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from smartcycle.features.preprocess import PipelineConfig, Prepared

INVARIANT = ("a_vert", "a_hmag", "a_dev")
FIXED = ("ax", "ay", "az")
FEATURE_SETS: dict[str, tuple[str, ...]] = {"invariant": INVARIANT,
                                            "invariant+fixed": INVARIANT + FIXED}
BANDS = ((0.0, 3.0), (3.0, 8.0), (8.0, 15.0), (15.0, 25.0))
SPEED_FEATURES = ("speed", "speed_missing", "rms_vert_per_speed")


def channel_features(x: np.ndarray, fs: float, name: str) -> dict[str, np.ndarray]:
    """Features of (n_windows, W) samples of one channel."""
    out: dict[str, np.ndarray] = {}
    mean = x.mean(axis=1)
    xc = x - mean[:, None]
    out["mean"], out["std"] = mean, x.std(axis=1)
    out["rms"] = np.sqrt((x ** 2).mean(axis=1))
    out["min"], out["max"] = x.min(axis=1), x.max(axis=1)
    out["ptp"] = out["max"] - out["min"]
    flat = out["std"] < 1e-9
    with np.errstate(all="ignore"):
        out["skew"] = np.where(flat, 0.0, skew(x, axis=1))
        out["kurt"] = np.where(flat, 0.0, kurtosis(x, axis=1))
    out["jerk"] = np.abs(np.diff(x, axis=1)).mean(axis=1) * fs
    out["zcr"] = (np.diff(np.signbit(xc).astype(np.int8), axis=1) != 0).sum(axis=1) \
        / (x.shape[1] / fs)
    spec = np.abs(np.fft.rfft(xc, axis=1)) ** 2 / x.shape[1]
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / fs)
    nyq = fs / 2
    for lo, hi in BANDS:
        hi_c = min(hi, nyq)
        m = (freqs >= lo) & (freqs < hi_c) if hi_c < nyq else (freqs >= lo) & (freqs <= nyq)
        out[f"band_{lo:g}_{hi:g}"] = spec[:, m].sum(axis=1) if lo < nyq and m.any() \
            else np.zeros(len(x))
    out["dom_freq"] = freqs[1:][spec[:, 1:].argmax(axis=1)] if spec.shape[1] > 1 \
        else np.zeros(len(x))
    return {f"{name}__{k}": v for k, v in out.items()}


def window_features(prep: Prepared, idx: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """ALL features (both channel sets + speed features) for the windows of one ride.
    Selection of a feature set happens later with `select_columns`."""
    feats: dict[str, np.ndarray] = {}
    for ch in INVARIANT + FIXED:
        feats.update(channel_features(prep.signals[ch][idx], prep.fs, ch))
    n = len(idx)
    v = prep.signals["a_vert"][idx] if n else np.zeros((0, 1))
    h = prep.signals["a_hmag"][idx] if n else np.zeros((0, 1))
    peak_v = np.abs(v).max(axis=1) if n else np.zeros(0)
    feats["peak_ratio_h_v"] = h.max(axis=1) / np.maximum(peak_v, 1e-3) if n else np.zeros(0)
    feats["t_peak_rel"] = (np.abs(v).argmax(axis=1) / max(v.shape[1] - 1, 1)) if n \
        else np.zeros(0)
    speed = meta["speed"].to_numpy(dtype="float64")
    missing = ~np.isfinite(speed)
    med = np.nanmedian(speed) if (~missing).any() else np.nan
    speed_imp = np.where(missing, med, speed)  # ride median; NaN left for the model imputer
    feats["speed"] = speed_imp
    feats["speed_missing"] = missing.astype(float)
    with np.errstate(all="ignore"):
        feats["rms_vert_per_speed"] = feats["a_vert__rms"] / np.maximum(speed_imp, 1.0)
    return pd.DataFrame(feats, index=meta.index)


def select_columns(columns: list[str], feature_set: str, include_speed: bool) -> list[str]:
    """Columns of the full feature table used for a given feature set / speed option."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set {feature_set!r}; use one of {list(FEATURE_SETS)}")
    chans = FEATURE_SETS[feature_set]
    keep = []
    for c in columns:
        if "__" in c:
            if c.split("__")[0] in chans:
                keep.append(c)
        elif c in SPEED_FEATURES:
            if include_speed:
                keep.append(c)
        else:
            keep.append(c)
    return keep


def config_columns(columns: list[str], cfg: PipelineConfig) -> list[str]:
    return select_columns(columns, cfg.feature_set, cfg.include_speed)
