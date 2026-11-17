"""SYNTHETIC windows calibrated on real labelled data (provenance `synthetic-calibrated`).

Calibration (from the allowed = TRAINING samples only when used inside a CV fold):
- per class Smooth / Rough: per-axis power spectrum (Welch) of dyn = acc − g, in the gravity
  frame (vertical, lateral, longitudinal), and its per-axis std;
- Pothole / Irregularity: per labelled interval, peak |a_vert|, peak a_hmag, duration and
  dominant frequency of a_vert;
- gravity vectors and speeds of the training windows.
Generation of one window: background = white noise shaped by the class spectrum (events use the
Smooth spectrum); Pothole = damped vertical oscillation with amplitude / frequency / duration
drawn from the real Pothole intervals; Irregularity = either a real-like obstacle response
(vertical + horizontal) or a lateral swerve / longitudinal braking pulse (0.3–0.8 s) — the
plan's definition of Irregularity, ABSENT from the real data — with amplitude from the real
Irregularity a_hmag peaks.
This is a plausible model fitted to ~4 min of real data, not a validated simulator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from smartcycle.features.augment import allowed_masks, window_len
from smartcycle.features.features import features_from_arrays
from smartcycle.features.preprocess import PipelineConfig, channels_from

CALIB_VERSION = "0.1.0"
PROVENANCE = "synthetic-calibrated"
SWERVE_SHARE = 0.5  # share of synthetic Irregularity windows that are swerve / braking pulses


@dataclass
class Calibration:
    fs: float
    psd_f: np.ndarray = field(default_factory=lambda: np.zeros(0))
    psd: dict[str, np.ndarray] = field(default_factory=dict)  # class -> (n_f, 3) frame axes
    std: dict[str, np.ndarray] = field(default_factory=dict)  # class -> (3,)
    events: dict[str, list[dict]] = field(default_factory=dict)  # class -> list of stats
    gravity: list[np.ndarray] = field(default_factory=list)
    speeds: dict[str, np.ndarray] = field(default_factory=dict)

    def summary(self) -> dict:
        return {"fs": self.fs, "std_per_class": {k: v.round(4).tolist()
                                                 for k, v in self.std.items()},
                "n_events": {k: len(v) for k, v in self.events.items()},
                "event_peaks": {k: [round(e["peak_v"], 2) for e in v]
                                for k, v in self.events.items()},
                "mean_speed": {k: round(float(np.mean(v)), 3) for k, v in self.speeds.items()
                               if len(v)}}


def frame(g: np.ndarray) -> np.ndarray:
    """3x3 rows: vertical (gravity), lateral, longitudinal unit vectors in the phone frame."""
    v = g / max(np.linalg.norm(g), 1e-9)
    x = np.array([1.0, 0.0, 0.0])
    if abs(x @ v) > 0.9:
        x = np.array([0.0, 1.0, 0.0])
    lat = x - (x @ v) * v
    lat /= np.linalg.norm(lat)
    return np.vstack([v, lat, np.cross(v, lat)])


def calibrate(wt, tr: np.ndarray, cfg: PipelineConfig) -> Calibration:
    meta = wt.meta
    masks = allowed_masks(wt, tr, cfg)
    fs = next(iter(wt.preps.values())).fs
    cal = Calibration(fs=fs)
    w = window_len(cfg, fs)
    nper = min(64, w)
    segs: dict[str, list[np.ndarray]] = {"Smooth": [], "Rough": []}
    lab = meta["label"].to_numpy()
    for k in tr:
        rid, i0 = meta["ride_id"].iloc[k], int(meta["i0"].iloc[k])
        p = wt.preps[rid]
        idx = np.arange(i0, i0 + w)
        dyn = np.column_stack([p.signals[c][idx] for c in ("ax", "ay", "az")])
        g = np.column_stack([p.signals[c][idx] for c in ("gx", "gy", "gz")]).mean(axis=0)
        cal.gravity.append(g)
        if lab[k] in segs:
            segs[lab[k]].append(dyn @ frame(g).T)  # to (vertical, lateral, longitudinal)
        cal.speeds.setdefault(lab[k], [])
        cal.speeds[lab[k]].append(float(meta["speed"].iloc[k]))
    cal.speeds = {k: np.array(v, float) for k, v in cal.speeds.items()}
    for c, lst in segs.items():
        if not lst:
            continue
        arr = np.stack(lst)  # (n, W, 3)
        f, P = welch(arr, fs=fs, nperseg=nper, axis=1)
        cal.psd_f, cal.psd[c] = f, P.mean(axis=0)
        cal.std[c] = arr.std(axis=1).mean(axis=0)
    for rid, prep in wt.preps.items():
        m = masks.get(rid)
        if m is None:
            continue
        for a, b, lbl in prep.label_runs:
            if lbl not in ("Pothole", "Irregularity"):
                continue
            idx = np.flatnonzero((prep.t >= a) & (prep.t < b))
            if len(idx) < 4 or not m[idx].all():
                continue
            v, h = prep.signals["a_vert"][idx], prep.signals["a_hmag"][idx]
            spec = np.abs(np.fft.rfft(v - v.mean()))
            fr = np.fft.rfftfreq(len(v), 1 / fs)
            cal.events.setdefault(lbl, []).append({
                "peak_v": float(np.abs(v).max()), "peak_h": float(h.max()),
                "dur_s": float(b - a),
                "freq": float(fr[1:][spec[1:].argmax()]) if len(fr) > 1 else 5.0})
    return cal


def _shaped_noise(rng, n: int, cal: Calibration, cls: str) -> np.ndarray:
    """(n, 3) noise in the frame axes with the class spectrum and std."""
    c = cls if cls in cal.psd else next(iter(cal.psd))
    out = np.empty((n, 3))
    fr = np.fft.rfftfreq(n, 1 / cal.fs)
    for j in range(3):
        amp = np.sqrt(np.interp(fr, cal.psd_f, cal.psd[c][:, j]))
        spec = np.fft.rfft(rng.standard_normal(n)) * amp
        x = np.fft.irfft(spec, n)
        out[:, j] = x / max(x.std(), 1e-9) * cal.std[c][j]
    return out


def _pulse(rng, n: int, fs: float, dur_s: float, freq: float, amp: float) -> np.ndarray:
    """Damped oscillation of peak ~amp starting at a random position, inside n samples."""
    L = max(4, min(n, int(dur_s * fs)))
    t = np.arange(L) / fs
    tau = max(dur_s / 3, 0.03)
    p = np.exp(-t / tau) * np.sin(2 * np.pi * max(freq, 2.0) * t + np.pi / 2)
    p = p / max(np.abs(p).max(), 1e-9) * amp
    out = np.zeros(n)
    off = int(rng.integers(0, n - L + 1))
    out[off:off + L] = p
    return out


def generate_window(rng, cal: Calibration, cls: str, w: int, g: np.ndarray
                    ) -> tuple[np.ndarray, str]:
    """(dyn in the phone frame (w, 3), kind)."""
    base = "Rough" if cls == "Rough" else "Smooth"
    fr = _shaped_noise(rng, w, cal, base)
    kind = cls
    if cls in ("Pothole", "Irregularity"):
        evs = cal.events.get(cls) or cal.events.get("Pothole") or [
            {"peak_v": 8.0, "peak_h": 4.0, "dur_s": 0.3, "freq": 8.0}]
        e = evs[int(rng.integers(len(evs)))]
        s = float(rng.uniform(0.85, 1.15))
        if cls == "Irregularity" and rng.random() < SWERVE_SHARE:
            axis = int(rng.integers(1, 3))  # 1 lateral (swerve), 2 longitudinal (braking)
            dur = float(rng.uniform(0.3, 0.8))
            fr[:, axis] += _pulse(rng, w, cal.fs, dur, 1.0 / dur, e["peak_h"] * s)
            kind = "Irregularity-swerve" if axis == 1 else "Irregularity-braking"
        else:
            dur = min(e["dur_s"], 0.8 * w / cal.fs)
            fr[:, 0] += _pulse(rng, w, cal.fs, dur, e["freq"], e["peak_v"] * s)
            if cls == "Irregularity":
                fr[:, 1] += _pulse(rng, w, cal.fs, dur, e["freq"], 0.5 * e["peak_h"] * s)
    return fr @ frame(g), kind


def calibrated_windows(wt, tr: np.ndarray, cfg: PipelineConfig, seed: int = 0,
                       per_class: dict[str, int] | None = None
                       ) -> tuple[pd.DataFrame, np.ndarray]:
    """Calibrate on the TRAINING windows `tr` only, then generate as many synthetic windows per
    class as there are training windows of that class (or `per_class`)."""
    rng = np.random.default_rng(seed)
    cal = calibrate(wt, tr, cfg)
    lab = wt.meta["label"].to_numpy()[tr]
    per_class = per_class or {c: int((lab == c).sum()) for c in set(lab)}
    w = window_len(cfg, cal.fs)
    dyns, gs, speeds, ys = [], [], [], []
    for cls, n in per_class.items():
        for _ in range(n):
            g = cal.gravity[int(rng.integers(len(cal.gravity)))]
            dyn, _ = generate_window(rng, cal, cls, w, g)
            dyns.append(dyn)
            gs.append(np.tile(g, (w, 1)))
            sp = cal.speeds.get(cls, np.array([2.0]))
            speeds.append(float(sp[int(rng.integers(len(sp)))]))
            ys.append(cls)
    if not dyns:
        return wt.X.iloc[:0].copy(), np.array([], dtype=object)
    D, G = np.stack(dyns), np.stack(gs)
    gm = float(np.linalg.norm(G[:, 0], axis=1).mean())
    ch = channels_from(D, G, gm, float(np.linalg.norm(G + D, axis=2).mean()))
    X = features_from_arrays(ch, cal.fs, np.array(speeds))[list(wt.X.columns)]
    return X, np.array(ys, dtype=object)


def load_for_calibration(src: str | Path, labels: str | Path | None = None):
    """`src` = a data dir with rides.csv, or a folder of canonical rides whose PARENT has the
    rides.csv (e.g. data/raw/canonical -> data/raw). `labels` overrides the dataset labels."""
    from smartcycle.io import load_dataset
    from smartcycle.io.labels import load_labels, merge_labels

    root = Path(src)
    if not (root / "rides.csv").exists() and (root.parent / "rides.csv").exists():
        root = root.parent
    ds = load_dataset(root)
    if labels is not None:
        lab = load_labels(labels)
        ds.labels = lab
        ds.rides = {rid: merge_labels(df.drop(columns=["label"], errors="ignore")
                                      .assign(label=pd.array([pd.NA] * len(df), dtype="string")),
                                      lab, rid) for rid, df in ds.rides.items()}
    return ds


def write_calibrated_dataset(data_dir: str | Path, out: str | Path, n_rides: int = 6,
                             minutes: float = 1.0, seed: int = 0,
                             cfg: PipelineConfig | None = None, mount: str = "handlebar",
                             labels: str | Path | None = None) -> dict:
    """Calibrate on ALL labelled moving windows of `data_dir` (the final-model setting) and
    write canonical rides + labels.csv + rides.csv + MANIFEST.json (provenance
    synthetic-calibrated). Rides have no GPS (speed features are taken from the real speeds
    only in the in-memory windows)."""
    from smartcycle.features.build import build_windows
    from smartcycle.io.dataset import select_mount

    cfg = cfg or PipelineConfig()
    wt = build_windows(select_mount(load_for_calibration(data_dir, labels), mount), cfg)
    tr = np.arange(len(wt.meta))
    cal = calibrate(wt, tr, cfg)
    rng = np.random.default_rng(seed)
    out = Path(out)
    (out / "rides").mkdir(parents=True, exist_ok=True)
    w = window_len(cfg, cal.fs)
    classes = [c for c in ("Smooth", "Rough", "Pothole", "Irregularity")
               if c in cal.speeds]
    rides, labels, kinds = [], [], {}
    for r in range(n_rides):
        rid = f"cal_{r:02d}"
        g = cal.gravity[int(rng.integers(len(cal.gravity)))]
        n_win = int(minutes * 60 * cal.fs / w)
        acc, t0 = [], 0.0
        for _ in range(n_win):
            cls = classes[int(rng.integers(len(classes)))]
            dyn, kind = generate_window(rng, cal, cls, w, g)
            kinds[kind] = kinds.get(kind, 0) + 1
            acc.append(dyn + g)
            labels.append({"ride_id": rid, "t_start_s": round(t0, 3),
                           "t_end_s": round(t0 + w / cal.fs, 3), "label": cls})
            t0 += w / cal.fs
        a = np.vstack(acc)
        t_ms = np.round(np.arange(len(a)) / cal.fs * 1000).astype(int)
        pd.DataFrame({"timestamp_ms": t_ms, "accel_x": a[:, 0], "accel_y": a[:, 1],
                      "accel_z": a[:, 2], "latitude": np.nan, "longitude": np.nan,
                      "gps_accuracy_m": np.nan, "speed_mps": np.nan, "label": ""}
                     ).to_csv(out / "rides" / f"{rid}.csv", index=False)
        rides.append({"ride_id": rid, "file": f"{rid}.csv", "mount": mount,
                      "device": "synthetic", "notes": "synthetic-calibrated"})
    pd.DataFrame(rides).to_csv(out / "rides.csv", index=False)
    pd.DataFrame(labels).to_csv(out / "labels.csv", index=False)
    manifest = {"provenance": PROVENANCE, "version": CALIB_VERSION, "seed": seed,
                "calibrated_from": str(data_dir), "mount": mount, "n_rides": n_rides,
                "windows_per_kind": kinds, "calibration": cal.summary(),
                "note": "SYNTHETIC windows shaped on real labelled data. Swerve/braking "
                        "Irregularity pulses are NOT present in the real data. No GPS. Use as "
                        "TRAINING data only, never in test folds."}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    return manifest
