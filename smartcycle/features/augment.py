"""Data augmentation for TRAINING folds only.

Every augmented window is built from samples of TRAINING windows of the current fold:
`allowed_masks` marks those grid samples per ride (test-window samples are explicitly removed
again), and every source range is checked against that mask. Test windows are never augmented.

Transforms of one training window (dyn = acc − g, g = gravity estimate, see preprocess):
- time-warp: a span of round(W·f) samples, f ~ U(0.9, 1.1), centred on the window and fully
  inside allowed samples, resampled to W samples (simulates a different speed; the window's
  speed feature is multiplied by f). If the span is not allowed, f = 1;
- rotation of the whole acceleration (dyn AND g) by a random angle about the window's mean
  gravity direction: invariant channels unchanged, fixed-axes channels rotated;
- amplitude scaling of dyn by U(0.85, 1.15);
- Gaussian noise on dyn, sigma = 0.05 x the per-axis std of the window.
Event paste: a Hann-tapered labelled Pothole / Irregularity interval (dyn) from a TRAINING part
of a ride is added into a TRAINING Smooth / Rough window at a random position; the new window
takes the event label and the background's speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smartcycle.features.features import features_from_arrays
from smartcycle.features.preprocess import PipelineConfig, Prepared, channels_from

AUGMENT_PARAMS = {"n_per_window": 1, "time_warp": [0.9, 1.1], "amplitude": [0.85, 1.15],
                  "noise_frac_of_std": 0.05, "rotation": "uniform 0-360 deg about gravity",
                  "event_paste_per_event_window": 2, "event_classes": ["Pothole", "Irregularity"],
                  "background_classes": ["Smooth", "Rough"]}
EVENT_CLASSES = ("Pothole", "Irregularity")
BACKGROUND = ("Smooth", "Rough")


@dataclass
class Augmented:
    X: pd.DataFrame  # all feature columns (same order as the window table)
    y: np.ndarray
    sources: list[list[tuple[str, np.ndarray]]]  # per window: (ride_id, grid indices used)
    kind: list[str]  # "transform" | "paste"


def window_len(cfg: PipelineConfig, fs: float) -> int:
    return int(round(cfg.window_s * fs))


def allowed_masks(wt, tr: np.ndarray, cfg: PipelineConfig, exclude: np.ndarray | None = None
                  ) -> dict[str, np.ndarray]:
    """Per ride: True on grid samples covered by a TRAINING window (and not by an excluded /
    test window)."""
    meta = wt.meta
    rid_arr, i0_arr = meta["ride_id"].to_numpy(), meta["i0"].to_numpy()
    masks: dict[str, np.ndarray] = {}
    for k in tr:
        prep = wt.preps[rid_arr[k]]
        m = masks.setdefault(rid_arr[k], np.zeros(len(prep.t), bool))
        m[int(i0_arr[k]):int(i0_arr[k]) + window_len(cfg, prep.fs)] = True
    for k in (exclude if exclude is not None else []):
        if rid_arr[k] in masks:
            w = window_len(cfg, wt.preps[rid_arr[k]].fs)
            masks[rid_arr[k]][int(i0_arr[k]):int(i0_arr[k]) + w] = False
    for rid, m in masks.items():
        m &= wt.preps[rid].valid
    return masks


def _dyn_g(prep: Prepared, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = prep.signals
    dyn = np.column_stack([s["ax"][idx], s["ay"][idx], s["az"][idx]])
    g = np.column_stack([s["gx"][idx], s["gy"][idx], s["gz"][idx]])
    return dyn, g


def _ride_means(prep: Prepared) -> tuple[float, float]:
    s = prep.signals
    g = np.column_stack([s["gx"], s["gy"], s["gz"]])
    acc = g + np.column_stack([s["ax"], s["ay"], s["az"]])
    return float(np.linalg.norm(g, axis=1).mean()), float(np.linalg.norm(acc, axis=1).mean())


def _resample(x: np.ndarray, n: int) -> np.ndarray:
    src = np.linspace(0.0, 1.0, len(x))
    dst = np.linspace(0.0, 1.0, n)
    return np.column_stack([np.interp(dst, src, x[:, k]) for k in range(x.shape[1])])


def rotate_about(v: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation of the rows of v (n, 3) about the unit vector `axis`."""
    k = axis / max(np.linalg.norm(axis), 1e-9)
    c, s = np.cos(theta), np.sin(theta)
    return v * c + np.cross(k, v) * s + np.outer(v @ k, k) * (1 - c)


def transform_window(prep: Prepared, i0: int, w: int, allowed: np.ndarray,
                     rng: np.random.Generator, warp: bool = True, rotate: bool = True,
                     scale: bool = True, noise: bool = True
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(dyn, g, source indices, speed factor) of one transformed window."""
    f = float(rng.uniform(*AUGMENT_PARAMS["time_warp"])) if warp else 1.0
    L = int(round(w * f))
    s0 = i0 + (w - L) // 2
    idx = np.arange(s0, s0 + L)
    if L != w and (s0 < 0 or s0 + L > len(allowed) or not allowed[idx].all()):
        f, L, idx = 1.0, w, np.arange(i0, i0 + w)
    dyn, g = _dyn_g(prep, idx)
    if L != w:
        dyn, g = _resample(dyn, w), _resample(g, w)
    if rotate:
        theta = float(rng.uniform(0.0, 2 * np.pi))
        axis = g.mean(axis=0)
        dyn, g = rotate_about(dyn, axis, theta), rotate_about(g, axis, theta)
    if scale:
        dyn = dyn * float(rng.uniform(*AUGMENT_PARAMS["amplitude"]))
    if noise:
        sd = dyn.std(axis=0) * AUGMENT_PARAMS["noise_frac_of_std"]
        dyn = dyn + rng.standard_normal(dyn.shape) * sd
    return dyn, g, idx, w / L if L else 1.0


def _event_sources(wt, masks: dict[str, np.ndarray], min_len_s: float
                   ) -> list[tuple[str, np.ndarray, str]]:
    """Labelled Pothole / Irregularity intervals fully inside allowed samples."""
    out = []
    for rid, prep in wt.preps.items():
        m = masks.get(rid)
        if m is None:
            continue
        for a, b, lbl in prep.label_runs:
            if lbl not in EVENT_CLASSES or b - a < min_len_s:
                continue
            idx = np.flatnonzero((prep.t >= a) & (prep.t < b))
            if len(idx) and m[idx].all():
                out.append((rid, idx, lbl))
    return out


def augment_training(wt, tr: np.ndarray, cfg: PipelineConfig, seed: int = 0,
                     exclude: np.ndarray | None = None, n_per_window: int | None = None,
                     return_sources: bool = False):
    """Augmented windows derived from the TRAINING windows `tr` of `wt` only.
    Returns (X, y) or an `Augmented` when `return_sources`."""
    rng = np.random.default_rng(seed)
    n_per_window = AUGMENT_PARAMS["n_per_window"] if n_per_window is None else n_per_window
    meta = wt.meta
    masks = allowed_masks(wt, tr, cfg, exclude)
    means = {rid: _ride_means(p) for rid, p in wt.preps.items() if rid in masks}
    dyns, gs, speeds, ys, rids, sources, kind = [], [], [], [], [], [], []
    rid_arr, i0_arr = meta["ride_id"].to_numpy(), meta["i0"].to_numpy()
    lab, spd = meta["label"].to_numpy(), meta["speed"].to_numpy(dtype=float)

    for _ in range(n_per_window):
        for k in tr:
            prep = wt.preps[rid_arr[k]]
            w = window_len(cfg, prep.fs)
            dyn, g, idx, fac = transform_window(prep, int(i0_arr[k]), w, masks[rid_arr[k]], rng)
            dyns.append(dyn), gs.append(g), speeds.append(spd[k] / fac), ys.append(lab[k])
            rids.append(rid_arr[k]), sources.append([(rid_arr[k], idx)]), kind.append("transform")

    events = _event_sources(wt, masks, cfg.event_overlap_s)
    bg = [k for k in tr if lab[k] in BACKGROUND]
    n_ev_windows = int(np.isin(lab[tr], EVENT_CLASSES).sum())
    n_paste = AUGMENT_PARAMS["event_paste_per_event_window"] * n_ev_windows
    if events and bg:
        for _ in range(n_paste):
            e_rid, e_idx, e_lbl = events[int(rng.integers(len(events)))]
            k = bg[int(rng.integers(len(bg)))]
            prep_b, prep_e = wt.preps[rid_arr[k]], wt.preps[e_rid]
            w = window_len(cfg, prep_b.fs)
            if len(e_idx) > int(0.8 * w):  # long event: keep 0.8 W around its peak
                dv = np.hypot(prep_e.signals["a_vert"][e_idx], prep_e.signals["a_hmag"][e_idx])
                L = int(0.8 * w)
                c = int(np.clip(dv.argmax() - L // 2, 0, len(e_idx) - L))
                e_idx = e_idx[c:c + L]
            if len(e_idx) < cfg.event_overlap_s * prep_b.fs:
                continue
            b_idx = np.arange(int(i0_arr[k]), int(i0_arr[k]) + w)
            dyn_b, g_b = _dyn_g(prep_b, b_idx)
            dyn_e, _ = _dyn_g(prep_e, e_idx)
            dyn_e = (dyn_e - dyn_e.mean(axis=0)) * np.hanning(len(e_idx))[:, None] \
                * float(rng.uniform(*AUGMENT_PARAMS["amplitude"]))
            off = int(rng.integers(0, w - len(e_idx) + 1))
            dyn = dyn_b.copy()
            dyn[off:off + len(e_idx)] += dyn_e
            dyns.append(dyn), gs.append(g_b), speeds.append(spd[k]), ys.append(e_lbl)
            rids.append(rid_arr[k])
            sources.append([(rid_arr[k], b_idx), (e_rid, e_idx)])
            kind.append("paste")

    cols = list(wt.X.columns)
    if not dyns:
        X = wt.X.iloc[:0].copy()
        y = np.array([], dtype=object)
        return Augmented(X, y, [], []) if return_sources else (X, y)
    # one feature pass per ride (ride means of |g| and |acc| are needed by the channels)
    order, Xs, y_out = [], [], []
    for rid in dict.fromkeys(rids):
        ii = [i for i, r in enumerate(rids) if r == rid]
        D, G = np.stack([dyns[i] for i in ii]), np.stack([gs[i] for i in ii])
        ch = channels_from(D, G, *means[rid])
        Xs.append(features_from_arrays(ch, wt.preps[rid].fs, np.array([speeds[i] for i in ii])))
        order += ii
    X = pd.concat(Xs, ignore_index=True)[cols]
    y = np.array(ys, dtype=object)[order]
    X = X.iloc[np.argsort(order)].reset_index(drop=True)
    y = y[np.argsort(order)]
    if return_sources:
        return Augmented(X, y, sources, kind)
    return X, y
