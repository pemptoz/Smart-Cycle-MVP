"""`validate-labels`: sanity checks of a labelled dataset before training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from smartcycle.features.build import build_windows, class_counts, prepare_dataset
from smartcycle.features.preprocess import PipelineConfig
from smartcycle.features.windows import EVENT_CLASSES, make_windows
from smartcycle.io.dataset import load_dataset
from smartcycle.io.labels import LABELS

MIN_WINDOWS = 30
MIN_EVENT_RIDES = 3
SEGMENT_CLASSES = ("Smooth", "Rough")


def interval_issues(labels: pd.DataFrame, durations: dict[str, float]) -> list[str]:
    """Overlaps that are suspicious (segment vs segment, or two intervals of the same class)
    and intervals outside the ride. An event inside a Smooth/Rough segment is normal."""
    out = []
    unknown = sorted(set(labels["ride_id"]) - set(durations))
    if unknown:
        out.append(f"WARNING: labels for unknown ride_id(s) {unknown} (not in rides.csv)")
    for rid, g in labels.groupby("ride_id"):
        g = g.sort_values("t_start_s").reset_index(drop=True)
        dur = durations.get(str(rid))
        if dur is not None:
            bad = g[(g["t_start_s"] < 0) | (g["t_end_s"] > dur + 0.5)]
            for _, r in bad.iterrows():
                out.append(f"WARNING: {rid} {r['label']} [{r['t_start_s']:.2f}, "
                           f"{r['t_end_s']:.2f}] s outside the ride (0–{dur:.1f} s)")
        rows = list(g.itertuples(index=False))
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if b.t_start_s >= a.t_end_s:
                    break
                both_seg = a.label in SEGMENT_CLASSES and b.label in SEGMENT_CLASSES
                if both_seg or a.label == b.label:
                    out.append(f"WARNING: {rid} overlapping {a.label} [{a.t_start_s:.2f}, "
                               f"{a.t_end_s:.2f}] and {b.label} [{b.t_start_s:.2f}, "
                               f"{b.t_end_s:.2f}] (resolved by priority "
                               f"Pothole > Irregularity > Rough > Smooth)")
    return out


def gps_table(preps: dict, cfg: PipelineConfig) -> pd.DataFrame:
    """Per ride: duration, GPS fixes kept / dropped on accuracy, speed source, mean moving speed,
    windows and stationary windows (before any label filtering)."""
    rows = {}
    for rid, prep in preps.items():
        meta, _ = make_windows(prep, cfg, labelled=False)
        inf = prep.info
        sp = meta.loc[~meta["stationary"], "speed"] if len(meta) else pd.Series(dtype=float)
        rows[rid] = {"duration_s": round(inf["duration_s"], 1), "fixes": inf["n_fixes"],
                     f"dropped_acc>{cfg.max_gps_accuracy_m}m": inf["n_fixes_dropped_accuracy"],
                     "speed_source": inf["speed_source"],
                     "mean_moving_speed": round(float(sp.mean()), 2) if len(sp) else None,
                     "windows": len(meta), "stationary": int(meta["stationary"].sum())
                     if len(meta) else 0}
    return pd.DataFrame(rows).T


def validate_labels(data_dir: str | Path, cfg: PipelineConfig | None = None) -> tuple[str, int]:
    """Returns (report text, number of warnings). Unknown label names raise ValueError
    (from the loader) with the list of allowed labels."""
    cfg = cfg or PipelineConfig()
    ds = load_dataset(data_dir)
    lines = [f"Dataset {data_dir} — provenance: {ds.provenance.upper()}, {len(ds.rides)} rides"]
    durations = {rid: float(df["t_s"].iloc[-1]) for rid, df in ds.rides.items()}
    warnings: list[str] = []
    if ds.labels is not None:
        lines.append(f"labels.csv: {len(ds.labels)} intervals, label names OK "
                     f"({sorted(set(ds.labels['label']))})")
        warnings += interval_issues(ds.labels, durations)
    else:
        lines.append("No labels.csv: using the label column of the ride files")
    unl = [rid for rid, df in ds.rides.items() if df["label"].isna().all()]
    if unl:
        warnings.append(f"WARNING: rides without any label: {unl}")
    preps = prepare_dataset(ds, cfg)
    lines += ["", "GPS / motion per ride (all windows, labelled or not):", gps_table(preps, cfg)
              .to_string()]
    try:
        wt = build_windows(ds, cfg, preps=preps, drop_stationary=True)
    except ValueError as e:
        warnings.append(f"WARNING: {e}")
        lines += [""] + warnings
        return "\n".join(lines), len(warnings)
    per = (wt.meta.groupby(["ride_id", "label"]).size().unstack(fill_value=0)
           .reindex(columns=list(LABELS), fill_value=0))
    per["stationary_dropped"] = pd.Series(wt.n_stationary)
    lines += ["", f"Windows per ride and class ({cfg.window_s} s, step {cfg.step_s} s, moving "
              f"windows only):", per.to_string(), ""]
    tot = class_counts(wt.meta["label"])
    lines.append("Total: " + ", ".join(f"{k} {v}" for k, v in tot.items())
                 + f" | stationary dropped {sum(wt.n_stationary.values())}")
    for c, n in tot.items():
        if n < MIN_WINDOWS:
            warnings.append(f"WARNING: class {c} has only {n} windows (< {MIN_WINDOWS})")
    for c in EVENT_CLASSES:
        n_r = int((per[c] > 0).sum())
        if n_r < MIN_EVENT_RIDES:
            warnings.append(f"WARNING: {c} events come from only {n_r} ride(s) "
                            f"(< {MIN_EVENT_RIDES}): ride-grouped CV will be unstable")
    if len(ds.rides) < 3:
        warnings.append("WARNING: < 3 rides -> evaluation falls back to blocked time-series CV "
                        "(optimistic)")
    lines += [""] + (warnings or ["No warnings."])
    return "\n".join(lines), len(warnings)
