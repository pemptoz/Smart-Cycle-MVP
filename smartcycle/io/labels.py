"""Ground-truth labels: the 4 plan classes, labels.csv intervals, merge into a ride."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from smartcycle.io.csvsniff import read_csv_any

log = logging.getLogger(__name__)

LABELS: tuple[str, ...] = ("Smooth", "Rough", "Pothole", "Irregularity")
# Higher index = higher priority when intervals overlap.
PRIORITY: tuple[str, ...] = ("Smooth", "Rough", "Irregularity", "Pothole")
LABEL_COLUMNS = ["ride_id", "t_start_s", "t_end_s", "label"]

_CANON = {lbl.lower(): lbl for lbl in LABELS}


def normalise_labels(values: pd.Series) -> pd.Series:
    """Map label strings to the canonical spelling (case-insensitive); empty -> NA.
    Raises ValueError on anything outside the 4 plan labels."""
    s = values.astype("string").str.strip()
    s = s.mask(s == "")
    out = s.str.lower().map(_CANON, na_action="ignore")
    bad = s.notna() & out.isna()
    if bad.any():
        raise ValueError(f"Unknown label(s) {sorted(set(s[bad]))}; allowed: {list(LABELS)}")
    return out.astype("string")


def load_labels(path: str | Path) -> pd.DataFrame:
    """Read labels.csv (`ride_id,t_start_s,t_end_s,label`, seconds from the ride's first sample)."""
    df = read_csv_any(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}; found {list(df.columns)}")
    df = df[LABEL_COLUMNS].copy()
    df["ride_id"] = df["ride_id"].astype("string").str.strip()
    df["t_start_s"] = pd.to_numeric(df["t_start_s"]).astype("float64")
    df["t_end_s"] = pd.to_numeric(df["t_end_s"]).astype("float64")
    df["label"] = normalise_labels(df["label"])
    if df["label"].isna().any():
        raise ValueError(f"{path}: empty label on some rows")
    bad = df["t_end_s"] <= df["t_start_s"]
    if bad.any():
        raise ValueError(f"{path}: t_end_s <= t_start_s on rows {list(df.index[bad])}")
    return df


def merge_labels(ride: pd.DataFrame, labels: pd.DataFrame, ride_id: str) -> pd.DataFrame:
    """Return a copy of `ride` whose `label` column comes from the intervals of `ride_id`.

    Samples with t_start_s <= t_s < t_end_s get the interval label; on overlaps the
    priority is Pothole > Irregularity > Rough > Smooth. Uncovered samples are NA.
    If the ride has no interval in `labels`, it is returned unchanged.
    """
    sub = labels[labels["ride_id"] == str(ride_id)]
    if sub.empty:
        return ride
    out = ride.copy()
    if out["label"].notna().any():
        log.warning("ride %s: labels.csv overrides the label column of the file", ride_id)
    t = out["t_s"].to_numpy()
    rank = np.full(len(out), -1)
    for _, row in sub.iterrows():
        r = PRIORITY.index(row["label"])
        m = (t >= row["t_start_s"]) & (t < row["t_end_s"]) & (rank < r)
        rank[m] = r
    names = np.array(PRIORITY, dtype=object)
    lab = np.where(rank >= 0, names[np.clip(rank, 0, None)], None)
    out["label"] = pd.array(lab, dtype="string")
    return out
