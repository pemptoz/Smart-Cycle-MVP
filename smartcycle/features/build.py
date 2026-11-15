"""Ride(s) -> window table (meta + all features)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from smartcycle.features.features import window_features
from smartcycle.features.preprocess import PipelineConfig, Prepared, preprocess_ride
from smartcycle.features.windows import EVENT_CLASSES, make_windows
from smartcycle.io.dataset import Dataset

log = logging.getLogger(__name__)


@dataclass
class WindowTable:
    meta: pd.DataFrame  # ride_id, t_start, t_end, t_peak, lat, lon, speed, stationary, ...
    X: pd.DataFrame  # all features (select with features.select_columns)
    events_true: dict[str, list[tuple[float, float, str]]]  # labelled event intervals per ride
    n_stationary: dict[str, int]
    preps: dict[str, Prepared]


def ride_windows(prep: Prepared, cfg: PipelineConfig, labelled: bool
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta, idx = make_windows(prep, cfg, labelled)
    X = window_features(prep, idx, meta)
    return meta, X


def prepare_dataset(ds: Dataset, cfg: PipelineConfig) -> dict[str, Prepared]:
    return {rid: preprocess_ride(df, cfg, rid) for rid, df in ds.rides.items()}


def build_windows(ds: Dataset, cfg: PipelineConfig,
                  preps: dict[str, Prepared] | None = None,
                  drop_stationary: bool = True) -> WindowTable:
    """Labelled windows of every ride of the dataset. Stationary windows are dropped (counted)."""
    preps = preps or prepare_dataset(ds, cfg)
    metas, Xs, n_stat, events = [], [], {}, {}
    for rid, prep in preps.items():
        meta, X = ride_windows(prep, cfg, labelled=True)
        n_stat[rid] = int(meta["stationary"].sum())
        if drop_stationary:
            keep = ~meta["stationary"].to_numpy()
            meta, X = meta[keep], X[keep]
        mount = str(ds.meta.loc[rid, "mount"]) if rid in ds.meta.index else ""
        meta = meta.assign(mount=mount or "unknown")
        metas.append(meta)
        Xs.append(X)
        events[rid] = [r for r in prep.label_runs if r[2] in EVENT_CLASSES]
    meta = pd.concat(metas, ignore_index=True)
    X = pd.concat(Xs, ignore_index=True)
    meta["label"] = meta["label"].astype(str)
    if meta.empty:
        raise ValueError("No labelled, moving windows in the dataset (check labels and speed)")
    return WindowTable(meta=meta, X=X, events_true=events, n_stationary=n_stat, preps=preps)


def class_counts(labels: pd.Series) -> dict[str, int]:
    from smartcycle.io.labels import LABELS
    vc = labels.value_counts()
    return {c: int(vc.get(c, 0)) for c in LABELS}

