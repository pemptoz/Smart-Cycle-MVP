"""Full inference chain on one ride: preprocess -> windows -> features -> model -> events."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartcycle.features.build import ride_windows
from smartcycle.features.preprocess import PipelineConfig, preprocess_ride
from smartcycle.geo.events import merge_events


def predict_ride(df: pd.DataFrame, bundle: dict, ride_id: str | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (windows, events).

    windows: t_start, t_end, t_peak, lat, lon, speed, stationary, peak_mag, pred, confidence,
    p_<class>. Stationary windows get pred = "stationary" (not a class, never an event).
    events: see `smartcycle.geo.events.merge_events`.
    """
    cfg = PipelineConfig.from_dict(bundle["config"])
    prep = preprocess_ride(df, cfg, ride_id)
    meta, X = ride_windows(prep, cfg, labelled=False)
    meta = meta.drop(columns=["label"])
    classes = list(bundle["classes"])
    if len(meta) == 0:
        cols = list(meta.columns) + ["pred", "confidence"] + [f"p_{c}" for c in classes]
        return pd.DataFrame(columns=cols), merge_events(pd.DataFrame())
    P = bundle["model"].predict_proba(X[bundle["feature_columns"]])
    pred = np.array(classes, dtype=object)[P.argmax(axis=1)]
    stat = meta["stationary"].to_numpy()
    meta["pred"] = np.where(stat, "stationary", pred)
    meta["confidence"] = np.where(stat, np.nan, P.max(axis=1))
    for j, c in enumerate(classes):
        meta[f"p_{c}"] = P[:, j]
    events = merge_events(meta, gap_s=1.0)
    return meta, events
