"""Leakage-free evaluation: ride-grouped cross-validation, pooled out-of-fold metrics,
event-level metrics, ablations.

Leakage rules enforced here:
- folds are grouped by ride (StratifiedGroupKFold): no ride has windows in both train and
  test; asserted at runtime for every fold;
- with < 3 rides, blocked time-series CV inside each ride with a gap of 2*window_s between
  train and test windows (LOUD warning: optimistic estimate);
- imputation, rule thresholds and models are fitted on the training fold only; fixed
  hyperparameters (no tuning at all), fixed seeds; no augmentation is used.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, \
    precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from smartcycle.features.build import WindowTable, build_windows, class_counts
from smartcycle.features.features import config_columns
from smartcycle.features.preprocess import PipelineConfig
from smartcycle.geo.events import POINT_CLASSES, event_recall, false_alarms, merge_events
from smartcycle.io.dataset import Dataset
from smartcycle.io.labels import LABELS
from smartcycle.models.zoo import MODEL_NAMES, make_model

log = logging.getLogger(__name__)
SEED = 0
BLOCKS_FALLBACK = 3


@dataclass
class Splits:
    folds: list[tuple[np.ndarray, np.ndarray]]
    mode: str  # "grouped" | "blocked-time"
    warning: str | None = None


def make_splits(meta: pd.DataFrame, window_s: float, n_splits_max: int = 5) -> Splits:
    """Ride-grouped folds (k = min(5, n_rides)); blocked time-series CV if n_rides < 3."""
    rides = meta["ride_id"].to_numpy()
    y = meta["label"].to_numpy()
    n_rides = len(np.unique(rides))
    if n_rides >= 3:
        k = min(n_splits_max, n_rides)
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
        folds = [(tr, te) for tr, te in cv.split(np.zeros(len(y)), y, groups=rides)]
        sp = Splits(folds=folds, mode="grouped")
    else:
        gap = 2 * window_s
        folds = []
        for b in range(BLOCKS_FALLBACK):
            test = np.zeros(len(meta), bool)
            train = np.zeros(len(meta), bool)
            for rid in np.unique(rides):
                m = rides == rid
                t0 = meta["t_start"].to_numpy()[m]
                t1 = meta["t_end"].to_numpy()[m]
                edges = np.quantile(t0, np.linspace(0, 1, BLOCKS_FALLBACK + 1))
                lo, hi = edges[b], edges[b + 1]
                in_test = (t0 >= lo) & ((t0 < hi) if b < BLOCKS_FALLBACK - 1 else (t0 <= hi))
                if not in_test.any():
                    continue
                a, z = t0[in_test].min(), t1[in_test].max()
                near = (t1 > a - gap) & (t0 < z + gap)
                test[np.flatnonzero(m)[in_test]] = True
                train[np.flatnonzero(m)[~near]] = True
            folds.append((np.flatnonzero(train), np.flatnonzero(test)))
        msg = (f"!!! ONLY {n_rides} RIDE(S): falling back to BLOCKED TIME-SERIES CV inside rides "
               f"(gap {gap:.1f} s). Train and test come from the SAME ride(s): the estimate is "
               f"OPTIMISTIC. Collect >= 3 rides for an honest ride-grouped CV. !!!")
        print(msg, file=sys.stderr)
        sp = Splits(folds=folds, mode="blocked-time", warning=msg)
    assert_no_leakage(meta, sp, window_s)
    return sp


def assert_no_leakage(meta: pd.DataFrame, sp: Splits, window_s: float) -> None:
    rides = meta["ride_id"].to_numpy()
    for i, (tr, te) in enumerate(sp.folds):
        assert len(np.intersect1d(tr, te)) == 0, f"fold {i}: same window in train and test"
        if sp.mode == "grouped":
            common = set(rides[tr]) & set(rides[te])
            assert not common, f"fold {i}: rides {common} in both train and test"
        else:
            gap = 2 * window_s
            t0, t1 = meta["t_start"].to_numpy(), meta["t_end"].to_numpy()
            for rid in np.unique(rides[te]):
                a = te[rides[te] == rid]
                b = tr[rides[tr] == rid]
                if len(b):
                    lo, hi = t0[a].min(), t1[a].max()
                    assert not ((t1[b] > lo - gap) & (t0[b] < hi + gap)).any(), \
                        f"fold {i}: train windows within the gap of the test block"


@dataclass
class ModelResult:
    name: str
    oof_pred: np.ndarray
    oof_conf: np.ndarray
    fold_metrics: list[dict]
    summary: dict = field(default_factory=dict)
    impurity_importance: pd.Series | None = None
    permutation_importance: pd.Series | None = None


def cross_validate(wt: WindowTable, cfg: PipelineConfig, models=MODEL_NAMES,
                   importance: bool = False) -> tuple[dict[str, ModelResult], Splits, list[str]]:
    """Out-of-fold predictions for every model; train-only fitting in each fold."""
    meta = wt.meta
    cols = config_columns(list(wt.X.columns), cfg)
    X, y = wt.X[cols], meta["label"].to_numpy()
    sp = make_splits(meta, cfg.window_s)
    results = {}
    for name in models:
        pred = np.empty(len(y), dtype=object)
        conf = np.full(len(y), np.nan)
        folds, imp, perm = [], [], []
        for k, (tr, te) in enumerate(sp.folds):
            if len(te) == 0 or len(tr) == 0:
                continue
            model = make_model(name).fit(X.iloc[tr], y[tr])
            P = model.predict_proba(X.iloc[te])
            classes = model.classes_
            pred[te] = classes[P.argmax(axis=1)]
            conf[te] = P.max(axis=1)
            folds.append(fold_metrics(y[te], pred[te], k, meta["ride_id"].iloc[te]))
            if importance and name == "rf":
                imp.append(pd.Series(model[-1].feature_importances_, index=cols))
                pi = permutation_importance(model, X.iloc[te], y[te], scoring="f1_macro",
                                            n_repeats=3, random_state=SEED)
                perm.append(pd.Series(pi.importances_mean, index=cols))
        r = ModelResult(name, pred.astype(str), conf, folds)
        if imp:
            r.impurity_importance = pd.concat(imp, axis=1).mean(axis=1)
            r.permutation_importance = pd.concat(perm, axis=1).mean(axis=1)
        r.summary = summarise(r, wt, cfg)
        results[name] = r
    return results, sp, cols


def fold_metrics(y_true, y_pred, k: int, rides: pd.Series) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(LABELS),
                                                 zero_division=0)
    present = s > 0  # classes absent from the test fold: NaN (not 0) in per-fold tables
    nan = np.where(present, 1.0, np.nan)
    return {"fold": k, "n_test": int(len(y_true)), "test_rides": sorted(set(rides)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=[c for c, ok in zip(LABELS, present)
                                                               if ok],
                                       average="macro", zero_division=0)),
            "per_class": {c: {"precision": float(p[i] * nan[i]), "recall": float(r[i] * nan[i]),
                              "f1": float(f[i] * nan[i]), "support": int(s[i])}
                          for i, c in enumerate(LABELS)}}


def _ms(vals) -> dict:
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    return {"mean": float(v.mean()) if v.size else None,
            "std": float(v.std()) if v.size else None, "n_folds": int(v.size)}


def summarise(r: ModelResult, wt: WindowTable, cfg: PipelineConfig) -> dict:
    meta = wt.meta
    y = meta["label"].to_numpy()
    pred = r.oof_pred
    p, rc, f, s = precision_recall_fscore_support(y, pred, labels=list(LABELS), zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(LABELS))
    with np.errstate(all="ignore"):
        cmn = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    per_class = {}
    for i, c in enumerate(LABELS):
        per_class[c] = {
            "pooled": {"precision": float(p[i]), "recall": float(rc[i]), "f1": float(f[i]),
                       "support": int(s[i])},
            **{m: _ms(fm["per_class"][c][m] for fm in r.fold_metrics)
               for m in ("precision", "recall", "f1")}}
    per_ride = {}
    for rid, g in meta.groupby("ride_id"):
        ii = g.index.to_numpy()
        labs = sorted(set(y[ii]))
        per_ride[str(rid)] = float(f1_score(y[ii], pred[ii], labels=labs, average="macro",
                                            zero_division=0))
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=list(LABELS), average="macro",
                                       zero_division=0)),
            "macro_f1_folds": _ms(fm["macro_f1"] for fm in r.fold_metrics),
            "accuracy_folds": _ms(fm["accuracy"] for fm in r.fold_metrics),
            "per_class": per_class,
            "confusion": {"labels": list(LABELS), "counts": cm.tolist(),
                          "row_normalised": np.round(cmn, 4).tolist()},
            "events": event_metrics(meta, pred, r.oof_conf, wt.events_true, cfg),
            "per_ride_macro_f1": per_ride}


def event_metrics(meta: pd.DataFrame, pred: np.ndarray, conf: np.ndarray,
                  events_true: dict, cfg: PipelineConfig, tol_s: float = 1.0) -> dict:
    """Event recall (±tol_s) and false alarms per km for Pothole / Irregularity, computed on
    out-of-fold predictions of the evaluated (moving, labelled) windows."""
    w = meta.assign(pred=pred, confidence=conf)
    hits = {c: [0, 0] for c in POINT_CLASSES}
    fa = {c: 0 for c in POINT_CLASSES}
    dist_m = 0.0
    for rid, g in w.groupby("ride_id"):
        truth = events_true.get(rid, [])
        if len(g):  # only labelled events inside the evaluated span count
            lo, hi = g["t_start"].min(), g["t_end"].max()
            truth = [e for e in truth if e[1] > lo and e[0] < hi]
        ev = merge_events(g, gap_s=1.0, classes=POINT_CLASSES)
        for c in POINT_CLASSES:
            h, n = event_recall(truth, g, c, tol_s)
            hits[c][0] += h
            hits[c][1] += n
            fa[c] += false_alarms(truth, ev, c, tol_s)
        dist_m += float(np.nansum(g["speed"].to_numpy(dtype=float)) * cfg.step_s)
    km = dist_m / 1000.0
    out = {"tolerance_s": tol_s, "distance_km": round(km, 3)}
    for c in POINT_CLASSES:
        out[c] = {"detected": hits[c][0], "n_labelled": hits[c][1],
                  "event_recall": (hits[c][0] / hits[c][1]) if hits[c][1] else None,
                  "false_alarms": fa[c],
                  "false_alarms_per_km": (fa[c] / km) if km > 0 else None}
    return out


def run_ablations(ds: Dataset, cfg: PipelineConfig, base: WindowTable,
                  base_rf: ModelResult | None = None, model: str = "rf") -> pd.DataFrame:
    """RandomForest ablations (grouped CV each): feature set, window length, speed features,
    and macro-F1 per mount subset of the base out-of-fold predictions."""
    rows = []

    def add(name: str, c: PipelineConfig, wt: WindowTable, r: ModelResult | None = None):
        if r is None:
            res, _, cols = cross_validate(wt, c, models=(model,))
            r = res[model]
        s = r.summary
        rows.append({"ablation": name, "feature_set": c.feature_set, "window_s": c.window_s,
                     "speed_features": c.include_speed, "n_windows": len(wt.meta),
                     "macro_f1": s["macro_f1"], "macro_f1_fold_mean": s["macro_f1_folds"]["mean"],
                     "macro_f1_fold_std": s["macro_f1_folds"]["std"],
                     "pothole_event_recall": s["events"]["Pothole"]["event_recall"],
                     "irregularity_event_recall": s["events"]["Irregularity"]["event_recall"]})
        return r

    c_inv = replace(cfg, feature_set="invariant", include_speed=True)
    r_base = add("baseline (invariant, 1 s, speed)", c_inv, base,
                 base_rf if cfg.feature_set == "invariant" and cfg.include_speed else None)
    add("+ fixed axes", replace(c_inv, feature_set="invariant+fixed"), base)
    add("no speed features", replace(c_inv, include_speed=False), base)
    c2 = replace(c_inv, window_s=2.0)
    add("2 s windows", c2, build_windows(ds, c2, preps=base.preps))
    mounts = base.meta["mount"]
    if mounts.nunique() > 1:
        y = base.meta["label"].to_numpy()
        for m in sorted(mounts.unique()):
            ii = np.flatnonzero(mounts.to_numpy() == m)
            rows.append({"ablation": f"mount={m} (subset of baseline OOF)",
                         "feature_set": "invariant", "window_s": cfg.window_s,
                         "speed_features": True, "n_windows": len(ii),
                         "macro_f1": float(f1_score(y[ii], r_base.oof_pred[ii],
                                                    labels=list(LABELS), average="macro",
                                                    zero_division=0)),
                         "macro_f1_fold_mean": None, "macro_f1_fold_std": None,
                         "pothole_event_recall": None, "irregularity_event_recall": None})
    return pd.DataFrame(rows)


def dataset_summary(ds: Dataset, wt: WindowTable) -> dict:
    return {"data_provenance": ds.provenance, "n_rides": int(wt.meta["ride_id"].nunique()),
            "n_windows": int(len(wt.meta)), "n_windows_per_class": class_counts(wt.meta["label"]),
            "n_stationary_windows_dropped": int(sum(wt.n_stationary.values())),
            "rides_per_class": {c: int(wt.meta.loc[wt.meta["label"] == c, "ride_id"].nunique())
                                for c in LABELS},
            "mounts": {str(k): int(v) for k, v in
                       wt.meta.groupby("mount")["ride_id"].nunique().items()}}
