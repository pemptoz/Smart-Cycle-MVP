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
from smartcycle.io.dataset import Dataset, select_mount
from smartcycle.io.labels import LABELS
from smartcycle.models.zoo import MODEL_NAMES, make_model

log = logging.getLogger(__name__)
SEED = 0
BLOCKS_FALLBACK = 3


BLOCKED_HEADER = ("blocked CV on a tiny single-session dataset: optimistic, no ride-level "
                  "generalisation claim")
_WARNED: dict[str, bool] = {}  # print the blocked-CV header once per process
SPLIT_SEEDS = range(50)  # tried in order; the choice depends on labels only, never on scores


@dataclass
class CVConfig:
    """Evaluation protocol. `grouped` = ride-grouped k-fold, `logo` = leave-one-ride-out,
    `blocked` = contiguous time blocks inside rides, grouped k-fold over blocks, with a purge
    of training windows near test windows of the same ride."""
    cv: str = "grouped"  # "grouped" | "blocked" | "logo"
    block_s: float = 15.0
    purge_s: float = 2.0
    n_splits_max: int = 5
    augment: bool = False
    extra_train: str | None = None  # None | "calibrated"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class Splits:
    folds: list[tuple[np.ndarray, np.ndarray]]
    mode: str  # "grouped" | "blocked-time" | "blocked" | "logo"
    warning: str | None = None
    protocol: dict = field(default_factory=dict)
    purge_s: float = 0.0


def rides_per_class(meta: pd.DataFrame) -> dict[str, list[str]]:
    return {c: sorted(meta.loc[meta["label"] == c, "ride_id"].astype(str).unique())
            for c in LABELS if (meta["label"] == c).any()}


def _require_two_rides_per_class(meta: pd.DataFrame, cv: str) -> None:
    rpc = rides_per_class(meta)
    thin = {c: r for c, r in rpc.items() if len(r) < 2}
    if thin:
        detail = "; ".join(f"{c} only in {r}" for c, r in thin.items())
        raise ValueError(
            f"--cv {cv} refused: some classes appear in fewer than 2 rides ({detail}). A test "
            f"ride would contain a class never seen in training. Use `--cv blocked` "
            f"(within-ride time blocks; optimistic, no ride-level generalisation claim).")


def block_ids(meta: pd.DataFrame, block_s: float) -> np.ndarray:
    """Block = ride + floor(t_start / block_s) (t from the first sample of the ride, so the
    grid is the same for every window length)."""
    b = np.floor(meta["t_start"].to_numpy(dtype=float) / block_s).astype(int)
    return np.array([f"{r}#{k}" for r, k in zip(meta["ride_id"].astype(str), b)])


def purge(meta: pd.DataFrame, tr: np.ndarray, te: np.ndarray, purge_s: float) -> np.ndarray:
    """Training indices minus every window within `purge_s` of a test window of the same ride."""
    rides = meta["ride_id"].to_numpy()
    t0, t1 = meta["t_start"].to_numpy(dtype=float), meta["t_end"].to_numpy(dtype=float)
    keep = np.ones(len(tr), bool)
    for rid in np.unique(rides[te]):
        a = te[rides[te] == rid]
        in_r = rides[tr] == rid
        b = tr[in_r]
        if len(b) == 0:
            continue
        near = ((t1[b][:, None] > t0[a][None, :] - purge_s)
                & (t0[b][:, None] < t1[a][None, :] + purge_s)).any(axis=1)
        keep[np.flatnonzero(in_r)[near]] = False
    return tr[keep]


def _blocked_splits(meta: pd.DataFrame, block_s: float, purge_s: float,
                    n_splits_max: int) -> Splits:
    y = meta["label"].to_numpy()
    groups = block_ids(meta, block_s)
    classes = sorted(set(y))
    blocks_per_class = {c: int(len(set(groups[y == c]))) for c in classes}
    n_blocks = int(len(set(groups)))
    k = min(n_splits_max, n_blocks, min(blocks_per_class.values()))
    if k < 2:
        raise ValueError(f"--cv blocked impossible: blocks per class {blocks_per_class} "
                         f"(block_s={block_s}); at least 2 blocks per class are needed. "
                         f"Try a smaller --block-s.")
    chosen, folds = None, None
    for seed in SPLIT_SEEDS:
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        cand = list(cv.split(np.zeros(len(y)), y, groups=groups))
        if all(set(y[te]) == set(classes) for _, te in cand):
            chosen, folds = seed, cand
            break
    warning = None
    if folds is None:
        chosen = 0
        folds = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0)
                     .split(np.zeros(len(y)), y, groups=groups))
        missing = {i: sorted(set(classes) - set(y[te])) for i, (_, te) in enumerate(folds)}
        warning = (f"!!! blocked CV: no split (seeds 0-{len(SPLIT_SEEDS) - 1}) has every class in "
                   f"every test fold; missing classes per fold: {missing} !!!")
        print(warning, file=sys.stderr)
    out, fold_info = [], []
    for i, (tr, te) in enumerate(folds):
        tr2 = purge(meta, tr, te, purge_s)
        out.append((tr2, te))
        fold_info.append({"fold": i, "n_train": int(len(tr2)), "n_purged": int(len(tr) - len(tr2)),
                          "n_test": int(len(te)), "test_blocks": sorted(set(groups[te])),
                          "test_classes": sorted(set(y[te])),
                          "missing_test_classes": sorted(set(classes) - set(y[te]))})
    if not _WARNED.get("blocked"):
        print(f"!!! {BLOCKED_HEADER} !!!", file=sys.stderr)
        _WARNED["blocked"] = True
    protocol = {"mode": "blocked", "block_s": block_s, "purge_s": purge_s, "n_splits": k,
                "seed": chosen, "n_blocks": n_blocks, "blocks_per_class": blocks_per_class,
                "header": BLOCKED_HEADER, "folds": fold_info}
    return Splits(folds=out, mode="blocked", warning=warning, protocol=protocol,
                  purge_s=purge_s)


def make_splits(meta: pd.DataFrame, window_s: float, n_splits_max: int = 5,
                cv: str = "grouped", block_s: float = 15.0, purge_s: float = 2.0) -> Splits:
    """Folds for one evaluation protocol (see CVConfig). Leakage is asserted on every fold."""
    rides = meta["ride_id"].to_numpy()
    y = meta["label"].to_numpy()
    n_rides = len(np.unique(rides))
    if cv == "blocked":
        sp = _blocked_splits(meta, block_s, purge_s, n_splits_max)
    elif cv == "logo":
        _require_two_rides_per_class(meta, cv)
        folds = [(np.flatnonzero(rides != r), np.flatnonzero(rides == r))
                 for r in np.unique(rides)]
        sp = Splits(folds=folds, mode="logo",
                    protocol={"mode": "logo", "n_splits": len(folds)})
    elif cv != "grouped":
        raise ValueError(f"unknown --cv {cv!r}; use grouped, blocked or logo")
    elif n_rides >= 3:
        _require_two_rides_per_class(meta, cv)
        k = min(n_splits_max, n_rides)
        cvs = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
        folds = [(tr, te) for tr, te in cvs.split(np.zeros(len(y)), y, groups=rides)]
        sp = Splits(folds=folds, mode="grouped",
                    protocol={"mode": "grouped", "n_splits": k, "seed": SEED})
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
        sp = Splits(folds=folds, mode="blocked-time", warning=msg,
                    protocol={"mode": "blocked-time", "n_splits": BLOCKS_FALLBACK, "gap_s": gap})
    assert_no_leakage(meta, sp, window_s)
    return sp


def assert_no_leakage(meta: pd.DataFrame, sp: Splits, window_s: float) -> None:
    rides = meta["ride_id"].to_numpy()
    t0, t1 = meta["t_start"].to_numpy(dtype=float), meta["t_end"].to_numpy(dtype=float)
    for i, (tr, te) in enumerate(sp.folds):
        assert len(np.intersect1d(tr, te)) == 0, f"fold {i}: same window in train and test"
        if sp.mode in ("grouped", "logo"):
            common = set(rides[tr]) & set(rides[te])
            assert not common, f"fold {i}: rides {common} in both train and test"
        elif sp.mode == "blocked":
            for rid in np.unique(rides[te]):
                a = te[rides[te] == rid]
                b = tr[rides[tr] == rid]
                if len(b):
                    near = ((t1[b][:, None] > t0[a][None, :] - sp.purge_s)
                            & (t0[b][:, None] < t1[a][None, :] + sp.purge_s))
                    assert not near.any(), f"fold {i}: train window within the purge of a test one"
        else:
            gap = 2 * window_s
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


def extra_training(wt: WindowTable, cfg: PipelineConfig, cvc: CVConfig, tr: np.ndarray,
                   te: np.ndarray, fold: int) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Augmented / calibrated-synthetic windows for ONE fold, derived from its TRAINING windows
    only (test windows are never touched). Returns (X_extra with ALL feature columns, y_extra,
    n_synthetic)."""
    Xs, ys, n_syn = [], [], 0
    if cvc.augment:
        from smartcycle.features.augment import augment_training
        Xa, ya = augment_training(wt, tr, cfg, seed=SEED + fold, exclude=te)
        Xs.append(Xa)
        ys.append(ya)
    if cvc.extra_train == "calibrated":
        from smartcycle.synth.calibrate import calibrated_windows
        Xc, yc = calibrated_windows(wt, tr, cfg, seed=SEED + fold)
        Xs.append(Xc)
        ys.append(yc)
        n_syn = len(yc)
    if not Xs:
        return wt.X.iloc[:0], np.array([], dtype=object), 0
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), n_syn


def cross_validate(wt: WindowTable, cfg: PipelineConfig, models=MODEL_NAMES,
                   importance: bool = False, cvc: CVConfig | None = None,
                   sp: Splits | None = None, columns: list[str] | None = None
                   ) -> tuple[dict[str, ModelResult], Splits, list[str]]:
    """Out-of-fold predictions for every model; train-only fitting in each fold. Augmentation /
    synthetic windows (if enabled) are added to the TRAINING part of each fold only: test folds
    contain only the original windows."""
    cvc = cvc or CVConfig()
    meta = wt.meta
    cols = columns or config_columns(list(wt.X.columns), cfg)
    X, y = wt.X[cols], meta["label"].to_numpy()
    if sp is None:
        sp = make_splits(meta, cfg.window_s, cvc.n_splits_max, cvc.cv, cvc.block_s, cvc.purge_s)
    pred = {n: np.empty(len(y), dtype=object) for n in models}
    conf = {n: np.full(len(y), np.nan) for n in models}
    folds = {n: [] for n in models}
    imp, perm = [], []
    n_extra = []
    for k, (tr, te) in enumerate(sp.folds):
        if len(te) == 0 or len(tr) == 0:
            continue
        Xe, ye, _ = extra_training(wt, cfg, cvc, tr, te, k)
        X_tr = pd.concat([X.iloc[tr], Xe[cols]], ignore_index=True) if len(ye) else X.iloc[tr]
        y_tr = np.concatenate([y[tr], ye]) if len(ye) else y[tr]
        n_extra.append(int(len(ye)))
        for name in models:
            model = make_model(name).fit(X_tr, y_tr)
            P = model.predict_proba(X.iloc[te])
            classes = model.classes_
            pred[name][te] = classes[P.argmax(axis=1)]
            conf[name][te] = P.max(axis=1)
            folds[name].append(fold_metrics(y[te], pred[name][te], k, meta["ride_id"].iloc[te]))
            if importance and name == "rf":
                imp.append(pd.Series(model[-1].feature_importances_, index=cols))
                pi = permutation_importance(model, X.iloc[te], y[te], scoring="f1_macro",
                                            n_repeats=3, random_state=SEED)
                perm.append(pd.Series(pi.importances_mean, index=cols))
    sp.protocol.update({"augment": cvc.augment, "extra_train": cvc.extra_train})
    if cvc.augment or cvc.extra_train:
        sp.protocol["n_extra_train_windows_per_fold"] = n_extra
    results = {}
    for name in models:
        r = ModelResult(name, pred[name].astype(str), conf[name], folds[name])
        if name == "rf" and imp:
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
    per_ride, per_ride_detail = {}, {}
    for rid, g in meta.groupby("ride_id"):
        ii = g.index.to_numpy()
        labs = sorted(set(y[ii]))
        per_ride[str(rid)] = float(f1_score(y[ii], pred[ii], labels=labs, average="macro",
                                            zero_division=0))
        _, rr, _, ss = precision_recall_fscore_support(y[ii], pred[ii], labels=list(LABELS),
                                                       zero_division=0)
        per_ride_detail[str(rid)] = {
            "macro_f1": per_ride[str(rid)],
            "n": {c: int(ss[i]) for i, c in enumerate(LABELS)},
            "recall": {c: (float(rr[i]) if ss[i] else None) for i, c in enumerate(LABELS)},
            "predicted": {c: int((pred[ii] == c).sum()) for c in LABELS}}
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=list(LABELS), average="macro",
                                       zero_division=0)),
            "macro_f1_folds": _ms(fm["macro_f1"] for fm in r.fold_metrics),
            "accuracy_folds": _ms(fm["accuracy"] for fm in r.fold_metrics),
            "per_class": per_class,
            "confusion": {"labels": list(LABELS), "counts": cm.tolist(),
                          "row_normalised": np.round(cmn, 4).tolist()},
            "events": event_metrics(meta, pred, r.oof_conf, wt.events_true, cfg),
            "per_ride_macro_f1": per_ride, "per_ride": per_ride_detail}


def event_metrics(meta: pd.DataFrame, pred: np.ndarray, conf: np.ndarray,
                  events_true: dict, cfg: PipelineConfig, tol_s: float = 1.0,
                  classes: tuple[str, ...] = POINT_CLASSES) -> dict:
    """Event recall (±tol_s) and false alarms per km for Pothole / Irregularity, computed on
    out-of-fold predictions of the evaluated (moving, labelled) windows."""
    w = meta.assign(pred=pred, confidence=conf)
    hits = {c: [0, 0] for c in classes}
    fa = {c: 0 for c in classes}
    dist_m = 0.0
    for rid, g in w.groupby("ride_id"):
        truth = events_true.get(rid, [])
        if len(g):  # only labelled events inside the evaluated span count
            lo, hi = g["t_start"].min(), g["t_end"].max()
            truth = [e for e in truth if e[1] > lo and e[0] < hi]
        ev = merge_events(g, gap_s=1.0, classes=classes)
        for c in classes:
            h, n = event_recall(truth, g, c, tol_s)
            hits[c][0] += h
            hits[c][1] += n
            fa[c] += false_alarms(truth, ev, c, tol_s)
        dist_m += float(np.nansum(g["speed"].to_numpy(dtype=float)) * cfg.step_s)
    km = dist_m / 1000.0
    out = {"tolerance_s": tol_s, "distance_km": round(km, 3)}
    for c in classes:
        out[c] = {"detected": hits[c][0], "n_labelled": hits[c][1],
                  "detected_of": f"{hits[c][0]}/{hits[c][1]}",
                  "event_recall": (hits[c][0] / hits[c][1]) if hits[c][1] else None,
                  "false_alarms": fa[c],
                  "false_alarms_per_km": (fa[c] / km) if km > 0 else None}
    return out


SPEED_ONLY_F1_WARN = 0.40  # chance level for 4 balanced classes is ~0.25
SPEED_RATIO_WARN = 1.20


def speed_by_class(meta: pd.DataFrame) -> pd.DataFrame:
    g = meta.groupby("label")["speed"]
    return pd.DataFrame({"n": g.size(), "mean_mps": g.mean(), "median_mps": g.median(),
                         "std_mps": g.std()}).reindex([c for c in LABELS
                                                       if c in set(meta["label"])])


def pocket_test(full: Dataset, cfg: PipelineConfig, base: WindowTable, cvc: CVConfig | None,
                feature_set: str, model: str = "rf") -> dict | None:
    """Train on every handlebar window (+ augmentation if enabled, train side only), test on the
    labelled moving windows of the pocket ride(s). Metrics only over the classes labelled in
    pocket. None when there is no labelled pocket window."""
    if "pocket" not in set(full.meta["mount"].astype(str).str.strip()):
        return None
    try:
        pw = build_windows(select_mount(full, "pocket"), cfg)
    except ValueError:
        return None
    hb = base.meta["mount"].to_numpy() == "handlebar"
    if not hb.any():
        return None
    c = replace(cfg, feature_set=feature_set)
    cols = config_columns(list(base.X.columns), c)
    tr = np.flatnonzero(hb)
    X_tr, y_tr = base.X.iloc[tr][cols], base.meta["label"].to_numpy()[tr]
    if cvc is not None and (cvc.augment or cvc.extra_train):
        Xe, ye, _ = extra_training(base, c, cvc, tr, np.array([], int), 0)
        if len(ye):
            X_tr = pd.concat([X_tr, Xe[cols]], ignore_index=True)
            y_tr = np.concatenate([y_tr, ye])
    m = make_model(model).fit(X_tr, y_tr)
    y_te = pw.meta["label"].to_numpy()
    pred = m.predict(pw.X[cols])
    present = [k for k in LABELS if (y_te == k).any()]
    per_class = {}
    for k in present:
        m_k = y_te == k
        per_class[k] = {"windows_correct": int((pred[m_k] == k).sum()),
                        "n_windows": int(m_k.sum()),
                        "predicted_as": {c: int((pred[m_k] == c).sum()) for c in LABELS}}
    ev = {}
    w = pw.meta.assign(pred=pred)
    for k in present:
        if k in POINT_CLASSES:
            h = n = 0
            for rid, g in w.groupby("ride_id"):
                hh, nn = event_recall(pw.events_true.get(rid, []), g, k, 1.0)
                h, n = h + hh, n + nn
            ev[k] = f"{h}/{n}"
    # Deliberately NO overall score: only the classes labelled in pocket are meaningful.
    return {"classes_in_pocket": present, "n_windows": int(len(y_te)), "per_class": per_class,
            "events_detected": ev,
            "note": "Only the classes labelled in pocket are meaningful; no overall score."}


IMPACT = "Impact"
THREE_CLASSES = ("Smooth", "Rough", IMPACT)


def to_three(labels) -> np.ndarray:
    """Pothole / Irregularity -> Impact (DIAGNOSTIC only; the product keeps the 4 classes)."""
    a = np.asarray(labels, dtype=object)
    return np.where(np.isin(a, POINT_CLASSES), IMPACT, a)


def three_class_diagnostic(wt: WindowTable, cfg: PipelineConfig, cvc: CVConfig | None,
                           model: str = "rf") -> dict:
    """Same folds as the 4-class run (splits computed on the 4-class labels), RF trained and
    scored on Smooth / Rough / Impact. Shows what the data can support; never shipped."""
    cvc = cvc or CVConfig()
    meta = wt.meta
    cols = config_columns(list(wt.X.columns), cfg)
    sp = make_splits(meta, cfg.window_s, cvc.n_splits_max, cvc.cv, cvc.block_s, cvc.purge_s)
    X, y = wt.X[cols], to_three(meta["label"].to_numpy())
    pred = np.empty(len(y), dtype=object)
    conf = np.full(len(y), np.nan)
    folds = []
    for k, (tr, te) in enumerate(sp.folds):
        Xe, ye, _ = extra_training(wt, cfg, cvc, tr, te, k)
        X_tr = pd.concat([X.iloc[tr], Xe[cols]], ignore_index=True) if len(ye) else X.iloc[tr]
        y_tr = np.concatenate([y[tr], to_three(ye)]) if len(ye) else y[tr]
        m = make_model(model).fit(X_tr, y_tr)
        P = m.predict_proba(X.iloc[te])
        pred[te] = m.classes_[P.argmax(axis=1)]
        conf[te] = P.max(axis=1)
        present = [c for c in THREE_CLASSES if (y[te] == c).any()]
        folds.append(float(f1_score(y[te], pred[te], labels=present, average="macro",
                                    zero_division=0)))
    pred = pred.astype(str)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(THREE_CLASSES),
                                                 zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(THREE_CLASSES))
    ev_true = {rid: [(a, b, IMPACT) for a, b, _ in evs] for rid, evs in wt.events_true.items()}
    return {"classes": list(THREE_CLASSES), "macro_f1": float(f1_score(
                y, pred, labels=list(THREE_CLASSES), average="macro", zero_division=0)),
            "macro_f1_folds": _ms(folds),
            "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]),
                              "f1": float(f[i]), "support": int(s[i])}
                          for i, c in enumerate(THREE_CLASSES)},
            "confusion": {"labels": list(THREE_CLASSES), "counts": cm.tolist()},
            "events": event_metrics(meta, pred, conf, ev_true, cfg, classes=(IMPACT,)),
            "note": "Diagnostic only: Pothole and Irregularity merged into Impact. The product "
                    "and the final model keep the 4 plan classes."}


def run_ablations(ds: Dataset, cfg: PipelineConfig, base: WindowTable,
                  base_rf: ModelResult | None = None, model: str = "rf",
                  cvc: CVConfig | None = None, mount: str = "all") -> pd.DataFrame:
    """RandomForest experiments, same CV protocol as the main run:
    (i) speed features vs none (+ speed-only diagnostic and per-class speed table),
    (ii) invariant vs invariant+fixed, (iii) 1 s vs 2 s windows, (iv) train on handlebar,
    test on pocket (only the classes labelled in pocket). `ds` is the FULL dataset (pocket
    included); `base` the windows of the main run. Extra info in `df.attrs`."""
    rows = []
    notes: list[str] = []
    for c in POINT_CLASSES:
        n_ev = sum(1 for evs in base.events_true.values() for e in evs if e[2] == c)
        if n_ev < 10:
            notes.append(f"{c} has fewer than 10 labelled events ({n_ev}): every {c} result "
                         f"is anecdotal; counts are reported as k/n, never as rates.")

    def add(name: str, c: PipelineConfig, wt: WindowTable, r: ModelResult | None = None,
            columns: list[str] | None = None):
        if r is None:
            res, _, _ = cross_validate(wt, c, models=(model,), cvc=cvc, columns=columns)
            r = res[model]
        s = r.summary
        rows.append({"ablation": name, "feature_set": c.feature_set, "window_s": c.window_s,
                     "speed_features": c.include_speed, "n_windows": len(wt.meta),
                     "macro_f1": s["macro_f1"], "macro_f1_fold_mean": s["macro_f1_folds"]["mean"],
                     "macro_f1_fold_std": s["macro_f1_folds"]["std"],
                     "pothole_events": s["events"]["Pothole"]["detected_of"],
                     "irregularity_events": s["events"]["Irregularity"]["detected_of"],
                     **{f"f1_{k}": s["per_class"][k]["pooled"]["f1"] for k in LABELS}})
        return r

    c_inv = replace(cfg, feature_set="invariant")
    reuse = base_rf if cfg.feature_set == "invariant" else None
    sp_txt = "speed" if c_inv.include_speed else "no speed"
    add(f"baseline (invariant, {cfg.window_s:g} s, {sp_txt})", c_inv, base, reuse)
    # (i) speed: the other option
    if c_inv.include_speed:
        add("(i) WITHOUT speed features", replace(c_inv, include_speed=False), base)
    else:
        add("(i) WITH speed features", replace(c_inv, include_speed=True), base)
    r_sp = add("(i) speed features ONLY (confound diagnostic)", c_inv, base,
               columns=[k for k in ("speed", "speed_missing") if k in base.X.columns])
    spd = speed_by_class(base.meta)
    ratio = float(spd["mean_mps"].max() / max(spd["mean_mps"].min(), 1e-6)) if len(spd) else 1.0
    if r_sp.summary["macro_f1"] > SPEED_ONLY_F1_WARN or ratio > SPEED_RATIO_WARN:
        notes.append(f"CONFOUNDING WARNING: speed alone gives macro-F1 "
                     f"{r_sp.summary['macro_f1']:.3f} (chance ~0.25) and class mean speeds differ "
                     f"by x{ratio:.2f}: part of what the model learns may be 'which recording / "
                     f"which speed', not the road. Compare with the no-speed row.")
    # (ii) fixed axes
    add("(ii) invariant+fixed", replace(c_inv, feature_set="invariant+fixed"), base)
    # (iii) window length
    c2 = replace(c_inv, window_s=2.0)
    add("(iii) 2 s windows", c2, build_windows(select_mount(ds, mount), c2, preps=base.preps))
    # mount subsets of the baseline OOF (only meaningful when several mounts are in the CV)
    mounts = base.meta["mount"]
    if mounts.nunique() > 1 and base_rf is not None:
        y = base.meta["label"].to_numpy()
        for m in sorted(mounts.unique()):
            ii = np.flatnonzero(mounts.to_numpy() == m)
            rows.append({"ablation": f"mount={m} (subset of baseline OOF)",
                         "feature_set": "invariant", "window_s": cfg.window_s,
                         "speed_features": True, "n_windows": len(ii),
                         "macro_f1": float(f1_score(y[ii], base_rf.oof_pred[ii],
                                                    labels=list(LABELS), average="macro",
                                                    zero_division=0))})
    # 3-class diagnostic (Smooth / Rough / Impact): what the data can support. NOT the product.
    three = three_class_diagnostic(base, c_inv, cvc, model)
    rows.append({"ablation": "DIAGNOSTIC 3 classes (Smooth/Rough/Impact=Pothole+Irregularity)",
                 "feature_set": c_inv.feature_set, "window_s": c_inv.window_s,
                 "speed_features": c_inv.include_speed, "n_windows": len(base.meta),
                 "macro_f1": three["macro_f1"],
                 "macro_f1_fold_mean": three["macro_f1_folds"]["mean"],
                 "macro_f1_fold_std": three["macro_f1_folds"]["std"],
                 "impact_events": three["events"]["Impact"]["detected_of"]})
    # (iv) orientation test: handlebar -> pocket
    pocket = {}
    for fs_name in ("invariant", "invariant+fixed"):
        res = pocket_test(ds, c_inv, base, cvc, fs_name, model)
        if res is None:
            notes.append("(iv) pocket test skipped: no labelled moving pocket window or no "
                         "handlebar training window.")
            break
        pocket[fs_name] = res
        rows.append({"ablation": f"(iv) train handlebar -> test pocket ({fs_name}); ONLY "
                                 f"{'/'.join(res['classes_in_pocket'])} labelled, no overall "
                                 f"score",
                     "feature_set": fs_name, "window_s": cfg.window_s,
                     "speed_features": c_inv.include_speed, "n_windows": res["n_windows"],
                     "pothole_events": res["events_detected"].get("Pothole"),
                     "irregularity_events": res["events_detected"].get("Irregularity")})
    df = pd.DataFrame(rows)
    df.attrs["speed_by_class"] = spd.round(3).reset_index().to_dict(orient="records")
    df.attrs["pocket"] = pocket
    df.attrs["three_class"] = three
    df.attrs["notes"] = notes
    for n in notes:
        print(n, file=sys.stderr)
    return df


def training_provenance(base: str, cvc: CVConfig | None) -> str:
    """real | mixed | synthetic. Real data + augmentation or synthetic windows = mixed."""
    if base == "real" and cvc is not None and (cvc.augment or cvc.extra_train):
        return "mixed"
    return base


def dataset_summary(ds: Dataset, wt: WindowTable, cvc: CVConfig | None = None) -> dict:
    return {"data_provenance": training_provenance(ds.provenance, cvc),
            "test_data_provenance": ds.provenance, "n_rides": int(wt.meta["ride_id"].nunique()),
            "n_windows": int(len(wt.meta)), "n_windows_per_class": class_counts(wt.meta["label"]),
            "n_stationary_windows_dropped": int(sum(wt.n_stationary.values())),
            "rides_per_class": {c: int(wt.meta.loc[wt.meta["label"] == c, "ride_id"].nunique())
                                for c in LABELS},
            "mounts": {str(k): int(v) for k, v in
                       wt.meta.groupby("mount")["ride_id"].nunique().items()}}
